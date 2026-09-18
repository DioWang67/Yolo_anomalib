"""Re-deriving the exposure when nobody asked, because the light moved.

This station is lit by daylight and room fluorescents, so the exposure that
was right yesterday is wrong today. Measured on Cable1/A over one working
day: 20149 us at 15:30, 22006 us at 16:02 --- 9.2% in half an hour. The
operator's workaround was to open 光源校正 and run it by hand after every
restart. Nothing was failing to persist; the light had genuinely changed.
This runs the same loop for them.

**It moves brightness, not colour.** Over that day's pre-shift checks the
error decomposed into 28.9 of lightness against 262.3 of chroma --- nine
parts in ten sit in a*/b*, where exposure has no reach at all, because what
is moving is the spectrum as daylight and fluorescent trade places. A
shroud and a controlled LED are the fix for that. This removes a manual
step and the tenth part; it is not a colour control.

The result is deliberately **not** written to the station's ``config.yaml``.
Under uncontrolled light the right exposure is a measurement of today rather
than a setting, and that file keeps its meaning as the last exposure a
person deliberately chose --- which is also why a human calibration clears
whatever this arrived at.

Only stations that already record ``calibration.target_luma`` take part.
There is no toggle and no new config key: a station nobody has ever
calibrated has no target to drive toward, and inventing one would be this
module deciding how bright a station it has never seen ought to be.
"""

from __future__ import annotations

from PyQt5.QtCore import QTimer

from app.gui.calibration_dialog import AutoCalibrateWorker
from app.gui.i18n import tr
from core.detection_system import calibration_scope
from core.services.auto_calibrator import CalibrationTarget
from core.services.calibration_session import CalibrationSession

#: Long enough to swallow the product -> area -> inference cascade, which
#: fires three scope changes for one operator action, short enough that the
#: loop is over before anyone reaches for 開始檢測.
_AUTOCALIB_DEBOUNCE_MS = 400

#: The loop is bounded at 20 iterations of roughly 0.2s, so this is slack
#: over the worst case rather than a guess.
_AUTOCALIB_SHUTDOWN_WAIT_MS = 5000


class AutoCalibrationHandlerMixin:
    """Run the illumination loop at camera-ready and on station switch.

    Relies on ``DetectionSystemGUI`` host attributes: ``controller``, the
    product/area/inference combos, ``statusBar``, ``log_message``,
    ``is_detection_running``, ``_logger``, and two mixins it must follow in
    the MRO --- ``CalibrationHandlerMixin`` for :meth:`_active_camera` and
    :meth:`_existing_calibration`, and ``LightHandlerMixin`` for the LED
    keepalive.

    Every entry point is written so that failing is quiet. This runs without
    anybody asking for it, so it may never raise into a Qt slot, never open a
    modal, and never stop the station from being used: when it cannot run, or
    does not converge, the station is left exactly as it behaves today.
    """

    def _t(self, key: str, **kwargs: object) -> str:  # pragma: no cover - trivial
        text = tr(getattr(self, "current_language", "en"), key)
        return text.format(**kwargs) if kwargs else text

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def _maybe_autocalibrate(self, reason: str = "") -> None:
        """Ask for a calibration; the guards decide whether one happens.

        Called from two independent async callbacks --- the camera finishing
        initialisation and the station combos being filled --- which can
        arrive in either order and neither of which is sufficient alone. Made
        idempotent and self-guarding rather than sequenced, so whichever
        arrives second is the one that runs.
        """
        try:
            timer = getattr(self, "_autocalib_timer", None)
            if timer is None:
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self._run_scheduled_autocalibration)
                self._autocalib_timer = timer
            self._autocalib_reason = reason
            # Re-arming an armed timer is what collapses the cascade.
            timer.start(_AUTOCALIB_DEBOUNCE_MS)
        except Exception as exc:  # noqa: BLE001 - never raise into a Qt slot
            self._logger.debug("Auto-calibration could not be scheduled: %s", exc)

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------
    def _current_calibration_scope(self) -> tuple[str, str, str] | None:
        """The selected station, or None while the combos are still empty."""
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        inference_type = self.inference_combo.currentText().strip()
        if not all([product, area, inference_type]):
            return None
        return calibration_scope(product, area, inference_type)

    def _run_scheduled_autocalibration(self) -> None:
        try:
            self._run_scheduled_autocalibration_guarded()
        except Exception as exc:  # noqa: BLE001 - never raise into a Qt slot
            self.log_message(self._t("autocalib_error", error=exc))

    def _run_scheduled_autocalibration_guarded(self) -> None:
        if getattr(self, "_autocalib_worker", None) is not None:
            return
        if getattr(self, "_closing", False):
            return
        # The camera has no lock; the whole codebase serialises on this.
        if self.is_detection_running():
            return
        scope = self._current_calibration_scope()
        if scope is None:
            return
        if not self.controller.has_system():
            return
        camera = self._active_camera()
        if camera is None:
            # Checked before the scope is ever marked, so an early station
            # change cannot consume the run that camera-ready will ask for.
            return

        if scope != getattr(self, "_last_autocalibrated_scope", None):
            # Stop lighting this station with a number measured at another
            # one, now rather than at the next inspection.
            self.controller.clear_runtime_exposure_override()
        else:
            return

        target, tolerance = self._existing_calibration(*scope)
        if target is None:
            # The entire rollout gate. A station with no recorded target has
            # never been calibrated, and this is not the thing to decide how
            # bright it should be.
            self._last_autocalibrated_scope = scope
            self.log_message(self._t("autocalib_skipped_no_target"))
            return

        self._start_autocalibration(scope, camera, target, tolerance)

    # ------------------------------------------------------------------
    # The run
    # ------------------------------------------------------------------
    def _start_autocalibration(self, scope, camera, target, tolerance) -> None:
        light = getattr(self, "_light_controller", None)
        session = CalibrationSession(
            camera, light if (light and light.is_open) else None
        )
        # The keepalive would re-send the stored brightness mid-loop and
        # overwrite what the calibrator just set -- same reason
        # open_calibration_dialog suspends it.
        self._stop_light_keepalive()
        # Whatever the loop leaves on the hardware, the next apply re-pushes
        # the effective value rather than skipping on a stale match.
        self.controller.invalidate_applied_camera_settings()

        calibration_target = (
            CalibrationTarget(target_luma=float(target))
            if tolerance is None
            else CalibrationTarget(
                target_luma=float(target), tolerance=float(tolerance)
            )
        )
        worker = AutoCalibrateWorker(session, calibration_target, self)
        worker.finished_ok.connect(self._on_autocalibration_finished)
        worker.failed.connect(self._on_autocalibration_failed)
        worker.progress.connect(self._on_autocalibration_progress)
        worker.finished.connect(self._finish_autocalibration)
        self._autocalib_worker = worker
        self._autocalib_scope = scope
        self.update_start_enabled()
        self.statusBar().showMessage(self._t("autocalib_running"))
        worker.start()

    def _on_autocalibration_progress(self, iteration, phase, luma, error) -> None:
        # Log only. A status bar that redraws twenty times in four seconds
        # reads as a fault rather than as progress.
        self._logger.debug(
            "auto-calibration step %s (%s): luma %.1f, error %.1f",
            iteration, phase, luma, error,
        )

    def _on_autocalibration_finished(self, outcome) -> None:
        scope = getattr(self, "_autocalib_scope", None)
        # Marked whatever the outcome: a station that cannot converge must not
        # re-run the loop every time the combos repaint.
        self._last_autocalibrated_scope = scope

        if not getattr(outcome, "success", False):
            self.log_message(
                self._t(
                    "autocalib_not_converged",
                    reason=getattr(outcome.reason, "value", outcome.reason),
                    luma=f"{outcome.final_luma:.1f}",
                )
            )
            self.statusBar().showMessage(
                self._t(
                    "autocalib_not_converged",
                    reason=getattr(outcome.reason, "value", outcome.reason),
                    luma=f"{outcome.final_luma:.1f}",
                )
            )
            self._restore_station_light(scope)
            return

        if scope is not None:
            self.controller.set_runtime_exposure_override(
                scope, float(outcome.final_exposure)
            )
        # The loop's own LED result, not the file's: re-reading the config
        # here would undo what it just converged on.
        led = int(getattr(outcome, "final_led_brightness", 0) or 0)
        if led > 0:
            self._start_light_keepalive(led)
        else:
            self._stop_light_keepalive()

        message = self._t(
            "autocalib_success",
            luma=f"{outcome.final_luma:.1f}",
            exposure=f"{outcome.final_exposure:.0f}",
            iterations=outcome.iterations,
        )
        self.log_message(message)
        self.statusBar().showMessage(message)

    def _on_autocalibration_failed(self, message: str) -> None:
        self._last_autocalibrated_scope = getattr(self, "_autocalib_scope", None)
        text = self._t("autocalib_error", error=message)
        self.log_message(text)
        self.statusBar().showMessage(text)
        self._restore_station_light(self._last_autocalibrated_scope)

    def _finish_autocalibration(self) -> None:
        """Connected to QThread.finished, so it runs on every exit path."""
        self._autocalib_worker = None
        self._autocalib_scope = None
        try:
            self.update_camera_controls()
            self.update_start_enabled()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Post-calibration UI refresh failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers used by the window
    # ------------------------------------------------------------------
    def _restore_station_light(self, scope) -> None:
        """Put the configured brightness back after a run that did not land."""
        if scope is None:
            return
        try:
            self._apply_model_light_brightness(*scope)
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Could not restore station light: %s", exc)

    def _autocalibration_in_progress(self) -> bool:
        return getattr(self, "_autocalib_worker", None) is not None

    def _stop_autocalibration_for_shutdown(
        self, timeout_ms: int = _AUTOCALIB_SHUTDOWN_WAIT_MS
    ) -> bool:
        """Wait out a running loop so its thread is not destroyed mid-step."""
        worker = getattr(self, "_autocalib_worker", None)
        if worker is None:
            return True
        try:
            return bool(worker.wait(timeout_ms))
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Waiting for auto-calibration failed: %s", exc)
            return False

    def _reset_autocalibration_for_camera_change(self) -> None:
        """A different camera session has measured nothing yet."""
        self._last_autocalibrated_scope = None
        try:
            self.controller.clear_runtime_exposure_override()
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("Could not clear the session exposure: %s", exc)
