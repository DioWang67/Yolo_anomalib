"""Dialog for illumination calibration: record current values and auto-calibrate.

The heavy lifting lives in :class:`core.services.calibration_session.CalibrationSession`
(pure, tested). This dialog is a thin shell that shows live brightness, lets the
operator set a target, and drives record / auto-calibrate, persisting results to
the per-model ``config.yaml`` via an injected save function.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

from app.gui.i18n import normalize_language, tr
from core.services.auto_calibrator import (
    CalibrationOutcome,
    CalibrationPhase,
    CalibrationTarget,
)
from core.services.calibration_session import CalibrationSession

_LIVE_LUMA_INTERVAL_MS = 500
_DEFAULT_TARGET_LUMA = 128.0
_DEFAULT_TOLERANCE = 4.0
_PREVIEW_MIN_WIDTH = 360
_PREVIEW_MIN_HEIGHT = 240


def _frame_to_pixmap(frame: np.ndarray, target_size) -> QPixmap | None:
    """Convert a BGR (or grayscale) uint8 frame to a scaled QPixmap.

    Returns ``None`` on an empty/malformed frame so the caller can keep the
    previous preview instead of blanking it.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    try:
        if frame.ndim == 2:
            gray = np.ascontiguousarray(frame)
            h, w = gray.shape
            qt_image = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        # QPixmap.fromImage copies the buffer, so the source array may be freed.
        return QPixmap.fromImage(qt_image).scaled(
            target_size, Qt.KeepAspectRatio, Qt.FastTransformation
        )
    except Exception:  # noqa: BLE001 - preview must never crash the dialog
        return None


class _AutoCalibrateWorker(QThread):
    """Runs the (blocking) calibration loop off the UI thread."""

    finished_ok = pyqtSignal(object)   # CalibrationOutcome
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str, float, float)  # iteration, phase, luma, error

    def __init__(
        self,
        session: CalibrationSession,
        target: CalibrationTarget,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._target = target

    def _emit_step(
        self, iteration: int, phase: CalibrationPhase, luma: float, error: float
    ) -> None:
        self.progress.emit(iteration, phase.value, luma, error)

    def run(self) -> None:  # pragma: no cover - exercised only with hardware
        try:
            outcome = self._session.run_auto(self._target, on_step=self._emit_step)
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(outcome)


class CalibrationDialog(QDialog):
    """Record current illumination values and run brightness auto-calibration.

    Args:
        config_path: Per-model ``config.yaml`` to persist results into.
        session: Bound :class:`CalibrationSession` (camera + optional light).
        save_fn: Callable persisting settings, i.e.
            ``save_calibration_settings`` (injected for testability).
        initial_target: Pre-filled target luma (from existing config).
        initial_tolerance: Pre-filled tolerance (from existing config).
        language: UI language code.
        parent: Parent widget.
    """

    def __init__(
        self,
        *,
        config_path: Path,
        session: CalibrationSession,
        save_fn: Callable[..., object],
        initial_target: float | None = None,
        initial_tolerance: float | None = None,
        language: str = "en",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config_path = config_path
        self._session = session
        self._save_fn = save_fn
        self._language = normalize_language(language)
        self._worker: _AutoCalibrateWorker | None = None

        self.setWindowTitle(self._t("calib_title"))
        self.setMinimumWidth(380)
        self._build_ui()
        if initial_target is not None:
            self._target_spin.setValue(float(initial_target))
        if initial_tolerance is not None:
            self._tolerance_spin.setValue(float(initial_tolerance))

        self._timer = QTimer(self)
        self._timer.setInterval(_LIVE_LUMA_INTERVAL_MS)
        self._timer.timeout.connect(self._refresh_luma)
        self._timer.start()
        self._refresh_luma()

    # ------------------------------------------------------------------
    def _t(self, key: str, **kwargs: object) -> str:
        text = tr(self._language, key)
        return text.format(**kwargs) if kwargs else text

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self._preview_label = QLabel()
        self._preview_label.setMinimumSize(_PREVIEW_MIN_WIDTH, _PREVIEW_MIN_HEIGHT)
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setStyleSheet(
            "background-color: #111827; color: #9ca3af; border: 1px solid #374151;"
        )
        self._preview_label.setText(self._t("calib_preview_waiting"))
        layout.addWidget(self._preview_label)

        self._sample_hint = QLabel(self._t("calib_sample_hint"))
        self._sample_hint.setWordWrap(True)
        self._sample_hint.setStyleSheet("color: #6b7280; font-size: 8pt;")
        layout.addWidget(self._sample_hint)

        form = QFormLayout()
        self._luma_label = QLabel("--")
        form.addRow(self._t("calib_current_luma"), self._luma_label)

        self._target_spin = QDoubleSpinBox()
        self._target_spin.setRange(0.0, 255.0)
        self._target_spin.setDecimals(1)
        self._target_spin.setValue(_DEFAULT_TARGET_LUMA)
        form.addRow(self._t("calib_target_luma"), self._target_spin)

        self._tolerance_spin = QDoubleSpinBox()
        self._tolerance_spin.setRange(0.5, 64.0)
        self._tolerance_spin.setDecimals(1)
        self._tolerance_spin.setValue(_DEFAULT_TOLERANCE)
        form.addRow(self._t("calib_tolerance"), self._tolerance_spin)
        layout.addLayout(form)

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Per-step convergence log, shown only while auto-calibrating.
        self._progress_log = QPlainTextEdit()
        self._progress_log.setReadOnly(True)
        self._progress_log.setMaximumHeight(96)
        self._progress_log.setVisible(False)
        layout.addWidget(self._progress_log)

        buttons = QHBoxLayout()
        self._record_btn = QPushButton(self._t("calib_record_btn"))
        self._record_btn.clicked.connect(self._on_record)
        self._auto_btn = QPushButton(self._t("calib_auto_btn"))
        self._auto_btn.clicked.connect(self._on_auto)
        buttons.addWidget(self._record_btn)
        buttons.addWidget(self._auto_btn)
        layout.addLayout(buttons)

        close_box = QDialogButtonBox(QDialogButtonBox.Close)
        close_box.button(QDialogButtonBox.Close).setText(self._t("calib_close_btn"))
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)

    # ------------------------------------------------------------------
    def _current_target(self) -> CalibrationTarget:
        return CalibrationTarget(
            target_luma=float(self._target_spin.value()),
            tolerance=float(self._tolerance_spin.value()),
        )

    def _refresh_luma(self) -> None:
        try:
            frame, luma = self._session.capture_and_measure()
        except Exception:  # noqa: BLE001 - live poll must never crash the dialog
            frame, luma = None, None
        self._luma_label.setText("--" if luma is None else f"{luma:.1f}")
        self._update_preview(frame)

    def _update_preview(self, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        pixmap = _frame_to_pixmap(frame, self._preview_label.size())
        if pixmap is not None:
            self._preview_label.setPixmap(pixmap)

    def _set_busy(self, busy: bool) -> None:
        self._record_btn.setEnabled(not busy)
        self._auto_btn.setEnabled(not busy)
        if busy:
            self._timer.stop()
        else:
            self._timer.start()

    # ------------------------------------------------------------------
    def _on_record(self) -> None:
        try:
            snapshot = self._session.record_current()
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(self._t("calib_error", error=exc))
            return

        self._target_spin.setValue(float(snapshot["target_luma"]))
        try:
            result = self._save_fn(
                self._config_path,
                exposure_time=snapshot["exposure_time"],
                gain=snapshot["gain"],
                light_brightness=snapshot["light_brightness"],
                target_luma=snapshot["target_luma"],
                tolerance=float(self._tolerance_spin.value()),
            )
        except Exception as exc:  # noqa: BLE001
            self._status_label.setText(self._t("calib_error", error=exc))
            return

        self._status_label.setText(
            self._t(
                "calib_recorded",
                exposure=snapshot["exposure_time"],
                gain=snapshot["gain"],
                light=snapshot["light_brightness"],
                luma=snapshot["target_luma"],
            )
            + "\n"
            + self._t("calib_saved", backup=getattr(result, "backup_path", ""))
        )

    def _on_auto(self) -> None:
        self._set_busy(True)
        self._status_label.setText(self._t("calib_running"))
        self._progress_log.clear()
        self._progress_log.setVisible(True)
        worker = _AutoCalibrateWorker(self._session, self._current_target(), self)
        worker.finished_ok.connect(self._on_auto_finished)
        worker.failed.connect(self._on_auto_failed)
        worker.progress.connect(self._on_auto_progress)
        worker.finished.connect(lambda: self._set_busy(False))
        self._worker = worker
        worker.start()

    def _on_auto_progress(
        self, iteration: int, phase: str, luma: float, error: float
    ) -> None:
        """Append one convergence step; shrinking error means it is converging."""
        phase_label = self._t(f"calib_phase_{phase}")
        self._progress_log.appendPlainText(
            self._t(
                "calib_step",
                step=iteration,
                phase=phase_label,
                luma=f"{luma:.1f}",
                target=f"{self._target_spin.value():.0f}",
                error=f"{error:.1f}",
            )
        )

    def _on_auto_finished(self, outcome: CalibrationOutcome) -> None:
        if not outcome.success:
            message = self._t(
                "calib_failed",
                reason=outcome.reason.value,
                luma=f"{outcome.final_luma:.1f}",
            )
            self._status_label.setText(message)
            QMessageBox.warning(self, self._t("calib_title"), message)
            return
        try:
            snapshot = self._session.snapshot_after_run()
            result = self._save_fn(
                self._config_path,
                exposure_time=snapshot["exposure_time"],
                gain=snapshot["gain"],
                light_brightness=snapshot["light_brightness"],
                target_luma=float(self._target_spin.value()),
                tolerance=float(self._tolerance_spin.value()),
            )
        except Exception as exc:  # noqa: BLE001
            message = self._t("calib_error", error=exc)
            self._status_label.setText(message)
            QMessageBox.critical(self, self._t("calib_title"), message)
            return
        message = (
            self._t(
                "calib_success",
                luma=f"{outcome.final_luma:.1f}",
                iterations=outcome.iterations,
            )
            + "\n"
            + self._t("calib_saved", backup=getattr(result, "backup_path", ""))
        )
        self._status_label.setText(message)
        QMessageBox.information(self, self._t("calib_title"), message)

    def _on_auto_failed(self, message: str) -> None:
        text = self._t("calib_error", error=message)
        self._status_label.setText(text)
        QMessageBox.critical(self, self._t("calib_title"), text)

    def reject(self) -> None:  # noqa: D102 - ensure timer/worker cleanup
        self._timer.stop()
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.wait(5000)
        super().reject()
