from __future__ import annotations

"""Mixin wiring the 'Illumination Calibration' menu action to its dialog."""

from PyQt5.QtWidgets import QDialog, QMessageBox

from app.gui.calibration_dialog import CalibrationDialog
from app.gui.i18n import tr
from core.services.calibration_session import CalibrationSession
from core.services.model_config_editor import save_calibration_settings


class CalibrationHandlerMixin:
    """Open the illumination calibration dialog for the selected model.

    Relies on ``DetectionSystemGUI`` host attributes: ``controller``,
    ``_catalog``, the product/area/inference combos, ``current_language``,
    ``log_message``, and the light controller helpers from
    ``LightHandlerMixin`` (``_light_controller`` / ``_ensure_light_controller``).
    """

    def _t(self, key: str, **kwargs: object) -> str:  # pragma: no cover - trivial
        text = tr(getattr(self, "current_language", "en"), key)
        return text.format(**kwargs) if kwargs else text

    def open_calibration_dialog(self) -> None:
        """Validate preconditions and open the calibration dialog."""
        if self.is_detection_running():
            QMessageBox.warning(
                self, self._t("calib_title"), self._t("calib_no_camera")
            )
            return

        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        inference_type = self.inference_combo.currentText().strip()
        if not all([product, area, inference_type]):
            QMessageBox.warning(self, self._t("calib_title"), self._t("calib_no_camera"))
            return
        if inference_type.lower() == "fusion":
            inference_type = "yolo"

        camera = self._active_camera()
        if camera is None:
            QMessageBox.warning(
                self, self._t("calib_title"), self._t("calib_no_camera")
            )
            return

        config_path = self._catalog.config_path(product, area, inference_type)
        light = getattr(self, "_light_controller", None)
        session = CalibrationSession(
            camera, light if (light and light.is_open) else None
        )

        target, tolerance = self._existing_calibration(product, area, inference_type)
        dialog = CalibrationDialog(
            config_path=config_path,
            session=session,
            save_fn=save_calibration_settings,
            initial_target=target,
            initial_tolerance=tolerance,
            language=getattr(self, "current_language", "en"),
            parent=self,
        )
        dialog.exec_()

        # A save may have changed exposure/gain/light or the target; hot-reload.
        try:
            self.controller.reload_model_settings(product, area, inference_type)
        except Exception as exc:  # noqa: BLE001
            self.log_message(self._t("calib_error", error=exc))

    # ------------------------------------------------------------------
    def _active_camera(self):
        """Return the live CameraController, or None when unavailable."""
        controller = getattr(self, "controller", None)
        if controller is None or not controller.has_system():
            return None
        camera = getattr(controller.detection_system, "camera", None)
        if camera is None or not getattr(camera, "is_initialized", False):
            return None
        return camera

    def _existing_calibration(
        self, product: str, area: str, inference_type: str
    ) -> tuple[float | None, float | None]:
        """Read the model config's saved target/tolerance, if any."""
        try:
            from core.services.model_config_editor import load_model_config

            config_path = self._catalog.config_path(product, area, inference_type)
            cfg = load_model_config(config_path)
        except Exception:  # noqa: BLE001
            return None, None
        calibration = cfg.get("calibration")
        if not isinstance(calibration, dict):
            return None, None
        target = calibration.get("target_luma")
        tolerance = calibration.get("tolerance")
        return (
            float(target) if target is not None else None,
            float(tolerance) if tolerance is not None else None,
        )
