"""Mixin wiring the 'Pre-shift Color Check' menu action to its dialog."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox

from app.gui.color_preflight_dialog import ColorPreflightDialog
from app.gui.i18n import tr
from core.services.color_preflight_store import ColorPreflightLedger
from core.station_data import load_station_data_paths


class ColorPreflightHandlerMixin:
    """Open the pre-shift color check for the selected model.

    Relies on ``DetectionSystemGUI`` host attributes: ``_catalog``, the
    product/area/inference combos, ``current_language`` and ``log_message``.

    Unlike the illumination calibration this needs no camera and no running
    detection -- it reads an inspection the station already saved -- so the
    only precondition is a selected scope.
    """

    def _t(self, key: str, **kwargs: object) -> str:  # pragma: no cover - trivial
        text = tr(getattr(self, "current_language", "en"), key)
        return text.format(**kwargs) if kwargs else text

    def open_color_preflight_dialog(self) -> None:
        """Validate the scope and open the pre-shift color check."""
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        inference_type = self.inference_combo.currentText().strip()
        if not all([product, area, inference_type]):
            QMessageBox.warning(
                self,
                self._t("preflight_title"),
                self._t("calib_no_selection"),
            )
            return
        # Fusion's color stage is the YOLO one, so it shares that scope.
        if inference_type.lower() == "fusion":
            inference_type = "yolo"

        try:
            paths = load_station_data_paths()
            config_path = self._catalog.config_path(product, area, inference_type)
        except Exception as exc:  # noqa: BLE001
            self.log_message(self._t("preflight_unavailable", error=exc))
            QMessageBox.warning(
                self,
                self._t("preflight_title"),
                self._t("preflight_unavailable", error=exc),
            )
            return

        dialog = ColorPreflightDialog(
            config_path=config_path,
            results_root=paths.results,
            product=product,
            area=area,
            model_type=inference_type,
            ledger=ColorPreflightLedger(paths.color_preflight),
            language=getattr(self, "current_language", "en"),
            parent=self,
        )
        # Shown non-modally, and kept on the host so it is not collected: the
        # golden sample is already in the fixture, so the operator should be
        # able to press 開始檢測 with this open and watch the table update,
        # rather than close it, inspect, and reopen.
        self._color_preflight_dialog = dialog
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.destroyed.connect(self._forget_color_preflight_dialog)
        dialog.show()
        dialog.raise_()

    def _forget_color_preflight_dialog(self) -> None:
        self._color_preflight_dialog = None
