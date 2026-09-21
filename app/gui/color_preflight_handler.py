"""Mixin wiring the 'Pre-shift Color Check' menu action to its dialog."""

from __future__ import annotations

from typing import NamedTuple

from PyQt5.QtWidgets import QMessageBox

from app.gui.golden_sample_dialog import GoldenSampleDialog
from app.gui.i18n import tr
from core.services.color_preflight_store import ColorPreflightLedger
from core.services.golden_sample import GoldenSampleError
from core.station_data import load_station_data_paths


class ReadinessCheck(NamedTuple):
    """One station precondition, phrased for the person who has to fix it."""

    label: str
    ok: bool
    hint: str


def golden_sample_readiness(host, product, area, model_type) -> list[ReadinessCheck]:
    """The preconditions for triggering a golden sample capture.

    Single source for both the dialog's readiness panel and the refusal in
    :meth:`_capture_golden_sample`. Keeping them apart let the panel say ready
    while the capture refused -- and the operator only found out after
    committing to a run.
    """
    current_type = host.inference_combo.currentText().strip().lower()
    if current_type == "fusion":
        current_type = "yolo"
    scope = (
        host.product_combo.currentText().strip(),
        host.area_combo.currentText().strip(),
        current_type,
    )
    auto = getattr(host, "_auto_controller", None)
    return [
        ReadinessCheck(
            "站點一致",
            scope == (product, area, model_type.lower()),
            "主畫面已切換到其他站點，請切回原站點",
        ),
        ReadinessCheck(
            "生產自動模式已停止",
            not (auto is not None and auto.is_running()),
            "請先停止生產自動模式",
        ),
        ReadinessCheck(
            "相機輸入",
            host.use_camera_chk.isChecked(),
            "請啟用相機輸入；不能用同一張檔案模擬多次取樣",
        ),
        ReadinessCheck(
            "檢測系統已初始化",
            host.controller.has_system(),
            "請先完成相機／檢測系統初始化",
        ),
        _illumination_check(host, (product, area, model_type.lower())),
    ]


def _illumination_check(host, scope) -> ReadinessCheck:
    """Whether illumination has settled into this station's luma band.

    Without this the other four can all be green while auto-calibration is
    still stepping — or has already given up — and the check then measures a
    golden sample at the wrong brightness. It fails, and it fails as "position
    N colour deviation", which sends the operator to inspect the board and the
    colour model instead of the light.

    Reads the recorded outcome rather than measuring: this runs on a 500 ms
    GUI-thread poll, where grabbing a frame would both stall the interface and
    contend with the camera.

    Never blocks on an unknown. A station that has not calibrated may simply
    have no target recorded, and refusing the daily check on a missing reading
    would stop a line to protect it from a measurement nobody asked for.
    """
    if getattr(host, "_autocalib_worker", None) is not None:
        return ReadinessCheck(
            "照明已收斂", False, "照明自動校正進行中，請等待完成"
        )

    status = getattr(host, "_last_autocalibration_status", None)
    if status is None or status.scope != scope:
        return ReadinessCheck("照明已收斂", True, "")
    if status.converged:
        return ReadinessCheck("照明已收斂", True, "")

    if status.final_luma is None or status.target_luma is None:
        detail = f"自動校正未完成（{status.reason}）"
    else:
        detail = (
            f"目前亮度 {status.final_luma:.1f}，目標 {status.target_luma:.1f}"
            f" ± {status.tolerance:.1f}"
            f"（偏離 {status.final_luma - status.target_luma:+.1f}，{status.reason}）"
        )
    return ReadinessCheck(
        "照明已收斂",
        False,
        f"{detail}。請先讓照明穩定或重跑照明校正，否則量到的是亮度問題而非顏色問題",
    )


class ColorPreflightHandlerMixin:
    """Open the pre-shift color check for the selected model.

    Relies on ``DetectionSystemGUI`` host attributes: ``_catalog``, the
    product/area/inference combos, ``current_language``, ``log_message`` and
    ``controller`` (for the camera values in force that the station config
    does not record).

    Opening requires only a selected scope. Once armed, the dialog collects
    new inspections triggered through the station's normal detection path.
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

        existing = getattr(self, "_color_preflight_dialog", None)
        if existing is not None:
            if existing._scope == (product, area, inference_type) or existing._worker is not None:
                existing.show()
                existing.raise_()
                existing.activateWindow()
                if existing._scope != (product, area, inference_type):
                    QMessageBox.warning(
                        self,
                        self._t("preflight_title"),
                        "另一站點的開線檢查仍在收集，請先完成或取消該次收集。",
                    )
                return
            # Not deleted until its threads have stopped. deleteLater destroys
            # the dialog's children, and a GoldenPreviewWorker still running
            # when its QThread is destroyed takes the process down with it --
            # reachable here because the retired scope is only known to have
            # no capture worker, not to have no preview workers.
            existing.hide()
            if existing.prepare_shutdown():
                existing.deleteLater()
            else:
                # Still working after the bounded wait. Left parented to the
                # window instead: an orphaned hidden dialog costs memory until
                # the window closes, destroying a live thread costs the shift.
                self.log_message(
                    "前一站點的開線檢查預覽仍在收尾，已隱藏而未銷毀"
                )

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

        dialog = GoldenSampleDialog(
            config_path=config_path,
            results_root=paths.results,
            product=product,
            area=area,
            model_type=inference_type,
            ledger=ColorPreflightLedger(paths.color_preflight),
            language=getattr(self, "current_language", "en"),
            request_capture_fn=lambda: self._capture_golden_sample(product, area, inference_type),
            readiness_fn=lambda: self._golden_sample_readiness(product, area, inference_type),
            authorize_fn=self._authorize_golden_sample_maintenance,
            # The exposure actually in force, which the station config does not
            # record once a session auto-calibration has moved it.
            observed_override_fn=self.controller.effective_camera_settings,
            parent=self,
        )
        # Shown non-modally, and kept on the host so it is not collected: the
        # golden sample is already in the fixture, so the operator should be
        # able to press 開始檢測 with this open and watch the table update,
        # rather than close it, inspect, and reopen.
        self._color_preflight_dialog = dialog
        dialog.destroyed.connect(lambda: self._forget_color_preflight_dialog(dialog))
        dialog.show()
        dialog.raise_()

    def _forget_color_preflight_dialog(self, dialog=None) -> None:
        # A retired scope can be deleted after the replacement has been opened.
        if dialog is None or getattr(self, "_color_preflight_dialog", None) is dialog:
            self._color_preflight_dialog = None

    def _golden_sample_readiness(self, product, area, model_type):
        return golden_sample_readiness(self, product, area, model_type)

    def _authorize_golden_sample_maintenance(self) -> bool:
        """Gate baseline creation behind the station's engineering PIN.

        Replacing a baseline is an engineering decision -- an operator who can
        reach it can quietly promote the day's drift into the new normal.
        """
        return self.control_panel.authorize_engineering_action()

    def _capture_golden_sample(self, product, area, model_type) -> bool:
        for check in golden_sample_readiness(self, product, area, model_type):
            if not check.ok:
                raise GoldenSampleError(f"{check.label}：{check.hint}")
        if self.is_detection_running():
            return False
        self.start_detection(golden_sample=True)
        if not self.is_detection_running():
            raise GoldenSampleError("單次檢測未成功啟動，請檢查主畫面的錯誤訊息")
        return True
