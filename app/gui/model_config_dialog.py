from __future__ import annotations

"""Dialog for editing common per-model settings without opening YAML."""

from pathlib import Path
from typing import Any

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import normalize_language
from core.services.model_config_editor import load_model_config


class ModelConfigDialog(QDialog):
    """Edit common model config fields for one product/area/type."""

    def __init__(
        self,
        *,
        product: str,
        area: str,
        inference_type: str,
        config_path: Path,
        language: str = "en",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.product = product
        self.area = area
        self.inference_type = inference_type
        self.config_path = config_path
        self.language = normalize_language(language)
        self._config = load_model_config(config_path)
        self.setWindowTitle(
            f"{self._l('Edit Model Config', '編輯機種設定')} - "
            f"{product}/{area}/{inference_type}"
        )
        self.setMinimumWidth(560)
        self.resize(640, 720)
        self._build_ui()
        self._load_values()

    def _l(self, en: str, zh: str) -> str:
        """Return local dialog text for the active language."""
        return zh if self.language == "zh" else en

    def changes(self) -> dict[str, Any]:
        """Return normalized values from the form."""
        return {
            "weights": self.weights_edit.text().strip(),
            "device": self.device_combo.currentText().strip(),
            "conf_thres": self.conf_spin.value(),
            "iou_thres": self.iou_spin.value(),
            "imgsz": [self.imgsz_w_spin.value(), self.imgsz_h_spin.value()],
            "timeout": self.timeout_spin.value(),
            "output_dir": self.output_dir_edit.text().strip(),
            "enable_yolo": self.enable_yolo_chk.isChecked(),
            "enable_anomalib": self.enable_anomalib_chk.isChecked(),
            "enable_color_check": self.enable_color_chk.isChecked(),
            "color_model_path": self.color_model_edit.text().strip() or None,
            "color_checker_type": self.color_checker_combo.currentText().strip(),
            "color_score_threshold": self.color_score_spin.value(),
            "position_check_enabled": self.position_check_chk.isChecked(),
            "position_mode": self.position_mode_combo.currentText().strip(),
            "position_tolerance": self.position_tolerance_spin.value(),
            "position_tolerance_unit": self.position_unit_combo.currentText().strip(),
            "position_alignment_enabled": self.position_alignment_chk.isChecked(),
            "missing_slot_check_enabled": self.missing_slot_chk.isChecked(),
            "count_check_strict": self.count_check_strict_chk.isChecked(),
            "fail_on_unexpected": self.fail_on_unexpected_chk.isChecked(),
            "save_original": self.save_original_chk.isChecked(),
            "save_processed": self.save_processed_chk.isChecked(),
            "save_annotated": self.save_annotated_chk.isChecked(),
            "save_crops": self.save_crops_chk.isChecked(),
            "save_fail_only": self.save_fail_only_chk.isChecked(),
            "expected_items": self.expected_items_edit.toPlainText(),
        }

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.addWidget(
            QLabel(f"{self._l('Config file', '設定檔')}: {self.config_path}")
        )

        model_group = QGroupBox(self._l("Model", "模型"))
        model_form = QFormLayout(model_group)
        self.weights_edit = QLineEdit()
        weights_row = QHBoxLayout()
        weights_row.addWidget(self.weights_edit)
        browse_weights_btn = QPushButton(self._l("Browse", "選擇"))
        browse_weights_btn.clicked.connect(self._browse_weights)
        weights_row.addWidget(browse_weights_btn)
        model_form.addRow(self._l("Weights", "權重檔"), weights_row)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda:0"])
        self.device_combo.setEditable(True)
        model_form.addRow(self._l("Device", "運算裝置"), self.device_combo)

        self.conf_spin = self._ratio_spin()
        self.iou_spin = self._ratio_spin()
        model_form.addRow(self._l("conf threshold", "conf 閾值"), self.conf_spin)
        model_form.addRow(self._l("iou threshold", "iou 閾值"), self.iou_spin)

        imgsz_row = QHBoxLayout()
        self.imgsz_w_spin = self._size_spin()
        self.imgsz_h_spin = self._size_spin()
        imgsz_row.addWidget(self.imgsz_w_spin)
        imgsz_row.addWidget(QLabel("x"))
        imgsz_row.addWidget(self.imgsz_h_spin)
        model_form.addRow("imgsz", imgsz_row)

        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 3600)
        model_form.addRow(self._l("timeout seconds", "timeout 秒"), self.timeout_spin)
        content_layout.addWidget(model_group)

        behavior_group = QGroupBox(self._l("Detection Behavior", "檢測行為"))
        behavior_form = QFormLayout(behavior_group)
        self.enable_yolo_chk = QCheckBox(self._l("Enable YOLO", "啟用 YOLO"))
        self.enable_anomalib_chk = QCheckBox(self._l("Enable Anomalib", "啟用 Anomalib"))
        self.enable_color_chk = QCheckBox(self._l("Enable color check", "啟用顏色檢查"))
        behavior_form.addRow(self.enable_yolo_chk)
        behavior_form.addRow(self.enable_anomalib_chk)
        behavior_form.addRow(self.enable_color_chk)

        self.color_model_edit = QLineEdit()
        color_row = QHBoxLayout()
        color_row.addWidget(self.color_model_edit)
        browse_color_btn = QPushButton(self._l("Browse", "選擇"))
        browse_color_btn.clicked.connect(self._browse_color_model)
        color_row.addWidget(browse_color_btn)
        behavior_form.addRow(self._l("Color model", "顏色模型"), color_row)

        self.color_checker_combo = QComboBox()
        self.color_checker_combo.addItems(["color_qc", "stats"])
        self.color_checker_combo.setEditable(True)
        behavior_form.addRow(
            self._l("Color checker", "顏色檢查器"), self.color_checker_combo
        )

        self.color_score_spin = self._ratio_spin()
        behavior_form.addRow(
            self._l("Color score threshold", "顏色分數閾值"),
            self.color_score_spin,
        )

        self.fail_on_unexpected_chk = QCheckBox(
            self._l(
                "Fail when unexpected classes appear",
                "出現非預期類別時判 FAIL",
            )
        )
        behavior_form.addRow(self.fail_on_unexpected_chk)

        self.count_check_strict_chk = QCheckBox(
            self._l(
                "Strict count check (extra parts also fail)",
                "數量檢查嚴格模式（多件也判 FAIL）",
            )
        )
        behavior_form.addRow(self.count_check_strict_chk)
        content_layout.addWidget(behavior_group)

        position_group = QGroupBox(self._l("Position Check", "位置檢測"))
        position_form = QFormLayout(position_group)
        self.position_check_chk = QCheckBox(self._l("Enable position check", "啟用位置檢測"))
        position_form.addRow(self.position_check_chk)

        self.position_mode_combo = QComboBox()
        self.position_mode_combo.addItems(["center", "region", "iou"])
        position_form.addRow(self._l("Decision mode", "判定模式"), self.position_mode_combo)

        tolerance_row = QHBoxLayout()
        self.position_tolerance_spin = QDoubleSpinBox()
        self.position_tolerance_spin.setRange(0.0, 10000.0)
        self.position_tolerance_spin.setSingleStep(0.1)
        self.position_tolerance_spin.setDecimals(3)
        tolerance_row.addWidget(self.position_tolerance_spin)
        self.position_unit_combo = QComboBox()
        self.position_unit_combo.addItems(["percent", "pixel"])
        tolerance_row.addWidget(self.position_unit_combo)
        position_form.addRow(self._l("Tolerance", "容許偏差"), tolerance_row)

        self.position_alignment_chk = QCheckBox(
            self._l(
                "Enable fixture shift auto-correction",
                "啟用治具偏移自動校正",
            )
        )
        self.missing_slot_chk = QCheckBox(self._l("Enable missing-slot check", "啟用缺槽補判"))
        position_form.addRow(self.position_alignment_chk)
        position_form.addRow(self.missing_slot_chk)
        content_layout.addWidget(position_group)

        output_group = QGroupBox(self._l("Output", "輸出"))
        output_form = QFormLayout(output_group)
        self.output_dir_edit = QLineEdit()
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit)
        browse_output_btn = QPushButton(self._l("Browse", "選擇"))
        browse_output_btn.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_output_btn)
        output_form.addRow(self._l("Output folder", "輸出資料夾"), output_row)

        self.save_original_chk = QCheckBox(self._l("Save original image", "儲存原圖"))
        self.save_processed_chk = QCheckBox(
            self._l("Save processed image", "儲存處理後影像")
        )
        self.save_annotated_chk = QCheckBox(self._l("Save annotated image", "儲存標註圖"))
        self.save_crops_chk = QCheckBox(self._l("Save crop images", "儲存裁切圖"))
        self.save_fail_only_chk = QCheckBox(self._l("Save FAIL only", "只存 FAIL"))
        for checkbox in (
            self.save_original_chk,
            self.save_processed_chk,
            self.save_annotated_chk,
            self.save_crops_chk,
            self.save_fail_only_chk,
        ):
            output_form.addRow(checkbox)
        content_layout.addWidget(output_group)

        items_group = QGroupBox(self._l("Expected Items", "應檢項目"))
        items_layout = QVBoxLayout(items_group)
        self.expected_items_edit = QPlainTextEdit()
        self.expected_items_edit.setPlaceholderText(
            self._l("One class per line, for example J5-1", "每行一個類別，例如 J5-1")
        )
        self.expected_items_edit.setMinimumHeight(80)
        items_layout.addWidget(self.expected_items_edit)
        content_layout.addWidget(items_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText(self._l("Save Update", "儲存更新"))
        buttons.button(QDialogButtonBox.Cancel).setText(self._l("Cancel", "取消"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_values(self) -> None:
        cfg = self._config
        self.weights_edit.setText(str(cfg.get("weights", "")))
        self.device_combo.setCurrentText(str(cfg.get("device", "cpu")))
        self.conf_spin.setValue(float(cfg.get("conf_thres", 0.25) or 0.25))
        self.iou_spin.setValue(float(cfg.get("iou_thres", 0.45) or 0.45))
        imgsz = cfg.get("imgsz") or [640, 640]
        if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
            self.imgsz_w_spin.setValue(int(imgsz[0]))
            self.imgsz_h_spin.setValue(int(imgsz[1]))
        self.timeout_spin.setValue(int(cfg.get("timeout", 2) or 0))
        self.output_dir_edit.setText(str(cfg.get("output_dir", "Result")))
        self.enable_yolo_chk.setChecked(bool(cfg.get("enable_yolo", True)))
        self.enable_anomalib_chk.setChecked(bool(cfg.get("enable_anomalib", False)))
        self.enable_color_chk.setChecked(bool(cfg.get("enable_color_check", False)))
        self.color_model_edit.setText(str(cfg.get("color_model_path", "") or ""))
        self.color_checker_combo.setCurrentText(
            str(cfg.get("color_checker_type", "color_qc") or "color_qc")
        )
        self.color_score_spin.setValue(float(cfg.get("color_score_threshold", 0.0) or 0.0))

        position_cfg = self._position_area_config()
        self.position_check_chk.setChecked(bool(position_cfg.get("enabled", False)))
        self.position_mode_combo.setCurrentText(str(position_cfg.get("mode", "center") or "center"))
        self.position_tolerance_spin.setValue(float(position_cfg.get("tolerance", 0.0) or 0.0))
        self.position_unit_combo.setCurrentText(
            str(position_cfg.get("tolerance_unit", "percent") or "percent")
        )
        alignment_cfg = position_cfg.get("alignment", {})
        self.position_alignment_chk.setChecked(
            bool(alignment_cfg.get("enabled", True)) if isinstance(alignment_cfg, dict) else True
        )
        missing_slot_cfg = position_cfg.get("missing_slot_check", {})
        self.missing_slot_chk.setChecked(
            bool(missing_slot_cfg.get("enabled", False))
            if isinstance(missing_slot_cfg, dict)
            else False
        )
        steps_cfg = cfg.get("steps", {})
        count_cfg = steps_cfg.get("count_check", {}) if isinstance(steps_cfg, dict) else {}
        self.count_check_strict_chk.setChecked(
            bool(count_cfg.get("strict", False)) if isinstance(count_cfg, dict) else False
        )
        self.fail_on_unexpected_chk.setChecked(bool(cfg.get("fail_on_unexpected", True)))
        self.save_original_chk.setChecked(bool(cfg.get("save_original", True)))
        self.save_processed_chk.setChecked(bool(cfg.get("save_processed", True)))
        self.save_annotated_chk.setChecked(bool(cfg.get("save_annotated", True)))
        self.save_crops_chk.setChecked(bool(cfg.get("save_crops", True)))
        self.save_fail_only_chk.setChecked(bool(cfg.get("save_fail_only", False)))

        expected = cfg.get("expected_items", {})
        values: list[str] = []
        if isinstance(expected, dict):
            product_map = expected.get(self.product, {})
            if isinstance(product_map, dict):
                area_values = product_map.get(self.area, [])
                if isinstance(area_values, list):
                    values = [str(item) for item in area_values]
        self.expected_items_edit.setPlainText("\n".join(values))

    def _position_area_config(self) -> dict[str, Any]:
        position_cfg = self._config.get("position_config", {})
        if not isinstance(position_cfg, dict):
            return {}
        product_cfg = position_cfg.get(self.product, {})
        if not isinstance(product_cfg, dict):
            return {}
        area_cfg = product_cfg.get(self.area, {})
        return area_cfg if isinstance(area_cfg, dict) else {}

    def _browse_weights(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._l("Choose weights file", "選擇權重檔"),
            str(self.config_path.parent),
            "Model files (*.pt *.onnx *.ckpt);;All files (*)",
        )
        if path:
            self.weights_edit.setText(path)

    def _browse_color_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._l("Choose color model", "選擇顏色模型"),
            str(self.config_path.parent),
            "JSON files (*.json);;All files (*)",
        )
        if path:
            self.color_model_edit.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            self._l("Choose output folder", "選擇輸出資料夾"),
            self.output_dir_edit.text() or str(self.config_path.parent),
        )
        if path:
            self.output_dir_edit.setText(path)

    @staticmethod
    def _ratio_spin() -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        return spin

    @staticmethod
    def _size_spin() -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(1, 8192)
        spin.setSingleStep(32)
        return spin
