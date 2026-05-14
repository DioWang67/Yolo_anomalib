from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Preset definition — (product, area, inference_type)
# Edit here to match your actual model directory structure.
# ---------------------------------------------------------------------------
DEFAULT_PRESETS: dict[str, tuple[str, str, str]] = {
    "— 選擇預設 —": ("", "", ""),
}


class ControlPanel(QGroupBox):
    """
    Panel containing all controls for the detection system:
    - Quick-switch preset selector
    - Model selection (Product, Area, Type)
    - Action buttons (Start, Stop, Save)
    - Camera controls
    - Image selection
    - Output path display
    """

    # Signals for interactions
    product_changed = pyqtSignal(str)
    area_changed = pyqtSignal(str)
    inference_type_changed = pyqtSignal(str)
    preset_selected = pyqtSignal(str, str, str)   # product, area, type

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    save_requested = pyqtSignal()
    edit_model_config_requested = pyqtSignal()

    use_camera_toggled = pyqtSignal(bool)
    reconnect_camera_requested = pyqtSignal()
    disconnect_camera_requested = pyqtSignal()

    pick_image_requested = pyqtSignal()
    clear_image_requested = pyqtSignal()
    show_detection_boxes_toggled = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("控制面板", parent)
        self._presets: dict[str, tuple[str, str, str]] = dict(DEFAULT_PRESETS)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # ── 機種選擇 ────────────────────────────────────────────────
        model_group = QGroupBox("機種選擇")
        model_layout = QVBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("一鍵選擇常用的產品 / 區域 / 類型組合")
        self._rebuild_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        model_layout.addWidget(QLabel("產線預設："))
        model_layout.addWidget(self.preset_combo)

        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.product_changed.emit)
        model_layout.addWidget(QLabel("產品："))
        model_layout.addWidget(self.product_combo)

        self.area_combo = QComboBox()
        self.area_combo.currentTextChanged.connect(self.area_changed.emit)
        model_layout.addWidget(QLabel("區域："))
        model_layout.addWidget(self.area_combo)

        self.inference_combo = QComboBox()
        self.inference_combo.currentTextChanged.connect(self.inference_type_changed.emit)
        model_layout.addWidget(QLabel("模型類型："))
        model_layout.addWidget(self.inference_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # ── 操作按鈕 ─────────────────────────────────────────────────
        button_group = QGroupBox("OP 操作")
        button_layout = QVBoxLayout()

        self.start_btn = QPushButton("開始檢測")
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("QPushButton { background-color: #28a745; }")
        self.start_btn.clicked.connect(self.start_requested.emit)

        self.stop_btn = QPushButton("停止檢測")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #dc3545; }")
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        self.save_btn = QPushButton("儲存結果")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested.emit)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_btn)

        # ── 相機控制 ─────────────────────────────────────────────────
        self.use_camera_chk = QCheckBox("使用相機")
        self.use_camera_chk.toggled.connect(self.use_camera_toggled.emit)
        button_layout.addWidget(self.use_camera_chk)

        camera_btn_layout = QHBoxLayout()
        self.reconnect_camera_btn = QPushButton("重新連接相機")
        self.reconnect_camera_btn.clicked.connect(self.reconnect_camera_requested.emit)
        camera_btn_layout.addWidget(self.reconnect_camera_btn)

        self.disconnect_camera_btn = QPushButton("斷開相機")
        self.disconnect_camera_btn.clicked.connect(self.disconnect_camera_requested.emit)
        camera_btn_layout.addWidget(self.disconnect_camera_btn)
        button_layout.addLayout(camera_btn_layout)

        # ── 影像選擇 ─────────────────────────────────────────────────
        self.pick_image_btn = QPushButton("選擇影像...")
        self.pick_image_btn.setStyleSheet("QPushButton { background-color: #17a2b8; }")
        self.pick_image_btn.clicked.connect(self.pick_image_requested.emit)

        self.image_path_label = QLabel("未選擇影像；可使用相機或載入影像")
        self.image_path_label.setStyleSheet("color: #6c757d;")

        self.clear_image_btn = QPushButton("清除影像")
        self.clear_image_btn.setEnabled(False)
        self.clear_image_btn.clicked.connect(self.clear_image_requested.emit)

        button_layout.addWidget(self.pick_image_btn)
        button_layout.addWidget(self.image_path_label)
        button_layout.addWidget(self.clear_image_btn)

        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

        # ── 工程設定 ────────────────────────────────────────────────
        self.engineering_toggle_btn = QPushButton("工程設定 ▸")
        self.engineering_toggle_btn.setCheckable(True)
        self.engineering_toggle_btn.setToolTip("展開機種與顯示相關設定")
        self.engineering_toggle_btn.toggled.connect(self._set_engineering_visible)
        layout.addWidget(self.engineering_toggle_btn)

        self.engineering_panel = QWidget()
        engineering_layout = QVBoxLayout(self.engineering_panel)
        engineering_layout.setContentsMargins(0, 0, 0, 0)

        self.edit_model_config_btn = QPushButton("編輯機種設定")
        self.edit_model_config_btn.clicked.connect(self.edit_model_config_requested.emit)
        engineering_layout.addWidget(self.edit_model_config_btn)

        self.output_path_label = QLabel("儲存路徑：—")
        self.output_path_label.setStyleSheet(
            "color: #6c757d; font-size: 8pt; padding: 2px 4px;"
        )
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setToolTip("目前結果儲存位置")
        engineering_layout.addWidget(self.output_path_label)

        self.show_detection_boxes_chk = QCheckBox("顯示位置檢測框")
        self.show_detection_boxes_chk.setChecked(True)
        self.show_detection_boxes_chk.setToolTip("切換結果圖是否顯示位置檢測框線")
        self.show_detection_boxes_chk.toggled.connect(
            self.show_detection_boxes_toggled.emit
        )
        engineering_layout.addWidget(self.show_detection_boxes_chk)

        self.engineering_panel.setVisible(False)
        layout.addWidget(self.engineering_panel)

        layout.addStretch()
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Preset management
    # ------------------------------------------------------------------

    def set_presets(self, presets: dict[str, tuple[str, str, str]]) -> None:
        """Replace the preset list (called by MainWindow after models load)."""
        self._presets = {"— 選擇預設 —": ("", "", ""), **presets}
        self._rebuild_preset_combo()

    def _rebuild_preset_combo(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(list(self._presets.keys()))
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, name: str) -> None:
        entry = self._presets.get(name)
        if entry and entry != ("", "", ""):
            product, area, inf_type = entry
            self.preset_selected.emit(product, area, inf_type)

    def _set_engineering_visible(self, visible: bool) -> None:
        """Show or hide controls that are not part of normal OP flow."""
        self.engineering_panel.setVisible(visible)
        self.engineering_toggle_btn.setText("工程設定 ▾" if visible else "工程設定 ▸")

    # ------------------------------------------------------------------
    # Output path display
    # ------------------------------------------------------------------

    def set_output_path(self, path: str) -> None:
        """Show the current result output directory."""
        self.output_path_label.setText(f"儲存路徑：{path}")
        self.output_path_label.setToolTip(path)
