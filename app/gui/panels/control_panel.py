from __future__ import annotations

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

from app.gui.i18n import LANGUAGE_LABELS, normalize_language, tr

DEFAULT_PRESETS: dict[str, tuple[str, str, str]] = {
    "Select preset": ("", "", ""),
}


class ControlPanel(QGroupBox):
    """Left-side setup and operation controls."""

    product_changed = pyqtSignal(str)
    area_changed = pyqtSignal(str)
    inference_type_changed = pyqtSignal(str)
    preset_selected = pyqtSignal(str, str, str)
    language_changed = pyqtSignal(str)

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
        super().__init__("Inspection Setup", parent)
        self._presets: dict[str, tuple[str, str, str]] = dict(DEFAULT_PRESETS)
        self._language = "en"
        self._output_path = "--"
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(10)

        self.language_label = QLabel()
        self.language_combo = QComboBox()
        for code, label in LANGUAGE_LABELS.items():
            self.language_combo.addItem(label, code)
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        model_group = QGroupBox("Product")
        self.model_group = model_group
        model_layout = QVBoxLayout()
        model_layout.setSpacing(6)

        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Quick switch product / area / model")
        self._rebuild_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        self.preset_label = QLabel()
        model_layout.addWidget(self.preset_label)
        model_layout.addWidget(self.preset_combo)

        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.product_changed.emit)
        self.product_label = QLabel()
        model_layout.addWidget(self.product_label)
        model_layout.addWidget(self.product_combo)

        self.area_combo = QComboBox()
        self.area_combo.currentTextChanged.connect(self.area_changed.emit)
        self.area_label = QLabel()
        model_layout.addWidget(self.area_label)
        model_layout.addWidget(self.area_combo)

        self.inference_combo = QComboBox()
        self.inference_combo.currentTextChanged.connect(self.inference_type_changed.emit)
        self.model_label = QLabel()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.inference_combo)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        button_group = QGroupBox("Operation")
        self.button_group = button_group
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        self.start_btn = QPushButton("Start Inspection")
        self.start_btn.setObjectName("primaryAction")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_requested.emit)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        self.save_btn = QPushButton("Save Result")
        self.save_btn.setObjectName("secondaryAction")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested.emit)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_btn)

        self.use_camera_chk = QCheckBox("Use camera")
        self.use_camera_chk.toggled.connect(self.use_camera_toggled.emit)
        button_layout.addWidget(self.use_camera_chk)

        camera_btn_layout = QHBoxLayout()
        self.reconnect_camera_btn = QPushButton("Reconnect")
        self.reconnect_camera_btn.setObjectName("secondaryAction")
        self.reconnect_camera_btn.clicked.connect(self.reconnect_camera_requested.emit)
        camera_btn_layout.addWidget(self.reconnect_camera_btn)

        self.disconnect_camera_btn = QPushButton("Disconnect")
        self.disconnect_camera_btn.setObjectName("secondaryAction")
        self.disconnect_camera_btn.clicked.connect(self.disconnect_camera_requested.emit)
        camera_btn_layout.addWidget(self.disconnect_camera_btn)
        button_layout.addLayout(camera_btn_layout)

        self.pick_image_btn = QPushButton("Choose Image...")
        self.pick_image_btn.setObjectName("secondaryAction")
        self.pick_image_btn.clicked.connect(self.pick_image_requested.emit)

        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setStyleSheet("color: #6b7280; font-size: 9pt;")
        self.image_path_label.setWordWrap(True)

        self.clear_image_btn = QPushButton("Clear Image")
        self.clear_image_btn.setObjectName("secondaryAction")
        self.clear_image_btn.setEnabled(False)
        self.clear_image_btn.clicked.connect(self.clear_image_requested.emit)

        button_layout.addWidget(self.pick_image_btn)
        button_layout.addWidget(self.image_path_label)
        button_layout.addWidget(self.clear_image_btn)

        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

        self.engineering_toggle_btn = QPushButton("Engineer Settings >")
        self.engineering_toggle_btn.setObjectName("secondaryAction")
        self.engineering_toggle_btn.setCheckable(True)
        self.engineering_toggle_btn.setToolTip("Show model, output, and debug options")
        self.engineering_toggle_btn.toggled.connect(self._set_engineering_visible)
        layout.addWidget(self.engineering_toggle_btn)

        self.engineering_panel = QWidget()
        engineering_layout = QVBoxLayout(self.engineering_panel)
        engineering_layout.setContentsMargins(0, 0, 0, 0)
        engineering_layout.setSpacing(8)

        self.edit_model_config_btn = QPushButton("Edit Model Config")
        self.edit_model_config_btn.setObjectName("secondaryAction")
        self.edit_model_config_btn.clicked.connect(self.edit_model_config_requested.emit)
        engineering_layout.addWidget(self.edit_model_config_btn)

        self.output_path_label = QLabel("Output: --")
        self.output_path_label.setStyleSheet(
            "color: #6b7280; font-size: 8pt; padding: 2px 4px;"
        )
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setToolTip("Current result output directory")
        engineering_layout.addWidget(self.output_path_label)

        self.show_detection_boxes_chk = QCheckBox("Show detection boxes")
        self.show_detection_boxes_chk.setChecked(True)
        self.show_detection_boxes_chk.setToolTip("Toggle inspection overlays on result view")
        self.show_detection_boxes_chk.toggled.connect(
            self.show_detection_boxes_toggled.emit
        )
        engineering_layout.addWidget(self.show_detection_boxes_chk)

        self.engineering_panel.setVisible(False)
        layout.addWidget(self.engineering_panel)

        layout.addStretch()
        self.setLayout(layout)
        self.set_language(self._language)

    def set_presets(self, presets: dict[str, tuple[str, str, str]]) -> None:
        """Replace the preset list after model catalog loading."""
        self._presets = {tr(self._language, "select_preset"): ("", "", ""), **presets}
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
        """Show or hide controls outside the normal operator flow."""
        self.engineering_panel.setVisible(visible)
        self.engineering_toggle_btn.setText(
            tr(self._language, "engineer_settings_open")
            if visible
            else tr(self._language, "engineer_settings_closed")
        )

    def set_output_path(self, path: str) -> None:
        """Show the current result output directory."""
        self._output_path = path
        self.output_path_label.setText(f"{tr(self._language, 'output')}: {path}")
        self.output_path_label.setToolTip(path)

    def set_language(self, language: str) -> None:
        """Update visible control labels."""
        self._language = normalize_language(language)
        self.setTitle(tr(self._language, "inspection_setup"))
        self.language_label.setText(tr(self._language, "language"))
        self.model_group.setTitle(tr(self._language, "product_group"))
        self.preset_label.setText(tr(self._language, "preset"))
        self.product_label.setText(tr(self._language, "product"))
        self.area_label.setText(tr(self._language, "area"))
        self.model_label.setText(tr(self._language, "model"))
        self.button_group.setTitle(tr(self._language, "operation"))
        self.start_btn.setText(tr(self._language, "start"))
        self.stop_btn.setText(tr(self._language, "stop"))
        self.save_btn.setText(tr(self._language, "save_result"))
        self.use_camera_chk.setText(tr(self._language, "use_camera"))
        self.reconnect_camera_btn.setText(tr(self._language, "reconnect"))
        self.disconnect_camera_btn.setText(tr(self._language, "disconnect"))
        self.pick_image_btn.setText(tr(self._language, "choose_image"))
        if self.image_path_label.text() in {
            "No image selected",
            "尚未選擇影像",
        }:
            self.image_path_label.setText(tr(self._language, "no_image"))
        self.clear_image_btn.setText(tr(self._language, "clear_image"))
        self.edit_model_config_btn.setText(tr(self._language, "edit_model_config"))
        self.show_detection_boxes_chk.setText(tr(self._language, "show_detection_boxes"))
        self.output_path_label.setText(
            f"{tr(self._language, 'output')}: {self._output_path}"
        )
        self._set_engineering_visible(self.engineering_panel.isVisible())

        current_code = self.language_combo.currentData()
        if current_code != self._language:
            index = self.language_combo.findData(self._language)
            if index >= 0:
                self.language_combo.blockSignals(True)
                self.language_combo.setCurrentIndex(index)
                self.language_combo.blockSignals(False)

    def _on_language_selected(self) -> None:
        code = normalize_language(self.language_combo.currentData())
        self.set_language(code)
        self.language_changed.emit(code)
