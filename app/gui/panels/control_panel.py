from __future__ import annotations

from PyQt5.QtCore import QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import LANGUAGE_LABELS, normalize_language, tr

DEFAULT_PRESETS: dict[str, tuple[str, str, str]] = {
    "Select preset": ("", "", ""),
}

_PIN_SETTINGS_KEY = "engineer_pin"
_PIN_DEFAULT = "admin"


# ---------------------------------------------------------------------------
# PIN entry dialog
# ---------------------------------------------------------------------------

class _PinDialog(QDialog):
    """Simple PIN entry dialog used to unlock the engineer panel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Engineer Access")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._label = QLabel("Enter engineer PIN:")
        layout.addWidget(self._label)

        self._pin_edit = QLineEdit()
        self._pin_edit.setEchoMode(QLineEdit.Password)
        self._pin_edit.setMaxLength(16)
        self._pin_edit.setPlaceholderText("PIN")
        self._pin_edit.returnPressed.connect(self.accept)
        layout.addWidget(self._pin_edit)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #b42318; font-size: 9pt;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def pin_value(self) -> str:
        return self._pin_edit.text()

    def show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._pin_edit.clear()
        self._pin_edit.setFocus()


class _ChangePinDialog(QDialog):
    """Dialog to change the engineer PIN."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Change Engineer PIN")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedWidth(280)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Current PIN:"))
        self._current = QLineEdit()
        self._current.setEchoMode(QLineEdit.Password)
        self._current.setMaxLength(16)
        layout.addWidget(self._current)

        layout.addWidget(QLabel("New PIN:"))
        self._new = QLineEdit()
        self._new.setEchoMode(QLineEdit.Password)
        self._new.setMaxLength(16)
        layout.addWidget(self._new)

        layout.addWidget(QLabel("Confirm new PIN:"))
        self._confirm = QLineEdit()
        self._confirm.setEchoMode(QLineEdit.Password)
        self._confirm.setMaxLength(16)
        layout.addWidget(self._confirm)

        self._msg = QLabel("")
        self._msg.setStyleSheet("color: #b42318; font-size: 9pt;")
        self._msg.setWordWrap(True)
        layout.addWidget(self._msg)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._new.text():
            self._msg.setText("New PIN cannot be empty.")
            return
        if self._new.text() != self._confirm.text():
            self._msg.setText("New PIN and confirmation do not match.")
            return
        self.accept()

    def current_pin(self) -> str:
        return self._current.text()

    def new_pin(self) -> str:
        return self._new.text()


# ---------------------------------------------------------------------------
# Main control panel
# ---------------------------------------------------------------------------

class ControlPanel(QGroupBox):
    """Left-side controls, split into operator (always visible) and
    engineer (PIN-protected, collapsible) sections.

    All widget *names* are preserved from the previous layout so that
    external code (main_window.py aliases) requires no changes.
    """

    product_changed = pyqtSignal(str)
    area_changed = pyqtSignal(str)
    inference_type_changed = pyqtSignal(str)
    preset_selected = pyqtSignal(str, str, str)
    language_changed = pyqtSignal(str)

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    save_requested = pyqtSignal()
    edit_model_config_requested = pyqtSignal()
    auto_mode_toggled = pyqtSignal(bool)

    use_camera_toggled = pyqtSignal(bool)
    reconnect_camera_requested = pyqtSignal()
    disconnect_camera_requested = pyqtSignal()

    pick_image_requested = pyqtSignal()
    clear_image_requested = pyqtSignal()
    show_detection_boxes_toggled = pyqtSignal(bool)
    calib_sample_empty_requested = pyqtSignal()
    calib_sample_product_requested = pyqtSignal()
    calib_apply_requested = pyqtSignal(int)  # new threshold value

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Inspection Setup", parent)
        self._language = "en"
        self._presets: dict[str, tuple[str, str, str]] = {
            tr(self._language, "select_preset"): ("", "", ""),
        }
        self._output_path = "--"
        self._settings = QSettings("yolo11_inspection", "ui_settings")
        self._calib_empty_area: float | None = None
        self._calib_product_area: float | None = None
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self._setup_ui()

    # ------------------------------------------------------------------
    # PIN helpers
    # ------------------------------------------------------------------

    def _stored_pin(self) -> str:
        return str(self._settings.value(_PIN_SETTINGS_KEY, _PIN_DEFAULT))

    def _save_pin(self, pin: str) -> None:
        self._settings.setValue(_PIN_SETTINGS_KEY, pin)

    def _verify_pin(self) -> bool:
        """Show PIN dialog; return True when the correct PIN is entered."""
        dlg = _PinDialog(self)
        while True:
            if dlg.exec_() != QDialog.Accepted:
                return False
            if dlg.pin_value() == self._stored_pin():
                return True
            dlg.show_error("Incorrect PIN. Try again.")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout()
        root.setSpacing(10)

        # ── Operator section ──────────────────────────────────────────
        self._build_operator_section(root)

        # ── Engineer toggle (PIN gate) ────────────────────────────────
        self.engineering_toggle_btn = QPushButton("Engineer Settings >")
        self.engineering_toggle_btn.setObjectName("secondaryAction")
        self.engineering_toggle_btn.setCheckable(True)
        self.engineering_toggle_btn.setToolTip("PIN required to access engineer settings")
        self.engineering_toggle_btn.toggled.connect(self._on_engineer_toggle)
        root.addWidget(self.engineering_toggle_btn)

        # ── Engineer panel (inside a scroll area) ────────────────────
        self.engineering_panel = QScrollArea()
        self.engineering_panel.setWidgetResizable(True)
        self.engineering_panel.setFrameShape(self.engineering_panel.NoFrame)
        self.engineering_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _eng_inner = QWidget()
        self._build_engineer_panel(_eng_inner)
        self.engineering_panel.setWidget(_eng_inner)
        self.engineering_panel.setVisible(False)
        self.engineering_panel.setMaximumHeight(500)
        root.addWidget(self.engineering_panel)

        root.addStretch()
        self.setLayout(root)
        self.set_language(self._language)

    def _build_operator_section(self, layout: QVBoxLayout) -> None:
        """Widgets always visible to the operator."""

        # Language
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        for code, label in LANGUAGE_LABELS.items():
            self.language_combo.addItem(label, code)
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        # Product / Area / Model selection
        self.model_group = QGroupBox("Product")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(6)

        self.preset_label = QLabel()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Quick switch product / area / model")
        self._rebuild_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
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

        self.model_group.setLayout(model_layout)
        layout.addWidget(self.model_group)

        # Auto Mode
        self.auto_mode_chk = QCheckBox("Auto Mode")
        self.auto_mode_chk.setToolTip(
            "Automatically detect and inspect when product is stable in ROI.\n"
            "Requires camera. Button-triggered inspection is disabled while active."
        )
        self.auto_mode_chk.toggled.connect(self._on_auto_mode_changed)
        layout.addWidget(self.auto_mode_chk)

        self.auto_mode_status_label = QLabel("")
        self.auto_mode_status_label.setStyleSheet(
            "color: #0369a1; font-size: 9pt; padding: 2px 4px;"
        )
        self.auto_mode_status_label.setWordWrap(True)
        layout.addWidget(self.auto_mode_status_label)

        # Operation buttons
        self.button_group = QGroupBox("Operation")
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        self.start_btn = QPushButton("Start Inspection")
        self.start_btn.setObjectName("primaryAction")
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(48)
        font = self.start_btn.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.start_btn.setFont(font)
        self.start_btn.clicked.connect(self.start_requested.emit)

        # Ghost widgets: kept alive for main_window references but not in layout
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.stop_btn.hide()

        self.save_btn = QPushButton("Save Result")
        self.save_btn.setObjectName("secondaryAction")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.save_btn.hide()

        btn_layout.addWidget(self.start_btn)
        self.button_group.setLayout(btn_layout)
        layout.addWidget(self.button_group)

    def _build_engineer_panel(self, panel: QWidget) -> None:
        """Advanced controls inside the PIN-protected collapsible panel."""
        eng_layout = QVBoxLayout(panel)
        eng_layout.setContentsMargins(0, 0, 0, 0)
        eng_layout.setSpacing(10)

        # Camera
        camera_group = QGroupBox("Camera")
        self.camera_group = camera_group
        cam_layout = QVBoxLayout()
        cam_layout.setSpacing(6)

        self.use_camera_chk = QCheckBox("Use camera")
        self.use_camera_chk.toggled.connect(self.use_camera_toggled.emit)
        cam_layout.addWidget(self.use_camera_chk)

        self.reconnect_camera_btn = QPushButton("Reconnect")
        self.reconnect_camera_btn.setObjectName("secondaryAction")
        self.reconnect_camera_btn.clicked.connect(self.reconnect_camera_requested.emit)
        cam_layout.addWidget(self.reconnect_camera_btn)

        self.disconnect_camera_btn = QPushButton("Disconnect")
        self.disconnect_camera_btn.setObjectName("secondaryAction")
        self.disconnect_camera_btn.clicked.connect(self.disconnect_camera_requested.emit)
        cam_layout.addWidget(self.disconnect_camera_btn)

        self.pick_image_btn = QPushButton("Choose Image...")
        self.pick_image_btn.setObjectName("secondaryAction")
        self.pick_image_btn.clicked.connect(self.pick_image_requested.emit)

        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setStyleSheet("color: #6b7280; font-size: 9pt;")
        self.image_path_label.setWordWrap(True)

        # Ghost widget: referenced externally but not shown (pick_image_btn already replaces selection)
        self.clear_image_btn = QPushButton("Clear Image")
        self.clear_image_btn.setObjectName("secondaryAction")
        self.clear_image_btn.setEnabled(False)
        self.clear_image_btn.clicked.connect(self.clear_image_requested.emit)
        self.clear_image_btn.hide()

        cam_layout.addWidget(self.pick_image_btn)
        cam_layout.addWidget(self.image_path_label)
        camera_group.setLayout(cam_layout)
        eng_layout.addWidget(camera_group)

        # Debug / model config
        debug_group = QGroupBox("Debug / Config")
        self.debug_group = debug_group
        debug_layout = QVBoxLayout()
        debug_layout.setSpacing(6)

        self.edit_model_config_btn = QPushButton("Edit Model Config")
        self.edit_model_config_btn.setObjectName("secondaryAction")
        self.edit_model_config_btn.clicked.connect(self.edit_model_config_requested.emit)
        debug_layout.addWidget(self.edit_model_config_btn)

        self.output_path_label = QLabel("Output: --")
        self.output_path_label.setStyleSheet(
            "color: #6b7280; font-size: 8pt; padding: 2px 4px;"
        )
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setToolTip("Current result output directory")
        debug_layout.addWidget(self.output_path_label)

        self.show_detection_boxes_chk = QCheckBox("Show detection boxes")
        self.show_detection_boxes_chk.setChecked(True)
        self.show_detection_boxes_chk.setToolTip("Toggle inspection overlays on result view")
        self.show_detection_boxes_chk.toggled.connect(self.show_detection_boxes_toggled.emit)
        debug_layout.addWidget(self.show_detection_boxes_chk)

        debug_group.setLayout(debug_layout)
        eng_layout.addWidget(debug_group)

        # Auto-trigger calibration
        calib_group = QGroupBox("Auto-Trigger Calibration")
        self.calib_group = calib_group
        calib_layout = QVBoxLayout()
        calib_layout.setSpacing(6)

        self._calib_hint_label = QLabel()
        self._calib_hint_label.setStyleSheet("color: #6b7280; font-size: 8pt;")
        self._calib_hint_label.setWordWrap(True)
        calib_layout.addWidget(self._calib_hint_label)

        self._calib_empty_btn = QPushButton("Sample Empty")
        self._calib_empty_btn.setObjectName("secondaryAction")
        self._calib_empty_btn.clicked.connect(self.calib_sample_empty_requested.emit)
        calib_layout.addWidget(self._calib_empty_btn)

        self._calib_empty_val = QLabel("Empty area: --")
        self._calib_empty_val.setStyleSheet("font-size: 8pt; color: #374151;")
        calib_layout.addWidget(self._calib_empty_val)

        self._calib_product_btn = QPushButton("Sample Product")
        self._calib_product_btn.setObjectName("secondaryAction")
        self._calib_product_btn.clicked.connect(self.calib_sample_product_requested.emit)
        calib_layout.addWidget(self._calib_product_btn)

        self._calib_product_val = QLabel("Product area: --")
        self._calib_product_val.setStyleSheet("font-size: 8pt; color: #374151;")
        calib_layout.addWidget(self._calib_product_val)

        self._calib_threshold_val = QLabel("Threshold: --")
        self._calib_threshold_val.setStyleSheet("font-size: 8pt; font-weight: bold; color: #1d4ed8;")
        calib_layout.addWidget(self._calib_threshold_val)

        self._calib_apply_btn = QPushButton("Apply Threshold")
        self._calib_apply_btn.setObjectName("primaryAction")
        self._calib_apply_btn.setEnabled(False)
        self._calib_apply_btn.clicked.connect(self._on_calib_apply)
        calib_layout.addWidget(self._calib_apply_btn)

        calib_group.setLayout(calib_layout)
        eng_layout.addWidget(calib_group)

        # Security (change PIN / lock)
        sec_group = QGroupBox("Security")
        self.sec_group = sec_group
        sec_layout = QVBoxLayout()
        sec_layout.setSpacing(6)

        self._change_pin_btn = QPushButton("Change PIN...")
        self._change_pin_btn.setObjectName("secondaryAction")
        self._change_pin_btn.clicked.connect(self._on_change_pin)
        sec_layout.addWidget(self._change_pin_btn)

        self._lock_btn = QPushButton("Lock Engineer Mode")
        self._lock_btn.setObjectName("dangerAction")
        self._lock_btn.clicked.connect(self._lock_engineer)
        sec_layout.addWidget(self._lock_btn)

        sec_group.setLayout(sec_layout)
        eng_layout.addWidget(sec_group)

    # ------------------------------------------------------------------
    # Engineer panel toggle (PIN gate)
    # ------------------------------------------------------------------

    def _on_auto_mode_changed(self, enabled: bool) -> None:
        """Lock model-selection combos while Auto Mode is active."""
        for widget in (
            self.preset_combo,
            self.product_combo,
            self.area_combo,
            self.inference_combo,
        ):
            widget.setEnabled(not enabled)
        self.auto_mode_toggled.emit(enabled)

    def _on_engineer_toggle(self, checked: bool) -> None:
        if checked:
            if not self._verify_pin():
                # Revert toggle without re-triggering this slot
                self.engineering_toggle_btn.blockSignals(True)
                self.engineering_toggle_btn.setChecked(False)
                self.engineering_toggle_btn.blockSignals(False)
                self.engineering_panel.setVisible(False)
                return
            self._set_engineering_visible(True)
        else:
            self._set_engineering_visible(False)

    def _set_engineering_visible(self, visible: bool) -> None:
        self.engineering_panel.setVisible(visible)
        self.engineering_toggle_btn.setText(
            tr(self._language, "engineer_settings_open")
            if visible
            else tr(self._language, "engineer_settings_closed")
        )
        if not visible:
            # Sync toggle button state without triggering the slot again
            self.engineering_toggle_btn.blockSignals(True)
            self.engineering_toggle_btn.setChecked(False)
            self.engineering_toggle_btn.blockSignals(False)

    def _lock_engineer(self) -> None:
        """Close the engineer panel from within (Lock button)."""
        self._set_engineering_visible(False)

    # ------------------------------------------------------------------
    # PIN management
    # ------------------------------------------------------------------

    def _on_change_pin(self) -> None:
        dlg = _ChangePinDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        if dlg.current_pin() != self._stored_pin():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Change PIN", "Current PIN is incorrect.")
            return
        self._save_pin(dlg.new_pin())
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Change PIN", "PIN changed successfully.")

    # ------------------------------------------------------------------
    # Preset helpers
    # ------------------------------------------------------------------

    def set_presets(self, presets: dict[str, tuple[str, str, str]]) -> None:
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

    # ------------------------------------------------------------------
    # Public helpers called by main_window
    # ------------------------------------------------------------------

    def set_calib_empty(self, area: float) -> None:
        self._calib_empty_area = area
        prefix = tr(self._language, "empty_area_prefix")
        self._calib_empty_val.setText(f"{prefix} {area:,.0f}")
        self._refresh_calib_threshold()

    def set_calib_product(self, area: float) -> None:
        self._calib_product_area = area
        prefix = tr(self._language, "product_area_prefix")
        self._calib_product_val.setText(f"{prefix} {area:,.0f}")
        self._refresh_calib_threshold()

    def _refresh_calib_threshold(self) -> None:
        if self._calib_empty_area is not None and self._calib_product_area is not None:
            threshold = int((self._calib_empty_area + self._calib_product_area) / 2)
            prefix = tr(self._language, "threshold_prefix")
            self._calib_threshold_val.setText(f"{prefix} {threshold:,}")
            self._calib_apply_btn.setEnabled(True)
            self._calib_apply_btn.setToolTip(
                f"Set product_area_threshold = {threshold:,}\n"
                f"(empty={self._calib_empty_area:,.0f}, product={self._calib_product_area:,.0f})"
            )

    def _on_calib_apply(self) -> None:
        if self._calib_empty_area is None or self._calib_product_area is None:
            return
        threshold = int((self._calib_empty_area + self._calib_product_area) / 2)
        self.calib_apply_requested.emit(threshold)

    def set_output_path(self, path: str) -> None:
        self._output_path = path
        self.output_path_label.setText(f"{tr(self._language, 'output')}: {path}")
        self.output_path_label.setToolTip(path)

    def set_auto_mode_status(self, state_name: str) -> None:
        self.auto_mode_status_label.setText(state_name)

    # ------------------------------------------------------------------
    # Localisation
    # ------------------------------------------------------------------

    def set_language(self, language: str) -> None:
        self._language = normalize_language(language)
        self.setTitle(tr(self._language, "inspection_setup"))

        # Operator section
        self.preset_label.setText(tr(self._language, "preset"))
        self.auto_mode_chk.setText(tr(self._language, "auto_mode"))
        self.button_group.setTitle(tr(self._language, "operation"))
        self.start_btn.setText(tr(self._language, "start"))
        self.stop_btn.setText(tr(self._language, "stop"))
        self.save_btn.setText(tr(self._language, "save_result"))

        # Operator section (continued)
        self.language_label.setText(tr(self._language, "language"))
        self.model_group.setTitle(tr(self._language, "product_group"))
        self.product_label.setText(tr(self._language, "product"))
        self.area_label.setText(tr(self._language, "area"))
        self.model_label.setText(tr(self._language, "model"))

        # Engineer section labels (update even when hidden so they're correct on reveal)
        self.camera_group.setTitle(tr(self._language, "camera_group"))
        self.use_camera_chk.setText(tr(self._language, "use_camera"))
        self.reconnect_camera_btn.setText(tr(self._language, "reconnect"))
        self.disconnect_camera_btn.setText(tr(self._language, "disconnect"))
        self.pick_image_btn.setText(tr(self._language, "choose_image"))
        if self.image_path_label.text() in {"No image selected", "尚未選擇影像"}:
            self.image_path_label.setText(tr(self._language, "no_image"))
        self.clear_image_btn.setText(tr(self._language, "clear_image"))
        self.debug_group.setTitle(tr(self._language, "debug_config_group"))
        self.edit_model_config_btn.setText(tr(self._language, "edit_model_config"))
        self.show_detection_boxes_chk.setText(tr(self._language, "show_detection_boxes"))
        self.output_path_label.setText(
            f"{tr(self._language, 'output')}: {self._output_path}"
        )
        self.calib_group.setTitle(tr(self._language, "auto_trigger_calib"))
        self._calib_hint_label.setText(tr(self._language, "calib_hint"))
        self._calib_empty_btn.setText(tr(self._language, "sample_empty"))
        self._calib_product_btn.setText(tr(self._language, "sample_product"))
        self._calib_apply_btn.setText(tr(self._language, "apply_threshold"))
        if self._calib_empty_area is None:
            self._calib_empty_val.setText(tr(self._language, "empty_area_label"))
        else:
            self._calib_empty_val.setText(
                f"{tr(self._language, 'empty_area_prefix')} {self._calib_empty_area:,.0f}"
            )
        if self._calib_product_area is None:
            self._calib_product_val.setText(tr(self._language, "product_area_label"))
        else:
            self._calib_product_val.setText(
                f"{tr(self._language, 'product_area_prefix')} {self._calib_product_area:,.0f}"
            )
        if self._calib_empty_area is None or self._calib_product_area is None:
            self._calib_threshold_val.setText(tr(self._language, "threshold_label"))
        else:
            threshold = int((self._calib_empty_area + self._calib_product_area) / 2)
            self._calib_threshold_val.setText(
                f"{tr(self._language, 'threshold_prefix')} {threshold:,}"
            )
        self.sec_group.setTitle(tr(self._language, "security_group"))
        self._change_pin_btn.setText(tr(self._language, "change_pin"))
        self._lock_btn.setText(tr(self._language, "lock_engineer"))

        self._set_engineering_visible(self.engineering_panel.isVisible())

        # Sync language combo
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
