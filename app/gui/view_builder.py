from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenuBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import tr
from app.gui.widgets import BigStatusLabel, ImageViewer, ResultDisplayWidget

if TYPE_CHECKING:
    from app.gui.main_window import DetectionSystemGUI


def _lang(gui: DetectionSystemGUI) -> str:
    return getattr(gui, "current_language", "en")


def build_control_panel(gui: DetectionSystemGUI) -> QGroupBox:
    """Build the legacy control panel with localized labels."""
    language = _lang(gui)
    panel = QGroupBox(tr(language, "inspection_setup"))
    layout = QVBoxLayout()

    product_group = QGroupBox(tr(language, "product"))
    product_layout = QVBoxLayout()
    gui.product_combo = QComboBox()
    gui.product_combo.currentTextChanged.connect(gui.on_product_changed)
    product_layout.addWidget(QLabel(f"{tr(language, 'product')}:"))
    product_layout.addWidget(gui.product_combo)
    product_group.setLayout(product_layout)
    layout.addWidget(product_group)

    area_group = QGroupBox(tr(language, "area"))
    area_layout = QVBoxLayout()
    gui.area_combo = QComboBox()
    gui.area_combo.currentTextChanged.connect(gui.on_area_changed)
    area_layout.addWidget(QLabel(f"{tr(language, 'area')}:"))
    area_layout.addWidget(gui.area_combo)
    area_group.setLayout(area_layout)
    layout.addWidget(area_group)

    inference_group = QGroupBox(tr(language, "model"))
    inference_layout = QVBoxLayout()
    gui.inference_combo = QComboBox()
    gui.inference_combo.currentTextChanged.connect(gui.on_inference_changed)
    inference_layout.addWidget(QLabel(f"{tr(language, 'type')}:"))
    inference_layout.addWidget(gui.inference_combo)
    inference_group.setLayout(inference_layout)
    layout.addWidget(inference_group)

    button_group = QGroupBox(tr(language, "operation"))
    button_layout = QVBoxLayout()
    gui.start_btn = QPushButton(tr(language, "start"))
    gui.start_btn.setObjectName("primaryAction")
    gui.start_btn.clicked.connect(gui.start_detection)
    button_layout.addWidget(gui.start_btn)

    gui.stop_btn = QPushButton(tr(language, "stop"))
    gui.stop_btn.setObjectName("dangerAction")
    gui.stop_btn.clicked.connect(gui.stop_detection)
    gui.stop_btn.setEnabled(False)
    button_layout.addWidget(gui.stop_btn)

    gui.save_btn = QPushButton(tr(language, "save_result"))
    gui.save_btn.setObjectName("secondaryAction")
    gui.save_btn.clicked.connect(gui.save_results)
    gui.save_btn.setEnabled(False)
    button_layout.addWidget(gui.save_btn)

    if hasattr(gui, "on_use_camera_toggled"):
        gui.use_camera_chk = QCheckBox(tr(language, "use_camera"))
        gui.use_camera_chk.setChecked(True)
        gui.use_camera_chk.toggled.connect(gui.on_use_camera_toggled)
        button_layout.addWidget(gui.use_camera_chk)

    if hasattr(gui, "handle_reconnect_camera"):
        gui.reconnect_camera_btn = QPushButton(tr(language, "reconnect"))
        gui.reconnect_camera_btn.setObjectName("secondaryAction")
        gui.reconnect_camera_btn.clicked.connect(gui.handle_reconnect_camera)
        button_layout.addWidget(gui.reconnect_camera_btn)

        gui.disconnect_camera_btn = QPushButton(tr(language, "disconnect"))
        gui.disconnect_camera_btn.setObjectName("secondaryAction")
        gui.disconnect_camera_btn.clicked.connect(gui.handle_disconnect_camera)
        button_layout.addWidget(gui.disconnect_camera_btn)

    gui.pick_image_btn = QPushButton(tr(language, "choose_image"))
    gui.pick_image_btn.setObjectName("secondaryAction")
    gui.pick_image_btn.clicked.connect(gui.pick_image)
    gui.image_path_label = QLabel(tr(language, "no_image"))
    gui.image_path_label.setWordWrap(True)
    button_layout.addWidget(gui.pick_image_btn)
    button_layout.addWidget(gui.image_path_label)

    if hasattr(gui, "clear_selected_image"):
        gui.clear_image_btn = QPushButton(tr(language, "clear_image"))
        gui.clear_image_btn.setObjectName("secondaryAction")
        gui.clear_image_btn.clicked.connect(gui.clear_selected_image)
        gui.clear_image_btn.setEnabled(False)
        button_layout.addWidget(gui.clear_image_btn)

    button_group.setLayout(button_layout)
    layout.addWidget(button_group)
    layout.addStretch()
    panel.setLayout(layout)
    return panel


def build_image_area(gui: DetectionSystemGUI) -> QGroupBox:
    """Build the legacy image preview area."""
    language = _lang(gui)
    area = QGroupBox(tr(language, "viewer"))
    layout = QVBoxLayout()
    gui.image_tabs = QTabWidget()

    gui.original_image = ImageViewer(tr(language, "original_image"))
    gui.original_image.set_language(language)
    gui.image_tabs.addTab(gui.original_image, tr(language, "original"))

    gui.processed_image = ImageViewer(tr(language, "processed_image"))
    gui.processed_image.set_language(language)
    gui.image_tabs.addTab(gui.processed_image, tr(language, "processed"))

    gui.result_image = ImageViewer(tr(language, "result_image"))
    gui.result_image.set_language(language)
    gui.image_tabs.addTab(gui.result_image, tr(language, "result"))

    layout.addWidget(gui.image_tabs)
    area.setLayout(layout)
    return area


def build_info_panel(gui: DetectionSystemGUI) -> QWidget:
    """Build the legacy info panel."""
    language = _lang(gui)
    panel = QWidget()
    layout = QVBoxLayout()

    gui.big_status_label = BigStatusLabel()
    layout.addWidget(gui.big_status_label)

    gui.result_widget = ResultDisplayWidget()
    gui.result_widget.set_language(language)
    layout.addWidget(gui.result_widget)

    log_group = QGroupBox(tr(language, "debug_log"))
    log_layout = QVBoxLayout()
    gui.log_text = QTextEdit()
    gui.log_text.setFont(QFont("Consolas", 8))
    gui.log_text.setReadOnly(True)
    log_layout.addWidget(gui.log_text)
    log_group.setLayout(log_layout)
    layout.addWidget(log_group)

    panel.setLayout(layout)
    return panel


def build_menu_bar(gui: DetectionSystemGUI) -> QMenuBar:
    """Build a localized menu bar for the current GUI language."""
    language = _lang(gui)
    menubar = gui.menuBar()

    file_menu = menubar.addMenu(tr(language, "file_menu"))
    open_action = QAction(tr(language, "open_config"), gui)
    open_action.triggered.connect(gui.open_config)
    file_menu.addAction(open_action)

    save_action = QAction(tr(language, "save_config"), gui)
    save_action.triggered.connect(gui.save_config)
    file_menu.addAction(save_action)

    file_menu.addSeparator()

    exit_action = QAction(tr(language, "exit"), gui)
    exit_action.triggered.connect(gui.close)
    file_menu.addAction(exit_action)

    view_menu = menubar.addMenu(tr(language, "view_menu"))
    refresh_action = QAction(tr(language, "reload_models"), gui)
    refresh_action.triggered.connect(gui.load_available_models)
    view_menu.addAction(refresh_action)

    edit_model_action = QAction(tr(language, "edit_current_model"), gui)
    edit_model_action.triggered.connect(gui.edit_current_model_config)
    view_menu.addAction(edit_model_action)

    view_menu.addSeparator()
    reset_stats_action = QAction(tr(language, "reset_shift_stats"), gui)
    reset_stats_action.setShortcut("Ctrl+R")
    reset_stats_action.triggered.connect(
        lambda: gui.info_panel.session_stats.reset_session()
    )
    view_menu.addAction(reset_stats_action)

    lighting_menu = menubar.addMenu(tr(language, "lighting_menu"))
    light_on_action = QAction(tr(language, "light_on"), gui)
    light_on_action.triggered.connect(gui._light_turn_on)
    lighting_menu.addAction(light_on_action)

    light_off_action = QAction(tr(language, "light_off"), gui)
    light_off_action.triggered.connect(gui._light_turn_off)
    lighting_menu.addAction(light_off_action)

    brightness_action = QAction(tr(language, "light_brightness"), gui)
    brightness_action.triggered.connect(gui._light_open_brightness_dialog)
    lighting_menu.addAction(brightness_action)

    lighting_menu.addSeparator()

    # Dynamic submenu: repopulated with live COM ports each time it opens.
    port_menu = lighting_menu.addMenu(tr(language, "light_port"))
    port_menu.aboutToShow.connect(lambda: gui.populate_light_port_menu(port_menu))

    help_menu = menubar.addMenu(tr(language, "help_menu"))
    about_action = QAction(tr(language, "about"), gui)
    about_action.triggered.connect(gui.show_about)
    help_menu.addAction(about_action)

    return menubar
