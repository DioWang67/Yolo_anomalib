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

from app.gui.widgets import BigStatusLabel, ImageViewer, ResultDisplayWidget, StatusWidget

if TYPE_CHECKING:
    from app.gui.main_window import DetectionSystemGUI


def build_control_panel(gui: DetectionSystemGUI) -> QGroupBox:
    panel = QGroupBox("控制面板")
    layout = QVBoxLayout()

    product_group = QGroupBox("產品")
    product_layout = QVBoxLayout()
    gui.product_combo = QComboBox()
    gui.product_combo.currentTextChanged.connect(gui.on_product_changed)
    try:
        gui.product_combo.currentTextChanged.connect(
            lambda _=None: gui.update_start_enabled()
        )
    except Exception:
        pass
    product_layout.addWidget(QLabel("產品："))
    product_layout.addWidget(gui.product_combo)
    product_group.setLayout(product_layout)
    layout.addWidget(product_group)

    area_group = QGroupBox("檢測區域")
    area_layout = QVBoxLayout()
    gui.area_combo = QComboBox()
    gui.area_combo.currentTextChanged.connect(gui.on_area_changed)
    try:
        gui.area_combo.currentTextChanged.connect(
            lambda _=None: gui.update_start_enabled()
        )
    except Exception:
        pass
    area_layout.addWidget(QLabel("區域："))
    area_layout.addWidget(gui.area_combo)
    area_group.setLayout(area_layout)
    layout.addWidget(area_group)

    inference_group = QGroupBox("模型類型")
    inference_layout = QVBoxLayout()
    gui.inference_combo = QComboBox()
    try:
        gui.inference_combo.currentTextChanged.connect(
            lambda _=None: gui.update_start_enabled()
        )
    except Exception:
        pass
    inference_layout.addWidget(QLabel("類型："))
    inference_layout.addWidget(gui.inference_combo)
    inference_group.setLayout(inference_layout)
    layout.addWidget(inference_group)

    button_group = QGroupBox("操作")
    button_layout = QVBoxLayout()

    gui.start_btn = QPushButton("開始檢測")
    gui.start_btn.clicked.connect(gui.start_detection)
    gui.start_btn.setStyleSheet("QPushButton { background-color: #28a745; }")

    gui.stop_btn = QPushButton("停止檢測")
    gui.stop_btn.clicked.connect(gui.stop_detection)
    gui.stop_btn.setEnabled(False)
    gui.stop_btn.setStyleSheet("QPushButton { background-color: #dc3545; }")

    gui.save_btn = QPushButton("儲存結果")
    gui.save_btn.clicked.connect(gui.save_results)
    gui.save_btn.setEnabled(False)

    button_layout.addWidget(gui.start_btn)
    button_layout.addWidget(gui.stop_btn)
    button_layout.addWidget(gui.save_btn)

    try:
        gui.use_camera_chk = QCheckBox("使用相機")
        gui.use_camera_chk.toggled.connect(gui.on_use_camera_toggled)
        button_layout.addWidget(gui.use_camera_chk)
    except Exception:
        pass
    try:
        camera_btn_layout = QHBoxLayout()
        gui.reconnect_camera_btn = QPushButton("重新連接相機")
        gui.reconnect_camera_btn.clicked.connect(gui.handle_reconnect_camera)
        camera_btn_layout.addWidget(gui.reconnect_camera_btn)
        gui.disconnect_camera_btn = QPushButton("斷開相機")
        gui.disconnect_camera_btn.clicked.connect(gui.handle_disconnect_camera)
        camera_btn_layout.addWidget(gui.disconnect_camera_btn)
        button_layout.addLayout(camera_btn_layout)
    except Exception:
        pass

    gui.pick_image_btn = QPushButton("選擇影像...")
    gui.pick_image_btn.setStyleSheet("QPushButton { background-color: #17a2b8; }")
    gui.pick_image_btn.clicked.connect(gui.pick_image)
    gui.image_path_label = QLabel("未選擇影像；可使用相機或載入影像")
    gui.image_path_label.setStyleSheet("color: #6c757d;")
    button_layout.addWidget(gui.pick_image_btn)
    button_layout.addWidget(gui.image_path_label)
    try:
        gui.clear_image_btn = QPushButton("清除影像")
        gui.clear_image_btn.clicked.connect(gui.clear_selected_image)
        gui.clear_image_btn.setEnabled(False)
        button_layout.addWidget(gui.clear_image_btn)
    except Exception:
        pass

    button_group.setLayout(button_layout)
    layout.addWidget(button_group)

    layout.addStretch()
    panel.setLayout(layout)
    return panel


def build_image_area(gui: DetectionSystemGUI) -> QGroupBox:
    area = QGroupBox("影像預覽")
    layout = QVBoxLayout()

    gui.image_tabs = QTabWidget()
    gui.original_image = ImageViewer("原始影像")
    gui.image_tabs.addTab(gui.original_image, "原始")

    gui.processed_image = ImageViewer("處理後影像")
    gui.image_tabs.addTab(gui.processed_image, "處理後")

    gui.result_image = ImageViewer("結果影像")
    gui.image_tabs.addTab(gui.result_image, "結果")

    layout.addWidget(gui.image_tabs)
    area.setLayout(layout)
    return area


def build_info_panel(gui: DetectionSystemGUI) -> QWidget:
    panel = QWidget()
    layout = QVBoxLayout()

    gui.status_widget = StatusWidget()
    layout.addWidget(gui.status_widget)

    # Added BigStatusLabel for prominent feedback
    gui.big_status_label = BigStatusLabel()
    layout.addWidget(gui.big_status_label)

    gui.result_widget = ResultDisplayWidget()
    layout.addWidget(gui.result_widget)

    log_group = QGroupBox("系統日誌")
    log_layout = QVBoxLayout()

    gui.log_text = QTextEdit()
    gui.log_text.setFont(QFont("Consolas", 8))
    gui.log_text.setStyleSheet(
        """
        QTextEdit {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 4px;
        }
    """
    )
    gui.log_text.setReadOnly(True)

    log_layout.addWidget(gui.log_text)
    log_group.setLayout(log_layout)
    layout.addWidget(log_group)

    panel.setLayout(layout)
    return panel


def build_menu_bar(gui: DetectionSystemGUI) -> QMenuBar:
    menubar = gui.menuBar()

    file_menu = menubar.addMenu("檔案")
    open_action = QAction("開啟設定檔", gui)
    open_action.triggered.connect(gui.open_config)
    file_menu.addAction(open_action)

    save_action = QAction("儲存設定", gui)
    save_action.triggered.connect(gui.save_config)
    file_menu.addAction(save_action)

    file_menu.addSeparator()

    exit_action = QAction("結束", gui)
    exit_action.triggered.connect(gui.close)
    file_menu.addAction(exit_action)

    view_menu = menubar.addMenu("檢視")
    refresh_action = QAction("重新載入模型", gui)
    refresh_action.triggered.connect(gui.load_available_models)
    view_menu.addAction(refresh_action)

    help_menu = menubar.addMenu("說明")
    about_action = QAction("關於", gui)
    about_action.triggered.connect(gui.show_about)
    help_menu.addAction(about_action)

    return menubar
