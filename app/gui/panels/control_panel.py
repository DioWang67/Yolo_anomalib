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


class ControlPanel(QGroupBox):
    """
    Panel containing all controls for the detection system:
    - Model selection (Product, Area, Type)
    - Action buttons (Start, Stop, Save)
    - Camera controls
    - Image selection
    """

    # Signals for interactions
    product_changed = pyqtSignal(str)
    area_changed = pyqtSignal(str)
    inference_type_changed = pyqtSignal(str)
    
    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    save_requested = pyqtSignal()
    
    use_camera_toggled = pyqtSignal(bool)
    reconnect_camera_requested = pyqtSignal()
    disconnect_camera_requested = pyqtSignal()
    
    pick_image_requested = pyqtSignal()
    clear_image_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("控制面板", parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        # Product selection
        product_group = QGroupBox("產品")
        product_layout = QVBoxLayout()
        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.product_changed.emit)
        product_layout.addWidget(QLabel("產品："))
        product_layout.addWidget(self.product_combo)
        product_group.setLayout(product_layout)
        layout.addWidget(product_group)

        # Area selection
        area_group = QGroupBox("檢測區域")
        area_layout = QVBoxLayout()
        self.area_combo = QComboBox()
        self.area_combo.currentTextChanged.connect(self.area_changed.emit)
        area_layout.addWidget(QLabel("區域："))
        area_layout.addWidget(self.area_combo)
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)

        # Inference Type selection
        inference_group = QGroupBox("模型類型")
        inference_layout = QVBoxLayout()
        self.inference_combo = QComboBox()
        self.inference_combo.currentTextChanged.connect(self.inference_type_changed.emit)
        inference_layout.addWidget(QLabel("類型："))
        inference_layout.addWidget(self.inference_combo)
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)

        # Action Buttons
        button_group = QGroupBox("操作")
        button_layout = QVBoxLayout()

        self.start_btn = QPushButton("開始檢測")
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

        # Camera Controls
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

        # Image Selection
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

        layout.addStretch()
        self.setLayout(layout)
