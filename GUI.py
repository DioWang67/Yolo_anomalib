import sys
import os
import json
import logging
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
import traceback
import threading

# 導入 DetectionSystem 類
try:
    from core.detection_system import DetectionSystem
except Exception as e:
    print(f"無法導入 DetectionSystem: {e}")
    print("請確保 main.py 檔案位於正確的目錄中")
    sys.exit(1)

class DetectionWorker(QThread):
    """後台檢測處理執行緒"""
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detection_system, product, area, inference_type, frame: np.ndarray | None = None):
        super().__init__()
        self.detection_system = detection_system
        self.product = product
        self.area = area
        self.inference_type = inference_type
        self.frame = frame
        self._stop_event = threading.Event()
    
    def cancel(self):
        try:
            self._stop_event.set()
        except Exception:
            pass
    
    def run(self):
        try:
            result = self.detection_system.detect(
                self.product,
                self.area,
                self.inference_type,
                frame=self.frame,
                cancel_cb=self._stop_event.is_set,
            )
            if not self._stop_event.is_set():
                self.result_ready.emit(result)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())

class StatusWidget(QWidget):
    """系統狀態顯示元件"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 標題標籤
        title_label = QLabel("系統狀態")
        title_label.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        
        # 狀態指示器
        self.status_indicator = QLabel("●")
        self.status_indicator.setFont(QFont("Arial", 20))
        self.status_indicator.setAlignment(Qt.AlignCenter)
        
        # 狀態文字
        self.status_text = QLabel("系統就緒")
        self.status_text.setAlignment(Qt.AlignCenter)
        self.status_text.setFont(QFont("Microsoft JhengHei", 10))
        
        # 將所有元件添加到佈局中
        layout.addWidget(title_label)
        layout.addWidget(self.status_indicator)
        layout.addWidget(self.status_text)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setFixedWidth(150)
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        
        # 初始化狀態為待機狀態
        self.set_status("idle")
    
    def set_status(self, status):
        status_colors = {
            "idle": ("#6c757d", "系統待機中"),
            "running": ("#ffc107", "檢測執行中..."),
            "success": ("#28a745", "檢測完成"),
            "error": ("#dc3545", "檢測錯誤"),
            "warning": ("#fd7e14", "警告")
        }
        
        color, text = status_colors.get(status, ("#6c757d", "未知狀態"))
        self.status_indicator.setStyleSheet(f"color: {color};")
        self.status_text.setText(text)

class ImageViewer(QLabel):
    """圖像顯示區域"""
    def __init__(self, title="圖像"):
        super().__init__()
        self.title = title
        self.init_ui()
    
    def init_ui(self):
        self.setMinimumSize(300, 300)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #adb5bd;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
        """)
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"等待{self.title}")
        self.setScaledContents(True)
    
    def set_image(self, image_path):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        else:
            self.setText(f"無法載入{self.title}")

class ResultDisplayWidget(QWidget):
    """結果顯示區域"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 標題
        title = QLabel("檢測結果")
        title.setFont(QFont("Microsoft JhengHei", 12, QFont.Bold))
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        
        # 結果文字區域
        self.result_text = QTextEdit()
        self.result_text.setFont(QFont("Consolas", 9))
        self.result_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        self.result_text.setMaximumHeight(200)
        self.result_text.setReadOnly(True)
        
        layout.addWidget(title)
        layout.addWidget(self.result_text)
        self.setLayout(layout)
    
    def update_result(self, result):
        """更新檢測結果顯示"""
        text = "=== 檢測結果 ===\n"
        text += f"狀態: {result.get('status', 'N/A')}\n"
        text += f"產品: {result.get('product', 'N/A')}\n"
        text += f"區域: {result.get('area', 'N/A')}\n"
        text += f"推論類型: {result.get('inference_type', 'N/A')}\n"
        text += f"模型路徑: {result.get('ckpt_path', 'N/A')}\n"
        
        if result.get('anomaly_score') or result.get('anomaly_score') in (0, 0.0):
            text += f"異常分數: {result.get('anomaly_score')}\n"
        
        if result.get('detections'):
            text += f"檢測到的物件數量: {len(result.get('detections', []))}\n"
            for i, det in enumerate(result.get('detections', [])[:3]):  # 只顯示前3個
                text += f"  - 物件{i+1}: {det}\n"
        
        if result.get('missing_items'):
            text += f"缺失項目: {result.get('missing_items')}\n"
        
        # 補：將缺失項目以多行顯示（若為清單）
        try:
            items = result.get('missing_items')
            if isinstance(items, list) and items:
                text += "缺失項目:\n" + "\n".join(f"  - {x}" for x in items) + "\n"
        except Exception:
            pass

        text += "==================="
        
        self.result_text.setPlainText(text)

class DetectionSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_system = None
        self.worker = None
        self.available_products = []
        self.available_areas = {}
        self.current_result = None
        self.selected_image_path = None
        # Models base path and settings
        self._models_base = os.path.join(os.path.dirname(__file__), "models")
        self._settings = QSettings()
        
        self.init_ui()
        self.init_system()
        self.load_available_models()
        # 快捷鍵
        try:
            QShortcut(QKeySequence("F5"), self).activated.connect(self.load_available_models)
            QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_config)
            QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_config)
        except Exception:
            pass
        # Restore window geometry/state
        try:
            geo = self._settings.value("geometry")
            if geo is not None:
                self.restoreGeometry(geo)
            st = self._settings.value("windowState")
            if st is not None:
                self.restoreState(st)
        except Exception:
            pass
    
    def init_ui(self):
        self.setWindowTitle("AI 檢測系統 - PyQt 介面")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #dee2e6;
            }
            QComboBox {
                padding: 6px 12px;
                border: 1px solid #ced4da;
                border-radius: 4px;
                background-color: white;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #80bdff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # 中央元件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主要佈局
        main_layout = QHBoxLayout(central_widget)
        
        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 中間的圖像顯示區域
        image_area = self.create_image_area()
        main_layout.addWidget(image_area, 3)
        
        # 右側的資訊面板
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel, 1)
        
        # 狀態列
        self.statusBar().showMessage("系統就緒")
        
        # 選單列
        self.create_menu_bar()
    
    def create_control_panel(self):
        """建立控制面板"""
        panel = QGroupBox("控制面板")
        layout = QVBoxLayout()
        
        # 產品選擇
        product_group = QGroupBox("產品")
        product_layout = QVBoxLayout()
        
        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.on_product_changed)
        try:
            self.product_combo.currentTextChanged.connect(lambda _=None: self.update_start_enabled())
        except Exception:
            pass
        product_layout.addWidget(QLabel("產品:"))
        product_layout.addWidget(self.product_combo)
        
        product_group.setLayout(product_layout)
        layout.addWidget(product_group)
        
        # 區域選擇
        area_group = QGroupBox("檢測區域")
        area_layout = QVBoxLayout()
        
        self.area_combo = QComboBox()
        self.area_combo.currentTextChanged.connect(self.on_area_changed)
        try:
            self.area_combo.currentTextChanged.connect(lambda _=None: self.update_start_enabled())
        except Exception:
            pass
        area_layout.addWidget(QLabel("區域:"))
        area_layout.addWidget(self.area_combo)
        
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)
        
        # 推論類型選擇
        inference_group = QGroupBox("推論後端")
        inference_layout = QVBoxLayout()
        
        self.inference_combo = QComboBox()
        try:
            self.inference_combo.currentTextChanged.connect(lambda _=None: self.update_start_enabled())
        except Exception:
            pass
        inference_layout.addWidget(QLabel("後端:"))
        inference_layout.addWidget(self.inference_combo)
        
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)
        
        # 操作按鈕
        button_group = QGroupBox("操作")
        button_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("開始檢測")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("QPushButton { background-color: #28a745; }")
        
        self.stop_btn = QPushButton("停止檢測")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #dc3545; }")
        
        self.save_btn = QPushButton("儲存結果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_btn)

        # Camera/Image toggle
        try:
            self.use_camera_chk = QCheckBox("使用相機")
            self.use_camera_chk.setChecked(True)
            self.use_camera_chk.toggled.connect(self.on_use_camera_toggled)
            button_layout.addWidget(self.use_camera_chk)
        except Exception:
            pass
        
        # 影像選擇（可選）
        self.pick_image_btn = QPushButton("選擇影像...")
        self.pick_image_btn.setStyleSheet("QPushButton { background-color: #17a2b8; }")
        self.pick_image_btn.clicked.connect(self.pick_image)
        self.image_path_label = QLabel("未選擇影像 (使用攝影機/模擬影像)")
        self.image_path_label.setStyleSheet("color: #6c757d;")
        button_layout.addWidget(self.pick_image_btn)
        button_layout.addWidget(self.image_path_label)
        try:
            self.clear_image_btn = QPushButton("清除選圖")
            self.clear_image_btn.clicked.connect(self.clear_selected_image)
            self.clear_image_btn.setEnabled(False)
            button_layout.addWidget(self.clear_image_btn)
        except Exception:
            pass
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        panel.setMaximumWidth(250)
        
        return panel
    
    def create_image_area(self):
        """建立圖像顯示區域"""
        area = QGroupBox("圖像顯示")
        layout = QVBoxLayout()
        
        # 圖像顯示標籤頁
        self.image_tabs = QTabWidget()
        
        # 原始圖像
        self.original_image = ImageViewer("原始圖像")
        self.image_tabs.addTab(self.original_image, "原始")
        
        # 處理後圖像
        self.processed_image = ImageViewer("處理後圖像")
        self.image_tabs.addTab(self.processed_image, "處理後")

        # 檢測結果圖（YOLO 標註或 Anomalib 熱圖等）
        self.result_image = ImageViewer("結果圖")
        self.image_tabs.addTab(self.result_image, "結果")
        
        layout.addWidget(self.image_tabs)
        area.setLayout(layout)
        
        return area
    
    def create_info_panel(self):
        """建立資訊面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 系統狀態
        self.status_widget = StatusWidget()
        layout.addWidget(self.status_widget)
        
        # 結果顯示
        self.result_widget = ResultDisplayWidget()
        layout.addWidget(self.result_widget)
        
        # 日誌顯示
        log_group = QGroupBox("系統日誌")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setFont(QFont("Consolas", 8))
        self.log_text.setMaximumHeight(150)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 4px;
            }
        """)
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        panel.setLayout(layout)
        panel.setMaximumWidth(300)
        
        return panel
    
    def create_menu_bar(self):
        """建立選單列"""
        menubar = self.menuBar()
        
        # 檔案選單
        file_menu = menubar.addMenu('檔案')
        
        open_action = QAction('開啟設定檔', self)
        open_action.triggered.connect(self.open_config)
        file_menu.addAction(open_action)
        
        save_action = QAction('儲存設定', self)
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('結束', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 檢視選單
        view_menu = menubar.addMenu('檢視')
        
        refresh_action = QAction('重新載入模型', self)
        refresh_action.triggered.connect(self.load_available_models)
        view_menu.addAction(refresh_action)
        
        # 幫助選單
        help_menu = menubar.addMenu('幫助')
        
        about_action = QAction('關於', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_system(self):
        """初始化偵測系統"""
        try:
            self.detection_system = DetectionSystem()
            self.log_message("檢測系統初始化成功")
        except Exception as e:
            self.log_message(f"檢測系統初始化失敗: {str(e)}")
            QMessageBox.critical(self, "系統錯誤", f"無法初始化偵測系統\n{str(e)}")
    
    def load_available_models(self):
        """載入可用模型"""
        try:
            models_base = self._models_base
            if not os.path.exists(models_base):
                self.log_message("找不到 models 資料夾")
                return
            
            # 載入產品
            self.available_products = sorted([
                d for d in os.listdir(models_base)
                if os.path.isdir(os.path.join(models_base, d)) and not str(d).startswith('.')
            ])
            
            self.product_combo.clear()

            # 載入區域
            self.available_areas = {}
            for product in self.available_products:
                product_dir = os.path.join(models_base, product)
                areas = sorted([
                    d for d in os.listdir(product_dir)
                    if os.path.isdir(os.path.join(product_dir, d)) and not str(d).startswith('.')
                ])
                self.available_areas[product] = areas

            self.product_combo.addItems(self.available_products)

            if self.available_products:
                # restore last selections if any
                last_prod = str(self._settings.value("last_product", ""))
                last_area = str(self._settings.value("last_area", ""))
                last_infer = str(self._settings.value("last_infer", ""))

                if last_prod and last_prod in self.available_products:
                    self.product_combo.setCurrentText(last_prod)
                self.on_product_changed(self.product_combo.currentText())
                try:
                    if last_area and last_area in self.available_areas.get(self.product_combo.currentText(), []):
                        self.area_combo.setCurrentText(last_area)
                    self.on_area_changed(self.area_combo.currentText())
                    # set inference type after reload
                    if last_infer and last_infer in [self.inference_combo.itemText(i) for i in range(self.inference_combo.count())]:
                        self.inference_combo.setCurrentText(last_infer)
                except Exception:
                    pass
            
            self.log_message(f"載入了 {len(self.available_products)} 個模型")
            
        except Exception as e:
            self.log_message(f"載入模型時發生錯誤: {str(e)}")
    
    def on_product_changed(self, product):
        """產品選擇變更時的處理"""
        self.area_combo.clear()
        if product in self.available_areas:
            self.area_combo.addItems(self.available_areas[product])
    
    def on_area_changed(self, area):
        try:
            self.reload_inference_types()
        except Exception:
            pass

    def reload_inference_types(self):
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        self.inference_combo.clear()
        if not product or not area:
            return
        base = os.path.join(self._models_base, product, area)
        if os.path.isdir(base):
            types = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
            types.sort()
            if types:
                self.inference_combo.addItems(types)
        
    def update_start_enabled(self):
        """根據選擇是否完整，自動啟用/停用開始檢測按鈕"""
        try:
            ok = bool(
                self.product_combo.currentText().strip()
                and self.area_combo.currentText().strip()
                and self.inference_combo.currentText().strip()
            )
            self.start_btn.setEnabled(ok and not self.stop_btn.isEnabled())
        except Exception:
            pass

    def _load_image_with_retry(self, viewer, path, attempts=5, delay_ms=200, on_fail=None):
        """嘗試載入影像，若尚未寫入磁碟則以定時器重試。"""
        if not path:
            if on_fail:
                on_fail()
            else:
                viewer.setText(f'無可顯示{viewer.title}')
            return

        def attempt(remaining):
            if os.path.exists(path):
                viewer.set_image(path)
            elif remaining > 0:
                QTimer.singleShot(delay_ms, lambda: attempt(remaining - 1))
            else:
                if on_fail:
                    on_fail()
                else:
                    viewer.setText(f'無法載入{viewer.title}')

        attempt(max(0, int(attempts)))

    def start_detection(self):
        """開始檢測"""
        product = self.product_combo.currentText()
        area = self.area_combo.currentText()
        inference_type = self.inference_combo.currentText()
        
        if not all([product, area, inference_type]):
            QMessageBox.warning(self, "參數不完整", "請選擇完整的檢測參數")
            return
        
        # 檢查模型設定是否存在
        config_path = os.path.join(self._models_base, product, area, inference_type, "config.yaml")
        if not os.path.exists(config_path):
            QMessageBox.critical(self, "模型不存在", f"找不到對應的模型設定檔:\n{config_path}")
            return
        
        # 更新界面
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_widget.set_status("running")
        
        # 清空圖像
        self.original_image.clear()
        self.processed_image.clear()
        self.result_image.clear()
        
        self.log_message(f"開始檢測 - 產品: {product}, 區域: {area}, 類型: {inference_type}")
        
        # 建立背景執行緒
        frame = None
        # 若選擇使用相機，frame 保持 None，交由 CameraController 擷取
        use_cam = True
        try:
            use_cam = bool(self.use_camera_chk.isChecked())
        except Exception:
            pass
        if not use_cam:
            if getattr(self, 'selected_image_path', None) and os.path.exists(self.selected_image_path):
                try:
                    img = cv2.imread(self.selected_image_path)
                    if img is not None:
                        frame = img
                except Exception:
                    pass
            else:
                QMessageBox.warning(self, "未選影像", "請選擇影像，或切回使用相機")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_widget.set_status("idle")
                return
        
        self.worker = DetectionWorker(self.detection_system, product, area, inference_type, frame=frame)
        self.worker.result_ready.connect(self.on_detection_complete)
        self.worker.error_occurred.connect(self.on_detection_error)
        self.worker.start()
    
    def stop_detection(self):
        """停止檢測"""
        if self.worker and self.worker.isRunning():
            try:
                self.worker.cancel()
                self.worker.wait(3000)
            except Exception:
                self.worker.terminate()
                self.worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_widget.set_status("idle")
        self.log_message("檢測已取消")
    
    def on_detection_complete(self, result):

        """檢測完成回調"""

        self.current_result = result



        # 更新界面

        self.start_btn.setEnabled(True)

        self.stop_btn.setEnabled(False)

        self.save_btn.setEnabled(True)



        # 根據結果設定狀態

        status = result.get('status', 'ERROR')

        if status == 'PASS':

            self.status_widget.set_status("success")

        elif status == 'FAIL':

            self.status_widget.set_status("warning")

        elif status == 'ERROR':

            self.status_widget.set_status("error")

        else:

            self.status_widget.set_status("warning")



        # 更新結果顯示

        self.result_widget.update_result(result)



        # 顯示圖像（考慮非同步寫檔延遲）

        self._load_image_with_retry(

            self.original_image,

            result.get('original_image_path'),

            on_fail=lambda: self.original_image.setText('尚無原始影像（可能未保存）')

        )



        self._load_image_with_retry(

            self.processed_image,

            result.get('preprocessed_image_path'),

            on_fail=lambda: self.processed_image.setText('尚無預處理影像')

        )



        annotated_path = result.get('annotated_path')

        heatmap_path = result.get('heatmap_path')



        def show_result_frame():

            rf = result.get('result_frame', None)

            if isinstance(rf, np.ndarray) and getattr(rf, 'size', 0) > 0:

                try:

                    import tempfile

                    temp_dir = os.path.join(tempfile.gettempdir(), 'ai_detect_preview')

                    os.makedirs(temp_dir, exist_ok=True)

                    fname = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"

                    temp_path = os.path.join(temp_dir, fname)

                    cv2.imwrite(temp_path, rf)

                    self.result_image.set_image(temp_path)

                except Exception:

                    self.result_image.setText('無法載入結果影像')

            else:

                self.result_image.setText('無法載入結果影像')



        def load_heatmap():

            if heatmap_path:

                self._load_image_with_retry(

                    self.result_image,

                    heatmap_path,

                    attempts=3,

                    delay_ms=200,

                    on_fail=show_result_frame

                )

            else:

                show_result_frame()



        self._load_image_with_retry(

            self.result_image,

            annotated_path,

            attempts=5,

            delay_ms=200,

            on_fail=load_heatmap

        )



        self.log_message(f"檢測完成 - 狀態: {status}")



        # 狀態列

        self.statusBar().showMessage(f"檢測完成 - {status}", 5000)



    def on_detection_error(self, error_msg):
        """檢測錯誤回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_widget.set_status("error")
        
        self.log_message(f"檢測錯誤: {error_msg}")
        QMessageBox.critical(self, "檢測錯誤", f"檢測過程中發生錯誤\n{error_msg}")
    
    def save_results(self):
        """儲存結果"""
        if not self.current_result:
            QMessageBox.warning(self, "沒有結果", "沒有可供儲存的檢測結果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "儲存檢測結果", 
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False, indent=2)
                self.log_message(f"結果已儲存: {file_path}")
                QMessageBox.information(self, "儲存成功", f"檢測結果已儲存至:\n{file_path}")
            except Exception as e:
                self.log_message(f"儲存時發生錯誤: {str(e)}")
                QMessageBox.critical(self, "儲存錯誤", f"無法儲存結果:\n{str(e)}")
    
    def pick_image(self):
        """選擇影像"""
        fname, _ = QFileDialog.getOpenFileName(self, "選擇影像", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not fname:
            return
        self.selected_image_path = fname
        try:
            self.image_path_label.setText(os.path.basename(fname))
        except Exception:
            pass
        try:
            self.original_image.set_image(fname)
        except Exception:
            pass

    def clear_selected_image(self):
        """清除當前選擇影像並切回相機"""
        self.selected_image_path = None
        try:
            self.image_path_label.setText("尚未選擇影像（可切換使用相機）")
        except Exception:
            pass
        try:
            if getattr(self, 'clear_image_btn', None):
                self.clear_image_btn.setEnabled(False)
            if getattr(self, 'use_camera_chk', None):
                self.use_camera_chk.setChecked(True)
        except Exception:
            pass

    def on_use_camera_toggled(self, checked):
        """切換 使用相機 / 使用影像 模式"""
        try:
            if checked:
                # 簡易偵測相機可用性
                ok = False
                try:
                    cap = cv2.VideoCapture(0)
                    ok = cap is not None and cap.isOpened()
                    if cap:
                        cap.release()
                except Exception:
                    ok = False
                if not ok:
                    QMessageBox.warning(self, "相機不可用", "找不到可用相機，請改用選圖模式。")
                    self.use_camera_chk.setChecked(False)
                    return
                # 使用相機 → 停用清除鈕與選圖鈕
                self.selected_image_path = None
                if getattr(self, 'pick_image_btn', None):
                    self.pick_image_btn.setEnabled(False)
                if getattr(self, 'clear_image_btn', None):
                    self.clear_image_btn.setEnabled(False)
                if getattr(self, 'image_path_label', None):
                    self.image_path_label.setText("使用相機輸入（可切換選圖）")
            else:
                # 使用影像 → 啟用選圖鈕
                if getattr(self, 'pick_image_btn', None):
                    self.pick_image_btn.setEnabled(True)
                if getattr(self, 'clear_image_btn', None):
                    self.clear_image_btn.setEnabled(bool(self.selected_image_path))
                if not self.selected_image_path and getattr(self, 'image_path_label', None):
                    self.image_path_label.setText("請選擇影像（或切回相機）")
        except Exception:
            pass

    def open_config(self):
        """開啟設定檔"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟設定檔", "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.log_message(f"載入設定檔: {file_path}")
    
    def save_config(self):
        """儲存設定"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "儲存設定", "config.yaml", "YAML files (*.yaml)"
        )
        if file_path:
            self.log_message(f"設定已儲存: {file_path}")
    
    def show_about(self):
        """顯示關於資訊"""
        QMessageBox.about(self, "關於", 
                         "AI 檢測系統 PyQt 介面\n\n"
                         "版本: 1.0\n"
                         "支援 YOLO 和 Anomalib 模型\n"
                         "提供完整的視覺化檢測結果顯示")
    
    def log_message(self, message):
        """記錄日誌訊息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """關閉事件處理"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '確認離開', 
                                       '檢測正在執行中，是否確定要離開？',
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.Yes:
                try:
                    self.worker.cancel()
                    self.worker.wait(3000)
                except Exception:
                    self.worker.terminate()
                    self.worker.wait()
            else:
                event.ignore()
                return
        
        if self.detection_system:
            self.detection_system.shutdown()
        # Save window + last selections
        try:
            self._settings.setValue("geometry", self.saveGeometry())
            self._settings.setValue("windowState", self.saveState())
            self._settings.setValue("last_product", self.product_combo.currentText())
            self._settings.setValue("last_area", self.area_combo.currentText())
            self._settings.setValue("last_infer", self.inference_combo.currentText())
        except Exception:
            pass

        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 設定應用程式資訊
    app.setApplicationName("AI 檢測系統")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AI Detection Lab")
    
    # 設定預設字體
    font = QFont("Microsoft JhengHei", 9)
    app.setFont(font)
    
    # 建立主視窗
    window = DetectionSystemGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
