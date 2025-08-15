import sys
import os
import json
import logging
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from PIL import Image
import traceback

# 導入您的 DetectionSystem 類別
try:
    from main import DetectionSystem
except ImportError as e:
    print(f"無法導入 DetectionSystem: {e}")
    print("請確保 main.py 檔案在同一目錄中")
    sys.exit(1)

class DetectionWorker(QThread):
    """背景執行檢測的工作執行緒"""
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, detection_system, product, area, inference_type):
        super().__init__()
        self.detection_system = detection_system
        self.product = product
        self.area = area
        self.inference_type = inference_type
    
    def run(self):
        try:
            result = self.detection_system.detect(self.product, self.area, self.inference_type)
            self.result_ready.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

class StatusWidget(QWidget):
    """狀態顯示元件"""
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 狀態標題
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
        
        # 先創建所有元件，再設定初始狀態
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
        
        # 在所有元件創建完成後設定初始狀態
        self.set_status("idle")
    
    def set_status(self, status):
        status_colors = {
            "idle": ("#6c757d", "系統就緒"),
            "running": ("#ffc107", "檢測中..."),
            "success": ("#28a745", "檢測完成"),
            "error": ("#dc3545", "檢測錯誤"),
            "warning": ("#fd7e14", "警告")
        }
        
        color, text = status_colors.get(status, ("#6c757d", "未知狀態"))
        self.status_indicator.setStyleSheet(f"color: {color};")
        self.status_text.setText(text)

class ImageViewer(QLabel):
    """圖像顯示元件"""
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
        self.setText(f"暫無{self.title}")
        self.setScaledContents(True)
    
    def set_image(self, image_path):
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
        else:
            self.setText(f"無法載入{self.title}")

class ResultDisplayWidget(QWidget):
    """結果顯示元件"""
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
        
        layout.addWidget(title)
        layout.addWidget(self.result_text)
        self.setLayout(layout)
    
    def update_result(self, result):
        """更新檢測結果顯示"""
        text = "=== 檢測結果 ===\n"
        text += f"狀態: {result.get('status', 'N/A')}\n"
        text += f"產品: {result.get('product', 'N/A')}\n"
        text += f"區域: {result.get('area', 'N/A')}\n"
        text += f"推理類型: {result.get('inference_type', 'N/A')}\n"
        text += f"模型路徑: {result.get('ckpt_path', 'N/A')}\n"
        
        if result.get('anomaly_score'):
            text += f"異常分數: {result.get('anomaly_score')}\n"
        
        if result.get('detections'):
            text += f"檢測到的物件: {len(result.get('detections', []))}\n"
            for i, det in enumerate(result.get('detections', [])[:3]):  # 只顯示前3個
                text += f"  - 物件{i+1}: {det}\n"
        
        if result.get('missing_items'):
            text += f"缺少項目: {result.get('missing_items')}\n"
        
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
        
        self.init_ui()
        self.init_system()
        self.load_available_models()
    
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
        
        # 左側控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # 中間圖像顯示區域
        image_area = self.create_image_area()
        main_layout.addWidget(image_area, 3)
        
        # 右側狀態和結果面板
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel, 1)
        
        # 狀態列
        self.statusBar().showMessage("系統初始化完成")
        
        # 選單列
        self.create_menu_bar()
    
    def create_control_panel(self):
        """創建控制面板"""
        panel = QGroupBox("檢測控制")
        layout = QVBoxLayout()
        
        # 產品選擇
        product_group = QGroupBox("產品選擇")
        product_layout = QVBoxLayout()
        
        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(self.on_product_changed)
        product_layout.addWidget(QLabel("選擇產品:"))
        product_layout.addWidget(self.product_combo)
        
        product_group.setLayout(product_layout)
        layout.addWidget(product_group)
        
        # 區域選擇
        area_group = QGroupBox("檢測區域")
        area_layout = QVBoxLayout()
        
        self.area_combo = QComboBox()
        area_layout.addWidget(QLabel("選擇區域:"))
        area_layout.addWidget(self.area_combo)
        
        area_group.setLayout(area_layout)
        layout.addWidget(area_group)
        
        # 推理類型選擇
        inference_group = QGroupBox("推理類型")
        inference_layout = QVBoxLayout()
        
        self.inference_combo = QComboBox()
        self.inference_combo.addItems(["yolo", "anomalib"])
        inference_layout.addWidget(QLabel("選擇推理類型:"))
        inference_layout.addWidget(self.inference_combo)
        
        inference_group.setLayout(inference_layout)
        layout.addWidget(inference_group)
        
        # 控制按鈕
        button_group = QGroupBox("操作控制")
        button_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("開始檢測")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setStyleSheet("QPushButton { background-color: #28a745; }")
        
        self.stop_btn = QPushButton("停止檢測")
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #dc3545; }")
        
        self.save_btn = QPushButton("保存結果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.save_btn)
        
        button_group.setLayout(button_layout)
        layout.addWidget(button_group)
        
        layout.addStretch()
        panel.setLayout(layout)
        panel.setMaximumWidth(250)
        
        return panel
    
    def create_image_area(self):
        """創建圖像顯示區域"""
        area = QGroupBox("圖像顯示")
        layout = QVBoxLayout()
        
        # 圖像顯示標籤頁
        self.image_tabs = QTabWidget()
        
        # 原始圖像
        self.original_image = ImageViewer("原始圖像")
        self.image_tabs.addTab(self.original_image, "原始圖像")
        
        # 處理後圖像
        self.processed_image = ImageViewer("處理後圖像")
        self.image_tabs.addTab(self.processed_image, "處理後圖像")

        # 檢測圖像
        self.heatmap_image = ImageViewer("檢測圖像")
        self.image_tabs.addTab(self.heatmap_image, "檢測圖像")
        
        layout.addWidget(self.image_tabs)
        area.setLayout(layout)
        
        return area
    
    def create_info_panel(self):
        """創建資訊面板"""
        panel = QWidget()
        layout = QVBoxLayout()
        
        # 狀態顯示
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
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        panel.setLayout(layout)
        panel.setMaximumWidth(300)
        
        return panel
    
    def create_menu_bar(self):
        """創建選單列"""
        menubar = self.menuBar()
        
        # 檔案選單
        file_menu = menubar.addMenu('檔案')
        
        open_action = QAction('開啟設定檔', self)
        open_action.triggered.connect(self.open_config)
        file_menu.addAction(open_action)
        
        save_action = QAction('保存設定', self)
        save_action.triggered.connect(self.save_config)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('退出', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 檢視選單
        view_menu = menubar.addMenu('檢視')
        
        refresh_action = QAction('重新整理模型', self)
        refresh_action.triggered.connect(self.load_available_models)
        view_menu.addAction(refresh_action)
        
        # 說明選單
        help_menu = menubar.addMenu('說明')
        
        about_action = QAction('關於', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_system(self):
        """初始化檢測系統"""
        try:
            self.detection_system = DetectionSystem()
            self.log_message("檢測系統初始化成功")
        except Exception as e:
            self.log_message(f"檢測系統初始化失敗: {str(e)}")
            QMessageBox.critical(self, "初始化錯誤", f"無法初始化檢測系統:\n{str(e)}")
    
    def load_available_models(self):
        """載入可用的模型"""
        try:
            models_base = os.path.join(os.path.dirname(__file__), "models")
            if not os.path.exists(models_base):
                self.log_message("未找到 models 資料夾")
                return
            
            # 載入產品
            self.available_products = [
                d for d in os.listdir(models_base)
                if os.path.isdir(os.path.join(models_base, d))
            ]
            
            self.product_combo.clear()
            self.product_combo.addItems(self.available_products)
            
            # 載入區域
            self.available_areas = {}
            for product in self.available_products:
                product_dir = os.path.join(models_base, product)
                areas = [
                    d for d in os.listdir(product_dir)
                    if os.path.isdir(os.path.join(product_dir, d))
                ]
                self.available_areas[product] = areas
            
            self.log_message(f"載入了 {len(self.available_products)} 個產品模型")
            
        except Exception as e:
            self.log_message(f"載入模型失敗: {str(e)}")
    
    def on_product_changed(self, product):
        """產品選擇改變時的處理"""
        self.area_combo.clear()
        if product in self.available_areas:
            self.area_combo.addItems(self.available_areas[product])
    
    def start_detection(self):
        """開始檢測"""
        product = self.product_combo.currentText()
        area = self.area_combo.currentText()
        inference_type = self.inference_combo.currentText()
        
        if not all([product, area, inference_type]):
            QMessageBox.warning(self, "參數不完整", "請選擇完整的檢測參數")
            return
        
        # 檢查模型配置是否存在
        config_path = os.path.join("models", product, area, inference_type, "config.yaml")
        if not os.path.exists(config_path):
            QMessageBox.critical(self, "模型不存在", f"找不到模型配置檔案:\n{config_path}")
            return
        
        # 更新界面狀態
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_widget.set_status("running")
        
        # 清除之前的圖像
        self.original_image.clear()
        self.processed_image.clear()
        self.heatmap_image.clear()
        
        self.log_message(f"開始檢測 - 產品: {product}, 區域: {area}, 類型: {inference_type}")
        
        # 創建工作執行緒
        self.worker = DetectionWorker(self.detection_system, product, area, inference_type)
        self.worker.result_ready.connect(self.on_detection_complete)
        self.worker.error_occurred.connect(self.on_detection_error)
        self.worker.start()
    
    def stop_detection(self):
        """停止檢測"""
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_widget.set_status("idle")
        self.log_message("檢測已停止")
    
    def on_detection_complete(self, result):
        """檢測完成回調"""
        self.current_result = result
        
        # 更新界面狀態
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        
        # 根據結果設定狀態
        status = result.get('status', 'ERROR')
        if status == 'OK':
            self.status_widget.set_status("success")
        elif status == 'ERROR':
            self.status_widget.set_status("error")
        else:
            self.status_widget.set_status("warning")
        
        # 更新結果顯示
        self.result_widget.update_result(result)
        
        # 顯示圖像
        if result.get('original_image_path'):
            self.original_image.set_image(result['original_image_path'])
        
        if result.get('preprocessed_image_path'):
            self.processed_image.set_image(result['preprocessed_image_path'])
        
        if result.get('heatmap_path'):
            self.heatmap_image.set_image(result['heatmap_path'])
        
        self.log_message(f"檢測完成 - 狀態: {status}")
        
        # 狀態列訊息
        self.statusBar().showMessage(f"檢測完成 - {status}", 5000)
    
    def on_detection_error(self, error_msg):
        """檢測錯誤回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_widget.set_status("error")
        
        self.log_message(f"檢測錯誤: {error_msg}")
        QMessageBox.critical(self, "檢測錯誤", f"檢測過程中發生錯誤:\n{error_msg}")
    
    def save_results(self):
        """保存結果"""
        if not self.current_result:
            QMessageBox.warning(self, "無結果", "沒有可保存的檢測結果")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存檢測結果", 
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.current_result, f, ensure_ascii=False, indent=2)
                self.log_message(f"結果已保存至: {file_path}")
                QMessageBox.information(self, "保存成功", f"檢測結果已保存至:\n{file_path}")
            except Exception as e:
                self.log_message(f"保存失敗: {str(e)}")
                QMessageBox.critical(self, "保存錯誤", f"無法保存結果:\n{str(e)}")
    
    def open_config(self):
        """開啟設定檔"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "開啟設定檔", "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.log_message(f"載入設定檔: {file_path}")
    
    def save_config(self):
        """保存設定"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存設定", "config.yaml", "YAML files (*.yaml)"
        )
        if file_path:
            self.log_message(f"設定已保存至: {file_path}")
    
    def show_about(self):
        """顯示關於對話框"""
        QMessageBox.about(self, "關於", 
                         "AI 檢測系統 PyQt 介面\n\n"
                         "版本: 1.0\n"
                         "支援 YOLO 和 Anomalib 模型\n"
                         "提供完整的檢測功能和結果顯示")
    
    def log_message(self, message):
        """添加日誌訊息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
    
    def closeEvent(self, event):
        """關閉事件處理"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(self, '確認退出', 
                                       '檢測正在進行中，確定要退出嗎？',
                                       QMessageBox.Yes | QMessageBox.No,
                                       QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait()
            else:
                event.ignore()
                return
        
        if self.detection_system:
            self.detection_system.shutdown()
        
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # 設定應用程式資訊
    app.setApplicationName("AI 檢測系統")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AI Detection Lab")
    
    # 設定全域字體
    font = QFont("Microsoft JhengHei", 9)
    app.setFont(font)
    
    # 創建主視窗
    window = DetectionSystemGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()