"""以 PyQt5 實作的桌面介面，用於操作與監看檢測流程。"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from app.gui.controller import DetectionController, ModelCatalog
from app.gui.preferences import PreferencesManager
from app.gui.utils import load_image_with_retry
from app.gui.view_builder import (
    build_control_panel,
    build_image_area,
    build_info_panel,
    build_menu_bar,
)


class DetectionSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_system = None
        self.worker = None
        self.available_products = []
        self.available_areas = {}
        self.current_result = None
        self.selected_image_path = None
        self.use_camera_chk = None
        self.reconnect_camera_btn = None
        self.disconnect_camera_btn = None
        # Models base path and settings
        self._project_root = Path(__file__).resolve().parents[2]
        self._config_path = self._project_root / "config.yaml"
        self._models_base = self._project_root / "models"
        self.preferences = PreferencesManager(QSettings())
        self._logger = logging.getLogger(__name__)
        self._catalog = ModelCatalog(self._models_base)
        self.controller = DetectionController(
            self._config_path, self._catalog, logger=self._logger
        )

        self.init_ui()
        self.update_camera_controls()
        self.init_system()
        self.load_available_models()
        # 快捷鍵
        try:
            QShortcut(QKeySequence("F5"), self).activated.connect(
                self.load_available_models
            )
            QShortcut(QKeySequence("Ctrl+O"),
                      self).activated.connect(self.open_config)
            QShortcut(QKeySequence("Ctrl+S"),
                      self).activated.connect(self.save_config)
        except Exception:
            pass
        # Restore window geometry/state
        try:
            geo, window_state = self.preferences.restore_window_state()
            if geo is not None:
                self.restoreGeometry(geo)
            if window_state is not None:
                self.restoreState(window_state)
        except Exception:
            pass

    def init_ui(self):
        self.setWindowTitle("AI 檢測系統 - PyQt 介面")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(
            """
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
        """
        )

        # 中央元件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主要佈局
        main_layout = QHBoxLayout(central_widget)

        # 控制面板
        control_panel = build_control_panel(self)
        main_layout.addWidget(control_panel, 1)

        # 中間的圖像顯示區域
        image_area = build_image_area(self)
        main_layout.addWidget(image_area, 3)

        # 右側的資訊面板
        info_panel = build_info_panel(self)
        main_layout.addWidget(info_panel, 1)

        # 狀態列
        self.statusBar().showMessage("系統就緒")

        # 選單列
        build_menu_bar(self)

    def init_system(self):
        """初始化偵測系統"""
        try:
            self.detection_system = self.controller.detection_system
            self.log_message("檢測系統初始化成功")
            if (
                getattr(self, "use_camera_chk", None)
                and self.detection_system
                and self.detection_system.is_camera_connected()
            ):
                self.use_camera_chk.setChecked(True)
        except Exception as e:
            self.log_message(f"檢測系統初始化失敗: {str(e)}")
            QMessageBox.critical(self, "系統錯誤", f"無法初始化偵測系統\n{str(e)}")

        finally:
            try:
                self.update_camera_controls()
            except Exception:
                pass


    def load_available_models(self):
        """載入可用模型資訊"""
        try:
            if not self._models_base.exists():
                self.log_message("找不到 models 資料夾")
                return

            self._catalog.refresh()
            self.available_products = self._catalog.products()
            self.product_combo.clear()

            self.available_areas = {
                product: self._catalog.areas(product)
                for product in self.available_products
            }

            if self.available_products:
                self.product_combo.addItems(self.available_products)

                last_prod, last_area, last_infer = self.preferences.restore_last_selection()

                if last_prod and last_prod in self.available_products:
                    self.product_combo.setCurrentText(last_prod)

                self.on_product_changed(self.product_combo.currentText())

                try:
                    if last_area and last_area in self.available_areas.get(
                        self.product_combo.currentText(), []
                    ):
                        self.area_combo.setCurrentText(last_area)

                    self.on_area_changed(self.area_combo.currentText())

                    available_types = [
                        self.inference_combo.itemText(i)
                        for i in range(self.inference_combo.count())
                    ]
                    if last_infer and last_infer in available_types:
                        self.inference_combo.setCurrentText(last_infer)
                except Exception:
                    pass
            else:
                self.product_combo.clear()

            self.log_message(f"載入了 {len(self.available_products)} 組模型設定")

        except Exception as exc:
            self.log_message(f"載入模型資料發生錯誤: {exc}")
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
        types = self._catalog.inference_types(product, area)
        if types:
            self.inference_combo.addItems(types)


    def is_detection_running(self) -> bool:
        """Return True if a detection worker is running."""
        return bool(self.worker and self.worker.isRunning())

    def update_camera_controls(self):
        """Sync camera-related widgets with the current state."""
        try:
            running = self.is_detection_running()
            camera_connected = False
            if self.controller.has_system():
                camera_connected = self.controller.is_camera_connected()
            if getattr(self, "reconnect_camera_btn", None):
                self.reconnect_camera_btn.setEnabled(
                    not running
                )
            if getattr(self, "disconnect_camera_btn", None):
                self.disconnect_camera_btn.setEnabled(
                    camera_connected and not running
                )
            if getattr(self, "use_camera_chk", None):
                if not camera_connected and self.use_camera_chk.isChecked():
                    self.use_camera_chk.blockSignals(True)
                    self.use_camera_chk.setChecked(False)
                    self.use_camera_chk.blockSignals(False)
                self.use_camera_chk.setEnabled(camera_connected and not running)
            if not camera_connected:
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(True)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(
                        bool(getattr(self, "selected_image_path", None))
                    )
        except Exception:
            pass



    def handle_reconnect_camera(self):
        """Manually reconnect the camera via detection system."""
        if self.is_detection_running():
            QMessageBox.warning(self, "Camera Operation", "Detection in progress. Stop before reconnecting.")
            return
        if not self.controller.has_system():
            self.init_system()
            if not self.controller.has_system():
                return
        self.log_message("Attempting to reconnect the camera...")
        try:
            success = self.controller.reconnect_camera()
        except Exception as exc:
            success = False
            self.log_message(f"Camera reconnect failed: {exc}")
            QMessageBox.critical(self, "Camera Error", f"Reconnect failed:\n{exc}")
        else:
            if success:
                self.log_message("Camera reconnected successfully")
                QMessageBox.information(self, "Camera Status", "Camera reconnected.")
                if getattr(self, "use_camera_chk", None):
                    self.use_camera_chk.blockSignals(True)
                    self.use_camera_chk.setChecked(True)
                    self.use_camera_chk.blockSignals(False)
                    self.on_use_camera_toggled(True)
            else:
                self.log_message("Camera reconnect failed")
                QMessageBox.critical(
                    self,
                    "Camera Error",
                    "Reconnect failed. Please check hardware or settings.",
                )
        self.update_camera_controls()

    def handle_disconnect_camera(self):
        """Allow user to disconnect the camera manually."""
        if self.is_detection_running():
            QMessageBox.warning(self, "Camera Operation", "Detection in progress. Stop before disconnecting.")
            return
        if not self.controller.has_system():
            QMessageBox.information(self, "Camera Status", "Detection system is not initialized yet.")
            return
        self.log_message("Disconnecting camera...")
        try:
            self.controller.disconnect_camera()
            self.log_message("Camera disconnected")
            QMessageBox.information(self, "Camera Status", "Camera disconnected.")
        except Exception as exc:
            self.log_message(f"Camera disconnect failed: {exc}")
            QMessageBox.critical(self, "Camera Error", f"Disconnect failed:\n{exc}")
        if getattr(self, "use_camera_chk", None):
            self.use_camera_chk.blockSignals(True)
            self.use_camera_chk.setChecked(False)
            self.use_camera_chk.blockSignals(False)
            self.on_use_camera_toggled(False)
        self.update_camera_controls()
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



    def start_detection(self):
        """Launch detection workflow."""
        product = self.product_combo.currentText()
        area = self.area_combo.currentText()
        inference_type = self.inference_combo.currentText()

        if not all([product, area, inference_type]):
            QMessageBox.warning(self, "Missing Parameters", "Select product, area, and inference type before starting detection.")
            return

        if not self._catalog.config_exists(product, area, inference_type):
            config_path = self._catalog.config_path(product, area, inference_type)
            QMessageBox.critical(
                self,
                "Model Not Found",
                f"Configuration file not found:\n{config_path}",
            )
            return

        if not self.controller.has_system():
            QMessageBox.critical(
                self,
                "Detection System",
                "Detection system not initialized. Please verify configuration and try again.",
            )
            self.init_system()
            if not self.controller.has_system():
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_widget.set_status("error")
                self.update_camera_controls()
                return
        else:
            self.detection_system = self.controller.detection_system

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_widget.set_status("running")
        self.update_camera_controls()

        self.original_image.clear()
        self.processed_image.clear()
        self.result_image.clear()

        self.log_message(
            f"Start detection - product: {product}, area: {area}, type: {inference_type}"
        )

        frame = None
        use_cam = True
        try:
            use_cam = bool(self.use_camera_chk.isChecked())
        except Exception:
            pass

        if not use_cam:
            selected = getattr(self, "selected_image_path", None)
            if selected:
                image = self.controller.load_image(Path(selected))
                if image is not None:
                    frame = image
            if frame is None:
                QMessageBox.warning(self, "Input Source", "Select an image file or enable camera input.")
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.status_widget.set_status("idle")
                self.update_camera_controls()
                return

        self.worker = self.controller.build_worker(
            product,
            area,
            inference_type,
            frame=frame,
        )
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
        self.update_camera_controls()
        self.log_message("檢測已取消")

    def on_detection_complete(self, result):
        """檢測完成回調"""

        self.current_result = result

        # 更新界面

        self.start_btn.setEnabled(True)

        self.stop_btn.setEnabled(False)

        self.save_btn.setEnabled(True)
        self.update_camera_controls()

        # 根據結果設定狀態

        status = result.get("status", "ERROR")

        if status == "PASS":

            self.status_widget.set_status("success")

        elif status == "FAIL":

            self.status_widget.set_status("warning")

        elif status == "ERROR":

            self.status_widget.set_status("error")

        else:

            self.status_widget.set_status("warning")

        # 更新結果顯示

        self.result_widget.update_result(result)

        # 顯示圖像（考慮非同步寫檔延遲）

        load_image_with_retry(
            self.original_image,
            result.get("original_image_path"),
            on_fail=lambda: self.original_image.setText("尚無原始影像（可能未保存）"),
        )

        load_image_with_retry(
            self.processed_image,
            result.get("preprocessed_image_path"),
            on_fail=lambda: self.processed_image.setText("尚無預處理影像"),
        )

        annotated_path = result.get("annotated_path")

        heatmap_path = result.get("heatmap_path")

        def show_result_frame():

            rf = result.get("result_frame", None)

            if isinstance(rf, np.ndarray) and getattr(rf, "size", 0) > 0:

                try:

                    import tempfile

                    temp_dir = os.path.join(
                        tempfile.gettempdir(), "ai_detect_preview")

                    os.makedirs(temp_dir, exist_ok=True)

                    fname = (
                        f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    )

                    temp_path = os.path.join(temp_dir, fname)

                    cv2.imwrite(temp_path, rf)

                    self.result_image.set_image(temp_path)

                except Exception:

                    self.result_image.setText("無法載入結果影像")

            else:

                self.result_image.setText("無法載入結果影像")

        def load_heatmap():

            if heatmap_path:

                load_image_with_retry(
                    self.result_image,
                    heatmap_path,
                    attempts=3,
                    delay_ms=200,
                    on_fail=show_result_frame,
                )

            else:

                show_result_frame()

        load_image_with_retry(
            self.result_image,
            annotated_path,
            attempts=5,
            delay_ms=200,
            on_fail=load_heatmap,
        )

        self.log_message(f"檢測完成 - 狀態: {status}")

        # 狀態列

        self.statusBar().showMessage(f"檢測完成 - {status}", 5000)

    def on_detection_error(self, error_msg):
        """檢測錯誤回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_widget.set_status("error")
        self.update_camera_controls()

        self.log_message(f"檢測錯誤: {error_msg}")
        QMessageBox.critical(self, "檢測錯誤", f"檢測過程中發生錯誤\n{error_msg}")


        
    def save_results(self):
        """Save the latest detection result to disk."""
        if not self.current_result:
            QMessageBox.warning(self, "No Result", "There is no detection result to save yet.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detection Result",
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)",
        )

        if file_path:
            try:
                self.controller.save_result_json(Path(file_path), self.current_result)
                self.log_message(f"Result saved to {file_path}")
                QMessageBox.information(
                    self, "Save Successful", f"Detection result saved to:\n{file_path}"
                )
            except Exception as exc:
                self.log_message(f"Failed to save result: {exc}")
                QMessageBox.critical(self, "Save Error", f"Unable to save result:\n{exc}")
    def pick_image(self):
        """選擇影像"""
        fname, _ = QFileDialog.getOpenFileName(
            self, "選擇影像", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
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
            if getattr(self, "clear_image_btn", None):
                self.clear_image_btn.setEnabled(False)
            if getattr(self, "use_camera_chk", None):
                camera_ready = self.controller.is_camera_connected() if self.controller.has_system() else False
                if camera_ready:
                    self.use_camera_chk.setChecked(True)
                else:
                    self.use_camera_chk.setChecked(False)
            self.update_camera_controls()
        except Exception:
            pass

    def on_use_camera_toggled(self, checked):
        """切換 使用相機 / 使用影像 模式"""
        try:
            if self.is_detection_running():
                return
            camera_ready = False
            if self.controller.has_system():
                camera_ready = self.controller.is_camera_connected()
            if checked:
                if not camera_ready:
                    QMessageBox.warning(
                        self, "相機不可用", "目前相機尚未連線，請先重新連線。"
                    )
                    if getattr(self, "use_camera_chk", None):
                        self.use_camera_chk.blockSignals(True)
                        self.use_camera_chk.setChecked(False)
                        self.use_camera_chk.blockSignals(False)
                    self.log_message("切換相機模式失敗：相機未連線")
                    return
                self.selected_image_path = None
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(False)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(False)
                if getattr(self, "image_path_label", None):
                    self.image_path_label.setText("使用相機輸入（檢測時自動拍攝）")
            else:
                if getattr(self, "pick_image_btn", None):
                    self.pick_image_btn.setEnabled(True)
                if getattr(self, "clear_image_btn", None):
                    self.clear_image_btn.setEnabled(
                        bool(self.selected_image_path)
                    )
                if not self.selected_image_path and getattr(
                    self, "image_path_label", None
                ):
                    self.image_path_label.setText("請選擇影像或重新連線相機")
        except Exception:
            pass
        finally:
            try:
                self.update_camera_controls()
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
        QMessageBox.about(
            self,
            "關於",
            "AI 檢測系統 PyQt 介面\n\n"
            "版本: 1.0\n"
            "支援 YOLO 和 Anomalib 模型\n"
            "提供完整的視覺化檢測結果顯示",
        )

    def log_message(self, message):
        """記錄日誌訊息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event):
        """關閉事件處理"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "確認離開",
                "檢測正在執行中，是否確定要離開？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
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

        if self.controller.has_system():
            self.controller.shutdown()
        try:
            self.preferences.save_window_state(self.saveGeometry(), self.saveState())
            self.preferences.save_last_selection(
                self.product_combo.currentText(),
                self.area_combo.currentText(),
                self.inference_combo.currentText(),
            )
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
