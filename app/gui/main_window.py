"""以 PyQt5 實作的桌面介面，用於操作與監看檢測流程。"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.types import DetectionResult


def _get_detection_class():
    try:
        mod = sys.modules.get("GUI")
        if mod:
            det_cls = getattr(mod, "DetectionSystem", None)
            if det_cls:
                return det_cls
        gui_mod = importlib.import_module("GUI")
        det_cls = getattr(gui_mod, "DetectionSystem", None)
        if det_cls:
            return det_cls
    except Exception:
        pass
    from core.detection_system import DetectionSystem as _CoreDetectionSystem

    return _CoreDetectionSystem


import cv2
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from app.gui.controller import DetectionController
from core.services.model_catalog import ModelCatalog
from app.gui.preferences import PreferencesManager
from app.gui.panels.control_panel import ControlPanel
from app.gui.panels.image_panel import ImagePanel
from app.gui.panels.info_panel import InfoPanel
from app.gui.utils import load_image_with_retry
from app.gui.view_builder import build_menu_bar


class DetectionSystemGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detection_system = None
        self.worker = None
        self.available_products = []
        self.available_areas = {}
        self.current_result: DetectionResult | None = None
        self.selected_image_path = None
        self.model_loader = None
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
            self._config_path,
            self._catalog,
            logger=self._logger,
            detection_cls=_get_detection_class(),
        )
        self._skip_system_init = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        self.init_ui()
        self.update_camera_controls()
        if not self._skip_system_init:
            self.init_system()
            self.load_available_models()
        else:
            # In test mode, prepare a lightweight stub system if available
            stub_cls = getattr(self.controller, "_detection_cls", None)
            if stub_cls:
                try:
                    self.controller._system = stub_cls()  # type: ignore[attr-defined]
                except Exception:
                    try:
                        self.controller._system = stub_cls(
                            config_path=str(self._config_path)
                        )  # type: ignore[attr-defined]
                    except Exception:
                        self.controller._system = None
            try:
                self.detection_system = self.controller.detection_system
            except Exception:
                self.detection_system = None
            self.log_message("Skip init_system (test mode)")
        
        # 快捷鍵
        try:
            QShortcut(QKeySequence("F5"), self).activated.connect(
                self.load_available_models
            )
            QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_config)
            QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_config)
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
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Instantiate Panels
        self.control_panel = ControlPanel()
        self.image_panel = ImagePanel()
        self.info_panel = InfoPanel()

        # Add to Splitter
        main_splitter.addWidget(self.control_panel)
        main_splitter.addWidget(self.image_panel)
        main_splitter.addWidget(self.info_panel)
        
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setStretchFactor(2, 1)
        main_layout.addWidget(main_splitter)

        # Aliases for compatibility with existing methods
        self.product_combo = self.control_panel.product_combo
        self.area_combo = self.control_panel.area_combo
        self.inference_combo = self.control_panel.inference_combo
        self.start_btn = self.control_panel.start_btn
        self.stop_btn = self.control_panel.stop_btn
        self.save_btn = self.control_panel.save_btn
        self.use_camera_chk = self.control_panel.use_camera_chk
        self.reconnect_camera_btn = self.control_panel.reconnect_camera_btn
        self.disconnect_camera_btn = self.control_panel.disconnect_camera_btn
        self.pick_image_btn = self.control_panel.pick_image_btn
        self.image_path_label = self.control_panel.image_path_label
        self.clear_image_btn = self.control_panel.clear_image_btn

        self.original_image = self.image_panel.original_image
        self.processed_image = self.image_panel.processed_image
        self.result_image = self.image_panel.result_image

        self.big_status_label = self.info_panel.big_status_label
        self.result_widget = self.info_panel.result_widget
        self.log_text = self.info_panel.log_text

        # Connect Signals
        self.control_panel.product_changed.connect(self.on_product_changed)
        self.control_panel.product_changed.connect(lambda _: self.update_start_enabled())
        
        self.control_panel.area_changed.connect(self.on_area_changed)
        self.control_panel.area_changed.connect(lambda _: self.update_start_enabled())
        
        self.control_panel.inference_type_changed.connect(lambda _: self.update_start_enabled())
        
        self.control_panel.start_requested.connect(self.start_detection)
        self.control_panel.stop_requested.connect(self.stop_detection)
        self.control_panel.save_requested.connect(self.save_results)
        
        self.control_panel.use_camera_toggled.connect(self.on_use_camera_toggled)
        self.control_panel.reconnect_camera_requested.connect(self.handle_reconnect_camera)
        self.control_panel.disconnect_camera_requested.connect(self.handle_disconnect_camera)
        
        self.control_panel.pick_image_requested.connect(self.pick_image)
        self.control_panel.clear_image_requested.connect(self.clear_selected_image)
        self.statusBar().showMessage("系統就緒")
        self.update_start_enabled()
        build_menu_bar(self)

    def init_system(self):
        """非同步初始化偵測系統與相機"""
        self.log_message("正在初始化檢測系統...")
        self.camera_worker = self.controller.build_camera_initializer()
        self.camera_worker.finished.connect(self._on_system_init_finished)
        self.camera_worker.start()

    def _on_system_init_finished(self, camera_success):
        try:
            self.detection_system = self.controller.detection_system
            self.log_message("檢測系統初始化完成")
            
            if camera_success:
                self.log_message("相機連接成功")
                if getattr(self, "use_camera_chk", None):
                    self.use_camera_chk.setChecked(True)
            else:
                self.log_message("相機未連接或初始化失敗")
        except Exception as e:
            self.log_message(f"系統回調錯誤: {e}")
        finally:
            self.update_camera_controls()

    def _update_model_combos(self):
        """Populates and sets the product, area, and inference type combo boxes."""
        self.available_products = self._catalog.products()
        self.product_combo.clear()
        self.area_combo.clear()
        self.inference_combo.clear()

        self.available_areas = {
            product: self._catalog.areas(product) for product in self.available_products
        }

        if not self.available_products:
            self.log_message("No models found in the models directory.")
            return

        self.product_combo.addItems(self.available_products)
        last_prod, last_area, last_infer = self.preferences.restore_last_selection()

        if last_prod and last_prod in self.available_products:
            self.product_combo.setCurrentText(last_prod)

        self.on_product_changed(self.product_combo.currentText())

        if last_area and last_area in self.available_areas.get(self.product_combo.currentText(), []):
            self.area_combo.setCurrentText(last_area)

        self.on_area_changed(self.area_combo.currentText())

        available_types = [self.inference_combo.itemText(i) for i in range(self.inference_combo.count())]
        if last_infer and last_infer in available_types:
            self.inference_combo.setCurrentText(last_infer)

    def load_available_models(self):
        """Async Load available model information from the filesystem."""
        self.log_message("正在載入模型清單...")
        
        # Check if base path exists first to fail fast
        base_path = Path(self._models_base)
        if not base_path.exists():
            self.log_message(f"Models directory not found at: {base_path}")
            return

        self.model_loader = self.controller.build_model_loader()
        self.model_loader.models_ready.connect(self._on_models_loaded)
        self.model_loader.error_occurred.connect(self._on_model_load_error)
        self.model_loader.start()

    def _on_models_loaded(self):
        try:
            self._update_model_combos()
            self.log_message(f"模型清單載入完成: {len(self.available_products)} products found.")
        except Exception as e:
            self.log_message(f"Error updating model combos: {e}")

    def _on_model_load_error(self, error_msg):
        self.log_message(f"Error loading model data: {error_msg}")
        QMessageBox.critical(self, "Model Loading Error", f"Failed to load model data:\n{error_msg}")

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
                self.reconnect_camera_btn.setEnabled(not running)
            if getattr(self, "disconnect_camera_btn", None):
                self.disconnect_camera_btn.setEnabled(camera_connected and not running)
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
            QMessageBox.warning(
                self,
                "Camera Operation",
                "Detection in progress. Stop before reconnecting.",
            )
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
            QMessageBox.warning(
                self,
                "Camera Operation",
                "Detection in progress. Stop before disconnecting.",
            )
            return
        if not self.controller.has_system():
            QMessageBox.information(
                self, "Camera Status", "Detection system is not initialized yet."
            )
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
            QMessageBox.warning(
                self,
                "Missing Parameters",
                "Select product, area, and inference type before starting detection.",
            )
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
                self.update_camera_controls()
                return
        else:
            self.detection_system = self.controller.detection_system
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_camera_controls()
        self.original_image.clear()
        self.processed_image.clear()
        self.result_image.clear()
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("RUNNING...")
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
                QMessageBox.warning(
                    self, "Input Source", "Select an image file or enable camera input."
                )
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                self.update_camera_controls()
                return
        self.worker = self.controller.build_worker(
            product,
            area,
            inference_type,
            frame=frame,
        )
        self.worker.result_ready.connect(self.on_detection_complete)
        self.worker.image_ready.connect(self.on_image_ready)
        self.worker.error_occurred.connect(self.on_detection_error)
        self.worker.start()

    def stop_detection(self):
        """停止檢測"""
        if self.worker and self.worker.isRunning():
            try:
                self.worker.stop()
                self.worker.wait(3000)
            except Exception:
                self.worker.terminate()
                self.worker.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("READY")
        self.update_camera_controls()
        self.log_message("檢測已取消")

    @pyqtSlot(object)
    def on_image_ready(self, image: np.ndarray) -> None:
        """Handle live image frame from worker."""
        if self.image_panel:
            self.image_panel.update_image(image)

    @pyqtSlot(object)
    def on_detection_complete(self, result: DetectionResult) -> None:
        """檢測完成回調"""
        self.current_result = result
        # 更新界面
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.update_camera_controls()
        
        # Update big status
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status(result.status)

        # 更新結果顯示
        self.info_panel.update_result(result)
        
        # 顯示圖像
        # Original and Preprocessed images should be paths if saved
        original_path = result.metadata.get("original_image_path") or result.image_path
        preprocessed_path = result.metadata.get("preprocessed_image_path")

        load_image_with_retry(
            self.original_image,
            original_path,
            on_fail=lambda: self.original_image.setText("尚無原始影像"),
        )
        load_image_with_retry(
            self.processed_image,
            preprocessed_path,
            on_fail=lambda: self.processed_image.setText("尚無預處理影像"),
        )
        
        # For result image, we favor finding annotations or heatmap paths
        annotated_path = result.metadata.get("annotated_path")
        heatmap_path = result.metadata.get("heatmap_path")
        
        # Check if result frame data is available in metadata to save temp file
        result_frame_data = result.metadata.get("result_frame")

        def show_result_frame_data():
            if isinstance(result_frame_data, np.ndarray) and result_frame_data.size > 0:
                  self.result_image.display_image(result_frame_data)
            else:
                 self.result_image.setText("無法載入結果影像")

        def load_heatmap():
             if heatmap_path:
                 load_image_with_retry(
                     self.result_image,
                     heatmap_path,
                     attempts=3,
                     delay_ms=200,
                     on_fail=show_result_frame_data,
                 )
             else:
                 show_result_frame_data()

        if annotated_path:
             load_image_with_retry(
                 self.result_image,
                 annotated_path,
                 attempts=5,
                 delay_ms=200,
                 on_fail=load_heatmap,
             )
        else:
             load_heatmap()

        self.log_message(f"檢測完成 - 狀態: {result.status}")
        self.statusBar().showMessage(f"檢測完成 - {result.status}", 5000)

    def on_detection_error(self, error_msg):
        """檢測錯誤回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("ERROR")
        self.update_camera_controls()
        self.log_message(f"檢測錯誤: {error_msg}")
        QMessageBox.critical(self, "檢測錯誤", f"檢測過程中發生錯誤\n{error_msg}")

    def save_results(self):
        """Save the latest detection result to disk."""
        if not self.current_result:
            QMessageBox.warning(
                self, "No Result", "There is no detection result to save yet."
            )
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Detection Result",
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)",
        )
        if file_path:
            try:
                # Use to_dict for serialization
                res_dict = self.current_result.to_dict()
                self.controller.save_result_json(Path(file_path), res_dict)
                self.log_message(f"Result saved to {file_path}")
                QMessageBox.information(
                    self, "Save Successful", f"Detection result saved to:\n{file_path}"
                )
            except Exception as exc:
                self.log_message(f"Failed to save result: {exc}")
                QMessageBox.critical(
                    self, "Save Error", f"Unable to save result:\n{exc}"
                )

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
                camera_ready = (
                    self.controller.is_camera_connected()
                    if self.controller.has_system()
                    else False
                )
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
                    self.clear_image_btn.setEnabled(bool(self.selected_image_path))
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
                    self.worker.stop()
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
    """Main entry point."""
    app = QApplication(sys.argv)
    window = DetectionSystemGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
