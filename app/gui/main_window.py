"""以 PyQt5 實作的桌面介面，用於操作與監看檢測流程。"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

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


import numpy as np
from PyQt5.QtCore import Qt, QTimer, QSettings, pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QShortcut,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.gui.camera_handler import CameraHandlerMixin
from app.gui.controller import DetectionController
from app.gui.panels.control_panel import ControlPanel
from app.gui.panels.image_panel import ImagePanel
from app.gui.panels.info_panel import InfoPanel
from app.gui.preferences import PreferencesManager
from app.gui.utils import load_image_with_retry
from app.gui.view_builder import build_menu_bar
from core.services.model_catalog import ModelCatalog


class DetectionSystemGUI(QMainWindow, CameraHandlerMixin):
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
        self.model_version_label = None  # Status bar version display
        # Models base path and settings
        from core.path_utils import project_root, resolve_path
        self._project_root = project_root()
        
        cfg_cand = resolve_path("config.yaml")
        self._config_path = cfg_cand if cfg_cand and cfg_cand.exists() else self._project_root / "config.yaml"
        
        mdl_cand = resolve_path("models")
        self._models_base = mdl_cand if mdl_cand and mdl_cand.is_dir() else self._project_root / "models"
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
        # NOTE: update_start_enabled() is already called at the end of
        # reload_inference_types(), which is the terminal handler in the
        # cascade: product_changed -> on_area_changed -> reload_inference_types.
        # Only inference_type_changed needs its own connection because the
        # user can change it directly without going through the cascade.
        self.control_panel.product_changed.connect(self.on_product_changed)

        self.control_panel.area_changed.connect(self.on_area_changed)

        self.control_panel.inference_type_changed.connect(lambda _: self.update_start_enabled())

        self.control_panel.start_requested.connect(self.start_detection)
        self.control_panel.stop_requested.connect(self.stop_detection)
        self.control_panel.save_requested.connect(self.save_results)

        self.control_panel.use_camera_toggled.connect(self.on_use_camera_toggled)
        self.control_panel.reconnect_camera_requested.connect(self.handle_reconnect_camera)
        self.control_panel.disconnect_camera_requested.connect(self.handle_disconnect_camera)

        self.control_panel.pick_image_requested.connect(self.pick_image)
        self.control_panel.clear_image_requested.connect(self.clear_selected_image)
        self.control_panel.preset_selected.connect(self._apply_preset)

        self.info_panel.session_stats.consecutive_fail_reached.connect(
            self._on_consecutive_fail_alert
        )

        # Add model version label to status bar (permanent widget on the right)
        self.model_version_label = QLabel("模型版本: --")
        self.model_version_label.setStyleSheet(
            "padding: 2px 8px; color: #6c757d; font-size: 11px; border-left: 1px solid #dee2e6;"
        )
        self.statusBar().addPermanentWidget(self.model_version_label)
        
        # --- New: Pipeline Bridge & Stats ---
        self.controller.bridge.image_ready.connect(self.on_image_ready)
        self.controller.bridge.result_ready.connect(self.on_pipeline_result)
        self.controller.bridge.error_occurred.connect(self.on_detection_error)
        self.controller.bridge.camera_disconnected.connect(self._on_camera_disconnected)
        
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(1000)
        self.stats_timer.timeout.connect(self.update_pipeline_stats)
        
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
        
        self.product_combo.blockSignals(True)
        self.area_combo.blockSignals(True)
        self.inference_combo.blockSignals(True)
        
        self.product_combo.clear()
        self.area_combo.clear()
        self.inference_combo.clear()

        self.available_areas = {
            product: self._catalog.areas(product) for product in self.available_products
        }

        if not self.available_products:
            self.product_combo.blockSignals(False)
            self.area_combo.blockSignals(False)
            self.inference_combo.blockSignals(False)
            self.log_message("模型目錄中未找到任何模型。")
            return

        self.product_combo.addItems(self.available_products)
        last_prod, last_area, last_infer = self.preferences.restore_last_selection()

        if last_prod and last_prod in self.available_products:
            self.product_combo.setCurrentText(last_prod)

        self.product_combo.blockSignals(False)
        self.on_product_changed(self.product_combo.currentText())

        if last_area and last_area in self.available_areas.get(self.product_combo.currentText(), []):
            self.area_combo.setCurrentText(last_area)

        self.area_combo.blockSignals(False)
        self.on_area_changed(self.area_combo.currentText())

        available_types = [self.inference_combo.itemText(i) for i in range(self.inference_combo.count())]
        if last_infer and last_infer in available_types:
            self.inference_combo.setCurrentText(last_infer)
            
        self.inference_combo.blockSignals(False)

    def load_available_models(self):
        """Async Load available model information from the filesystem."""
        self.log_message("正在載入模型清單...")

        # Check if base path exists first to fail fast
        base_path = Path(self._models_base)
        if not base_path.exists():
            self.log_message(f"找不到模型目錄：{base_path}")
            return

        self.model_loader = self.controller.build_model_loader()
        self.model_loader.models_ready.connect(self._on_models_loaded)
        self.model_loader.error_occurred.connect(self._on_model_load_error)
        self.model_loader.start()

    def _on_models_loaded(self):
        try:
            self._update_model_combos()
            self.log_message(f"模型清單載入完成：共 {len(self.available_products)} 個產品")
            self._rebuild_presets()
            self._update_output_path_label()
        except Exception as e:
            self.log_message(f"更新模型選單時發生錯誤：{e}")

    def _rebuild_presets(self) -> None:
        """Build preset combos from the loaded model catalogue."""
        presets: dict[str, tuple[str, str, str]] = {}
        for product in self.available_products:
            for area in self.available_areas.get(product, []):
                types = self._catalog.inference_types(product, area)
                for inf_type in types:
                    label = f"{product} / {area} / {inf_type}"
                    presets[label] = (product, area, inf_type)
        self.control_panel.set_presets(presets)

    def _update_output_path_label(self) -> None:
        """Show the current output directory in the control panel."""
        try:
            if self.controller.has_system():
                output_dir = self.controller.detection_system.config.output_dir
                self.control_panel.set_output_path(str(output_dir))
        except Exception as e:
            self.log_message(f"無法取得輸出路徑：{e}")

    # ------------------------------------------------------------------
    # Combo helpers — shared by preset application and cascade handlers
    # ------------------------------------------------------------------

    def _rebuild_area_combo(self, product: str, *, select: str | None = None) -> None:
        """Repopulate area_combo for *product*, optionally pre-selecting *select*."""
        self.area_combo.clear()
        self.area_combo.addItems(self.available_areas.get(product, []))
        if select is not None:
            idx = self.area_combo.findText(select)
            if idx >= 0:
                self.area_combo.setCurrentIndex(idx)

    def _rebuild_inference_combo(self, product: str, area: str, *, select: str | None = None) -> None:
        """Repopulate inference_combo for *product*+*area*, optionally pre-selecting *select*."""
        self.inference_combo.clear()
        if product and area:
            self.inference_combo.addItems(self._catalog.inference_types(product, area))
        if select is not None:
            idx = self.inference_combo.findText(select)
            if idx >= 0:
                self.inference_combo.setCurrentIndex(idx)

    def _apply_preset(self, product: str, area: str, inf_type: str) -> None:
        """Apply a quick-switch preset to the three selection combos atomically."""
        if product not in self.available_products:
            return
        self.product_combo.blockSignals(True)
        self.area_combo.blockSignals(True)
        self.inference_combo.blockSignals(True)

        idx = self.product_combo.findText(product)
        if idx >= 0:
            self.product_combo.setCurrentIndex(idx)
        self._rebuild_area_combo(product, select=area)
        self._rebuild_inference_combo(product, area, select=inf_type)

        self.product_combo.blockSignals(False)
        self.area_combo.blockSignals(False)
        self.inference_combo.blockSignals(False)

        self.update_start_enabled()
        self.log_message(f"套用預設：{product} / {area} / {inf_type}")

    def _on_model_load_error(self, error_msg):
        self.log_message(f"載入模型資料時發生錯誤：{error_msg}")
        QMessageBox.critical(self, "模型載入錯誤", f"無法載入模型資料：\n{error_msg}")

    def on_product_changed(self, product):
        """產品選擇變更時的處理"""
        self.area_combo.blockSignals(True)
        self._rebuild_area_combo(product)
        # Call on_area_changed BEFORE unblocking signals to prevent the
        # area_combo.currentTextChanged signal from triggering a duplicate call.
        self.on_area_changed(self.area_combo.currentText())
        self.area_combo.blockSignals(False)

    def on_area_changed(self, area):
        try:
            self.reload_inference_types()
        except Exception as e:
            self.log_message(f"載入推論類型時發生錯誤：{e}")

    def reload_inference_types(self):
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        self.inference_combo.blockSignals(True)
        self._rebuild_inference_combo(product, area)
        self.inference_combo.blockSignals(False)
        self.update_start_enabled()

    def is_detection_running(self) -> bool:
        """Return True if a detection worker is running."""
        return bool(self.worker and self.worker.isRunning())

    # update_camera_controls, handle_reconnect_camera, handle_disconnect_camera
    # → moved to CameraHandlerMixin (app/gui/camera_handler.py)

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
                "缺少參數",
                "請先選擇產品、區域及推理類型再開始檢測。",
            )
            return
        if not self._catalog.config_exists(product, area, inference_type):
            config_path = self._catalog.config_path(product, area, inference_type)
            QMessageBox.critical(
                self,
                "找不到模型",
                f"找不到設定檔：\n{config_path}",
            )
            return
        if not self.controller.has_system():
            QMessageBox.critical(
                self,
                "檢測系統",
                "檢測系統尚未初始化，請確認設定後再試。",
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
        # Clear FAIL reason on new run
        self.info_panel.fail_reason_label.clear_reason()
        self.log_message(
            f"開始檢測 - 產品：{product}，區域：{area}，類型：{inference_type}"
        )
        
        use_cam = self.use_camera_chk.isChecked()
        if use_cam:
            # --- ASYNC PIPELINE MODE ---
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.stats_timer.start()
            
            # Use DetectionWorker as a start proxy (runs in separate thread to avoid UI lag)
            self.worker = self.controller.build_worker(
                product, area, inference_type,
                capture_interval=0.1 # Example interval
            )
            self.worker.error_occurred.connect(self.on_detection_error)
            self.worker.start()
        else:
            # --- SINGLE SHOT MODE (Synchronous) ---
            selected = getattr(self, "selected_image_path", None)
            if not selected:
                QMessageBox.warning(self, "輸入來源", "請先選擇影像檔案。")
                self._reset_ui_state()
                return

            image = self.controller.load_image(Path(selected))
            if image is None:
                QMessageBox.warning(self, "載入錯誤", "無法載入選取的影像。")
                self._reset_ui_state()
                return

            self.image_panel.update_image(image)
            # Run one-off detection in a daemon thread to keep UI alive
            threading.Thread(
                target=self._run_single_shot,
                args=(image, product, area, inference_type),
                daemon=True,
            ).start()

    def _run_single_shot(self, frame, product, area, inference_type):
        """Execute a single synchronous detect() call off the main thread."""
        try:
            result = self.controller.detection_system.detect(
                product, area, inference_type, frame=frame
            )
            # Use bridge signal to safely cross back to the main thread
            self.controller.bridge.result_ready.emit(result)
        except Exception as e:
            self.controller.bridge.error_occurred.emit(str(e))

    def stop_detection(self):
        """優雅停止管線 (非阻塞)"""
        if not self.controller.has_system():
            return
            
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("正在停止...")
        
        # Use ShutdownWorker to handle blocking stop_pipeline()
        self._shutdown_worker = self.controller.build_shutdown_worker()
        self._shutdown_worker.shutdown_complete.connect(self._on_pipeline_stopped)
        self._shutdown_worker.start()

    def _on_pipeline_stopped(self):
        """Pipeline stopped callback."""
        self.stats_timer.stop()
        self.stop_btn.setText("停止檢測")
        self._reset_ui_state()
        self.log_message("檢測管線已關閉 (IO 已落盤)")

    def _reset_ui_state(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("READY")
        self.update_camera_controls()

    def update_pipeline_stats(self):
        """Update FPS/Counters on the status bar."""
        if not self.controller.has_system():
            return
        sys = self.controller.detection_system
        if not sys.pipeline_running:
            return
        stats = sys.pipeline_stats()
        dropped_info = ""
        tasks_dropped = stats.get("tasks_dropped", 0)
        if tasks_dropped > 0:
            dropped_info = f" | 遺失：{tasks_dropped}"
        msg = (f"擷取：{stats['frames_captured']} | "
               f"丟棄：{stats['frames_dropped']} | "
               f"已存：{stats['tasks_saved']} | "
               f"佇列：{stats['inference_queue_size']}/{stats['io_queue_size']}"
               f"{dropped_info}")
        self.statusBar().showMessage(msg)

    @pyqtSlot(object)
    def on_pipeline_result(self, result_or_task):
        """Handle result from pipeline (via Bridge).

        Receives either a ``DetectionResult`` (single-shot mode) or a
        ``DetectionTask`` (pipeline mode). Converts to ``DetectionResult``
        and delegates to ``on_detection_complete``.
        """
        from core.types import DetectionResult, DetectionTask, DetectionItem

        if isinstance(result_or_task, DetectionResult):
            # Single-shot mode emits DetectionResult directly
            self.on_detection_complete(result_or_task)
            return

        # Pipeline mode: convert DetectionTask → DetectionResult
        task = result_or_task
        if not isinstance(task, DetectionTask) or task.result is None:
            return

        res = task.result
        items = [
            DetectionItem(
                label=d.get("class", "unknown"),
                confidence=float(d.get("confidence", 0.0)),
                bbox_xyxy=tuple(d.get("bbox", (0, 0, 0, 0))),
                metadata={k: v for k, v in d.items()
                          if k not in ("class", "confidence", "bbox")},
            )
            for d in res.get("detections", [])
        ]

        result = DetectionResult(
            status=res.get("status", "ERROR"),
            items=items,
            latency=time.time() - task.timestamp,
            product=task.product,
            area=task.area,
            inference_type=task.inference_type,
            error=res.get("error"),
            anomaly_score=res.get("anomaly_score"),
            missing_items=res.get("missing_items", []),
            unexpected_items=res.get("unexpected_items", []),
            annotated_path=res.get("annotated_path", ""),
            heatmap_path=res.get("heatmap_path", ""),
            original_image_path=res.get("original_image_path", ""),
            preprocessed_image_path=res.get("preprocessed_image_path", ""),
            result_frame=res.get("result_frame"),
        )
        self.on_detection_complete(result)

    def _on_worker_finished(self) -> None:
        """Restore UI when worker finishes for any reason."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.update_camera_controls()

    @pyqtSlot(object)
    def on_image_ready(self, image: np.ndarray) -> None:
        """Handle live image frame from worker."""
        if self.image_panel:
            self.image_panel.update_image(image)

    @pyqtSlot(object)
    def on_detection_complete(self, result: DetectionResult) -> None:
        """檢測完成回調"""
        self.current_result = result

        # Only reset buttons if pipeline is NOT running (single-shot mode)
        is_pipeline_running = (
            self.controller.has_system()
            and self.controller.detection_system.pipeline_running
        )
        if not is_pipeline_running:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_camera_controls()

        self.save_btn.setEnabled(True)

        # Update big status
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status(result.status)
        
        # Update version label after model is loaded
        product = result.product or self.product_combo.currentText()
        area = result.area or self.area_combo.currentText()
        inference_type = result.inference_type or self.inference_combo.currentText()
        self._update_version_label(product, area, inference_type)

        # 更新結果顯示
        self.info_panel.update_result(result)

        # 顯示圖像 — use explicit attributes instead of metadata.get()
        original_path = result.original_image_path or result.image_path
        preprocessed_path = result.preprocessed_image_path

        load_image_with_retry(
            self.original_image,
            original_path,
            on_fail=lambda: self.original_image.setText("尚無原始影像 — 請重新選擇影像"),
        )
        load_image_with_retry(
            self.processed_image,
            preprocessed_path,
            on_fail=lambda: self.processed_image.setText("尚無預處理影像"),
        )

        # For result image, we favor finding annotations or heatmap paths
        annotated_path = result.annotated_path
        heatmap_path = result.heatmap_path

        # Check if result frame data is available
        result_frame_data = result.result_frame

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
                  attempts=2,
                  delay_ms=150,
                 on_fail=load_heatmap,
             )
        else:
             load_heatmap()

        self.log_message(f"檢測完成 - 狀態: {result.status}")
        self.statusBar().showMessage(f"檢測完成 - {result.status}", 5000)

        # Consecutive FAIL alert is handled via SessionStatsWidget.consecutive_fail_reached signal.

    @pyqtSlot(int)
    def _on_consecutive_fail_alert(self, count: int) -> None:
        """Slot for SessionStatsWidget.consecutive_fail_reached signal."""
        self.log_message(f"⚠ 警告：已連續 {count} 次 NG！請確認產線狀況。")
        self.statusBar().showMessage(f"⚠ 連續 {count} 次 NG！請確認產線狀況。", 8000)

    def on_detection_error(self, error_msg):
        """檢測錯誤回調"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("ERROR")
        self.update_camera_controls()
        self.log_message(f"檢測錯誤: {error_msg}")
        QMessageBox.critical(self, "檢測錯誤", f"檢測過程中發生錯誤\n{error_msg}")

    # _on_camera_disconnected, _on_camera_lost_pipeline_stopped → CameraHandlerMixin
    # The pyqtSlot decorator is not needed on mixin methods — Qt resolves slots
    # by name at runtime regardless of where the method is defined in the MRO.

    def save_results(self):
        """Save the latest detection result to disk."""
        if not self.current_result:
            QMessageBox.warning(
                self, "尚無結果", "目前沒有可儲存的檢測結果。"
            )
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存檢測結果",
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)",
        )
        if file_path:
            try:
                # Use to_dict for serialization
                res_dict = self.current_result.to_dict()
                self.controller.save_result_json(Path(file_path), res_dict)
                self.log_message(f"結果已儲存至 {file_path}")
                QMessageBox.information(
                    self, "儲存成功", f"檢測結果已儲存至：\n{file_path}"
                )
            except Exception as exc:
                self.log_message(f"儲存結果失敗：{exc}")
                QMessageBox.critical(
                    self, "儲存錯誤", f"無法儲存結果：\n{exc}"
                )

    def pick_image(self):
        """選擇影像"""
        options = QFileDialog.Options()
        # Bypass Windows Native dialog, which freezes the app when network drives or Quick Access paths are unresponsive
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getOpenFileName(
            self, "選擇影像", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options
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

    # on_use_camera_toggled → moved to CameraHandlerMixin

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
        # 改用管線真實狀態來判斷是否正在檢測
        is_pipeline_running = False
        if self.controller.has_system():
            is_pipeline_running = getattr(self.controller.detection_system, "pipeline_running", False)

        if is_pipeline_running:
            reply = QMessageBox.question(
                self,
                "確認離開",
                "檢測管線正在執行中，是否確定要強制關閉？這可能需要幾秒鐘完成存檔。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            # 顯示等待游標，因為接下來的 shutdown 是同步阻塞的
            QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            if self.controller.has_system():
                # 這裡的 shutdown 在上一階段已經被我們改寫過，會優先等待 stop_pipeline() 完成
                self.controller.shutdown()
            
            self.preferences.save_window_state(self.saveGeometry(), self.saveState())
            self.preferences.save_last_selection(
                self.product_combo.currentText(),
                self.area_combo.currentText(),
                self.inference_combo.currentText(),
            )
        except Exception as e:
            self._logger.error(f"Shutdown error: {e}")
        finally:
            if is_pipeline_running:
                QApplication.restoreOverrideCursor()
            event.accept()
    
    def _update_version_label(self, product: str, area: str, inference_type: str) -> None:
        """Update the model version display in status bar."""
        if not self.model_version_label:
            return
        
        try:
            # Try to get version from detection system's loaded model
            if self.detection_system and hasattr(self.detection_system, "model_manager"):
                manager = self.detection_system.model_manager
                cache_key = (product, area)
                
                # Check if model is cached
                if hasattr(manager, "_cache") and cache_key in manager._cache:
                    cached = manager._cache[cache_key].get(inference_type)
                    if cached:
                        _, config = cached
                        weights = getattr(config, "weights", "")
                        if weights:
                            # Extract version from filename using version_utils
                            from core.version_utils import parse_model_version, version_to_string
                            version = parse_model_version(weights)
                            if version:
                                version_str = version_to_string(version)
                                self.model_version_label.setText(f"模型版本: v{version_str}")
                                self.model_version_label.setToolTip(
                                    f"當前載入模型:\n{product}/{area}/{inference_type}\n版本: v{version_str}"
                                )
                                return
            
            # Fallback: show model info without version
            self.model_version_label.setText(f"{product}/{area}")
            self.model_version_label.setToolTip(f"{product}/{area}/{inference_type}")
        except Exception as e:
            self._logger.debug(f"Failed to update version label: {e}")
            self.model_version_label.setText("模型版本: --")

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = DetectionSystemGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
