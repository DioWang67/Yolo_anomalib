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
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
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
    QDialog,
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
from app.gui.i18n import normalize_language, tr
from app.gui.model_config_dialog import ModelConfigDialog
from app.gui.panels.control_panel import ControlPanel
from app.gui.panels.image_panel import ImagePanel
from app.gui.panels.info_panel import InfoPanel
from app.gui.preferences import PreferencesManager
from app.gui.utils import load_image_with_retry
from app.gui.view_builder import build_menu_bar
from core.services.model_catalog import ModelCatalog
from core.services.model_config_editor import ModelConfigEditError, update_model_config


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
        self.show_detection_boxes_chk = None
        self._run_generation = 0
        self._single_shot_running = False
        self._single_shot_cancel_event = threading.Event()
        self._shutdown_in_progress = False
        self._stopping_generation: int | None = None
        # Models base path and settings
        from core.path_utils import project_root, resolve_path
        self._project_root = project_root()
        
        cfg_cand = resolve_path("config.yaml")
        self._config_path = cfg_cand if cfg_cand and cfg_cand.exists() else self._project_root / "config.yaml"
        
        mdl_cand = resolve_path("models")
        self._models_base = mdl_cand if mdl_cand and mdl_cand.is_dir() else self._project_root / "models"
        self.preferences = PreferencesManager(QSettings())
        self.current_language = normalize_language(self.preferences.restore_language())
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
            self.detection_system = self.controller._system
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
        self.setWindowTitle(tr(self.current_language, "window_title"))
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f4f6f8;
            }
            QWidget {
                font-family: "Microsoft JhengHei", "Segoe UI", Arial;
                color: #1f2933;
            }
            QPushButton {
                background-color: #eef2f6;
                color: #1f2933;
                border: 1px solid #cbd5df;
                padding: 8px 12px;
                border-radius: 6px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #e4ebf2;
                border-color: #9fb0c3;
            }
            QPushButton:pressed {
                background-color: #d7e1ec;
            }
            QPushButton#primaryAction {
                background-color: #16794c;
                color: white;
                border: none;
            }
            QPushButton#primaryAction:hover {
                background-color: #12643f;
            }
            QPushButton#dangerAction {
                background-color: #b42318;
                color: white;
                border: none;
            }
            QPushButton#dangerAction:hover {
                background-color: #971d14;
            }
            QPushButton#secondaryAction {
                background-color: #ffffff;
                color: #243b53;
                border: 1px solid #bcccdc;
            }
            QPushButton:disabled {
                background-color: #d9e2ec;
                color: #829ab1;
                border-color: #d9e2ec;
            }
            QComboBox {
                padding: 6px 10px;
                border: 1px solid #bcccdc;
                border-radius: 6px;
                background-color: white;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #829ab1;
            }
            QGroupBox {
                font-weight: 700;
                border: 1px solid #d9e2ec;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 12px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px 0 6px;
                color: #334e68;
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
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setStretchFactor(2, 1)
        main_splitter.setSizes([280, 760, 360])
        main_layout.addWidget(main_splitter)

        # Aliases for compatibility with existing methods
        self.product_combo = self.control_panel.product_combo
        self.area_combo = self.control_panel.area_combo
        self.inference_combo = self.control_panel.inference_combo
        self.start_btn = self.control_panel.start_btn
        self.stop_btn = self.control_panel.stop_btn
        self.save_btn = self.control_panel.save_btn
        self.edit_model_config_btn = self.control_panel.edit_model_config_btn
        self.use_camera_chk = self.control_panel.use_camera_chk
        self.reconnect_camera_btn = self.control_panel.reconnect_camera_btn
        self.disconnect_camera_btn = self.control_panel.disconnect_camera_btn
        self.pick_image_btn = self.control_panel.pick_image_btn
        self.image_path_label = self.control_panel.image_path_label
        self.clear_image_btn = self.control_panel.clear_image_btn
        self.show_detection_boxes_chk = self.control_panel.show_detection_boxes_chk

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
        self.control_panel.edit_model_config_requested.connect(self.edit_current_model_config)

        self.control_panel.use_camera_toggled.connect(self.on_use_camera_toggled)
        self.control_panel.reconnect_camera_requested.connect(self.handle_reconnect_camera)
        self.control_panel.disconnect_camera_requested.connect(self.handle_disconnect_camera)

        self.control_panel.pick_image_requested.connect(self.pick_image)
        self.control_panel.clear_image_requested.connect(self.clear_selected_image)
        self.control_panel.preset_selected.connect(self._apply_preset)
        self.control_panel.show_detection_boxes_toggled.connect(
            self._on_show_detection_boxes_toggled
        )
        self.control_panel.language_changed.connect(self.on_language_changed)

        self.info_panel.session_stats.consecutive_fail_reached.connect(
            self._on_consecutive_fail_alert
        )

        # Add model version label to status bar (permanent widget on the right)
        self.model_version_label = QLabel(
            f"{tr(self.current_language, 'model_version')}: --"
        )
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
        
        self.apply_language(self.current_language)
        self.statusBar().showMessage(tr(self.current_language, "ready"))
        self.update_start_enabled()
        self.show_detection_boxes_chk.setChecked(
            self.preferences.restore_show_detection_boxes()
        )

    def apply_language(self, language: str) -> None:
        """Apply the selected language to operator-facing GUI text."""
        self.current_language = normalize_language(language)
        self.setWindowTitle(tr(self.current_language, "window_title"))
        self.control_panel.set_language(self.current_language)
        self.image_panel.set_language(self.current_language)
        self.info_panel.set_language(self.current_language)
        if self.model_version_label:
            text = self.model_version_label.text()
            suffix = text.split(":", 1)[1].strip() if ":" in text else "--"
            self.model_version_label.setText(
                f"{tr(self.current_language, 'model_version')}: {suffix}"
            )
        if hasattr(self, "menuBar"):
            self.menuBar().clear()
            build_menu_bar(self)

    def on_language_changed(self, language: str) -> None:
        """Persist and apply language changes from the control panel."""
        self.apply_language(language)
        self.preferences.save_language(self.current_language)
        self.statusBar().showMessage(tr(self.current_language, "ready"), 3000)

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
        if self._single_shot_running or self._shutdown_in_progress:
            return True
        if self.worker and self.worker.isRunning():
            return True
        if self.controller.has_system():
            try:
                return bool(self.controller.detection_system.pipeline_running)
            except Exception:
                return False
        return False

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
            self.start_btn.setEnabled(
                ok and not self.stop_btn.isEnabled() and not self.is_detection_running()
            )
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
        
        self._run_generation += 1
        run_generation = self._run_generation
        self._shutdown_in_progress = False
        self._stopping_generation = None
        self._single_shot_cancel_event.clear()

        use_cam = self.use_camera_chk.isChecked()
        if use_cam:
            # --- CAMERA SINGLE-SHOT PIPELINE MODE ---
            self.controller.bridge.begin_run(run_generation)
            self._single_shot_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.stats_timer.start()
            
            # Use DetectionWorker as a start proxy (runs in separate thread to avoid UI lag)
            self.worker = self.controller.build_worker(
                product, area, inference_type,
                capture_interval=0.1, # Example interval
                mode="single",
                run_id=run_generation,
            )
            self.worker.error_occurred.connect(
                lambda msg, gen=run_generation: (
                    self.on_detection_error(msg)
                    if gen == self._run_generation else None
                )
            )
            self.worker.finished.connect(
                lambda gen=run_generation: self._on_start_worker_finished(gen)
            )
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
            self._single_shot_running = True
            # Run one-off detection in a daemon thread to keep UI alive
            threading.Thread(
                target=self._run_single_shot,
                args=(image, product, area, inference_type, run_generation),
                daemon=True,
            ).start()

    def _run_single_shot(self, frame, product, area, inference_type, run_generation):
        """Execute a single synchronous detect() call off the main thread."""
        try:
            result = self.controller.detection_system.detect(
                product,
                area,
                inference_type,
                frame=frame,
                cancel_cb=self._single_shot_cancel_event.is_set,
            )
            if (
                self._single_shot_cancel_event.is_set()
                or run_generation != self._run_generation
            ):
                return
            # Use bridge signal to safely cross back to the main thread
            self.controller.bridge.result_ready.emit(result)
        except Exception as e:
            if (
                not self._single_shot_cancel_event.is_set()
                and run_generation == self._run_generation
            ):
                self.controller.bridge.error_occurred.emit(str(e))
        finally:
            if run_generation == self._run_generation:
                self._single_shot_running = False

    def stop_detection(self):
        """優雅停止管線 (非阻塞)"""
        if not self.controller.has_system():
            return

        stopped_generation = self._run_generation
        self._single_shot_cancel_event.set()
        self.controller.bridge.end_run(stopped_generation)
        self._stopping_generation = stopped_generation
        self._run_generation += 1
        self._shutdown_in_progress = True
        self.stop_btn.setEnabled(False)
        startup_worker_running = bool(self.worker and self.worker.isRunning())
        if startup_worker_running and hasattr(self.worker, "cancel"):
            self.worker.cancel()
        self.stop_btn.setText(tr(self.current_language, "stopping"))
        
        if (
            not self.controller.detection_system.pipeline_running
            and self._single_shot_running
        ):
            self._single_shot_running = False
            self._shutdown_in_progress = False
            self._on_pipeline_stopped()
            return

        if (
            not self.controller.detection_system.pipeline_running
            and startup_worker_running
        ):
            self.log_message("正在取消啟動中的檢測...")
            return

        if not self.controller.detection_system.pipeline_running:
            self._shutdown_in_progress = False
            self._on_pipeline_stopped()
            return

        self._begin_pipeline_shutdown()

    def _begin_pipeline_shutdown(self):
        """Start a bounded background shutdown if one is not already active."""
        shutdown_worker = getattr(self, "_shutdown_worker", None)
        if shutdown_worker is not None and shutdown_worker.isRunning():
            return
        self._shutdown_worker = self.controller.build_shutdown_worker()
        self._shutdown_worker.shutdown_complete.connect(self._on_pipeline_stopped)
        self._shutdown_worker.start()

    def _on_start_worker_finished(self, run_generation):
        """Finish a stop request that happened while start_pipeline() was loading."""
        if self._stopping_generation != run_generation:
            return
        if (
            self.controller.has_system()
            and self.controller.detection_system.pipeline_running
        ):
            self._begin_pipeline_shutdown()
            return
        self._on_pipeline_stopped()

    def _on_pipeline_stopped(self):
        """Pipeline stopped callback."""
        self.controller.bridge.end_run()
        self._shutdown_in_progress = False
        self._stopping_generation = None
        self.stats_timer.stop()
        self.stop_btn.setText(tr(self.current_language, "stop"))
        self._reset_ui_state()
        self.log_message("檢測管線已關閉 (IO 已落盤)")

    def _reset_ui_state(self):
        self.controller.bridge.end_run()
        self._single_shot_cancel_event.set()
        self._single_shot_running = False
        self._shutdown_in_progress = False
        self._stopping_generation = None
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
        if (
            self.controller.has_system()
            and not self.controller.detection_system.pipeline_running
            and str(res.get("status", "")).upper()
            not in {"PASS", "DETECTION_FAIL", "INFERENCE_ERROR", "FAIL"}
        ):
            return
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
            missing_locations=res.get("missing_locations", []),
            unexpected_items=res.get("unexpected_items", []),
            annotated_path=res.get("annotated_path", ""),
            heatmap_path=res.get("heatmap_path", ""),
            original_image_path=res.get("original_image_path", ""),
            preprocessed_image_path=res.get("preprocessed_image_path", ""),
            cropped_paths=res.get("cropped_paths", []),
            color_check=res.get("color_check"),
            sequence_check=res.get("sequence_check"),
            result_frame=res.get("result_frame"),
            metadata={
                "decision": res.get("decision"),
                "slot_check": res.get("slot_check"),
                "slot_mismatches": res.get("slot_mismatches", []),
                "layout_alignment": res.get("layout_alignment"),
                "alignment_quality": res.get("alignment_quality"),
                "aligned_expected_boxes": res.get("aligned_expected_boxes", {}),
            },
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
        if not (
            self.controller.has_system()
            and self.controller.detection_system.pipeline_running
        ):
            return
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
        if self._single_shot_running or not is_pipeline_running:
            self._single_shot_running = False
            self.stats_timer.stop()
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
            on_fail=lambda: self.original_image.setText("No original image available"),
        )
        load_image_with_retry(
            self.processed_image,
            preprocessed_path,
            on_fail=lambda: self.processed_image.setText("No processed image available"),
        )
        self._refresh_result_image()

        self.log_message(f"檢測完成 - 狀態: {result.status}")
        self.statusBar().showMessage(f"檢測完成 - {result.status}", 5000)

        # Consecutive FAIL alert is handled via SessionStatsWidget.consecutive_fail_reached signal.

    @pyqtSlot(bool)
    def _on_show_detection_boxes_toggled(self, checked: bool) -> None:
        """Update the result tab when detection box visibility changes."""
        self.preferences.save_show_detection_boxes(checked)
        if self.current_result is not None:
            self._refresh_result_image()

    def _refresh_result_image(self) -> None:
        """Render the result tab using either annotated or clean imagery."""
        result = self.current_result
        if result is None:
            return

        show_boxes = (
            self.show_detection_boxes_chk.isChecked()
            if self.show_detection_boxes_chk is not None
            else True
        )
        inference_type = str(result.inference_type or "").lower()
        is_yolo_like = inference_type in {"", "yolo"}

        if not show_boxes and is_yolo_like:
            clean_path = result.preprocessed_image_path or result.original_image_path
            if clean_path:
                load_image_with_retry(
                    self.result_image,
                    clean_path,
                    attempts=2,
                    delay_ms=150,
                    on_fail=lambda: self.result_image.setText("Unable to load result image"),
                )
            else:
                self.result_image.setText("Unable to load result image")
            return

        annotated_path = result.annotated_path
        heatmap_path = result.heatmap_path
        result_frame_data = result.result_frame

        def show_result_frame_data() -> None:
            if isinstance(result_frame_data, np.ndarray) and result_frame_data.size > 0:
                self.result_image.display_image(result_frame_data)
            else:
                self.result_image.setText("Unable to load result image")

        def load_heatmap() -> None:
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

    @pyqtSlot(int)
    def _on_consecutive_fail_alert(self, count: int) -> None:
        """Slot for SessionStatsWidget.consecutive_fail_reached signal."""
        self.log_message(f"⚠ 警告：已連續 {count} 次 NG！請確認產線狀況。")
        self.statusBar().showMessage(f"⚠ 連續 {count} 次 NG！請確認產線狀況。", 8000)

    def on_detection_error(self, error_msg):
        """檢測錯誤回調"""
        self.controller.bridge.end_run()
        self._single_shot_cancel_event.set()
        self._single_shot_running = False
        self._shutdown_in_progress = False
        self._stopping_generation = None
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
            self.image_path_label.setText(tr(self.current_language, "no_image"))
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

    def edit_current_model_config(self):
        """Open a guarded editor for the selected model config and hot-reload it."""
        if self.is_detection_running():
            QMessageBox.warning(self, "機種設定", "檢測進行中，請先停止後再修改設定。")
            return

        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        inference_type = self.inference_combo.currentText().strip()
        if not all([product, area, inference_type]):
            QMessageBox.warning(self, "機種設定", "請先選擇產品、區域與模型類型。")
            return

        if inference_type.lower() == "fusion":
            QMessageBox.information(
                self,
                "機種設定",
                "Fusion 由 YOLO 與 Anomalib 兩份設定組成，請先分別編輯 yolo 或 anomalib。",
            )
            return

        config_path = self._catalog.config_path(product, area, inference_type)
        try:
            dialog = ModelConfigDialog(
                product=product,
                area=area,
                inference_type=inference_type,
                config_path=config_path,
                parent=self,
            )
        except ModelConfigEditError as exc:
            QMessageBox.critical(self, "機種設定", str(exc))
            return

        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            result = update_model_config(
                config_path,
                dialog.changes(),
                product=product,
                area=area,
            )
            self.controller.reload_model_settings(product, area, inference_type)
            self.load_available_models()
        except Exception as exc:
            QMessageBox.critical(self, "機種設定", f"儲存或熱更新失敗：\n{exc}")
            return

        self.log_message(
            f"機種設定已更新並熱更新: {product}/{area}/{inference_type} "
            f"(備份: {result.backup_path})"
        )
        QMessageBox.information(
            self,
            "機種設定",
            "設定已儲存，下一次檢測會使用新設定。\n"
            f"備份檔：{result.backup_path}",
        )

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
            self.preferences.save_show_detection_boxes(
                self.show_detection_boxes_chk.isChecked()
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
                                self.model_version_label.setText(
                                    f"{tr(self.current_language, 'model_version')}: v{version_str}"
                                )
                                self.model_version_label.setToolTip(
                                    f"當前載入模型:\n{product}/{area}/{inference_type}\n版本: v{version_str}"
                                )
                                return
            
            # Fallback: show model info without version
            self.model_version_label.setText(f"{product}/{area}")
            self.model_version_label.setToolTip(f"{product}/{area}/{inference_type}")
        except Exception as e:
            self._logger.debug(f"Failed to update version label: {e}")
            self.model_version_label.setText(
                f"{tr(self.current_language, 'model_version')}: --"
            )

    # ------------------------------------------------------------------
    # Localized operator-facing overrides
    # ------------------------------------------------------------------

    def _t(self, key: str, **kwargs: object) -> str:
        """Translate a GUI string for the active language."""
        text = tr(self.current_language, key)
        return text.format(**kwargs) if kwargs else text

    def init_system(self):
        """Initialize detection system and camera asynchronously."""
        self.log_message(self._t("init_system"))
        self.camera_worker = self.controller.build_camera_initializer()
        self.camera_worker.finished.connect(self._on_system_init_finished)
        self.camera_worker.start()

    def _on_system_init_finished(self, camera_success):
        try:
            self.detection_system = self.controller.detection_system
            self.log_message(self._t("system_initialized"))
            if camera_success:
                self.log_message(self._t("camera_connected"))
                if getattr(self, "use_camera_chk", None):
                    self.use_camera_chk.setChecked(True)
            else:
                self.log_message(self._t("camera_init_failed"))
        except Exception as exc:
            self.log_message(self._t("system_callback_error", error=exc))
        finally:
            self.update_camera_controls()

    def load_available_models(self):
        """Async load available model information from the filesystem."""
        self.log_message(self._t("loading_models"))
        base_path = Path(self._models_base)
        if not base_path.exists():
            self.log_message(self._t("models_dir_missing", path=base_path))
            return
        self.model_loader = self.controller.build_model_loader()
        self.model_loader.models_ready.connect(self._on_models_loaded)
        self.model_loader.error_occurred.connect(self._on_model_load_error)
        self.model_loader.start()

    def _on_models_loaded(self):
        try:
            self._update_model_combos()
            self.log_message(self._t("models_loaded", count=len(self.available_products)))
            self._rebuild_presets()
            self._update_output_path_label()
        except Exception as exc:
            self.log_message(self._t("model_menu_error", error=exc))

    def _on_model_load_error(self, error_msg):
        self.log_message(self._t("model_load_error", error=error_msg))
        QMessageBox.critical(
            self,
            self._t("model_load_error_title"),
            self._t("model_load_error", error=error_msg),
        )

    def start_detection(self):
        """Launch detection workflow."""
        product = self.product_combo.currentText()
        area = self.area_combo.currentText()
        inference_type = self.inference_combo.currentText()
        if not all([product, area, inference_type]):
            QMessageBox.warning(
                self,
                self._t("missing_params_title"),
                self._t("missing_params"),
            )
            return
        if not self._catalog.config_exists(product, area, inference_type):
            config_path = self._catalog.config_path(product, area, inference_type)
            QMessageBox.critical(
                self,
                self._t("model_missing_title"),
                self._t("model_missing", path=config_path),
            )
            return
        if not self.controller.has_system():
            QMessageBox.critical(
                self,
                self._t("system_not_ready_title"),
                self._t("system_not_ready"),
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
        self.info_panel.fail_reason_label.clear_reason()
        self.log_message(
            self._t("start_log", product=product, area=area, model=inference_type)
        )

        self._run_generation += 1
        run_generation = self._run_generation
        self._shutdown_in_progress = False
        self._stopping_generation = None
        self._single_shot_cancel_event.clear()

        use_cam = self.use_camera_chk.isChecked()
        if use_cam:
            self.controller.bridge.begin_run(run_generation)
            self._single_shot_running = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.stats_timer.start()
            self.worker = self.controller.build_worker(
                product,
                area,
                inference_type,
                capture_interval=0.1,
                mode="single",
                run_id=run_generation,
            )
            self.worker.error_occurred.connect(
                lambda msg, gen=run_generation: (
                    self.on_detection_error(msg) if gen == self._run_generation else None
                )
            )
            self.worker.finished.connect(
                lambda gen=run_generation: self._on_start_worker_finished(gen)
            )
            self.worker.start()
            return

        selected = getattr(self, "selected_image_path", None)
        if not selected:
            QMessageBox.warning(
                self,
                self._t("input_source_title"),
                self._t("select_image_first"),
            )
            self._reset_ui_state()
            return

        image = self.controller.load_image(Path(selected))
        if image is None:
            QMessageBox.warning(
                self,
                self._t("load_error_title"),
                self._t("image_load_failed"),
            )
            self._reset_ui_state()
            return

        self.image_panel.update_image(image)
        self._single_shot_running = True
        threading.Thread(
            target=self._run_single_shot,
            args=(image, product, area, inference_type, run_generation),
            daemon=True,
        ).start()

    @pyqtSlot(object)
    def on_detection_complete(self, result: DetectionResult) -> None:
        """Handle a completed detection result."""
        self.current_result = result
        is_pipeline_running = (
            self.controller.has_system()
            and self.controller.detection_system.pipeline_running
        )
        if self._single_shot_running or not is_pipeline_running:
            self._single_shot_running = False
            self.stats_timer.stop()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.update_camera_controls()

        self.save_btn.setEnabled(True)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status(result.status)

        product = result.product or self.product_combo.currentText()
        area = result.area or self.area_combo.currentText()
        inference_type = result.inference_type or self.inference_combo.currentText()
        self._update_version_label(product, area, inference_type)
        self.info_panel.update_result(result)

        original_path = result.original_image_path or result.image_path
        preprocessed_path = result.preprocessed_image_path
        load_image_with_retry(
            self.original_image,
            original_path,
            on_fail=lambda: self.original_image.setText("No original image available"),
        )
        load_image_with_retry(
            self.processed_image,
            preprocessed_path,
            on_fail=lambda: self.processed_image.setText("No processed image available"),
        )
        self._refresh_result_image()

        self.log_message(self._t("detect_done_log", status=result.status))
        self.statusBar().showMessage(
            self._t("detect_done_status", status=result.status), 5000
        )

    @pyqtSlot(int)
    def _on_consecutive_fail_alert(self, count: int) -> None:
        """Slot for SessionStatsWidget.consecutive_fail_reached signal."""
        message = self._t("consecutive_fail", count=count)
        self.log_message(message)
        self.statusBar().showMessage(message, 8000)

    def on_detection_error(self, error_msg):
        """Handle detection error callback."""
        self.controller.bridge.end_run()
        self._single_shot_cancel_event.set()
        self._single_shot_running = False
        self._shutdown_in_progress = False
        self._stopping_generation = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("ERROR")
        self.update_camera_controls()
        self.log_message(f"{self._t('detect_error_title')}: {error_msg}")
        QMessageBox.critical(
            self,
            self._t("detect_error_title"),
            self._t("detect_error", error=error_msg),
        )

    def save_results(self):
        """Save the latest detection result to disk."""
        if not self.current_result:
            QMessageBox.warning(self, self._t("no_result_title"), self._t("no_result"))
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            self._t("save_result_dialog"),
            f"detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON files (*.json)",
        )
        if not file_path:
            return
        try:
            self.controller.save_result_json(Path(file_path), self.current_result.to_dict())
            self.log_message(self._t("save_success", path=file_path))
            QMessageBox.information(
                self,
                self._t("save_success_title"),
                self._t("save_success", path=file_path),
            )
        except Exception as exc:
            self.log_message(self._t("save_error", error=exc))
            QMessageBox.critical(
                self,
                self._t("save_error_title"),
                self._t("save_error", error=exc),
            )

    def pick_image(self):
        """Select an image file."""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fname, _ = QFileDialog.getOpenFileName(
            self,
            self._t("select_image_title"),
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
            options=options,
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

    def open_config(self):
        """Open a config file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, self._t("open_config_title"), "", "YAML files (*.yaml *.yml)"
        )
        if file_path:
            self.log_message(self._t("config_loaded", path=file_path))

    def save_config(self):
        """Save a config file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, self._t("save_config_title"), "config.yaml", "YAML files (*.yaml)"
        )
        if file_path:
            self.log_message(self._t("config_saved", path=file_path))

    def edit_current_model_config(self):
        """Open a guarded editor for the selected model config and hot-reload it."""
        if self.is_detection_running():
            QMessageBox.warning(
                self, self._t("model_config_title"), self._t("stop_before_edit")
            )
            return

        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        inference_type = self.inference_combo.currentText().strip()
        if not all([product, area, inference_type]):
            QMessageBox.warning(
                self, self._t("model_config_title"), self._t("select_model_first")
            )
            return

        if inference_type.lower() == "fusion":
            QMessageBox.information(
                self, self._t("model_config_title"), self._t("fusion_edit_hint")
            )
            return

        config_path = self._catalog.config_path(product, area, inference_type)
        try:
            dialog = ModelConfigDialog(
                product=product,
                area=area,
                inference_type=inference_type,
                config_path=config_path,
                language=self.current_language,
                parent=self,
            )
        except ModelConfigEditError as exc:
            QMessageBox.critical(self, self._t("model_config_title"), str(exc))
            return

        if dialog.exec_() != QDialog.Accepted:
            return

        try:
            result = update_model_config(
                config_path,
                dialog.changes(),
                product=product,
                area=area,
            )
            self.controller.reload_model_settings(product, area, inference_type)
            self.load_available_models()
        except Exception as exc:
            QMessageBox.critical(
                self,
                self._t("model_config_title"),
                self._t("model_config_save_error", error=exc),
            )
            return

        self.log_message(
            self._t(
                "model_config_updated",
                product=product,
                area=area,
                model=inference_type,
                backup=result.backup_path,
            )
        )
        QMessageBox.information(
            self,
            self._t("model_config_title"),
            self._t("model_config_saved", backup=result.backup_path),
        )

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(self, self._t("about_title"), self._t("about_body"))

    def closeEvent(self, event):
        """Close the GUI after persisting preferences."""
        is_pipeline_running = self.is_detection_running()
        if is_pipeline_running:
            reply = QMessageBox.question(
                self,
                self._t("exit_title"),
                self._t("exit_running"),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            if self.controller.has_system():
                self.controller.shutdown()
            self.preferences.save_window_state(self.saveGeometry(), self.saveState())
            self.preferences.save_last_selection(
                self.product_combo.currentText(),
                self.area_combo.currentText(),
                self.inference_combo.currentText(),
            )
            self.preferences.save_show_detection_boxes(
                self.show_detection_boxes_chk.isChecked()
            )
        except Exception as exc:
            self._logger.error(f"Shutdown error: {exc}")
        finally:
            if is_pipeline_running:
                QApplication.restoreOverrideCursor()
            event.accept()

    def _update_version_label(self, product: str, area: str, inference_type: str) -> None:
        """Update the model version display in status bar."""
        if not self.model_version_label:
            return
        try:
            if self.detection_system and hasattr(self.detection_system, "model_manager"):
                manager = self.detection_system.model_manager
                cache_key = (product, area)
                if hasattr(manager, "_cache") and cache_key in manager._cache:
                    cached = manager._cache[cache_key].get(inference_type)
                    if cached:
                        _, config = cached
                        weights = getattr(config, "weights", "")
                        if weights:
                            from core.version_utils import parse_model_version, version_to_string

                            version = parse_model_version(weights)
                            if version:
                                version_str = version_to_string(version)
                                self.model_version_label.setText(
                                    f"{self._t('model_version')}: v{version_str}"
                                )
                                self.model_version_label.setToolTip(
                                    self._t(
                                        "current_model_tooltip",
                                        product=product,
                                        area=area,
                                        model=inference_type,
                                        version=version_str,
                                    )
                                )
                                return
            self.model_version_label.setText(f"{product}/{area}")
            self.model_version_label.setToolTip(f"{product}/{area}/{inference_type}")
        except Exception as exc:
            self._logger.debug(f"Failed to update version label: {exc}")
            self.model_version_label.setText(f"{self._t('model_version')}: --")

def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = DetectionSystemGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
