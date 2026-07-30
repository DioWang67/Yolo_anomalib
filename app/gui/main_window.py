"""以 PyQt5 實作的桌面介面，用於操作與監看檢測流程。"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from app.gui.retraining_workspace_host import RetrainingWorkspaceHost
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
    except (ImportError, AttributeError, OSError, RuntimeError) as exc:
        logging.getLogger(__name__).warning(
            "GUI DetectionSystem is unavailable; using core implementation: %s",
            exc,
        )
    from core.detection_system import DetectionSystem as _CoreDetectionSystem

    return _CoreDetectionSystem


import numpy as np
from PyQt5.QtCore import (
    QIODevice,
    QSaveFile,
    QSettings,
    QTemporaryDir,
    Qt,
    QTimer,
    pyqtSlot,
)
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.auto_inspection_controller import (
    AutoInspectionController,
    DEFAULT_AUTO_TRIGGER_CONFIG,
)
from app.gui.calibration_handler import CalibrationHandlerMixin
from app.gui.camera_handler import CameraHandlerMixin
from app.gui.controller import DetectionController
from app.gui.engineering_settings_page import EngineeringSettingsPage
from app.gui.inspection_history_page import InspectionHistoryPage
from app.gui.light_handler import LightHandlerMixin
from app.gui.i18n import normalize_language, tr
from app.gui.model_config_dialog import ModelConfigDialog
from app.gui.panels.control_panel import ControlPanel
from app.gui.panels.image_panel import ImagePanel
from app.gui.panels.info_panel import InfoPanel
from app.gui.preferences import PreferencesManager
from app.gui.utils import load_image_with_retry
from app.gui.view_builder import (
    _open_model_update_status,
    _open_model_versions,
    _open_training_review,
    build_menu_bar,
)
from app.gui.widgets import CameraStatusIndicator
from core.auto_trigger import AutoTriggerConfig
from core.services.model_catalog import ModelCatalog
from core.services.model_config_editor import ModelConfigEditError, update_model_config


class DetectionSystemGUI(
    QMainWindow, CameraHandlerMixin, LightHandlerMixin, CalibrationHandlerMixin
):
    def __init__(
        self,
        *,
        control_panel_settings: QSettings | None = None,
    ):
        super().__init__()
        self._test_settings_dir: QTemporaryDir | None = None
        if (
            control_panel_settings is None
            and os.environ.get("PYTEST_CURRENT_TEST")
        ):
            self._test_settings_dir = QTemporaryDir()
            if not self._test_settings_dir.isValid():
                raise RuntimeError("Unable to create isolated GUI test settings.")
            control_panel_settings = QSettings(
                str(
                    Path(self._test_settings_dir.path())
                    / "control-panel.ini"
                ),
                QSettings.IniFormat,
            )
        self._control_panel_settings = control_panel_settings
        self.detection_system = None
        self._light_controller = None
        self.worker = None
        self.available_products = []
        self.available_areas = {}
        self.current_result: DetectionResult | None = None
        self.selected_image_path = None
        self.model_loader = None
        self.use_camera_chk = None
        self.reconnect_camera_btn = None
        self.disconnect_camera_btn = None
        self.camera_status_indicator: CameraStatusIndicator | None = None
        self.model_version_label = None  # Status bar version display
        self.show_detection_boxes_chk = None
        self.show_original_tab_chk = None
        self.show_processed_tab_chk = None
        self._run_generation = 0
        self._single_shot_running = False
        self._single_shot_thread: threading.Thread | None = None
        self._single_shot_cancel_event = threading.Event()
        self._shutdown_in_progress = False
        self._closing = False
        self._close_after_auto_stop = False
        self._closing_auto_generation: int | None = None
        self._stopping_generation: int | None = None
        self._auto_controller: AutoInspectionController | None = None
        self._pending_auto_restart: tuple[int, str, str, str] | None = None
        self._retraining_workspace: RetrainingWorkspaceHost | None = None
        self._retraining_workspace_key: tuple[str, ...] | None = None
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
            self._camera_check_ts = 0
            self.update_camera_controls()

        # 快捷鍵
        try:
            QShortcut(QKeySequence("F5"), self).activated.connect(
                self.load_available_models
            )
            QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.open_config)
            QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.save_config)
            # Space / Enter：只在 start_btn 可用時觸發，避免誤觸
            for key in ("Space", "Return"):
                shortcut = QShortcut(
                    QKeySequence(key),
                    self.inspection_workspace,
                )
                shortcut.setContext(Qt.WidgetWithChildrenShortcut)
                shortcut.activated.connect(self._trigger_start_if_ready)
        except (TypeError, RuntimeError) as exc:
            self._logger.warning("Keyboard shortcuts could not be registered: %s", exc)
        # Restore window geometry/state
        try:
            geo, window_state = self.preferences.restore_window_state()
            if geo is not None:
                self.restoreGeometry(geo)
            if window_state is not None:
                self.restoreState(window_state)
        except (TypeError, ValueError, RuntimeError) as exc:
            self._logger.warning("Window state could not be restored: %s", exc)

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
        self.workspace_stack = QStackedWidget()
        self.setCentralWidget(self.workspace_stack)
        self.inspection_workspace = QWidget()
        self.workspace_stack.addWidget(self.inspection_workspace)
        main_layout = QVBoxLayout(self.inspection_workspace)
        main_splitter = QSplitter(Qt.Horizontal)

        # Instantiate Panels
        self.control_panel = ControlPanel(
            settings=self._control_panel_settings,
        )
        self.image_panel = ImagePanel()
        self.info_panel = InfoPanel()
        self.engineering_settings_page = EngineeringSettingsPage(
            self.control_panel.engineering_panel,
            language=self.current_language,
            parent=self.workspace_stack,
        )
        self.inspection_history_page = InspectionHistoryPage(
            self._project_root / "Result" / "inspection_records.sqlite3",
            language=self.current_language,
            parent=self.workspace_stack,
        )
        self.engineering_settings_page.configure_tab_order(
            self.control_panel.engineering_focus_widgets()
        )
        self.workspace_stack.addWidget(self.engineering_settings_page)
        self.workspace_stack.addWidget(self.inspection_history_page)

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
        self.show_original_tab_chk = self.control_panel.show_original_tab_chk
        self.show_processed_tab_chk = self.control_panel.show_processed_tab_chk
        self.auto_mode_chk = self.control_panel.auto_mode_chk

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
        self.control_panel.edit_model_config_requested.connect(
            lambda: self._run_engineering_action(
                self.edit_current_model_config
            )
        )
        self.control_panel.model_versions_requested.connect(
            lambda: self._run_engineering_action(
                lambda: _open_model_versions(self)
            )
        )
        self.control_panel.retraining_workspace_requested.connect(
            lambda: self._run_engineering_action(
                lambda: _open_training_review(self)
            )
        )
        self.control_panel.model_update_status_requested.connect(
            lambda: self._run_engineering_action(
                lambda: _open_model_update_status(self)
            )
        )
        self.control_panel.engineering_settings_requested.connect(
            self.show_engineering_settings
        )
        self.control_panel.inspection_history_requested.connect(
            self.show_inspection_history
        )
        self.control_panel.engineering_settings_closed.connect(
            self.show_inspection_workspace
        )
        self.engineering_settings_page.back_to_inspection_requested.connect(
            self.show_inspection_workspace
        )
        self.inspection_history_page.back_to_inspection_requested.connect(
            self.show_inspection_workspace
        )

        self.control_panel.use_camera_toggled.connect(self.on_use_camera_toggled)
        self.control_panel.reconnect_camera_requested.connect(self.handle_reconnect_camera)
        self.control_panel.disconnect_camera_requested.connect(self.handle_disconnect_camera)

        self.control_panel.pick_image_requested.connect(self.pick_image)
        self.control_panel.clear_image_requested.connect(self.clear_selected_image)
        self.control_panel.preset_selected.connect(self._apply_preset)
        self.control_panel.show_detection_boxes_toggled.connect(
            self._on_show_detection_boxes_toggled
        )
        self.control_panel.show_original_tab_toggled.connect(
            self._on_image_tab_visibility_toggled
        )
        self.control_panel.show_processed_tab_toggled.connect(
            self._on_image_tab_visibility_toggled
        )
        self.control_panel.auto_mode_toggled.connect(self._on_auto_mode_toggled)
        self.control_panel.language_changed.connect(self.on_language_changed)
        self.control_panel.calib_sample_empty_requested.connect(self._on_calib_sample_empty)
        self.control_panel.calib_sample_product_requested.connect(self._on_calib_sample_product)
        self.control_panel.calib_apply_requested.connect(self._on_calib_apply)

        self.info_panel.session_stats.consecutive_fail_reached.connect(
            self._on_consecutive_fail_alert
        )

        self.camera_status_indicator = CameraStatusIndicator(
            self.current_language,
            self,
        )
        self.camera_status_indicator.reconnect_requested.connect(
            self.handle_reconnect_camera
        )
        self.statusBar().addPermanentWidget(self.camera_status_indicator)

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
        self.controller.bridge.storage_completed.connect(
            self.on_pipeline_storage_completed
        )
        self.controller.bridge.error_occurred.connect(self.on_detection_error)
        self.controller.bridge.camera_disconnected.connect(self._on_camera_disconnected)
        self.controller.bridge.single_shot_finished.connect(
            self._on_single_shot_thread_finished
        )
        
        self.stats_timer = QTimer(self)
        self.stats_timer.setInterval(1000)
        self.stats_timer.timeout.connect(self.update_pipeline_stats)
        
        self.apply_language(self.current_language)
        self.statusBar().showMessage(tr(self.current_language, "ready"))
        self.update_start_enabled()
        self.show_detection_boxes_chk.setChecked(
            self.preferences.restore_show_detection_boxes()
        )
        self.show_original_tab_chk.setChecked(
            self.preferences.restore_show_original_tab()
        )
        self.show_processed_tab_chk.setChecked(
            self.preferences.restore_show_processed_tab()
        )
        self._apply_image_tab_visibility()

    def _engineering_session_active(self) -> bool:
        return (
            self.control_panel.engineering_access_granted
            and self.workspace_stack.currentWidget()
            is self.engineering_settings_page
        )

    def _run_engineering_action(
        self,
        action: Callable[[], object],
    ) -> bool:
        """Execute a privileged UI command only inside an unlocked page visit."""
        if not self._engineering_session_active():
            self._logger.warning(
                "Blocked engineering action outside an authenticated page session."
            )
            return False
        action()
        return True

    def show_inspection_workspace(self) -> None:
        """Return to inspection and revoke any engineering page session."""
        self.control_panel.lock_engineering_access()
        self.engineering_settings_page.clear_preview()
        self.workspace_stack.setCurrentWidget(self.inspection_workspace)
        QTimer.singleShot(
            0,
            self.control_panel.engineering_toggle_btn.setFocus,
        )

    def show_engineering_settings(self) -> bool:
        """Authenticate and show the persistent engineering settings page."""
        if not self.control_panel.unlock_engineering_access():
            return False
        self.engineering_settings_page.set_target(
            self.product_combo.currentText(),
            self.area_combo.currentText(),
        )
        self.engineering_settings_page.clear_preview()
        self.workspace_stack.setCurrentWidget(self.engineering_settings_page)
        QTimer.singleShot(
            0,
            self.engineering_settings_page.back_button.setFocus,
        )
        return True

    def show_inspection_history(self) -> None:
        """Show persisted records for the currently selected target."""
        self.control_panel.lock_engineering_access()
        self.engineering_settings_page.clear_preview()
        if self.controller.has_system():
            config = self.controller.detection_system.config
            self.inspection_history_page.set_database_context(
                Path(config.output_dir) / "inspection_records.sqlite3",
                sync_enabled=bool(
                    getattr(config, "inspection_sync_enabled", False)
                ),
            )
        self.workspace_stack.setCurrentWidget(self.inspection_history_page)
        self.inspection_history_page.show_for_target(
            self.product_combo.currentText(),
            self.area_combo.currentText(),
        )
        QTimer.singleShot(
            0,
            self.inspection_history_page.back_button.setFocus,
        )

    def show_retraining_workspace(
        self,
        *,
        result_root: str | Path,
        manifest_path: str | Path,
        training_data_dir: str | Path,
        language: str,
        product: str | None,
        area: str | None,
        available_targets: tuple[tuple[str, str], ...] | None = None,
    ) -> RetrainingWorkspaceHost:
        """Show one persistent, in-window retraining workspace for a target."""
        from app.gui.retraining_workspace_host import RetrainingWorkspaceHost

        workspace_key = (
            str(Path(result_root).resolve()),
            str(Path(manifest_path).resolve()),
            str(Path(training_data_dir).resolve()),
            language,
            product or "",
            area or "",
            *(
                f"{target_product}\0{target_area}"
                for target_product, target_area in sorted(
                    available_targets or ()
                )
            ),
        )
        workspace = self._retraining_workspace
        if workspace is None or self._retraining_workspace_key != workspace_key:
            if workspace is not None:
                workspace.shutdown_workspace()
                self.workspace_stack.removeWidget(workspace)
                workspace.deleteLater()
            workspace = RetrainingWorkspaceHost(
                result_root=result_root,
                manifest_path=manifest_path,
                training_data_dir=training_data_dir,
                language=language,
                product=product,
                area=area,
                available_targets=available_targets,
                parent=self.workspace_stack,
            )
            workspace.back_to_inspection_requested.connect(
                self.show_inspection_workspace
            )
            workspace.workspace_ready.connect(
                lambda count: self.log_message(
                    f"補訓資料已在背景載入完成：{count} 筆"
                )
            )
            workspace.workspace_failed.connect(
                lambda message: self.log_message(
                    f"補訓資料背景載入失敗：{message}"
                )
            )
            self.workspace_stack.addWidget(workspace)
            self._retraining_workspace = workspace
            self._retraining_workspace_key = workspace_key
        self.control_panel.lock_engineering_access()
        self.engineering_settings_page.clear_preview()
        self.workspace_stack.setCurrentWidget(workspace)
        workspace.refresh_workspace()
        return workspace

    def apply_language(self, language: str) -> None:
        """Apply the selected language to operator-facing GUI text."""
        self.current_language = normalize_language(language)
        self.setWindowTitle(tr(self.current_language, "window_title"))
        self.control_panel.set_language(self.current_language)
        self.engineering_settings_page.set_language(self.current_language)
        self.inspection_history_page.set_language(self.current_language)
        self.image_panel.set_language(self.current_language)
        self.info_panel.set_language(self.current_language)
        if self.camera_status_indicator is not None:
            self.camera_status_indicator.set_language(self.current_language)
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
        if self._auto_controller is not None and self._auto_controller.is_running():
            return True
        if self.worker and self.worker.isRunning():
            return True
        if self.controller.has_system():
            try:
                return bool(self.controller.detection_system.pipeline_running)
            except (AttributeError, RuntimeError) as exc:
                self._logger.error(
                    "Detection state is unreadable; keeping start disabled: %s",
                    exc,
                )
                return True
        return False

    # update_camera_controls, handle_reconnect_camera, handle_disconnect_camera
    # → moved to CameraHandlerMixin (app/gui/camera_handler.py)

    def _trigger_start_if_ready(self) -> None:
        """Space / Enter shortcut: fire start_detection only when the button is active."""
        if self.workspace_stack.currentWidget() is not self.inspection_workspace:
            return
        if self.start_btn.isEnabled():
            self.start_detection()

    def update_start_enabled(self):
        """根據選擇是否完整，自動啟用/停用開始檢測按鈕"""
        ok = bool(
            self.product_combo.currentText().strip()
            and self.area_combo.currentText().strip()
            and self.inference_combo.currentText().strip()
        )
        self.start_btn.setEnabled(
            ok and not self.stop_btn.isEnabled() and not self.is_detection_running()
        )


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
            self.controller.bridge.single_shot_finished.emit(run_generation)

    @pyqtSlot(int)
    def _on_single_shot_thread_finished(self, run_generation: int) -> None:
        """Release single-shot ownership only after the backend really returned."""
        self._single_shot_thread = None
        if self._stopping_generation == run_generation:
            self._single_shot_running = False
            self._on_pipeline_stopped()
            return
        if run_generation == self._run_generation:
            self._single_shot_running = False

    def stop_detection(self):
        """優雅停止管線 (非阻塞)"""
        if self._auto_controller is not None and self._auto_controller.is_running():
            self._stop_auto_mode()
            return
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
            self.log_message("正在等待目前的模型推論安全結束...")
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
        if (
            self.controller.has_system()
            and self.controller.detection_system.pipeline_running
        ):
            self._shutdown_in_progress = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText(tr(self.current_language, "stopping"))
            QTimer.singleShot(250, self._on_pipeline_stopped)
            return
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
        self.engineering_settings_page.clear_preview()
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
            over_items=res.get("over_items", []),
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
                "task_id": task.task_id,
                "storage_completed": False,
                "decision": res.get("decision"),
                "slot_check": res.get("slot_check"),
                "slot_mismatches": res.get("slot_mismatches", []),
                "over_items": res.get("over_items", []),
                "layout_alignment": res.get("layout_alignment"),
                "alignment_quality": res.get("alignment_quality"),
                "aligned_expected_boxes": res.get("aligned_expected_boxes", {}),
                "duplicate_filter": res.get("duplicate_filter"),
                "raw_detection_count": len(
                    res.get("raw_detections")
                    or res.get("detections")
                    or []
                ),
            },
        )
        self.on_detection_complete(result)

    @pyqtSlot(object)
    def on_pipeline_storage_completed(self, task) -> None:
        """Attach durable artifact paths without replaying the UI verdict."""
        from core.types import DetectionTask

        if not isinstance(task, DetectionTask) or task.result is None:
            return
        self.inspection_history_page.mark_dirty()
        current = self.current_result
        if current is None or current.metadata.get("task_id") != task.task_id:
            return
        result = task.result
        current.original_image_path = result.get("original_image_path", "")
        current.preprocessed_image_path = result.get("preprocessed_image_path", "")
        current.annotated_path = result.get("annotated_path", "")
        current.heatmap_path = result.get("heatmap_path", "")
        current.cropped_paths = result.get("cropped_paths", [])
        current.metadata["storage_completed"] = True
        self._refresh_detection_images()

    def _on_worker_finished(self) -> None:
        """Restore UI when worker finishes for any reason."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.engineering_settings_page.clear_preview()
        self.update_camera_controls()

    @pyqtSlot(object)
    def on_image_ready(self, image: np.ndarray) -> None:
        """Handle live image frame from worker or auto-inspection preview."""
        if (
            isinstance(image, np.ndarray)
            and image.size > 0
            and self.controller.has_system()
            and self.use_camera_chk is not None
            and self.use_camera_chk.isChecked()
        ):
            self._set_camera_status("ready")
        auto_active = (
            self._auto_controller is not None and self._auto_controller.is_running()
        )
        pipeline_active = (
            self.controller.has_system()
            and self.controller.detection_system.pipeline_running
        )
        if not (auto_active or pipeline_active):
            return
        if self.image_panel:
            self.image_panel.update_image(image)
        if (
            self.workspace_stack.currentWidget()
            is self.engineering_settings_page
        ):
            self.engineering_settings_page.update_preview(image)

    @pyqtSlot(bool)
    def _on_show_detection_boxes_toggled(self, checked: bool) -> None:
        """Update the result tab when detection box visibility changes."""
        self.preferences.save_show_detection_boxes(checked)
        if self.current_result is not None:
            self._refresh_result_image()

    @pyqtSlot(bool)
    def _on_image_tab_visibility_toggled(self, _checked: bool) -> None:
        """Persist and apply original/processed image tab visibility."""
        self._apply_image_tab_visibility()

    def _apply_image_tab_visibility(self) -> None:
        """Apply image-viewer tab visibility from engineer settings."""
        show_original = (
            self.show_original_tab_chk.isChecked()
            if self.show_original_tab_chk is not None
            else True
        )
        show_processed = (
            self.show_processed_tab_chk.isChecked()
            if self.show_processed_tab_chk is not None
            else True
        )
        self.preferences.save_show_original_tab(show_original)
        self.preferences.save_show_processed_tab(show_processed)
        self.image_panel.set_optional_tabs_visible(
            show_original=show_original,
            show_processed=show_processed,
        )

    def _refresh_result_image(self) -> None:
        """Render the result tab from the current inspection artifacts."""
        result = self.current_result
        if result is None:
            return

        if result.metadata.get("storage_completed") is False:
            load_image_with_retry(
                self.result_image,
                None,
                on_fail=lambda: self.result_image.setText("Saving result image..."),
            )
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
                on_fail=lambda: self.result_image.setText(
                    "Unable to load saved result image"
                ),
            )
        elif heatmap_path:
            load_heatmap()
        else:
            show_result_frame_data()

    def _refresh_detection_images(self) -> None:
        """Apply all image tabs from one current result snapshot."""
        result = self.current_result
        if result is None:
            return

        if result.metadata.get("storage_completed") is False:
            self._refresh_result_image()
            return

        original_path = result.original_image_path or result.image_path
        load_image_with_retry(
            self.original_image,
            original_path,
            on_fail=lambda: self.original_image.setText(
                "No original image available"
            ),
        )
        load_image_with_retry(
            self.processed_image,
            result.preprocessed_image_path,
            on_fail=lambda: self.processed_image.setText(
                "No processed image available"
            ),
        )
        self._refresh_result_image()



    # _on_camera_disconnected, _on_camera_lost_pipeline_stopped → CameraHandlerMixin
    # The pyqtSlot decorator is not needed on mixin methods — Qt resolves slots
    # by name at runtime regardless of where the method is defined in the MRO.



    def clear_selected_image(self):
        """清除當前選擇影像並切回相機"""
        self.selected_image_path = None
        self.image_path_label.setText(tr(self.current_language, "no_image"))
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
        except (AttributeError, RuntimeError) as exc:
            self._logger.error("Could not reset image/camera controls: %s", exc)

    # on_use_camera_toggled → moved to CameraHandlerMixin





    def log_message(self, message):
        """記錄日誌訊息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    

    # ------------------------------------------------------------------
    # Localized operator-facing overrides
    # ------------------------------------------------------------------

    def _t(self, key: str, **kwargs: object) -> str:
        """Translate a GUI string for the active language."""
        text = tr(self.current_language, key)
        return text.format(**kwargs) if kwargs else text

    def init_system(self):
        """Initialize detection system and camera asynchronously."""
        self._set_camera_status("connecting")
        self.log_message(self._t("init_system"))
        self.camera_worker = self.controller.build_camera_initializer()
        self.camera_worker.finished.connect(self._on_system_init_finished)
        self.camera_worker.start()

    def _on_system_init_finished(self, camera_success):
        try:
            self.detection_system = self.controller.detection_system
            self.log_message(self._t("system_initialized"))
            if camera_success:
                self._set_camera_status("connected")
                self.log_message(self._t("camera_connected"))
                if getattr(self, "use_camera_chk", None):
                    self.use_camera_chk.setChecked(True)
            else:
                self._set_camera_status("unavailable")
                self.log_message(self._t("camera_init_failed"))
        except Exception as exc:
            self._set_camera_status("unavailable")
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
        # Auto Mode armed: Start launches the auto-inspection loop instead of a
        # single manual inspection.
        if self.auto_mode_chk.isChecked():
            self._start_auto_mode()
            return
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

        self._apply_model_light_brightness(product, area, inference_type)
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
        self._single_shot_thread = threading.Thread(
            target=self._run_single_shot,
            args=(image, product, area, inference_type, run_generation),
            daemon=True,
            name=f"single-inspection-{run_generation}",
        )
        self._single_shot_thread.start()

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

        # In Auto Mode, push the verdict onto the always-visible phase banner.
        # This runs before the state machine transitions to SHOW_RESULT, so the
        # banner's result-owned guard keeps PASS/FAIL on screen until removal.
        if self._auto_controller is not None and self._auto_controller.is_running():
            self.image_panel.auto_phase_banner.set_result(
                result.status, self.current_language
            )

        product = result.product or self.product_combo.currentText()
        area = result.area or self.area_combo.currentText()
        inference_type = result.inference_type or self.inference_combo.currentText()
        self._update_version_label(product, area, inference_type)
        self.info_panel.update_result(result)

        self._refresh_detection_images()

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
        if self._auto_controller is not None and self._auto_controller.is_running():
            self.image_panel.auto_phase_banner.set_result("ERROR", self.current_language)
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
        self._set_camera_status("image_mode")
        self.image_path_label.setText(os.path.basename(fname))
        try:
            self.original_image.set_image(fname)
        except (OSError, RuntimeError, ValueError) as exc:
            self._logger.error("Selected image could not be displayed: %s", exc)
            QMessageBox.warning(
                self,
                self._t("load_error_title"),
                str(exc),
            )

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

    # ------------------------------------------------------------------
    # Auto Mode
    # ------------------------------------------------------------------

    def _auto_trigger_config_path(self) -> Path | None:
        """Return models/{product}/{area}/auto_trigger.yaml for the current selection."""
        product = self.product_combo.currentText().strip()
        area = self.area_combo.currentText().strip()
        if not product or not area:
            return None
        return Path(self._models_base) / product / area / "auto_trigger.yaml"

    def _build_auto_trigger_config(self) -> AutoTriggerConfig:
        """Build AutoTriggerConfig with three-layer priority:
        1. Defaults (DEFAULT_AUTO_TRIGGER_CONFIG)
        2. Global config.yaml auto_trigger section
        3. Product/area-level models/{product}/{area}/auto_trigger.yaml  ← highest priority
        """
        cfg_dict = dict(DEFAULT_AUTO_TRIGGER_CONFIG)
        # Layer 2: global config
        try:
            if self.controller.has_system():
                raw = getattr(self.controller.detection_system.config, "auto_trigger", None)
                if isinstance(raw, dict):
                    cfg_dict.update(raw)
        except Exception as exc:
            self._logger.debug("Could not read global auto_trigger config: %s", exc)
        # Layer 3: product/area config
        try:
            path = self._auto_trigger_config_path()
            if path and path.exists():
                with open(path, encoding="utf-8") as f:
                    product_cfg = yaml.safe_load(f) or {}
                cfg_dict.update(product_cfg)
                self._logger.debug("Loaded auto_trigger from %s", path)
        except Exception as exc:
            self._logger.debug("Could not read product auto_trigger config: %s", exc)
        return AutoTriggerConfig.from_dict(cfg_dict)

    def _on_calib_sample_empty(self) -> None:
        if not self._engineering_session_active():
            return
        if self._auto_controller is None or not self._auto_controller.is_running():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, self._t("auto_trigger_calib"), self._t("calib_start_auto_first"))
            return
        area = self._auto_controller.get_current_contour_area()
        if area is not None:
            self.control_panel.set_calib_empty(
                area,
                target=(
                    self.product_combo.currentText(),
                    self.area_combo.currentText(),
                ),
            )

    def _on_calib_sample_product(self) -> None:
        if not self._engineering_session_active():
            return
        if self._auto_controller is None or not self._auto_controller.is_running():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, self._t("auto_trigger_calib"), self._t("calib_start_auto_first"))
            return
        area = self._auto_controller.get_current_contour_area()
        if area is not None:
            self.control_panel.set_calib_product(
                area,
                target=(
                    self.product_combo.currentText(),
                    self.area_combo.currentText(),
                ),
            )

    def _on_calib_apply(
        self,
        threshold: int,
        sampled_product: str,
        sampled_area: str,
    ) -> None:
        """Save calibrated threshold to product/area config and restart worker."""
        from PyQt5.QtWidgets import QMessageBox

        current_target = (
            self.product_combo.currentText().strip(),
            self.area_combo.currentText().strip(),
        )
        sampled_target = (
            str(sampled_product).strip(),
            str(sampled_area).strip(),
        )
        if (
            not self.control_panel.engineering_access_granted
            or self.workspace_stack.currentWidget()
            is not self.engineering_settings_page
            or not all(sampled_target)
            or sampled_target != current_target
        ):
            self.control_panel.clear_calibration()
            QMessageBox.warning(
                self,
                self._t("auto_trigger_calib"),
                self._t("calib_target_changed"),
            )
            return

        save_path = (
            Path(self._models_base)
            / sampled_target[0]
            / sampled_target[1]
            / "auto_trigger.yaml"
        )
        # Load existing file or start fresh, then update only product_area_threshold
        existing: dict = {}
        if save_path.exists():
            try:
                with open(save_path, encoding="utf-8") as f:
                    loaded = yaml.safe_load(f) or {}
                if not isinstance(loaded, dict):
                    raise ValueError("auto_trigger.yaml must contain a mapping.")
                existing = loaded
            except (OSError, ValueError, yaml.YAMLError) as exc:
                self._logger.error(
                    "Cannot read auto-trigger calibration config %s: %s",
                    save_path,
                    exc,
                )
                QMessageBox.critical(
                    self,
                    self._t("auto_trigger_calib"),
                    self._t("calib_save_failed").format(error=exc),
                )
                return
        existing["product_area_threshold"] = threshold
        try:
            serialized = yaml.safe_dump(
                existing,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            ).encode("utf-8")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_file = QSaveFile(str(save_path))
            if not save_file.open(QIODevice.WriteOnly):
                raise OSError(save_file.errorString())
            if save_file.write(serialized) != len(serialized):
                error = save_file.errorString()
                save_file.cancelWriting()
                raise OSError(
                    error or "Incomplete calibration configuration write."
                )
            if not save_file.commit():
                raise OSError(save_file.errorString())
        except (OSError, yaml.YAMLError) as exc:
            self._logger.error(
                "Cannot save auto-trigger calibration config %s: %s",
                save_path,
                exc,
            )
            QMessageBox.critical(
                self,
                self._t("auto_trigger_calib"),
                self._t("calib_save_failed").format(error=exc),
            )
            return

        # Restart only after the old worker has actually finished so the new
        # threshold cannot be reported active while the old state machine runs.
        was_running = self._auto_controller is not None and self._auto_controller.is_running()
        if was_running:
            self._stop_auto_mode(restart_after_stop=True)

        self.control_panel.clear_calibration()
        product, area = sampled_target
        QMessageBox.information(
            self,
            self._t("calib_saved_title"),
            self._t("calib_saved_msg").format(threshold=f"{threshold:,}", product=product, area=area),
        )

    def _on_auto_mode_toggled(self, enabled: bool) -> None:
        """Arm or disarm Auto Mode.

        Checking the box only *selects* the mode; the auto-inspection loop is
        started when the operator presses Start (see ``start_detection``).
        Unchecking stops a running loop and disarms.
        """
        if enabled:
            # Arm only. Stop any manual inspection so Start is free for Auto.
            if self.is_detection_running():
                self.stop_detection()
            self.log_message(self._t("auto_armed"))
            self.statusBar().showMessage(self._t("auto_armed"), 4000)
            self.update_start_enabled()
        else:
            if self._auto_controller is not None and self._auto_controller.is_running():
                self._stop_auto_mode()
            else:
                self.update_start_enabled()

    def _start_auto_mode(self) -> None:
        """Validate prerequisites and start AutoInspectionController."""
        product = self.product_combo.currentText()
        area = self.area_combo.currentText()
        inference_type = self.inference_combo.currentText()
        if not all([product, area, inference_type]):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                self._t("missing_params_title"),
                self._t("missing_params"),
            )
            self.auto_mode_chk.blockSignals(True)
            self.auto_mode_chk.setChecked(False)
            self.auto_mode_chk.blockSignals(False)
            return

        if not self.controller.has_system():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                self._t("system_not_ready_title"),
                self._t("system_not_ready"),
            )
            self.auto_mode_chk.blockSignals(True)
            self.auto_mode_chk.setChecked(False)
            self.auto_mode_chk.blockSignals(False)
            return

        if not self.controller.is_camera_connected():
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Auto Mode", self._t("auto_mode_camera_missing"))
            self.auto_mode_chk.blockSignals(True)
            self.auto_mode_chk.setChecked(False)
            self.auto_mode_chk.blockSignals(False)
            return

        self._apply_model_light_brightness(product, area, inference_type)

        # Stop any running pipeline/single-shot before taking exclusive camera access
        if self.is_detection_running():
            self.stop_detection()

        # Auto mode supplies preview frames directly to detect(). Prepare the
        # model-specific camera settings before the preview worker can capture
        # its first frame, otherwise that first inspection can use old exposure.
        try:
            self.controller.detection_system.prepare_auto_inspection(
                product, area, inference_type
            )
        except Exception as exc:
            self._logger.exception("Auto-mode preflight failed: %s", exc)
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Auto Mode", f"無法套用檢測參數：{exc}")
            return

        config = self._build_auto_trigger_config()
        if self._auto_controller is None:
            self._auto_controller = AutoInspectionController(
                detection_system=self.controller.detection_system,
                bridge=self.controller.bridge,
                config=config,
                show_debug_overlay=True,
            )
            self._auto_controller.auto_state_changed.connect(
                self._on_auto_state_changed
            )
            self._auto_controller.auto_error.connect(self._on_auto_error)
            self._auto_controller.fully_stopped.connect(
                self._on_auto_controller_stopped
            )
        else:
            self._auto_controller.set_config(config)

        ok = self._auto_controller.start(product, area, inference_type)
        if not ok:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Auto Mode", self._t("auto_mode_start_failed"))
            self.auto_mode_chk.blockSignals(True)
            self.auto_mode_chk.setChecked(False)
            self.auto_mode_chk.blockSignals(False)
            return

        # While Auto runs: Start is locked, Stop ends the loop.
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.use_camera_chk.setEnabled(False)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("AUTO")
        self.control_panel.set_auto_mode_status("WAIT_EMPTY")
        self.image_panel.auto_phase_banner.activate(self.current_language)
        self.log_message(
            f"自動模式已啟動 — {product}/{area}/{inference_type}"
        )

    def _stop_auto_mode(self, *, restart_after_stop: bool = False) -> bool:
        """Stop Auto Mode, optionally restarting after confirmed termination."""
        controller = self._auto_controller
        generation = (
            controller.active_generation
            if controller is not None
            else None
        )
        if restart_after_stop and generation is not None:
            self._pending_auto_restart = (
                generation,
                self.product_combo.currentText().strip(),
                self.area_combo.currentText().strip(),
                self.inference_combo.currentText().strip(),
            )
        else:
            self._pending_auto_restart = None
        stopped = True
        if controller is not None:
            stopped = controller.stop()
        self.engineering_settings_page.clear_preview()
        self.control_panel.set_auto_mode_status("")
        self.image_panel.auto_phase_banner.deactivate()
        self.statusBar().clearMessage()
        if not stopped:
            self.control_panel.set_auto_mode_status("STOPPING")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.use_camera_chk.setEnabled(False)
            self.statusBar().showMessage(tr(self.current_language, "stopping"))
            return False
        pending_restart = self._pending_auto_restart
        self._pending_auto_restart = None
        self._finish_auto_mode_stop_ui()
        self._restart_auto_mode_if_valid(pending_restart)
        return True

    def _finish_auto_mode_stop_ui(self) -> None:
        """Restore controls only after all work from the old run has ended."""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.use_camera_chk.setEnabled(True)
        if getattr(self, "big_status_label", None):
            self.big_status_label.set_status("READY")
        self.log_message("自動模式已停止")
        self.update_start_enabled()

    def _restart_auto_mode_if_valid(
        self,
        pending_restart: tuple[int, str, str, str] | None,
    ) -> bool:
        """Restart a saved target only when the original request is still valid."""
        if pending_restart is None:
            return False
        _generation, product, area, inference_type = pending_restart
        current_target = (
            self.product_combo.currentText().strip(),
            self.area_combo.currentText().strip(),
            self.inference_combo.currentText().strip(),
        )
        if (
            not self.auto_mode_chk.isChecked()
            or self._shutdown_in_progress
            or self._closing
            or current_target != (product, area, inference_type)
        ):
            return False
        self._start_auto_mode()
        return True

    @pyqtSlot(int)
    def _on_auto_controller_stopped(self, generation: int) -> None:
        """Complete a timed-out stop and any deferred calibration restart."""
        if self._close_after_auto_stop:
            if self._closing_auto_generation != generation:
                return
            self._close_after_auto_stop = False
            self._closing_auto_generation = None
            self._pending_auto_restart = None
            QTimer.singleShot(0, self.close)
            return
        pending_restart = self._pending_auto_restart
        if (
            pending_restart is not None
            and pending_restart[0] != generation
        ):
            return
        controller = self._auto_controller
        if (
            controller is not None
            and controller.active_generation is not None
        ):
            return
        self._pending_auto_restart = None
        self.engineering_settings_page.clear_preview()
        self.control_panel.set_auto_mode_status("")
        self.image_panel.auto_phase_banner.deactivate()
        self.statusBar().clearMessage()
        self._finish_auto_mode_stop_ui()
        self._restart_auto_mode_if_valid(pending_restart)

    def _on_auto_state_changed(self, state_name: str) -> None:
        """Update status label when auto-trigger state changes."""
        self.control_panel.set_auto_mode_status(state_name)
        self.statusBar().showMessage(f"Auto: {state_name}")
        self.image_panel.auto_phase_banner.set_phase(state_name, self.current_language)

    def _on_auto_error(self, msg: str) -> None:
        """Handle fatal auto-inspection error (e.g. camera lost)."""
        self._set_camera_status("lost")
        self.log_message(f"自動模式錯誤: {msg}")
        # Turn off auto mode checkbox to avoid a locked-down UI
        self.auto_mode_chk.blockSignals(True)
        self.auto_mode_chk.setChecked(False)
        self.auto_mode_chk.blockSignals(False)
        self._stop_auto_mode()

    def closeEvent(self, event):
        """Close the GUI after persisting preferences."""
        if self._close_after_auto_stop:
            event.ignore()
            return
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

        self._closing = True
        self._pending_auto_restart = None
        auto_controller = self._auto_controller
        closing_generation = (
            auto_controller.active_generation
            if auto_controller is not None
            else None
        )
        if auto_controller is not None and not auto_controller.stop():
            self._close_after_auto_stop = True
            self._closing_auto_generation = closing_generation
            self.statusBar().showMessage(tr(self.current_language, "stopping"))
            if is_pipeline_running:
                QApplication.restoreOverrideCursor()
            event.ignore()
            return
        try:
            if self._retraining_workspace is not None:
                self._retraining_workspace.shutdown_workspace()
            self.inspection_history_page.shutdown()
            self.shutdown_light()
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
            self.preferences.save_show_original_tab(
                self.show_original_tab_chk.isChecked()
            )
            self.preferences.save_show_processed_tab(
                self.show_processed_tab_chk.isChecked()
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
