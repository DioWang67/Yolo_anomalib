from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QAction,
    QMenuBar,
    QMessageBox,
)

from app.gui.i18n import tr
from core.workspace import load_workspace_paths

if TYPE_CHECKING:
    from app.gui.main_window import DetectionSystemGUI


def _lang(gui: DetectionSystemGUI) -> str:
    return getattr(gui, "current_language", "en")


def _retraining_targets(gui: DetectionSystemGUI) -> tuple[tuple[str, str], ...]:
    """Return deployed product/area pairs without an unsafe all-target option."""
    targets: set[tuple[str, str]] = set()
    available_areas = getattr(gui, "available_areas", {})
    if isinstance(available_areas, dict):
        for product, areas in available_areas.items():
            if not isinstance(areas, (list, tuple, set)):
                continue
            for area in areas:
                normalized = (str(product).strip(), str(area).strip())
                if all(normalized):
                    targets.add(normalized)
    selected = (
        gui.product_combo.currentText().strip(),
        gui.area_combo.currentText().strip(),
    )
    if all(selected):
        targets.add(selected)
    return tuple(sorted(targets))


def _reload_models(gui: DetectionSystemGUI) -> None:
    """Clear runtime model caches, then refresh the filesystem catalog."""
    if gui.is_detection_running():
        message = "Stop inspection before reloading models."
        gui.log_message(message)
        QMessageBox.warning(gui, "Model reload", message)
        return
    try:
        product = gui.product_combo.currentText().strip() or None
        area = gui.area_combo.currentText().strip() or None
        inference_type = gui.inference_combo.currentText().strip() or None
        gui.controller.reload_model_settings(product, area, inference_type)
        gui.load_available_models()
        gui.log_message("Model cache cleared; the next inspection will load deployed files.")
    except Exception as exc:
        gui.log_message(f"Model reload failed: {exc}")


def _open_training_review(gui: DetectionSystemGUI) -> None:
    """Open the button-based data review without interrupting inference state."""
    if gui.is_detection_running():
        language = _lang(gui)
        message = (
            "請先停止檢測，再複核訓練資料。"
            if language.lower().startswith("zh")
            else "Stop inspection before reviewing training data."
        )
        gui.log_message(message)
        QMessageBox.warning(
            gui,
            "訓練資料" if language.lower().startswith("zh") else "Training Data",
            message,
        )
        return
    try:
        project_root = getattr(gui, "_project_root", None)
        if project_root is None:
            project_root = Path.cwd()
        workspace = load_workspace_paths(project_root)
        workspace_args = {
            "result_root": project_root / "Result",
            "manifest_path": project_root / "review_manifest.csv",
            "training_data_dir": workspace.training_data,
            "language": _lang(gui),
            "product": gui.product_combo.currentText().strip() or None,
            "area": gui.area_combo.currentText().strip() or None,
        }
        if hasattr(gui, "show_retraining_workspace"):
            gui.show_retraining_workspace(
                **workspace_args,
                available_targets=_retraining_targets(gui),
            )
        else:
            _run_review_dialog(parent=gui, **workspace_args)
        gui.log_message("Training data review opened; decisions are saved immediately.")
    except (OSError, RuntimeError, ValueError, csv.Error) as exc:
        gui.log_message(f"Training data review failed: {exc}")


def _run_review_dialog(**kwargs) -> int:
    """Import the dialog lazily to keep standalone GUI startup acyclic."""
    from app.gui.review_cases_dialog import run_review_dialog

    return run_review_dialog(**kwargs)


def _open_model_versions(gui: DetectionSystemGUI) -> None:
    """Open the unified model and color component version inventory."""
    try:
        from app.gui.inspection_components_dialog import (
            InspectionComponentsDialog,
        )
        from core.services.inspection_component_catalog import (
            InspectionComponentCatalog,
        )

        project_root = getattr(
            gui,
            "_project_root",
            gui._models_base.parent,
        )
        catalog = InspectionComponentCatalog(
            models_root=gui._models_base,
            color_revisions_root=project_root / ".color_revisions",
            color_profiles_root=project_root / ".color_profiles",
            inspection_releases_root=project_root / ".inspection_releases",
        )

        def on_create_combination(record) -> None:
            _open_inspection_releases(
                gui,
                product=record.product,
                area=record.area,
                inference_type=record.inference_type,
                preferred_model_version=(
                    record.version
                    if record.category == "AI_MODEL"
                    else ""
                ),
                preferred_color_version=(
                    record.version
                    if record.category == "COLOR_REVISION"
                    else ""
                ),
                begin_creation=True,
            )

        _run_inspection_components_dialog(
            dialog_class=InspectionComponentsDialog,
            catalog=catalog,
            selected_product=gui.product_combo.currentText().strip(),
            selected_area=gui.area_combo.currentText().strip(),
            selected_inference_type=gui.inference_combo.currentText().strip(),
            on_create_combination=on_create_combination,
            parent=gui,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        gui.log_message(f"模型版本管理開啟失敗：{exc}")
        QMessageBox.critical(
            gui,
            "模型版本管理",
            f"無法載入模型版本資料：\n{exc}",
        )


def _open_inspection_releases(
    gui: DetectionSystemGUI,
    *,
    product: str | None = None,
    area: str | None = None,
    inference_type: str | None = None,
    preferred_model_version: str = "",
    preferred_color_version: str = "",
    begin_creation: bool = False,
) -> None:
    """Open the single control plane for model/color release combinations."""
    try:
        from app.gui.inspection_releases_dialog import InspectionReleasesDialog
        from core.services.inspection_release_store import InspectionReleaseStore

        store = InspectionReleaseStore(
            gui._project_root / ".inspection_releases"
        )

        def on_activated(release) -> None:
            gui.controller.reload_model_settings(
                release.scope.product,
                release.scope.area,
                release.scope.inference_type,
            )
            gui._catalog.refresh()
            gui.load_available_models()
            gui.log_message(
                "Inspection release activated: "
                f"{release.scope.product}/{release.scope.area} "
                f"{release.display_version}"
            )

        dialog = InspectionReleasesDialog(
            store=store,
            product=product or gui.product_combo.currentText().strip() or None,
            area=area or gui.area_combo.currentText().strip() or None,
            inference_type=(
                inference_type
                or gui.inference_combo.currentText().strip()
                or None
            ),
            preferred_model_version=preferred_model_version,
            preferred_color_version=preferred_color_version,
            is_inspection_running=gui.is_detection_running,
            on_activated=on_activated,
            parent=gui,
        )
        if begin_creation:
            QTimer.singleShot(0, dialog.begin_combination_creation)
        dialog.exec_()
        refresh_summary = getattr(
            gui,
            "refresh_engineering_version_summary",
            None,
        )
        if callable(refresh_summary):
            refresh_summary()
    except (OSError, RuntimeError, ValueError) as exc:
        gui.log_message(f"Inspection release manager failed: {exc}")
        QMessageBox.critical(gui, "檢測組合管理", str(exc))


def _run_model_versions_dialog(**kwargs) -> int:
    """Import the version dialog lazily to keep GUI startup lightweight."""
    from app.gui.model_versions_dialog import ModelVersionsDialog

    return ModelVersionsDialog(**kwargs).exec_()


def _run_inspection_components_dialog(*, dialog_class, **kwargs) -> int:
    """Construct the injected component inventory dialog and run it."""
    return dialog_class(**kwargs).exec_()


def _open_model_update_status(gui: DetectionSystemGUI) -> None:
    """Open the read-only cross-project model update status screen."""
    project_root = getattr(gui, "_project_root", Path.cwd())
    try:
        data_root = load_workspace_paths(project_root).training_data
        if hasattr(gui, "show_retraining_workspace"):
            workspace = gui.show_retraining_workspace(
                result_root=project_root / "Result",
                manifest_path=project_root / "review_manifest.csv",
                training_data_dir=data_root,
                language=_lang(gui),
                product=gui.product_combo.currentText().strip() or None,
                area=gui.area_combo.currentText().strip() or None,
                available_targets=_retraining_targets(gui),
            )
            workspace.show_progress_page()
            return
        _run_model_update_status_dialog(
            data_root=data_root,
            language=_lang(gui),
            selected_product=gui.product_combo.currentText().strip() or None,
            selected_area=gui.area_combo.currentText().strip() or None,
            parent=gui,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        gui.log_message(f"模型更新狀態載入失敗：{exc}")
        QMessageBox.critical(gui, "模型更新狀態", str(exc))


def _run_model_update_status_dialog(**kwargs) -> int:
    """Import the status dialog lazily to keep normal inference startup fast."""
    from app.gui.model_update_status_dialog import ModelUpdateStatusDialog

    kwargs.setdefault("background_refresh", True)
    return ModelUpdateStatusDialog(**kwargs).exec_()


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
    refresh_action.triggered.connect(lambda: _reload_models(gui))
    view_menu.addAction(refresh_action)

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

    lighting_menu.addSeparator()
    calibration_action = QAction(tr(language, "calibration_menu"), gui)
    calibration_action.triggered.connect(gui.open_calibration_dialog)
    lighting_menu.addAction(calibration_action)

    help_menu = menubar.addMenu(tr(language, "help_menu"))
    about_action = QAction(tr(language, "about"), gui)
    about_action.triggered.connect(gui.show_about)
    help_menu.addAction(about_action)

    return menubar
