"""Non-blocking host for the embedded retraining workspace."""

from __future__ import annotations

import csv
import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.hover_help import HoverHelpBadge
from core.station_data import resolve_inference_path_contract
from tools.retraining_workspaces import (
    RetrainingWorkspace,
    RetrainingWorkspaceError,
    create_retraining_workspace,
    list_retraining_workspaces,
    load_retraining_workspace,
)
from tools.submission_history import suggest_next_training_batch_version

logger = logging.getLogger(__name__)


class ReviewManifestPreloadWorker(QThread):
    """Prepare a review manifest without blocking Qt's main thread."""

    manifest_ready = pyqtSignal(int, object)
    manifest_failed = pyqtSignal(int, str)

    def __init__(
        self,
        *,
        generation: int,
        result_root: Path,
        manifest_path: Path,
        product: str | None,
        area: str | None,
        freeze_existing: bool = False,
    ) -> None:
        super().__init__()
        self._generation = generation
        self._result_root = result_root
        self._manifest_path = manifest_path
        self._product = product
        self._area = area
        self._freeze_existing = freeze_existing

    def run(self) -> None:
        try:
            from app.gui.review_cases_dialog import (
                ReviewManifestStore,
                _target_manifest_path,
                prepare_review_manifest,
            )

            if self._freeze_existing:
                target_path = _target_manifest_path(
                    self._manifest_path,
                    product=self._product,
                    area=self._area,
                )
                rows = [
                    dict(row)
                    for row in ReviewManifestStore(
                        target_path,
                        synchronize_repository=False,
                    ).rows
                ]
            else:
                _target_path, rows = prepare_review_manifest(
                    result_root=self._result_root,
                    manifest_path=self._manifest_path,
                    product=self._product,
                    area=self._area,
                )
        except (OSError, RuntimeError, ValueError, csv.Error, sqlite3.Error) as exc:
            self.manifest_failed.emit(self._generation, str(exc))
            return
        if not self.isInterruptionRequested():
            self.manifest_ready.emit(self._generation, rows)


_ACTIVE_PRELOAD_WORKERS: set[ReviewManifestPreloadWorker] = set()


def _retain_worker(worker: ReviewManifestPreloadWorker) -> None:
    """Keep an unparented QThread alive until its native thread exits."""
    _ACTIVE_PRELOAD_WORKERS.add(worker)

    def release() -> None:
        _ACTIVE_PRELOAD_WORKERS.discard(worker)
        worker.deleteLater()

    worker.finished.connect(release)


class RetrainingWorkspaceHost(QWidget):
    """Show loading/error state, then host one persistent review workspace."""

    back_to_inspection_requested = pyqtSignal()
    workspace_ready = pyqtSignal(int)
    workspace_failed = pyqtSignal(str)
    color_configuration_changed = pyqtSignal(str, str, str)

    def __init__(
        self,
        *,
        result_root: str | Path,
        manifest_path: str | Path,
        training_data_dir: str | Path,
        inference_models_dir: str | Path | None = None,
        inference_station_data_dir: str | Path | None = None,
        inference_project_root: str | Path | None = None,
        language: str,
        product: str | None,
        area: str | None,
        available_targets: Iterable[tuple[str, str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("RetrainingWorkspaceHost")
        self.setStyleSheet(
            "QWidget#RetrainingWorkspaceHost { background:#f3f6fa; }"
            "QWidget#RetrainingWorkspaceHost {"
            "font-family:'Segoe UI','Microsoft JhengHei';font-size:10pt;}"
            "QFrame#retrainingTargetSelector {"
            "background:white;border:1px solid #dce3ec;border-radius:9px;}"
            "QFrame#retrainingTargetSelector QComboBox {"
            "background:#f8fafc;border:1px solid #c7d1dc;border-radius:6px;"
            "padding:5px 10px;min-height:24px;}"
            "QPushButton#workspaceBackButton {"
            "background:transparent;color:#34506b;border:0;padding:7px 10px;"
            "font-weight:600;}"
            "QPushButton#workspaceBackButton:hover {background:#edf3f8;border-radius:6px;}"
        )
        self.result_root = Path(result_root)
        self.manifest_path = Path(manifest_path)
        self.training_data_dir = Path(training_data_dir)
        inference_paths = resolve_inference_path_contract(
            models_dir=inference_models_dir,
            station_data_dir=inference_station_data_dir,
            project_root=inference_project_root,
        )
        self.inference_models_dir = inference_paths.models_dir
        self.inference_station_data_dir = inference_paths.station_data_dir
        self.inference_project_root = inference_paths.project_root
        self.language = language
        self._available_targets = self._normalize_targets(
            available_targets,
            product=product,
            area=area,
        )
        self.product: str | None
        self.area: str | None
        if product and area:
            self.product, self.area = product, area
        elif self._available_targets:
            self.product, self.area = self._available_targets[0]
        else:
            self.product, self.area = None, None
        self._generation = 0
        self._closed = False
        self._pending_progress_page = False
        self._pending_target: tuple[str, str] | None = None
        self._pending_workspace: RetrainingWorkspace | None = None
        self.active_workspace: RetrainingWorkspace | None = None
        self._worker: ReviewManifestPreloadWorker | None = None
        self._workspace = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_target_selector())
        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)
        self.loading_page = self._build_loading_page()
        self.stack.addWidget(self.loading_page)
        self.stack.setCurrentWidget(self.loading_page)
        self._refresh_batch_filter()

    @staticmethod
    def _normalize_targets(
        targets: Iterable[tuple[str, str]] | None,
        *,
        product: str | None,
        area: str | None,
    ) -> tuple[tuple[str, str], ...]:
        normalized = {
            (str(target_product).strip(), str(target_area).strip())
            for target_product, target_area in (targets or ())
            if str(target_product).strip() and str(target_area).strip()
        }
        if product and area:
            normalized.add((str(product).strip(), str(area).strip()))
        return tuple(sorted(normalized))

    @property
    def workspace(self):
        """Return the loaded ReviewCasesDialog, or ``None`` while loading."""
        return self._workspace

    def _build_target_selector(self) -> QWidget:
        panel = QFrame(self)
        panel.setObjectName("retrainingTargetSelector")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(10, 8, 12, 8)
        layout.setSpacing(9)
        back_button = QPushButton(
            "← 返回檢測" if self.language.lower().startswith("zh") else "← Inspection"
        )
        back_button.setObjectName("workspaceBackButton")
        back_button.clicked.connect(self.back_to_inspection_requested.emit)
        layout.addWidget(back_button)
        divider = QFrame(panel)
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color:#dce3ec;")
        layout.addWidget(divider)
        title = QLabel(
            "補訓資料" if self.language.lower().startswith("zh") else "Retraining data"
        )
        title.setStyleSheet(
            "font-size:12pt;font-weight:600;color:#1f3347;padding:0 6px;border:0;"
        )
        layout.addWidget(title)
        layout.addWidget(
            QLabel("機種" if self.language.lower().startswith("zh") else "Product")
        )
        self.product_filter = QComboBox(panel)
        self.product_filter.setObjectName("retrainingProductFilter")
        self.product_filter.setMinimumWidth(170)
        layout.addWidget(self.product_filter)
        layout.addWidget(
            QLabel("區域" if self.language.lower().startswith("zh") else "Area")
        )
        self.area_filter = QComboBox(panel)
        self.area_filter.setObjectName("retrainingAreaFilter")
        self.area_filter.setMinimumWidth(120)
        layout.addWidget(self.area_filter)
        layout.addWidget(
            QLabel(
                "補訓資料夾"
                if self.language.lower().startswith("zh")
                else "Training folder"
            )
        )
        self.batch_filter = QComboBox(panel)
        self.batch_filter.setObjectName("retrainingBatchFilter")
        self.batch_filter.setMinimumWidth(220)
        layout.addWidget(self.batch_filter)
        self.create_batch_button = QPushButton(
            "＋ 建立資料夾"
            if self.language.lower().startswith("zh")
            else "+ Create folder"
        )
        self.create_batch_button.setObjectName("createRetrainingBatchButton")
        self.create_batch_button.clicked.connect(self._create_batch_workspace)
        layout.addWidget(self.create_batch_button)
        self.target_scope_label = HoverHelpBadge(
            self._target_scope_help_text(),
            language=self.language,
            parent=panel,
        )
        self.target_scope_label.setObjectName("retrainingTargetScopeHelp")
        layout.addWidget(self.target_scope_label)
        layout.addStretch()

        self.product_filter.blockSignals(True)
        self.product_filter.addItems(
            sorted({target[0] for target in self._available_targets})
        )
        if self.product:
            self.product_filter.setCurrentText(self.product)
        self.product_filter.blockSignals(False)
        self._populate_area_filter(self.product or "")
        self.product_filter.currentTextChanged.connect(self._on_product_changed)
        self.area_filter.currentTextChanged.connect(self._on_area_changed)
        self.batch_filter.currentIndexChanged.connect(self._on_batch_changed)
        self._update_target_scope_label()
        return panel

    def _populate_area_filter(self, product: str) -> None:
        areas = sorted(
            target_area
            for target_product, target_area in self._available_targets
            if target_product == product
        )
        preferred_area = self.area if self.area in areas else None
        self.area_filter.blockSignals(True)
        self.area_filter.clear()
        self.area_filter.addItems(areas)
        if preferred_area:
            self.area_filter.setCurrentText(preferred_area)
        self.area_filter.blockSignals(False)

    def _on_product_changed(self, product: str) -> None:
        self._populate_area_filter(str(product).strip())
        self._request_target(
            str(product).strip(),
            self.area_filter.currentText().strip(),
        )

    def _on_area_changed(self, area: str) -> None:
        self._request_target(
            self.product_filter.currentText().strip(),
            str(area).strip(),
        )

    def _refresh_batch_filter(self, *, preferred_version: str = "") -> None:
        """Refresh folders for the active target and open the selected one."""
        workspaces = list_retraining_workspaces(
            self.training_data_dir,
            product=self.product,
            area=self.area,
        )
        self.batch_filter.blockSignals(True)
        try:
            self.batch_filter.clear()
            for workspace in workspaces:
                state_label = (
                    "已送訓"
                    if workspace.state == "submitted"
                    and self.language.lower().startswith("zh")
                    else (
                        "編輯中"
                        if self.language.lower().startswith("zh")
                        else (
                            "Submitted"
                            if workspace.state == "submitted"
                            else "Draft"
                        )
                    )
                )
                self.batch_filter.addItem(
                    f"{workspace.batch_version}｜{state_label}",
                    str(workspace.root),
                )
            preferred_index = next(
                (
                    index
                    for index, workspace in enumerate(workspaces)
                    if workspace.batch_version == preferred_version
                ),
                0,
            )
            if workspaces:
                self.batch_filter.setCurrentIndex(preferred_index)
        finally:
            self.batch_filter.blockSignals(False)
        if not workspaces:
            self.active_workspace = None
            self._dispose_workspace()
            self.retry_button.setVisible(False)
            self.status_label.setText(
                "請先建立補訓資料夾，再進入選照片與複核。"
                if self.language.lower().startswith("zh")
                else "Create a training folder before selecting and reviewing images."
            )
            self.stack.setCurrentWidget(self.loading_page)
            return
        self._activate_workspace(workspaces[preferred_index])

    def _on_batch_changed(self, index: int) -> None:
        path = str(self.batch_filter.itemData(index) or "")
        if not path:
            return
        try:
            workspace = load_retraining_workspace(path)
        except (OSError, RetrainingWorkspaceError) as exc:
            self._show_failure(str(exc))
            return
        self._activate_workspace(workspace)

    def _activate_workspace(self, workspace: RetrainingWorkspace) -> None:
        if (workspace.product, workspace.area) != (self.product, self.area):
            self._show_failure("補訓資料夾與目前產品／工位不一致。")
            return
        if (
            self.active_workspace is not None
            and self.active_workspace.root == workspace.root
            and self._workspace is not None
        ):
            return
        self._generation += 1
        self._dispose_workspace()
        self.active_workspace = workspace
        self._update_target_scope_label()
        self.stack.setCurrentWidget(self.loading_page)
        if self._worker is not None and self._worker.isRunning():
            self._pending_workspace = workspace
            self._worker.requestInterruption()
            return
        self._pending_workspace = None
        self._start_loading()

    def _create_batch_workspace(self) -> None:
        product = str(self.product or "").strip()
        area = str(self.area or "").strip()
        if not product or not area:
            QMessageBox.warning(
                self,
                "建立補訓資料夾",
                "請先選擇產品與工位。",
            )
            return
        suggestion = suggest_next_training_batch_version(
            self.training_data_dir,
            product=product,
            area=area,
        )
        version, accepted = QInputDialog.getText(
            self,
            "建立補訓資料夾"
            if self.language.lower().startswith("zh")
            else "Create training folder",
            "資料夾版本："
            if self.language.lower().startswith("zh")
            else "Folder version:",
            text=suggestion,
        )
        if not accepted:
            return
        try:
            workspace = create_retraining_workspace(
                self.training_data_dir,
                product=product,
                area=area,
                batch_version=version,
            )
        except (OSError, ValueError, RetrainingWorkspaceError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self._refresh_batch_filter(preferred_version=workspace.batch_version)

    def _request_target(self, product: str, area: str) -> None:
        target = (product, area)
        if not all(target) or target not in self._available_targets:
            return
        if target == (self.product, self.area) and self._pending_target is None:
            return
        self._pending_target = target
        self._generation += 1
        self.stack.setCurrentWidget(self.loading_page)
        self.retry_button.setVisible(False)
        self.status_label.setText(
            f"正在切換到 {product}/{area}…"
            if self.language.lower().startswith("zh")
            else f"Switching to {product}/{area}…"
        )
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            return
        self._activate_pending_target()

    def _activate_pending_target(self) -> None:
        if self._closed or self._pending_target is None:
            return
        self.product, self.area = self._pending_target
        self._pending_target = None
        self._dispose_workspace()
        self.active_workspace = None
        self._update_target_scope_label()
        self._refresh_batch_filter()

    def _update_target_scope_label(self) -> None:
        text = self._target_scope_help_text()
        self.target_scope_label.setToolTip(text)
        self.target_scope_label.setAccessibleDescription(text)

    def _target_scope_help_text(self) -> str:
        if self.product and self.area:
            batch_text = (
                f"目前資料夾：{self.active_workspace.batch_version}。"
                if self.active_workspace is not None
                and self.language.lower().startswith("zh")
                else (
                    f"Folder: {self.active_workspace.batch_version}. "
                    if self.active_workspace is not None
                    else ""
                )
            )
            return (
                f"{batch_text}目前只顯示 {self.product}/{self.area}，"
                "照片與送訓資料不會混入其他機種。"
                if self.language.lower().startswith("zh")
                else f"{batch_text}Showing only {self.product}/{self.area}; photos and "
                "training data from other targets are excluded."
            )
        return (
            "請先選擇機種與區域。"
            if self.language.lower().startswith("zh")
            else "Select a product and area first."
        )

    def _dispose_workspace(self) -> None:
        if self._workspace is None:
            return
        self._workspace.shutdown_workspace()
        self.stack.removeWidget(self._workspace)
        self._workspace.deleteLater()
        self._workspace = None

    def _build_loading_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 36, 36, 36)
        layout.addStretch()
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "font-size:14pt;color:#334e68;background:#eef5fb;"
            "border:1px solid #b8d0e5;border-radius:8px;padding:24px;"
        )
        layout.addWidget(self.status_label)
        self.retry_button = QPushButton(
            "重新載入" if self.language.lower().startswith("zh") else "Retry"
        )
        self.retry_button.clicked.connect(self._start_loading)
        self.retry_button.setVisible(False)
        layout.addWidget(self.retry_button, alignment=Qt.AlignCenter)
        layout.addStretch()
        return page

    def _start_loading(self) -> None:
        if self._closed or (self._worker is not None and self._worker.isRunning()):
            return
        if not self.product or not self.area:
            self._show_failure(
                "請先選擇機種與區域。"
                if self.language.lower().startswith("zh")
                else "Select a product and area first."
            )
            return
        if self.active_workspace is None:
            self._show_failure(
                "請先建立或選擇補訓資料夾。"
                if self.language.lower().startswith("zh")
                else "Create or select a training folder first."
            )
            return
        self._generation += 1
        generation = self._generation
        self.retry_button.setVisible(False)
        self.status_label.setText(
            f"正在開啟 {self.active_workspace.batch_version}，整理可選照片…\n"
            "你可以立即返回檢測，整理不會卡住畫面。"
            if self.language.lower().startswith("zh")
            else f"Opening {self.active_workspace.batch_version} and preparing "
            f"selectable {self.product}/{self.area} inspection images "
            "in the background…\n"
            "You may return to inspection without blocking the window."
        )
        worker = ReviewManifestPreloadWorker(
            generation=generation,
            result_root=self.result_root,
            manifest_path=self.active_workspace.manifest_path,
            product=self.product,
            area=self.area,
            freeze_existing=self.active_workspace.state == "submitted",
        )
        self._worker = worker
        worker.finished.connect(
            lambda gen=generation, current=worker: self._on_worker_finished(
                gen, current
            )
        )
        _retain_worker(worker)
        worker.manifest_ready.connect(self._on_manifest_ready)
        worker.manifest_failed.connect(self._on_manifest_failed)
        worker.start()

    def _on_worker_finished(
        self, generation: int, worker: ReviewManifestPreloadWorker
    ) -> None:
        if self._worker is worker:
            self._worker = None
        if self._pending_target is not None:
            self._activate_pending_target()
        elif self._pending_workspace is not None:
            pending = self._pending_workspace
            self._pending_workspace = None
            self.active_workspace = pending
            self._start_loading()

    def _on_manifest_ready(self, generation: int, rows: object) -> None:
        if self._closed or generation != self._generation:
            return
        prepared_rows = [dict(row) for row in rows] if isinstance(rows, list) else []
        active_workspace = self.active_workspace
        if active_workspace is None:
            return
        try:
            from app.gui.review_cases_dialog import ReviewCasesDialog

            workspace = ReviewCasesDialog(
                result_root=self.result_root,
                manifest_path=active_workspace.manifest_path,
                training_data_dir=self.training_data_dir,
                inference_models_dir=self.inference_models_dir,
                inference_station_data_dir=self.inference_station_data_dir,
                inference_project_root=self.inference_project_root,
                language=self.language,
                product=self.product,
                area=self.area,
                start_in_overview=True,
                embedded=True,
                show_embedded_navigation=False,
                manifest_prepared=True,
                prepared_rows=prepared_rows,
                batch_version=active_workspace.batch_version,
                batch_workspace_dir=active_workspace.root,
                parent=self.stack,
            )
        except (OSError, RuntimeError, ValueError, csv.Error, sqlite3.Error) as exc:
            logger.exception("Could not construct the prepared review workspace")
            self._show_failure(str(exc))
            return
        workspace.back_to_inspection_requested.connect(
            self.back_to_inspection_requested.emit
        )
        workspace.color_configuration_changed.connect(
            self.color_configuration_changed.emit
        )
        self.stack.addWidget(workspace)
        self._workspace = workspace
        if self._pending_progress_page:
            workspace.show_progress_page()
            self._pending_progress_page = False
        self.stack.setCurrentWidget(workspace)
        self.workspace_ready.emit(len(prepared_rows))

    def _on_manifest_failed(self, generation: int, message: str) -> None:
        if self._closed or generation != self._generation:
            return
        self._show_failure(message)

    def _show_failure(self, message: str) -> None:
        self.status_label.setText(
            (f"補訓資料載入失敗：\n{message}")
            if self.language.lower().startswith("zh")
            else f"Could not load retraining data:\n{message}"
        )
        self.retry_button.setVisible(True)
        self.stack.setCurrentWidget(self.loading_page)
        self.workspace_failed.emit(message)

    def refresh_workspace(self) -> None:
        if self._workspace is not None:
            self._workspace.refresh_workspace()

    def show_progress_page(self) -> None:
        if self._workspace is None:
            self._pending_progress_page = True
            return
        self._workspace.show_progress_page()

    def shutdown_workspace(self) -> None:
        self._closed = True
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
        self._dispose_workspace()
