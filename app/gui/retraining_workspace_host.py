"""Non-blocking host for the embedded retraining workspace."""

from __future__ import annotations

import csv
import logging
import sqlite3
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

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
    ) -> None:
        super().__init__()
        self._generation = generation
        self._result_root = result_root
        self._manifest_path = manifest_path
        self._product = product
        self._area = area

    def run(self) -> None:
        try:
            from app.gui.review_cases_dialog import prepare_review_manifest

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

    def __init__(
        self,
        *,
        result_root: str | Path,
        manifest_path: str | Path,
        training_data_dir: str | Path,
        language: str,
        product: str | None,
        area: str | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.result_root = Path(result_root)
        self.manifest_path = Path(manifest_path)
        self.training_data_dir = Path(training_data_dir)
        self.language = language
        self.product = product
        self.area = area
        self._generation = 0
        self._closed = False
        self._pending_progress_page = False
        self._worker: ReviewManifestPreloadWorker | None = None
        self._workspace = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget(self)
        layout.addWidget(self.stack)
        self.loading_page = self._build_loading_page()
        self.stack.addWidget(self.loading_page)
        self.stack.setCurrentWidget(self.loading_page)
        self._start_loading()

    @property
    def workspace(self):
        """Return the loaded ReviewCasesDialog, or ``None`` while loading."""
        return self._workspace

    def _build_loading_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(36, 36, 36, 36)
        navigation = QHBoxLayout()
        back_button = QPushButton(
            "← 返回檢測主畫面"
            if self.language.lower().startswith("zh")
            else "← Back to inspection"
        )
        back_button.clicked.connect(self.back_to_inspection_requested.emit)
        navigation.addWidget(back_button)
        navigation.addStretch()
        layout.addLayout(navigation)
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
        self._generation += 1
        generation = self._generation
        self.retry_button.setVisible(False)
        self.status_label.setText(
            "正在背景整理歷史檢測資料…\n你可以立即返回檢測，整理不會卡住畫面。"
            if self.language.lower().startswith("zh")
            else "Preparing inspection history in the background…\n"
            "You may return to inspection without blocking the window."
        )
        worker = ReviewManifestPreloadWorker(
            generation=generation,
            result_root=self.result_root,
            manifest_path=self.manifest_path,
            product=self.product,
            area=self.area,
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
        if generation == self._generation and self._worker is worker:
            self._worker = None

    def _on_manifest_ready(self, generation: int, rows: object) -> None:
        if self._closed or generation != self._generation:
            return
        prepared_rows = [dict(row) for row in rows] if isinstance(rows, list) else []
        try:
            from app.gui.review_cases_dialog import ReviewCasesDialog

            workspace = ReviewCasesDialog(
                result_root=self.result_root,
                manifest_path=self.manifest_path,
                training_data_dir=self.training_data_dir,
                language=self.language,
                product=self.product,
                area=self.area,
                start_in_overview=True,
                embedded=True,
                manifest_prepared=True,
                prepared_rows=prepared_rows,
                parent=self.stack,
            )
        except (OSError, RuntimeError, ValueError, csv.Error, sqlite3.Error) as exc:
            logger.exception("Could not construct the prepared review workspace")
            self._show_failure(str(exc))
            return
        workspace.back_to_inspection_requested.connect(
            self.back_to_inspection_requested.emit
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
        if self._workspace is not None:
            self._workspace.shutdown_workspace()
