"""Operator status, reconnect, and safe-stop screen for model update jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QProcess, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from tools.operator_job_control import (
    OperatorJobControlError,
    request_operator_job_cancel,
)
from tools.process_liveness import heartbeat_is_stale, is_process_active
from tools.record_visibility import hide_record, load_hidden_record_ids

STATE_LABELS = {
    "queued": ("等待處理", "Queued"),
    "waiting_feedback": ("累積改善案例", "Collecting corrective cases"),
    "waiting_annotation": ("等待補標", "Waiting for annotation"),
    "preparing_dataset": ("準備資料", "Preparing dataset"),
    "training": ("模型訓練", "Training"),
    "evaluating": ("品質驗證", "Evaluating"),
    "deploying": ("部署模型", "Deploying"),
    "deployed": ("部署完成", "Deployed"),
    "failed": ("失敗", "Failed"),
    "cancelled": ("已停止", "Cancelled"),
    "cancelling": ("安全停止中", "Stopping safely"),
    "unresponsive": ("心跳中斷", "Heartbeat lost"),
    "invalid": ("紀錄損壞", "Invalid record"),
    "unknown": ("等待回報", "Awaiting status"),
}

STATE_COLORS = {
    "deployed": "#dff3e4",
    "failed": "#fde2e2",
    "cancelled": "#eeeeee",
    "cancelling": "#fff3cd",
    "unresponsive": "#fde2e2",
    "waiting_annotation": "#fff3cd",
    "waiting_feedback": "#fff3cd",
    "training": "#dceeff",
    "evaluating": "#e8e1f5",
    "deploying": "#dff0f0",
    "invalid": "#fde2e2",
}

OPERATOR_WORKFLOW_STEPS = (
    ("接收資料", "Receive data"),
    ("補齊標註", "Complete labels"),
    ("補訓模型", "Retrain model"),
    ("品質驗證", "Quality check"),
    ("安全部署", "Safe deployment"),
)

STATE_STEP_INDEX = {
    "queued": 0,
    "waiting_feedback": 0,
    "waiting_annotation": 1,
    "preparing_dataset": 2,
    "training": 2,
    "evaluating": 3,
    "deploying": 4,
    "deployed": 4,
    "cancelling": 2,
    "unresponsive": 2,
}

TASK_STEP_INDEX = {
    "dataset_lint": 2,
    "dataset_readiness": 2,
    "dataset_splitter": 2,
    "yolo_augmentation": 2,
    "yolo_train": 2,
    "yolo_evaluation": 3,
    "deploy": 4,
}

JOB_ROLE = Qt.UserRole
JOB_PAGE_SIZE = 100
NORMAL_REFRESH_INTERVAL_MS = 3000
LARGE_HISTORY_REFRESH_INTERVAL_MS = 15000
CLEARABLE_JOB_STATES = frozenset(
    {"deployed", "failed", "cancelled", "invalid", "waiting_feedback"}
)
PROCESS_BOUND_JOB_STATES = frozenset(
    {
        "queued",
        "waiting_annotation",
        "preparing_dataset",
        "training",
        "evaluating",
        "deploying",
        "cancelling",
        "unresponsive",
    }
)
POSITION_MODE_LABELS = {
    "auto": ("自動判斷", "Auto"),
    "yolo_only": ("僅 YOLO", "YOLO only"),
    "calibrate_validate": (
        "YOLO＋位置校正與驗證",
        "YOLO + position calibration/validation",
    ),
}
POSITION_ACTIVATION_LABELS = {
    "preserve": ("保留現場狀態", "Preserve station state"),
    "enable_after_gate": ("驗證通過後啟用", "Enable after gate"),
}


@dataclass(frozen=True)
class ModelUpdateJob:
    """One shared model update status record."""

    job_id: str
    state: str
    product: str
    area: str
    created_at: datetime | None
    updated_at: datetime | None
    progress: int
    ready_count: int
    pending_count: int
    message: str
    current_task: str
    error: str
    handoff_path: Path
    status_path: Path
    training_process_id: int
    training_process_host: str
    heartbeat_at: datetime | None
    cancel_request_pending: bool
    batch_version: str = ""
    epochs: int = 0
    augmentations_per_image: int = 0
    batch: int = 0
    imgsz: int = 0
    position_training_mode: str = "auto"
    position_activation: str = "preserve"


def workflow_step_index(state: str, current_task: str = "") -> int:
    """Map a shared job state to the stable operator workflow step."""
    normalized_state = str(state or "unknown").strip().lower()
    normalized_task = str(current_task or "").strip().lower()
    if normalized_state in {"failed", "cancelled", "cancelling", "unresponsive"}:
        return TASK_STEP_INDEX.get(normalized_task, 0)
    return STATE_STEP_INDEX.get(normalized_state, 0)


def load_model_update_jobs(data_root: str | Path) -> list[ModelUpdateJob]:
    """Load all job statuses without modifying training data.

    Args:
        data_root: Yolo11_auto_train ``data`` directory.

    Returns:
        Newest-first model update records. Corrupt records remain visible with
        the ``invalid`` state instead of disappearing silently.
    """
    root = Path(data_root).expanduser().resolve()
    jobs_root = root / ".operator_handoff" / "jobs"
    if not jobs_root.is_dir():
        return []
    hidden_job_ids = load_hidden_record_ids(root, "model_update_jobs")
    jobs: list[ModelUpdateJob] = []
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        if (
            (job_dir / "workspace.json").is_file()
            and not (job_dir / "handoff.json").is_file()
        ):
            # Draft folders belong to the selection/review workspace, not the
            # training-progress list yet.
            continue
        handoff, _handoff_valid = _read_json(job_dir / "handoff.json")
        status, status_valid = _read_json(job_dir / "status.json")
        job_id = str(status.get("job_id") or handoff.get("job_id") or job_dir.name)
        if job_id in hidden_job_ids:
            continue
        target = _single_target(handoff)
        state = str(status.get("state") or "unknown")
        if not status_valid:
            state = "invalid"
        training_process_id = _safe_int(
            status.get("training_process_id"), minimum=0
        )
        training_process_host = str(status.get("training_process_host") or "")
        heartbeat_at = _parse_datetime(status.get("heartbeat_at"))
        message = str(status.get("message") or "")
        error = str(status.get("error") or "")
        control, _control_valid = _read_json(job_dir / "control.json")
        control_request_id = str(control.get("request_id") or "").strip()
        cancel_request_pending = bool(
            control_request_id
            and str(control.get("job_id") or "") == job_id
            and str(control.get("action") or "").strip().lower() == "cancel"
            and control_request_id
            != str(status.get("handled_control_request_id") or "").strip()
        )
        if (
            state in PROCESS_BOUND_JOB_STATES
            and training_process_id > 0
            and not is_process_active(training_process_id, training_process_host)
        ):
            if state == "cancelling" or cancel_request_pending:
                state = "cancelled"
                message = "補訓程序已停止"
            else:
                state = "failed"
                message = "補訓程序已中斷，可從此工作重新嘗試"
                if not error:
                    error = "Recorded training process is no longer active."
        elif state in PROCESS_BOUND_JOB_STATES and heartbeat_is_stale(
            status.get("heartbeat_at"),
            status.get("heartbeat_timeout_seconds"),
        ):
            state = "unresponsive"
            message = "補訓心跳已中斷；可要求安全停止，但不會冒險啟動第二個任務"
        elif cancel_request_pending and state in PROCESS_BOUND_JOB_STATES:
            state = "cancelling"
            message = "安全停止要求已送出，等待訓練程序回應"
        training_options = handoff.get("training_options")
        if not isinstance(training_options, dict):
            training_options = {}
        jobs.append(
            ModelUpdateJob(
                job_id=job_id,
                state=state if state in STATE_LABELS else "unknown",
                product=str(status.get("product") or target.get("product") or ""),
                area=str(status.get("area") or target.get("area") or ""),
                created_at=_parse_datetime(status.get("created_at") or handoff.get("created_at")),
                updated_at=_parse_datetime(status.get("updated_at")),
                progress=_safe_int(status.get("progress"), minimum=0, maximum=100),
                ready_count=_safe_int(status.get("ready_count"), minimum=0),
                pending_count=_safe_int(status.get("pending_count"), minimum=0),
                message=message,
                current_task=str(status.get("current_task") or ""),
                error=error,
                handoff_path=(job_dir / "handoff.json").resolve(),
                status_path=(job_dir / "status.json").resolve(),
                training_process_id=training_process_id,
                training_process_host=training_process_host,
                heartbeat_at=heartbeat_at,
                cancel_request_pending=cancel_request_pending,
                batch_version=str(handoff.get("batch_version") or ""),
                epochs=_safe_int(training_options.get("epochs"), minimum=0),
                augmentations_per_image=_safe_int(
                    training_options.get("augmentations_per_image"), minimum=0
                ),
                batch=_safe_int(training_options.get("batch"), minimum=0),
                imgsz=_safe_int(training_options.get("imgsz"), minimum=0),
                position_training_mode=str(
                    training_options.get("position_training_mode") or "auto"
                ),
                position_activation=str(
                    training_options.get("position_activation") or "preserve"
                ),
            )
        )
    return sorted(
        jobs,
        key=lambda item: (
            item.created_at.timestamp() if item.created_at else 0.0,
            item.job_id,
        ),
        reverse=True,
    )


_ACTIVE_JOB_LOAD_WORKERS: set[QThread] = set()


class ModelUpdateJobLoadWorker(QThread):
    """Load status files outside the Qt main thread."""

    loaded = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, *, generation: int, data_root: Path) -> None:
        super().__init__()
        self.generation = generation
        self.data_root = data_root

    def run(self) -> None:
        try:
            jobs = load_model_update_jobs(self.data_root)
        except (OSError, RuntimeError, ValueError) as exc:
            self.failed.emit(self.generation, str(exc))
            return
        if not self.isInterruptionRequested():
            self.loaded.emit(self.generation, jobs)


def _retain_job_load_worker(worker: QThread) -> None:
    _ACTIVE_JOB_LOAD_WORKERS.add(worker)

    def release() -> None:
        _ACTIVE_JOB_LOAD_WORKERS.discard(worker)
        worker.deleteLater()

    worker.finished.connect(release)


class ModelUpdateStatusDialog(QDialog):
    """Display current and historical model update jobs for an operator."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        language: str = "zh_TW",
        selected_product: str | None = None,
        selected_area: str | None = None,
        background_refresh: bool = False,
        embedded: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data_root = Path(data_root).expanduser().resolve()
        self.language = language
        self.initial_product = selected_product or ""
        self.initial_area = selected_area or ""
        self.jobs: list[ModelUpdateJob] = []
        self._resume_requested_job_ids: set[str] = set()
        self._cancel_requested_job_ids: set[str] = set()
        self._background_refresh = bool(background_refresh)
        self._embedded = bool(embedded)
        self._load_generation = 0
        self._load_worker: ModelUpdateJobLoadWorker | None = None
        self._refresh_pending = False
        self._restore_initial_pending = True
        self._closed = False
        self._render_limit = JOB_PAGE_SIZE
        self.setWindowTitle(self._text("產線模型補訓｜進度總覽", "Production Retraining | Progress"))
        if self._embedded:
            self.setWindowFlags(Qt.Widget)
        else:
            configure_responsive_dialog(
                self,
                preferred=(1350, 780),
                minimum=(780, 520),
                parent=parent,
            )
        self._build_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(NORMAL_REFRESH_INTERVAL_MS)
        self.refresh_timer.timeout.connect(self.refresh_jobs)
        self.refresh_timer.start()
        self.refresh_jobs(restore_initial=True)

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            8 if self._embedded else 11,
            6 if self._embedded else 11,
            8 if self._embedded else 11,
            8 if self._embedded else 11,
        )
        layout.setSpacing(6 if self._embedded else 8)
        if not self._embedded:
            title = QLabel(
                self._text(
                    "從補標到部署都在同一個流程中；畫面會自動更新，不會中斷檢測或補訓。",
                    "Annotation through deployment stays in one workflow and refreshes automatically.",
                )
            )
            title.setStyleSheet(
                "QLabel { background: #243447; color: white; padding: 12px; "
                "font-size: 13pt; font-weight: bold; }"
            )
            layout.addWidget(title)
        layout.addWidget(self._build_workflow_panel())

        filters = QHBoxLayout()
        filters.addWidget(QLabel(self._text("產品", "Product")))
        self.product_filter = QComboBox()
        filters.addWidget(self.product_filter)
        filters.addWidget(QLabel(self._text("工位", "Station")))
        self.area_filter = QComboBox()
        filters.addWidget(self.area_filter)
        filters.addWidget(QLabel(self._text("狀態", "State")))
        self.state_filter = QComboBox()
        filters.addWidget(self.state_filter)
        filters.addStretch()
        refresh_button = QPushButton(self._text("重新整理", "Refresh"))
        refresh_button.clicked.connect(self.refresh_jobs)
        filters.addWidget(refresh_button)
        layout.addLayout(filters)

        headers = [
            self._text("狀態", "State"),
            self._text("批次版本", "Batch version"),
            self._text("產品", "Product"),
            self._text("工位", "Station"),
            self._text("送出時間", "Submitted"),
            self._text("最後更新", "Updated"),
            self._text("進度", "Progress"),
            self._text("可訓練", "Ready"),
            self._text("待補標", "Pending"),
            self._text("目前說明", "Message"),
            self._text("任務編號", "Job ID"),
        ]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_details)
        if self._embedded:
            self.table.setMinimumHeight(120)
        layout.addWidget(self.table, 1)
        self.load_more_button = QPushButton(
            self._text("載入更多歷史任務", "Load more history")
        )
        self.load_more_button.setVisible(False)
        self.load_more_button.clicked.connect(self._load_more_jobs)
        layout.addWidget(self.load_more_button)

        details_row = QHBoxLayout()
        details_row.setSpacing(10)
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("QLabel { background: #eef2f6; border: 1px solid #c8d1dc; padding: 8px; }")
        details_row.addWidget(self.details_label, 1)
        self.resume_button = QPushButton(
            self._text("繼續這筆補訓", "Continue this retraining job")
        )
        self.resume_button.setObjectName("resumeOperatorJobButton")
        self.resume_button.setMinimumHeight(44)
        self.resume_button.setEnabled(False)
        self.resume_button.setStyleSheet(
            "QPushButton#resumeOperatorJobButton {"
            "background:#237a3b;color:white;border:0;border-radius:6px;"
            "padding:8px 18px;font-weight:bold;}"
            "QPushButton#resumeOperatorJobButton:disabled {"
            "background:#aab2bd;color:#eef1f4;}"
        )
        self.resume_button.clicked.connect(self._resume_selected_job)
        details_row.addWidget(self.resume_button)
        layout.addLayout(details_row)

        footer = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-weight: bold;")
        footer.addWidget(self.summary_label)
        footer.addStretch()
        self.clear_record_button = QPushButton(
            self._text("清除選取紀錄", "Clear selected record")
        )
        self.clear_record_button.setEnabled(False)
        self.clear_record_button.setStyleSheet(
            "QPushButton { color:#a61b1b;background:#fff5f5;border:1px solid #d96c6c;"
            "border-radius:4px;padding:6px 10px;font-weight:bold; }"
            "QPushButton:disabled { color:#9aa0a6;background:#f3f4f6;border-color:#d1d5db; }"
        )
        self.clear_record_button.clicked.connect(self._clear_selected_record)
        self.cancel_button = QPushButton(
            self._text("安全停止補訓", "Stop retraining safely")
        )
        self.cancel_button.setEnabled(False)
        self.cancel_button.setStyleSheet(
            "QPushButton { color:#8a4b00;background:#fff7e6;border:1px solid #d6a756;"
            "border-radius:4px;padding:6px 10px;font-weight:bold; }"
            "QPushButton:disabled { color:#9aa0a6;background:#f3f4f6;border-color:#d1d5db; }"
        )
        self.cancel_button.clicked.connect(self._cancel_selected_job)
        footer.addWidget(self.clear_record_button)
        footer.addWidget(self.cancel_button)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        for combo in (self.product_filter, self.area_filter, self.state_filter):
            combo.currentIndexChanged.connect(self._on_filter_changed)

    def _build_workflow_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("OperatorWorkflowPanel")
        panel.setStyleSheet(
            "QFrame#OperatorWorkflowPanel { background: #111820; border: 1px solid #2f3b49; border-radius: 8px; }"
        )
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 14, 18, 14)
        panel_layout.setSpacing(10)

        header = QHBoxLayout()
        self.workflow_state_label = QLabel()
        self.workflow_state_label.setStyleSheet("color: #f0f6fc; font-size: 14px; font-weight: bold;")
        self.workflow_target_label = QLabel()
        self.workflow_target_label.setStyleSheet("color: #58a6ff; font-size: 12px; font-weight: bold;")
        header.addWidget(self.workflow_state_label)
        header.addStretch()
        header.addWidget(self.workflow_target_label)
        panel_layout.addLayout(header)

        steps = QHBoxLayout()
        steps.setSpacing(7)
        self.workflow_step_labels: list[QLabel] = []
        for number, names in enumerate(OPERATOR_WORKFLOW_STEPS, start=1):
            label = QLabel(f"{number}  {self._text(*names)}")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(32)
            steps.addWidget(label, 1)
            self.workflow_step_labels.append(label)
        panel_layout.addLayout(steps)

        self.workflow_progress = QProgressBar()
        self.workflow_progress.setRange(0, 100)
        self.workflow_progress.setFormat(self._text("模型更新 %p%", "Model update %p%"))
        self.workflow_progress.setStyleSheet(
            "QProgressBar { background: #202b36; border: 0; border-radius: 5px; "
            "color: white; text-align: center; min-height: 18px; }"
            "QProgressBar::chunk { background: #2f81f7; border-radius: 5px; }"
        )
        panel_layout.addWidget(self.workflow_progress)
        self._clear_workflow_panel()
        return panel

    def refresh_jobs(self, _checked=False, *, restore_initial: bool = False) -> None:
        """Refresh statuses while preserving the operator's current filters."""
        if self._background_refresh:
            self._restore_initial_pending = (
                self._restore_initial_pending or restore_initial
            )
            if self._load_worker is not None and self._load_worker.isRunning():
                self._refresh_pending = True
                return
            self._load_generation += 1
            worker = ModelUpdateJobLoadWorker(
                generation=self._load_generation,
                data_root=self.data_root,
            )
            self._load_worker = worker
            self._refresh_pending = False
            worker.loaded.connect(self._on_jobs_loaded)
            worker.failed.connect(self._on_jobs_load_failed)
            worker.finished.connect(
                lambda current=worker: self._on_job_load_finished(current)
            )
            _retain_job_load_worker(worker)
            if not self.jobs:
                self.summary_label.setText(
                    self._text("正在背景載入任務…", "Loading jobs in background…")
                )
            worker.start()
            return

        jobs = load_model_update_jobs(self.data_root)
        self._apply_loaded_jobs(jobs, restore_initial=restore_initial)

    def _on_jobs_loaded(self, generation: int, jobs: object) -> None:
        if self._closed or generation != self._load_generation:
            return
        loaded_jobs = jobs if isinstance(jobs, list) else []
        restore_initial = self._restore_initial_pending
        self._restore_initial_pending = False
        self._apply_loaded_jobs(loaded_jobs, restore_initial=restore_initial)

    def _on_jobs_load_failed(self, generation: int, message: str) -> None:
        if self._closed or generation != self._load_generation:
            return
        self.summary_label.setText(
            self._text(
                f"任務載入失敗：{message}",
                f"Could not load jobs: {message}",
            )
        )

    def _on_job_load_finished(self, worker: ModelUpdateJobLoadWorker) -> None:
        if self._load_worker is worker:
            self._load_worker = None
        if self._closed or not self._refresh_pending:
            return
        self._refresh_pending = False
        QTimer.singleShot(0, self.refresh_jobs)

    def _apply_loaded_jobs(
        self,
        jobs: list[ModelUpdateJob],
        *,
        restore_initial: bool,
    ) -> None:
        selected_job = self._selected_job()
        selected_job_id = selected_job.job_id if selected_job is not None else ""
        current = (
            (self.initial_product, self.initial_area, "")
            if restore_initial
            else (
                self.product_filter.currentData() or "",
                self.area_filter.currentData() or "",
                self.state_filter.currentData() or "",
            )
        )
        self.jobs = jobs
        self.refresh_timer.setInterval(
            LARGE_HISTORY_REFRESH_INTERVAL_MS
            if len(self.jobs) > 200
            else NORMAL_REFRESH_INTERVAL_MS
        )
        self._set_filter(
            self.product_filter,
            sorted({job.product for job in self.jobs if job.product}),
            current[0],
        )
        self._set_filter(
            self.area_filter,
            sorted({job.area for job in self.jobs if job.area}),
            current[1],
        )
        self._set_filter(
            self.state_filter,
            sorted({job.state for job in self.jobs}),
            current[2],
            state_labels=True,
        )
        self._apply_filters(preferred_job_id=selected_job_id)

    def _set_filter(
        self,
        combo: QComboBox,
        values: list[str],
        selected: str,
        *,
        state_labels: bool = False,
    ) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._text("全部", "All"), "")
            for value in values:
                label = self._state_label(value) if state_labels else value
                combo.addItem(label, value)
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _on_filter_changed(self, _index=None) -> None:
        self._render_limit = JOB_PAGE_SIZE
        self._apply_filters()

    def _load_more_jobs(self) -> None:
        selected = self._selected_job()
        selected_job_id = selected.job_id if selected is not None else ""
        self._render_limit += JOB_PAGE_SIZE
        self._apply_filters(preferred_job_id=selected_job_id)

    def _apply_filters(self, _index=None, *, preferred_job_id: str = "") -> None:
        product = str(self.product_filter.currentData() or "")
        area = str(self.area_filter.currentData() or "")
        state = str(self.state_filter.currentData() or "")
        visible = [
            job
            for job in self.jobs
            if (not product or job.product == product)
            and (not area or job.area == area)
            and (not state or job.state == state)
        ]
        preferred_index = next(
            (
                index
                for index, job in enumerate(visible)
                if job.job_id == preferred_job_id
            ),
            -1,
        )
        if preferred_index >= self._render_limit:
            self._render_limit = (
                (preferred_index // JOB_PAGE_SIZE) + 1
            ) * JOB_PAGE_SIZE
        rendered = visible[: self._render_limit]
        self.table.setRowCount(0)
        for job in rendered:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                self._state_label(job.state),
                job.batch_version or self._text("舊任務（未命名）", "Legacy (unnamed)"),
                job.product or "—",
                job.area or "—",
                _format_datetime(job.created_at),
                _format_datetime(job.updated_at),
                f"{job.progress}%",
                str(job.ready_count),
                str(job.pending_count),
                job.message or "—",
                job.job_id,
            ]
            color = QColor(STATE_COLORS.get(job.state, "#ffffff"))
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(JOB_ROLE, job)
                item.setBackground(color)
                self.table.setItem(row, column, item)
        active = sum(job.state not in {"deployed", "failed", "cancelled", "waiting_feedback"} for job in visible)
        collecting = sum(job.state == "waiting_feedback" for job in visible)
        self.summary_label.setText(
            self._text(
                f"共 {len(visible)} 筆；顯示 {len(rendered)} 筆；"
                f"進行中 {active} 筆；累積案例 {collecting} 筆",
                f"{len(visible)} jobs; showing {len(rendered)}; "
                f"{active} active; {collecting} collecting cases",
            )
        )
        remaining = len(visible) - len(rendered)
        self.load_more_button.setVisible(remaining > 0)
        self.load_more_button.setText(
            self._text(
                f"再載入 {min(JOB_PAGE_SIZE, remaining)} 筆（尚有 {remaining} 筆）",
                f"Load {min(JOB_PAGE_SIZE, remaining)} more ({remaining} remaining)",
            )
        )
        if rendered:
            selected_row = next(
                (
                    index
                    for index, job in enumerate(rendered)
                    if job.job_id == preferred_job_id
                ),
                0,
            )
            self.table.selectRow(selected_row)
        else:
            self._update_details()

    def _selected_job(self) -> ModelUpdateJob | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None
        item = self.table.item(selected[0].row(), 0)
        value = item.data(JOB_ROLE) if item else None
        return value if isinstance(value, ModelUpdateJob) else None

    def _update_details(self) -> None:
        job = self._selected_job()
        if job is None:
            self.resume_button.setEnabled(False)
            self.cancel_button.setEnabled(False)
            self.clear_record_button.setEnabled(False)
            self._clear_workflow_panel()
            self.details_label.setText(self._text("尚無模型更新紀錄。", "No model update jobs."))
            return
        self._render_workflow(job)
        can_resume = (
            job.state in {"queued", "waiting_annotation", "failed", "cancelled"}
            and job.job_id not in self._resume_requested_job_ids
        )
        process_active = operator_process_is_active(job) if can_resume else False
        self.resume_button.setEnabled(can_resume and not process_active)
        cancel_requested = (
            job.cancel_request_pending
            or job.job_id in self._cancel_requested_job_ids
        )
        can_cancel = job.state in PROCESS_BOUND_JOB_STATES and not cancel_requested
        self.cancel_button.setEnabled(can_cancel)
        self.cancel_button.setText(
            self._text("已要求安全停止", "Safe stop requested")
            if cancel_requested
            else self._text("安全停止補訓", "Stop retraining safely")
        )
        self.cancel_button.setToolTip(
            self._text(
                "要求訓練端在安全點停止；不會直接強制終止程序。",
                "Requests a cooperative stop at a safe point; the process is not force-killed.",
            )
        )
        can_clear = job.state in CLEARABLE_JOB_STATES
        self.clear_record_button.setEnabled(can_clear)
        self.clear_record_button.setToolTip(
            self._text(
                "只從操作介面清除工作紀錄，不刪除模型、主訓練集、圖片或標註。"
                if can_clear
                else "進行中的補訓紀錄不可清除。",
                "Removes only the operator-facing job record; models, main datasets, images, and labels remain."
                if can_clear
                else "Active retraining records cannot be cleared.",
            )
        )
        self.resume_button.setText(
            self._text(
                (
                    "開啟／繼續補標"
                    if job.pending_count > 0
                    else (
                        "修正後重新嘗試"
                        if job.state == "failed"
                        else (
                            "從中斷處繼續補訓"
                            if job.state == "cancelled"
                            else "繼續這筆補訓"
                        )
                    )
                ),
                (
                    "Open / continue annotation"
                    if job.pending_count > 0
                    else (
                        "Retry after correction"
                        if job.state == "failed"
                        else (
                            "Resume retraining from checkpoint"
                            if job.state == "cancelled"
                            else "Continue this retraining job"
                        )
                    )
                ),
            )
        )
        resume_requested = job.job_id in self._resume_requested_job_ids
        self.resume_button.setToolTip(
            self._text(
                "正在重新開啟補訓視窗"
                if resume_requested
                else ("補訓視窗仍在執行中" if process_active else "重新開啟這筆既有工作"),
                "Reopening the retraining window"
                if resume_requested
                else ("The retraining window is still running" if process_active else "Reopen this existing job"),
            )
        )
        version = job.batch_version or self._text(
            "舊任務（未命名）",
            "Legacy (unnamed)",
        )
        detail = self._text(
            f"補訓批次：{version}\n",
            f"Retraining batch: {version}\n",
        )
        detail += job.message or self._state_label(job.state)
        if job.epochs and job.batch and job.imgsz:
            detail += self._text(
                f"\n訓練設定：{job.epochs} Epochs／增強 {job.augmentations_per_image}／"
                f"Batch {job.batch}／{job.imgsz}px",
                f"\nTraining options: {job.epochs} epochs / "
                f"{job.augmentations_per_image} augmentations / batch {job.batch} / "
                f"{job.imgsz}px",
            )
        position_mode_labels = POSITION_MODE_LABELS.get(
            job.position_training_mode,
            (job.position_training_mode, job.position_training_mode),
        )
        activation_labels = POSITION_ACTIVATION_LABELS.get(
            job.position_activation,
            (job.position_activation, job.position_activation),
        )
        detail += self._text(
            f"\n位置設定：{position_mode_labels[0]}／{activation_labels[0]}",
            f"\nPosition: {position_mode_labels[1]} / {activation_labels[1]}",
        )
        if job.error:
            detail += f"\n{self._text('失敗原因', 'Error')}: {job.error}"
        if job.heartbeat_at is not None:
            detail += self._text(
                f"\n最後心跳：{_format_datetime(job.heartbeat_at)}",
                f"\nLast heartbeat: {_format_datetime(job.heartbeat_at)}",
            )
        self.details_label.setText(detail)

    def _cancel_selected_job(self) -> None:
        """Publish an idempotent cooperative stop request for the selected job."""
        job = self._selected_job()
        if job is None or job.state not in PROCESS_BOUND_JOB_STATES:
            return
        answer = QMessageBox.question(
            self,
            self.windowTitle(),
            self._text(
                f"要安全停止這筆補訓嗎？\n\n{job.product} / {job.area}\n{job.job_id}\n\n"
                "程序會在安全點結束，不會直接強制關閉。",
                f"Stop this retraining job safely?\n\n{job.product} / {job.area}\n{job.job_id}\n\n"
                "The process will stop at a safe point and will not be force-killed.",
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            request_operator_job_cancel(job.status_path, job_id=job.job_id)
        except OperatorJobControlError as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self._cancel_requested_job_ids.add(job.job_id)
        self.refresh_jobs()

    def _clear_selected_record(self) -> None:
        """Hide one safe-to-clear job record without deleting training artifacts."""
        job = self._selected_job()
        if job is None or job.state not in CLEARABLE_JOB_STATES:
            return
        answer = QMessageBox.question(
            self,
            self.windowTitle(),
            self._text(
                f"確定清除這筆補訓紀錄？\n\n{job.product or '—'} / {job.area or '—'}\n"
                f"{job.job_id}\n\n模型、主訓練集、圖片與標註不會被刪除。",
                f"Clear this retraining record?\n\n{job.product or '—'} / {job.area or '—'}\n"
                f"{job.job_id}\n\nModels, main datasets, images, and labels will not be deleted.",
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            hide_record(self.data_root, "model_update_jobs", job.job_id)
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self.refresh_jobs()

    def _resume_selected_job(self) -> None:
        job = self._selected_job()
        if job is None or job.state not in {
            "queued",
            "waiting_annotation",
            "failed",
            "cancelled",
        }:
            return
        if operator_process_is_active(job):
            self.resume_button.setEnabled(False)
            return
        training_root = self.data_root.parent
        launcher = training_root / "open_operator_training.bat"
        if not launcher.is_file() or not job.handoff_path.is_file():
            QMessageBox.critical(
                self,
                self.windowTitle(),
                self._text(
                    "找不到補訓啟動器或工作資料，請通知工程人員。",
                    "The retraining launcher or job data is missing.",
                ),
            )
            return
        needs_annotation = job.pending_count > 0
        arguments = [
            "/c",
            str(launcher),
            str(job.handoff_path),
            "--background",
        ]
        result = QProcess.startDetached("cmd.exe", arguments, str(training_root))
        started = result[0] if isinstance(result, tuple) else bool(result)
        if not started:
            QMessageBox.critical(
                self,
                self.windowTitle(),
                self._text("補訓視窗啟動失敗。", "Failed to start retraining."),
            )
            return
        self._resume_requested_job_ids.add(job.job_id)
        self.resume_button.setEnabled(False)
        self.details_label.setText(
            self._text(
                (
                    "正在開啟補標工具；完成最後一張並按 Ctrl+S 後會接續補訓…"
                    if needs_annotation
                    else "正在背景重新啟動這筆補訓…"
                ),
                (
                    "Opening annotation; save the final image with Ctrl+S to continue training…"
                    if needs_annotation
                    else "Restarting this retraining job in the background…"
                ),
            )
        )

    def _clear_workflow_panel(self) -> None:
        self.workflow_state_label.setText(self._text("尚未選擇模型更新工作", "No model update selected"))
        self.workflow_target_label.setText("—")
        self.workflow_progress.setValue(0)
        for label in self.workflow_step_labels:
            label.setStyleSheet("background: #202b36; color: #8b98a5; border: 1px solid #303b46; border-radius: 6px;")

    def _render_workflow(self, job: ModelUpdateJob) -> None:
        active_step = workflow_step_index(job.state, job.current_task)
        self.workflow_state_label.setText(self._state_label(job.state))
        self.workflow_target_label.setText(f"{job.product or '—'} / {job.area or '—'}")
        self.workflow_progress.setValue(job.progress)
        is_success = job.state == "deployed"
        is_failure = job.state in {"failed", "invalid"}
        is_cancelled = job.state == "cancelled"
        for index, label in enumerate(self.workflow_step_labels):
            if index < active_step or (is_success and index <= active_step):
                style = (
                    "background: #1f6f3e; color: #d9fbe5; border: 1px solid #2ea44f; "
                    "border-radius: 6px; font-weight: 600;"
                )
            elif index == active_step and is_failure:
                style = (
                    "background: #7a2525; color: white; border: 1px solid #e5534b; "
                    "border-radius: 6px; font-weight: 700;"
                )
            elif index == active_step and is_cancelled:
                style = (
                    "background: #4b5563; color: white; border: 1px solid #9ca3af; "
                    "border-radius: 6px; font-weight: 700;"
                )
            elif index == active_step and not is_success:
                style = (
                    "background: #174b7a; color: #e6f2ff; border: 1px solid #58a6ff; "
                    "border-radius: 6px; font-weight: 700;"
                )
            else:
                style = "background: #202b36; color: #8b98a5; border: 1px solid #303b46; border-radius: 6px;"
            label.setStyleSheet(style)

    def _shutdown_refresh(self) -> None:
        self._closed = True
        self.refresh_timer.stop()
        if self._load_worker is not None and self._load_worker.isRunning():
            self._load_worker.requestInterruption()

    def accept(self) -> None:
        self._shutdown_refresh()
        super().accept()

    def reject(self) -> None:
        self._shutdown_refresh()
        super().reject()

    def closeEvent(self, event) -> None:
        self._shutdown_refresh()
        super().closeEvent(event)

    def _state_label(self, state: str) -> str:
        labels = STATE_LABELS.get(state, STATE_LABELS["unknown"])
        return self._text(*labels)


def _single_target(handoff: dict[str, Any]) -> dict[str, Any]:
    targets = handoff.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        return {}
    return dict(targets[0]) if isinstance(targets[0], dict) else {}


def _read_json(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.is_file():
        return {}, False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}, False
    return (dict(payload), True) if isinstance(payload, dict) else ({}, False)


def _safe_int(value: Any, *, minimum: int, maximum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return minimum
    parsed = max(parsed, minimum)
    return min(parsed, maximum) if maximum is not None else parsed


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone() if parsed.tzinfo else parsed.astimezone()


def _format_datetime(value: datetime | None) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S") if value else "—"


def operator_process_is_active(job: ModelUpdateJob) -> bool:
    """Return whether the recorded local training process is still active."""
    return bool(
        is_process_active(
            job.training_process_id,
            job.training_process_host,
        )
    )
