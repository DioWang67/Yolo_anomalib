"""Read-only operator screen for cross-project model update jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QProcess, Qt, QTimer
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

from tools.process_liveness import is_process_active

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
    "invalid": ("紀錄損壞", "Invalid record"),
    "unknown": ("等待回報", "Awaiting status"),
}

STATE_COLORS = {
    "deployed": "#dff3e4",
    "failed": "#fde2e2",
    "cancelled": "#eeeeee",
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


def workflow_step_index(state: str, current_task: str = "") -> int:
    """Map a shared job state to the stable operator workflow step."""
    normalized_state = str(state or "unknown").strip().lower()
    normalized_task = str(current_task or "").strip().lower()
    if normalized_state in {"failed", "cancelled"}:
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
    jobs: list[ModelUpdateJob] = []
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        handoff, _handoff_valid = _read_json(job_dir / "handoff.json")
        status, status_valid = _read_json(job_dir / "status.json")
        target = _single_target(handoff)
        state = str(status.get("state") or "unknown")
        if not status_valid:
            state = "invalid"
        jobs.append(
            ModelUpdateJob(
                job_id=str(status.get("job_id") or handoff.get("job_id") or job_dir.name),
                state=state if state in STATE_LABELS else "unknown",
                product=str(status.get("product") or target.get("product") or ""),
                area=str(status.get("area") or target.get("area") or ""),
                created_at=_parse_datetime(status.get("created_at") or handoff.get("created_at")),
                updated_at=_parse_datetime(status.get("updated_at")),
                progress=_safe_int(status.get("progress"), minimum=0, maximum=100),
                ready_count=_safe_int(status.get("ready_count"), minimum=0),
                pending_count=_safe_int(status.get("pending_count"), minimum=0),
                message=str(status.get("message") or ""),
                current_task=str(status.get("current_task") or ""),
                error=str(status.get("error") or ""),
                handoff_path=(job_dir / "handoff.json").resolve(),
                status_path=(job_dir / "status.json").resolve(),
                training_process_id=_safe_int(status.get("training_process_id"), minimum=0),
                training_process_host=str(status.get("training_process_host") or ""),
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


class ModelUpdateStatusDialog(QDialog):
    """Display current and historical model update jobs for an operator."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        language: str = "zh_TW",
        selected_product: str | None = None,
        selected_area: str | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data_root = Path(data_root).expanduser().resolve()
        self.language = language
        self.initial_product = selected_product or ""
        self.initial_area = selected_area or ""
        self.jobs: list[ModelUpdateJob] = []
        self._resume_requested_job_ids: set[str] = set()
        self.setWindowTitle(self._text("產線模型補訓｜進度總覽", "Production Retraining | Progress"))
        self.resize(1350, 780)
        self._build_ui()
        self.refresh_jobs(restore_initial=True)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(3000)
        self.refresh_timer.timeout.connect(self.refresh_jobs)
        self.refresh_timer.start()

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel(
            self._text(
                "從補標到部署都在同一個流程中；畫面會自動更新，不會中斷檢測或補訓。",
                "Annotation through deployment stays in one workflow and refreshes automatically.",
            )
        )
        title.setStyleSheet(
            "QLabel { background: #243447; color: white; padding: 12px; font-size: 13pt; font-weight: bold; }"
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
        header.setSectionResizeMode(8, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_details)
        layout.addWidget(self.table, 1)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("QLabel { background: #eef2f6; border: 1px solid #c8d1dc; padding: 8px; }")
        layout.addWidget(self.details_label)

        footer = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-weight: bold;")
        footer.addWidget(self.summary_label)
        footer.addStretch()
        self.resume_button = QPushButton(self._text("繼續這筆補訓", "Continue this retraining job"))
        self.resume_button.setEnabled(False)
        self.resume_button.clicked.connect(self._resume_selected_job)
        footer.addWidget(self.resume_button)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        for combo in (self.product_filter, self.area_filter, self.state_filter):
            combo.currentIndexChanged.connect(self._apply_filters)

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
        current = (
            (self.initial_product, self.initial_area, "")
            if restore_initial
            else (
                self.product_filter.currentData() or "",
                self.area_filter.currentData() or "",
                self.state_filter.currentData() or "",
            )
        )
        self.jobs = load_model_update_jobs(self.data_root)
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
        self._apply_filters()

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

    def _apply_filters(self, _index=None) -> None:
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
        self.table.setRowCount(0)
        for job in visible:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                self._state_label(job.state),
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
                f"共 {len(visible)} 筆；進行中 {active} 筆；累積案例 {collecting} 筆",
                f"{len(visible)} jobs; {active} active; {collecting} collecting cases",
            )
        )
        if visible:
            self.table.selectRow(0)
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
            self._clear_workflow_panel()
            self.details_label.setText(self._text("尚無模型更新紀錄。", "No model update jobs."))
            return
        self._render_workflow(job)
        can_resume = (
            job.state in {"queued", "waiting_annotation", "failed"} and job.job_id not in self._resume_requested_job_ids
        )
        process_active = operator_process_is_active(job) if can_resume else False
        self.resume_button.setEnabled(can_resume and not process_active)
        self.resume_button.setText(
            self._text(
                "修正後重新嘗試" if job.state == "failed" else "繼續這筆補訓",
                "Retry after correction" if job.state == "failed" else "Continue this retraining job",
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
        detail = job.message or self._state_label(job.state)
        if job.error:
            detail += f"\n{self._text('失敗原因', 'Error')}: {job.error}"
        self.details_label.setText(detail)

    def _resume_selected_job(self) -> None:
        job = self._selected_job()
        if job is None or job.state not in {"queued", "waiting_annotation", "failed"}:
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
        result = QProcess.startDetached(
            "cmd.exe",
            ["/c", str(launcher), str(job.handoff_path)],
            str(training_root),
        )
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
        self.details_label.setText(self._text("正在重新開啟這筆補訓…", "Reopening this retraining job…"))

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
    return is_process_active(
        job.training_process_id,
        job.training_process_host,
    )
