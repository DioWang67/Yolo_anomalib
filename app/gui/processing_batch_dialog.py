"""Phase 3 processing UI; contains no routing or execution business rules."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from app.gui.processing_summary_view_model import ProcessingSummaryViewModel

PROCESSING_PIPELINE_ENV = "YOLO_PROCESSING_PIPELINE"
PROCESSING_EXECUTION_FRAMEWORK_ENV = "YOLO_PROCESSING_EXECUTION_FRAMEWORK"
PROCESSING_DATASET_STEP_ENV = "YOLO_PROCESSING_DATASET_STEP"
PROCESSING_DATASET_DRY_RUN_ENV = "YOLO_PROCESSING_DATASET_DRY_RUN"
PROCESSING_ANNOTATION_STEP_ENV = "YOLO_PROCESSING_ANNOTATION_STEP"
PROCESSING_COLOR_STEP_ENV = "YOLO_PROCESSING_COLOR_STEP"


def processing_pipeline_enabled(environment: dict[str, str] | None = None) -> bool:
    """Return whether the opt-in Phase 3A dialog is enabled."""
    values = os.environ if environment is None else environment
    return str(values.get(PROCESSING_PIPELINE_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def processing_execution_framework_enabled(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether Phase 3B execution replaces the Phase 3A stub."""
    values = os.environ if environment is None else environment
    return str(
        values.get(PROCESSING_EXECUTION_FRAMEWORK_ENV) or ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def processing_dataset_step_enabled(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether Phase 3C1 replaces only the READY NoOp batch step."""
    values = os.environ if environment is None else environment
    return str(values.get(PROCESSING_DATASET_STEP_ENV) or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def processing_dataset_dry_run_enabled(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether Phase 3C1 validates a preview without committing files."""
    values = os.environ if environment is None else environment
    return str(values.get(PROCESSING_DATASET_DRY_RUN_ENV) or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def processing_annotation_step_enabled(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether Phase 3C2 replaces the annotation NoOp batch step."""
    values = os.environ if environment is None else environment
    return str(values.get(PROCESSING_ANNOTATION_STEP_ENV) or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def processing_color_step_enabled(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether Phase 3C3 replaces only the color NoOp step."""
    values = os.environ if environment is None else environment
    return str(values.get(PROCESSING_COLOR_STEP_ENV) or "").strip().lower() in {
        "1", "true", "yes", "on"
    }


class ProcessingBatchDialog(QDialog):
    """Render and start one view-model-owned processing plan."""

    def __init__(
        self,
        view_model: ProcessingSummaryViewModel,
        *,
        language: str = "zh_TW",
        cleanup_launcher: Callable[[], None] | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.view_model = view_model
        self.language = language
        self._cleanup_launcher = cleanup_launcher
        self.setWindowTitle(self._text("處理批次", "Processing Batch"))
        configure_responsive_dialog(
            self,
            preferred=(760, 680),
            minimum=(560, 460),
            parent=parent,
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        body_scroll = QScrollArea()
        body_scroll.setObjectName("ProcessingBodyScroll")
        body_scroll.setWidgetResizable(True)
        body_scroll.setFrameShape(QFrame.NoFrame)
        body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)
        body_scroll.setWidget(body)
        root_layout.addWidget(body_scroll, 1)

        title = QLabel(self._text("Processing Summary", "Processing Summary"))
        title.setStyleSheet("font-size:20pt;font-weight:bold;color:#20354a;")
        layout.addWidget(title)

        metadata = QLabel(
            self._text(
                f"共 {self.view_model.sample_count} 筆｜模式 {self.view_model.execution_mode}",
                f"{self.view_model.sample_count} samples | Mode {self.view_model.execution_mode}",
            )
        )
        if getattr(self.view_model, "class_fix_count", 0):
            metadata.setText(
                metadata.text()
                + self._text(
                    f"｜需補框 {self.view_model.annotation_fix_count}｜需修正類別 {self.view_model.class_fix_count}",
                    f" | Annotation {self.view_model.annotation_fix_count} | Class Fix {self.view_model.class_fix_count}",
                )
            )
        metadata.setStyleSheet("color:#5b6573;font-size:10.5pt;")
        layout.addWidget(metadata)

        execution_details = getattr(self.view_model, "execution_details", "")
        self.execution_details_label = QLabel(execution_details)
        self.execution_details_label.setWordWrap(True)
        self.execution_details_label.setStyleSheet("color:#5b6573;font-size:9.5pt;")
        self.execution_details_label.setVisible(bool(execution_details))
        layout.addWidget(self.execution_details_label)

        summary = QFrame()
        summary.setObjectName("ProcessingSummaryPanel")
        summary.setStyleSheet(
            "QFrame#ProcessingSummaryPanel { background:white;border:1px solid #d8dee6;"
            "border-radius:8px; }"
        )
        grid = QGridLayout(summary)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        self.metric_value_labels: dict[str, QLabel] = {}
        for position, metric in enumerate(self.view_model.metrics):
            row, column = divmod(position, 3)
            card = QFrame()
            card.setObjectName(f"Metric_{metric.key}")
            card.setStyleSheet(
                f"QFrame#Metric_{metric.key} {{ background:#f7f9fc;"
                "border:1px solid #e1e6ed;border-radius:7px; }"
            )
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            label = QLabel(metric.label)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet("color:#5b6573;border:0;")
            value = QLabel(str(metric.value))
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet("font-size:22pt;font-weight:bold;color:#20354a;border:0;")
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            grid.addWidget(card, row, column)
            self.metric_value_labels[metric.key] = value
        layout.addWidget(summary)

        warning_title = QLabel(self._text("Warnings", "Warnings"))
        warning_title.setStyleSheet("font-size:12pt;font-weight:bold;color:#344054;")
        layout.addWidget(warning_title)
        warning_text = "\n".join(f"• {warning}" for warning in self.view_model.warnings)
        if not warning_text:
            warning_text = self._text("無警告。", "No warnings.")
        self.warning_label = QLabel(warning_text)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet(
            "background:#fff8e6;color:#6f4b00;border:1px solid #efd28a;"
            "border-radius:6px;padding:10px;"
        )
        layout.addWidget(self.warning_label)

        self.start_button = QPushButton(self._text("Start Processing", "Start Processing"))
        self.start_button.setMinimumHeight(52)
        self.start_button.setStyleSheet(
            "QPushButton { background:#237a3b;color:white;border:0;border-radius:6px;"
            "font-size:13pt;font-weight:bold;padding:10px 18px; }"
            "QPushButton:hover { background:#1d6932; }"
        )
        self.start_button.clicked.connect(self._start_processing)
        layout.addWidget(self.start_button)

        self.execution_panel = self._build_execution_panel()
        self.execution_panel.setVisible(False)
        layout.addWidget(self.execution_panel)

        self.advanced_button = QPushButton(self._text("Advanced ▼", "Advanced ▼"))
        self.advanced_button.setCheckable(True)
        self.advanced_button.setStyleSheet(
            "QPushButton { text-align:left;background:transparent;color:#365f87;"
            "border:0;font-weight:bold;padding:6px; }"
        )
        self.advanced_button.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_button)

        self.advanced_panel = self._build_advanced_panel()
        self.advanced_panel.setVisible(False)
        layout.addWidget(self.advanced_panel)
        layout.addStretch()

        footer = QHBoxLayout()
        footer.addStretch()
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.reject)
        footer.addWidget(close_button)
        footer_widget = QWidget()
        footer_widget.setLayout(footer)
        footer_widget.setContentsMargins(12, 0, 12, 8)
        root_layout.addWidget(footer_widget)

    def _build_advanced_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("AdvancedPanel")
        panel.setStyleSheet(
            "QFrame#AdvancedPanel { background:#f7f9fc;border:1px solid #d8dee6;"
            "border-radius:7px; }"
        )
        buttons = QHBoxLayout(panel)
        buttons.setContentsMargins(12, 10, 12, 10)
        self.offline_button = QPushButton(
            self._text("Export Offline Package", "Export Offline Package")
        )
        self.history_button = QPushButton(self._text("History", "History"))
        self.diagnostics_button = QPushButton(
            self._text("Diagnostics", "Diagnostics")
        )
        self.historical_cleanup_button = QPushButton(
            self._text("歷史資料清理", "Historical Cleanup")
        )
        self.rollback_color_button = QPushButton(
            self._text("回復色彩設定版本", "Rollback Color Revision")
        )
        for button in (
            self.offline_button,
            self.history_button,
            self.diagnostics_button,
        ):
            button.clicked.connect(self._show_advanced_notice)
            buttons.addWidget(button)
        self.historical_cleanup_button.setVisible(
            self._cleanup_launcher is not None
            and bool(getattr(self.view_model, "has_blocking_items", False))
        )
        self.historical_cleanup_button.clicked.connect(self._open_historical_cleanup)
        buttons.addWidget(self.historical_cleanup_button)
        self.rollback_color_button.clicked.connect(self._rollback_color_revision)
        self.rollback_color_button.setVisible(False)
        buttons.addWidget(self.rollback_color_button)
        return panel

    def _open_historical_cleanup(self) -> None:
        if self._cleanup_launcher is not None:
            self._cleanup_launcher()

    def _build_execution_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ExecutionReportPanel")
        panel.setStyleSheet(
            "QFrame#ExecutionReportPanel { background:#f7f9fc;"
            "border:1px solid #c8d2df;border-radius:7px; }"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        self.report_status_label = QLabel()
        self.report_status_label.setStyleSheet(
            "font-size:12pt;font-weight:bold;color:#20354a;border:0;"
        )
        self.report_summary_label = QLabel()
        self.report_summary_label.setWordWrap(True)
        self.report_summary_label.setStyleSheet("color:#344054;border:0;")
        self.event_log_view = QPlainTextEdit()
        self.event_log_view.setObjectName("ProcessingEventLog")
        self.event_log_view.setReadOnly(True)
        self.event_log_view.setMaximumBlockCount(2000)
        self.event_log_view.setMinimumHeight(130)
        self.report_path_label = QLabel()
        self.report_path_label.setWordWrap(True)
        self.report_path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.report_path_label.setStyleSheet("color:#365f87;border:0;")
        self.color_version_title = QLabel(
            self._text("顏色設定版本", "Color Configuration Versions")
        )
        self.color_version_title.setStyleSheet(
            "font-size:11pt;font-weight:bold;color:#20354a;border:0;"
        )
        self.color_version_table = QTableWidget(0, 6)
        self.color_version_table.setObjectName("ColorVersionHistoryTable")
        self.color_version_table.setHorizontalHeaderLabels(
            [
                self._text("範圍", "Scope"),
                self._text("版本", "Version"),
                self._text("狀態", "Status"),
                self._text("證據", "Evidence"),
                self._text("設定差異", "Changes"),
                self._text("建立時間", "Created"),
            ]
        )
        self.color_version_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.color_version_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.color_version_table.setMinimumHeight(145)
        version_header = self.color_version_table.horizontalHeader()
        version_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        version_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        version_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        version_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        version_header.setSectionResizeMode(4, QHeaderView.Stretch)
        version_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.color_version_title.setVisible(False)
        self.color_version_table.setVisible(False)
        layout.addWidget(self.report_status_label)
        layout.addWidget(self.report_summary_label)
        layout.addWidget(self.event_log_view)
        layout.addWidget(self.report_path_label)
        layout.addWidget(self.color_version_title)
        layout.addWidget(self.color_version_table)

        actions = QHBoxLayout()
        self.open_report_button = QPushButton(
            self._text("Open Report", "Open Report")
        )
        self.copy_report_path_button = QPushButton(
            self._text("Copy Report Path", "Copy Report Path")
        )
        self.build_retry_button = QPushButton(
            self._text("Build Retry Plan", "Build Retry Plan")
        )
        self.open_dataset_button = QPushButton(
            self._text("開啟 Dataset 資料夾", "Open Dataset Folder")
        )
        self.open_preparation_report_button = QPushButton(
            self._text("開啟準備報告", "Open Preparation Report")
        )
        self.copy_dataset_id_button = QPushButton(
            self._text("複製 Dataset ID", "Copy Dataset ID")
        )
        self.open_annotation_package_button = QPushButton(
            self._text("開啟補標工作包", "Open Annotation Package")
        )
        self.launch_annotation_tool_button = QPushButton(
            self._text("啟動補標工具", "Launch Annotation Tool")
        )
        self.resume_annotation_button = QPushButton(
            self._text("繼續補標結果", "Resume Annotation Results")
        )
        self.open_annotation_completion_button = QPushButton(
            self._text("開啟補標完成報告", "Open Annotation Completion")
        )
        self.open_color_package_button = QPushButton(
            self._text("開啟色彩校正套件", "Open Color Package")
        )
        self.open_color_preview_button = QPushButton(
            self._text("檢視校正預覽", "Review Color Proposal")
        )
        self.approve_color_button = QPushButton(self._text("核准 Scope", "Approve Scope"))
        self.reject_color_button = QPushButton(self._text("拒絕 Scope", "Reject Scope"))
        self.apply_color_button = QPushButton(
            self._text("套用已核准校正", "Apply Approved Calibration")
        )
        self.open_color_completion_button = QPushButton(
            self._text("開啟校正完成報告", "Open Color Completion")
        )
        self.open_report_button.clicked.connect(self._open_report)
        self.copy_report_path_button.clicked.connect(self._copy_report_path)
        self.build_retry_button.clicked.connect(self._build_retry_plan)
        self.open_dataset_button.clicked.connect(self._open_dataset)
        self.open_preparation_report_button.clicked.connect(self._open_preparation_report)
        self.copy_dataset_id_button.clicked.connect(self._copy_dataset_id)
        self.open_annotation_package_button.clicked.connect(self._open_annotation_package)
        self.launch_annotation_tool_button.clicked.connect(self._launch_annotation_tool)
        self.resume_annotation_button.clicked.connect(self._resume_annotation)
        self.open_annotation_completion_button.clicked.connect(self._open_annotation_completion)
        self.open_color_package_button.clicked.connect(self._open_color_package)
        self.open_color_preview_button.clicked.connect(self._open_color_preview)
        self.approve_color_button.clicked.connect(lambda: self._decide_color_scope(True))
        self.reject_color_button.clicked.connect(lambda: self._decide_color_scope(False))
        self.apply_color_button.clicked.connect(self._apply_color_calibration)
        self.open_color_completion_button.clicked.connect(self._open_color_completion)
        for button in (
            self.open_report_button,
            self.copy_report_path_button,
            self.build_retry_button,
            self.open_dataset_button,
            self.open_preparation_report_button,
            self.copy_dataset_id_button,
            self.open_annotation_package_button,
            self.launch_annotation_tool_button,
            self.resume_annotation_button,
            self.open_annotation_completion_button,
        ):
            actions.addWidget(button)
        for button in (
            self.open_dataset_button,
            self.open_preparation_report_button,
            self.copy_dataset_id_button,
            self.open_annotation_package_button,
            self.launch_annotation_tool_button,
            self.resume_annotation_button,
            self.open_annotation_completion_button,
            self.open_color_package_button,
            self.open_color_preview_button,
            self.approve_color_button,
            self.reject_color_button,
            self.apply_color_button,
            self.open_color_completion_button,
        ):
            button.setVisible(False)
        layout.addLayout(actions)
        color_actions = QHBoxLayout()
        for button in (
            self.open_color_package_button,
            self.open_color_preview_button,
            self.approve_color_button,
            self.reject_color_button,
            self.apply_color_button,
            self.open_color_completion_button,
        ):
            color_actions.addWidget(button)
        color_actions.addStretch()
        layout.addLayout(color_actions)
        return panel

    def _toggle_advanced(self, expanded: bool) -> None:
        self.advanced_panel.setVisible(expanded)
        self.advanced_button.setText(
            self._text("Advanced ▲", "Advanced ▲")
            if expanded
            else self._text("Advanced ▼", "Advanced ▼")
        )

    def _start_processing(self) -> None:
        if self.view_model.has_blocking_items and getattr(
            self.view_model,
            "block_before_start",
            True,
        ):
            self._show_blocking_summary()
            return
        if getattr(self.view_model, "requires_dataset_commit_confirmation", False):
            answer = QMessageBox.question(
                self,
                self._text("確認建立 Dataset", "Confirm Dataset Preparation"),
                self._text(
                    "將建立不可變 Dataset；本步驟不會開始訓練。是否繼續？",
                    "An immutable dataset will be created. Training will not start. Continue?",
                ),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        result = self.view_model.start_processing()
        self._render_execution_result(result)
        message = QMessageBox.information if result.accepted else QMessageBox.warning
        message(self, result.title, result.message)
        if result.accepted and result.close_dialog:
            self.accept()

    def _render_execution_result(self, result) -> None:
        if not result.report_status and not result.report_path:
            return
        self.execution_panel.setVisible(True)
        self.report_status_label.setText(
            self._text(
                f"Report Status：{result.report_status}",
                f"Report Status: {result.report_status}",
            )
        )
        self.report_summary_label.setText(
            " | ".join(
                f"{metric.label} {metric.value}" for metric in result.report_metrics
            )
            or result.message
        )
        self.event_log_view.setPlainText("\n".join(result.event_lines))
        self.report_path_label.setText(result.report_path)
        has_report = bool(result.report_path)
        self.open_report_button.setEnabled(has_report)
        self.copy_report_path_button.setEnabled(has_report)
        self.build_retry_button.setEnabled(result.can_build_retry)
        self._render_dataset_result()
        self._render_annotation_result()
        self._render_color_result()

    def _render_color_result(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is None or not color.visible:
            return
        scope_lines = [
            f"{item.label}: {item.proposal_status} / Gate {item.gate_status} / regressions {item.regression_count}"
            for item in color.scopes
        ]
        self.report_summary_label.setText(
            self.report_summary_label.text()
            + "\n"
            + self._text(
                f"色彩校正：待核准 {color.pending_count} / 已核准 {color.approved_count} / 已拒絕 {color.rejected_count}\n",
                f"Color calibration: pending {color.pending_count} / approved {color.approved_count} / rejected {color.rejected_count}\n",
            )
            + "\n".join(scope_lines)
        )
        for button in (
            self.open_color_package_button,
            self.open_color_preview_button,
            self.approve_color_button,
            self.reject_color_button,
            self.apply_color_button,
        ):
            button.setVisible(True)
        self.apply_color_button.setEnabled(color.pending_count == 0 and color.approved_count > 0)
        self.open_color_completion_button.setVisible(bool(color.completion_path))
        self._render_color_revision_history()

    def _render_color_revision_history(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        revisions = tuple(getattr(color, "revisions", ())) if color is not None else ()
        self.color_version_title.setVisible(bool(revisions))
        self.color_version_table.setVisible(bool(revisions))
        self.color_version_table.setRowCount(len(revisions))
        for row, revision in enumerate(revisions):
            values = (
                revision.scope_label,
                revision.display_version,
                self._text("使用中", "ACTIVE")
                if revision.active
                else self._text("可回退", "AVAILABLE"),
                revision.evidence_level,
                revision.changes,
                revision.created_at,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setToolTip(revision.revision_id)
                    item.setData(Qt.UserRole, revision.revision_id)
                self.color_version_table.setItem(row, column, item)
        self.rollback_color_button.setVisible(
            any(not revision.active for revision in revisions)
        )

    def _render_annotation_result(self) -> None:
        annotation = getattr(self.view_model, "annotation_package", None)
        if annotation is None or not annotation.visible:
            return
        self.report_summary_label.setText(
            self.report_summary_label.text()
            + "\n"
            + self._text(
                f"補標工作包：{annotation.package_id}｜狀態：{annotation.status}\n"
                "補標工作與 resume 為獨立操作；Training 尚未開始。",
                f"Annotation package: {annotation.package_id} | Status: {annotation.status}\n"
                "Annotation and resume are separate operations; training has not started.",
            )
        )
        item_lines = [
            f"{sample_id}: {operation}"
            for sample_id, operation in getattr(annotation, "items", ())
        ]
        item_lines.extend(getattr(annotation, "item_results", ()))
        if item_lines:
            self.report_summary_label.setText(
                self.report_summary_label.text() + "\n" + "\n".join(item_lines)
            )
        self.open_annotation_package_button.setVisible(True)
        self.launch_annotation_tool_button.setVisible(True)
        self.resume_annotation_button.setVisible(annotation.requires_resume)
        self.open_annotation_completion_button.setVisible(bool(annotation.completion_path))

    def _render_dataset_result(self) -> None:
        dataset = getattr(self.view_model, "dataset_preparation", None)
        if dataset is None or not dataset.visible:
            return
        mode = "Dry-run" if dataset.dry_run else "Actual"
        self.report_summary_label.setText(
            self.report_summary_label.text()
            + "\n"
            + self._text(
                f"Dataset ID: {dataset.dataset_id} | 模式: {mode} | 接受: {dataset.accepted_count} "
                f"(train {dataset.train_count}, val {dataset.val_count}, test {dataset.test_count})\n"
                f"Hash: {dataset.dataset_hash}\n位置: {dataset.artifact_path or '未建立（dry-run）'}\n"
                "Training not started",
                f"Dataset ID: {dataset.dataset_id} | Mode: {mode} | Accepted: {dataset.accepted_count} "
                f"(train {dataset.train_count}, val {dataset.val_count}, test {dataset.test_count})\n"
                f"Hash: {dataset.dataset_hash}\nPath: {dataset.artifact_path or 'Not created (dry-run)'}\n"
                "Training not started",
            )
        )
        self.open_dataset_button.setVisible(bool(dataset.artifact_path))
        self.open_preparation_report_button.setVisible(
            bool(dataset.preparation_report_path)
        )
        self.copy_dataset_id_button.setVisible(bool(dataset.dataset_id))

    def _open_dataset(self) -> None:
        dataset = getattr(self.view_model, "dataset_preparation", None)
        if dataset is not None and dataset.artifact_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(dataset.artifact_path))

    def _open_preparation_report(self) -> None:
        dataset = getattr(self.view_model, "dataset_preparation", None)
        if dataset is not None and dataset.preparation_report_path:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(dataset.preparation_report_path)
            )

    def _copy_dataset_id(self) -> None:
        dataset = getattr(self.view_model, "dataset_preparation", None)
        if dataset is not None and dataset.dataset_id:
            QApplication.clipboard().setText(dataset.dataset_id)

    def _open_annotation_package(self) -> None:
        annotation = getattr(self.view_model, "annotation_package", None)
        if annotation is not None and annotation.package_root:
            QDesktopServices.openUrl(QUrl.fromLocalFile(annotation.package_root))

    def _launch_annotation_tool(self) -> None:
        annotation = getattr(self.view_model, "annotation_package", None)
        if annotation is None:
            return
        result = annotation.launch_tool()
        (QMessageBox.information if result.accepted else QMessageBox.warning)(
            self, result.title, result.message
        )

    def _resume_annotation(self) -> None:
        annotation = getattr(self.view_model, "annotation_package", None)
        if annotation is None:
            return
        reason, accepted = QInputDialog.getText(
            self,
            self._text("補標修訂原因", "Annotation Revision Reason"),
            self._text("請輸入本批補標修訂原因：", "Enter the revision reason for this batch:"),
        )
        if not accepted:
            return
        if not reason.strip():
            QMessageBox.warning(
                self,
                self._text("需要修訂原因", "Revision Reason Required"),
                self._text("不得以空白原因提交補標 revision。", "A revision reason is required."),
            )
            return
        result = annotation.resume(reason.strip())
        self._render_annotation_result()
        (QMessageBox.information if result.accepted else QMessageBox.warning)(
            self, result.title, result.message
        )

    def _open_annotation_completion(self) -> None:
        annotation = getattr(self.view_model, "annotation_package", None)
        if annotation is not None and annotation.completion_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(annotation.completion_path))

    def _open_color_package(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is not None and color.package_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(os.path.dirname(color.package_path))))

    def _open_color_preview(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is not None and color.preview_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(color.preview_path))

    def _decide_color_scope(self, approved: bool) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is None or not color.scopes:
            return
        labels = [item.label for item in color.scopes]
        label, accepted = QInputDialog.getItem(
            self, self._text("選擇校正 Scope", "Select Calibration Scope"),
            self._text("Scope", "Scope"), labels, 0, False,
        )
        if not accepted:
            return
        scope = next(item for item in color.scopes if item.label == label)
        reviewer, accepted = QInputDialog.getText(
            self, self._text("覆核人員", "Reviewer"),
            self._text("請輸入具名覆核人員：", "Enter the named reviewer:"),
        )
        if not accepted:
            return
        reason, accepted = QInputDialog.getText(
            self, self._text("決策原因", "Decision Reason"),
            self._text("請輸入核准／拒絕原因：", "Enter the approval/rejection reason:"),
        )
        if not accepted:
            return
        try:
            color.decide(scope.scope_hash, approved=approved, reviewer=reviewer, reason=reason)
            self._render_color_result()
            QMessageBox.information(self, self._text("已記錄", "Recorded"), self._text("Scope 決策已記錄。", "Scope decision recorded."))
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, self._text("無法記錄", "Decision Rejected"), str(exc))

    def _apply_color_calibration(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is None:
            return
        regression_count = sum(item.regression_count for item in color.scopes)
        answer = QMessageBox.question(
            self, self._text("確認啟用色彩設定", "Confirm Color Activation"),
            self._text(
                f"將啟用 {color.approved_count} 個 scope；預覽 regression 數為 {regression_count}。繼續？",
                f"Activate {color.approved_count} scopes; preview regressions: {regression_count}. Continue?",
            ),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            outcome = color.apply_approved()
            self.open_color_completion_button.setVisible(bool(color.completion_path))
            self._render_color_revision_history()
            QMessageBox.information(
                self, self._text("校正處理完成", "Color Calibration Completed"),
                f"{outcome.status}\n{outcome.report_path}",
            )
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, self._text("校正啟用失敗", "Color Activation Failed"), str(exc))

    def _open_color_completion(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is not None and color.completion_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(color.completion_path))

    def _rollback_color_revision(self) -> None:
        color = getattr(self.view_model, "color_calibration", None)
        if color is None or not color.scopes:
            return
        labels = [item.label for item in color.scopes]
        label, accepted = QInputDialog.getItem(
            self,
            self._text("選擇回復 Scope", "Select Rollback Scope"),
            self._text("Scope", "Scope"),
            labels,
            0,
            False,
        )
        if not accepted:
            return
        scope = next(item for item in color.scopes if item.label == label)
        candidates = tuple(
            item
            for item in color.revisions_for_scope(scope.scope_hash)
            if not item.active
        )
        if not candidates:
            QMessageBox.information(
                self,
                self._text("沒有可回退版本", "No Rollback Version"),
                self._text(
                    "此範圍目前沒有較早的顏色設定版本。",
                    "This scope has no earlier color configuration version.",
                ),
            )
            return
        version_labels = [
            f"{item.display_version} | {item.changes}" for item in candidates
        ]
        selected_version, accepted = QInputDialog.getItem(
            self,
            self._text("選擇顏色版本", "Select Color Version"),
            self._text("回退目標", "Rollback target"),
            version_labels,
            0,
            False,
        )
        if not accepted:
            return
        target_revision = candidates[
            version_labels.index(selected_version)
        ].display_version
        operator, accepted = QInputDialog.getText(
            self,
            self._text("操作人員", "Operator"),
            self._text("輸入具名操作人員：", "Enter the named operator:"),
        )
        if not accepted:
            return
        reason, accepted = QInputDialog.getText(
            self,
            self._text("回復原因", "Rollback Reason"),
            self._text("輸入回復原因：", "Enter the rollback reason:"),
        )
        if not accepted:
            return
        try:
            pointer = color.rollback(
                scope.scope_hash,
                target_revision,
                operator.strip(),
                reason.strip(),
            )
            self._render_color_revision_history()
            QMessageBox.information(
                self,
                self._text("回復完成", "Rollback Completed"),
                str(pointer),
            )
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(
                self,
                self._text("回復失敗", "Rollback Failed"),
                str(exc),
            )

    def _open_report(self) -> None:
        path = self.report_path_label.text().strip()
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _copy_report_path(self) -> None:
        path = self.report_path_label.text().strip()
        if path:
            QApplication.clipboard().setText(path)

    def _build_retry_plan(self) -> None:
        builder = getattr(self.view_model, "build_retry_plan", None)
        if builder is None:
            return
        result = builder()
        message = QMessageBox.information if result.created else QMessageBox.warning
        details = result.message
        if result.plan_path:
            details += f"\n\n{result.plan_path}"
        message(self, result.title, details)

    def _show_blocking_summary(self) -> None:
        self._build_blocking_summary_dialog().exec_()

    def _build_blocking_summary_dialog(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setWindowTitle(self._text("Blocking Summary", "Blocking Summary"))
        configure_responsive_dialog(
            dialog,
            preferred=(860, 430),
            minimum=(520, 340),
            parent=self,
        )
        layout = QVBoxLayout(dialog)
        guidance = QLabel(
            self._text(
                "下列資料尚未符合處理條件；請依原因完成複核後再開始。",
                "These samples are not ready. Resolve the listed review issues before starting.",
            )
        )
        guidance.setWordWrap(True)
        layout.addWidget(guidance)
        table = QTableWidget(len(self.view_model.blocking_items), 3)
        table.setObjectName("BlockingSummaryTable")
        table.setHorizontalHeaderLabels(
            [
                self._text("Sample ID", "Sample ID"),
                self._text("Reason", "Reason"),
                self._text("Violation Code", "Violation Code"),
            ]
        )
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        for row, item in enumerate(self.view_model.blocking_items):
            table.setItem(row, 0, QTableWidgetItem(item.sample_id))
            table.setItem(row, 1, QTableWidgetItem(item.reason))
            table.setItem(row, 2, QTableWidgetItem(item.violation_code))
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(table, 1)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignRight)
        return dialog

    def _show_advanced_notice(self) -> None:
        QMessageBox.information(
            self,
            self._text("進階功能", "Advanced"),
            self._text(
                "Phase 3B 僅提供可稽核 dry-run；此進階動作尚未接入實際處理流程。",
                "Phase 3B provides an auditable dry-run only; this advanced action is not connected to downstream processing.",
            ),
        )

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en
