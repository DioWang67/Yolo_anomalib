"""Three-column RC-1 Historical Cleanup Assistant UI."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from app.gui.historical_cleanup_view_model import HistoricalCleanupViewModel
from tools.historical_cleanup import CleanupDecisionError, HistoricalCleanupError
from tools.review_repair import ReviewRepairError

GROUP_ID_ROLE = Qt.UserRole + 1
RECORD_ID_ROLE = Qt.UserRole + 2


class HistoricalCleanupDialog(QDialog):
    """Render cleanup groups without loading every blocking row at once."""

    def __init__(
        self,
        view_model: HistoricalCleanupViewModel,
        *,
        language: str = "zh_TW",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.view_model = view_model
        self.language = language
        self._current_group_id = ""
        self._current_page = 0
        self.setWindowTitle(self._text("歷史資料清理助理", "Historical Cleanup Assistant"))
        configure_responsive_dialog(
            self,
            preferred=(1180, 760),
            minimum=(760, 500),
            parent=parent,
        )
        self._build_ui()
        self._load_groups()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        title = QLabel(self.windowTitle())
        title.setStyleSheet("font-size:18pt;font-weight:bold;color:#20354a;")
        root.addWidget(title)
        stats = self.view_model.planner_statistics
        self.summary_label = QLabel(
            self._text(
                f"樣本 {self.view_model.sample_count}｜READY {stats['ready_count']}｜"
                f"BLOCKED {stats['blocking_count']}｜EXCLUDED {stats['excluded_count']}",
                f"Samples {self.view_model.sample_count} | READY {stats['ready_count']} | "
                f"BLOCKED {stats['blocking_count']} | EXCLUDED {stats['excluded_count']}",
            )
        )
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.summary_label)

        columns = QHBoxLayout()
        columns.setSpacing(10)
        columns.addWidget(self._build_groups_panel(), 2)
        columns.addWidget(self._build_records_panel(), 4)
        columns.addWidget(self._build_detail_panel(), 3)
        root.addLayout(columns, 1)

        identity = QHBoxLayout()
        identity.addWidget(QLabel(self._text("覆核者", "Reviewer")))
        self.reviewer_edit = QLineEdit()
        self.reviewer_edit.setPlaceholderText(self._text("必填", "Required"))
        identity.addWidget(self.reviewer_edit, 1)
        identity.addWidget(QLabel(self._text("決策原因", "Decision reason")))
        self.reason_edit = QLineEdit()
        self.reason_edit.setPlaceholderText(
            self._text("必填；會寫入稽核紀錄", "Required; recorded in audit")
        )
        identity.addWidget(self.reason_edit, 3)
        root.addLayout(identity)

        root.addLayout(self._build_action_bar())

    def _build_groups_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel(self._text("Blocking Groups", "Blocking Groups")))
        self.group_list = QListWidget()
        self.group_list.setObjectName("CleanupGroupList")
        self.group_list.currentItemChanged.connect(self._on_group_changed)
        layout.addWidget(self.group_list)
        return panel

    def _build_records_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel(self._text("Group Detail", "Group Detail")))
        self.record_table = QTableWidget(0, 4)
        self.record_table.setObjectName("CleanupRecordTable")
        self.record_table.setHorizontalHeaderLabels(
            [
                self._text("Sample", "Sample"),
                self._text("原因", "Root Cause"),
                self._text("信心", "Confidence"),
                self._text("決策", "Decision"),
            ]
        )
        self.record_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.record_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.record_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.record_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.record_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )
        self.record_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents
        )
        self.record_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeToContents
        )
        self.record_table.itemSelectionChanged.connect(self._on_record_changed)
        layout.addWidget(self.record_table)
        paging = QHBoxLayout()
        self.previous_page_button = QPushButton(self._text("上一頁", "Previous"))
        self.next_page_button = QPushButton(self._text("下一頁", "Next"))
        self.page_label = QLabel()
        self.previous_page_button.clicked.connect(lambda: self._move_page(-1))
        self.next_page_button.clicked.connect(lambda: self._move_page(1))
        paging.addWidget(self.previous_page_button)
        paging.addWidget(self.page_label)
        paging.addWidget(self.next_page_button)
        paging.addStretch()
        layout.addLayout(paging)
        return panel

    def _build_detail_panel(self) -> QWidget:
        panel = self._panel()
        layout = QVBoxLayout(panel)
        layout.addWidget(QLabel(self._text("Suggested Fix", "Suggested Fix")))
        self.detail_view = QPlainTextEdit()
        self.detail_view.setObjectName("CleanupDetailView")
        self.detail_view.setReadOnly(True)
        layout.addWidget(self.detail_view)
        open_row = QHBoxLayout()
        for label, target in (
            (self._text("Sample", "Sample"), "sample"),
            (self._text("Image", "Image"), "image"),
            (self._text("Annotation", "Annotation"), "annotation"),
            (self._text("Conflict", "Conflict"), "conflict_report"),
        ):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, key=target: self._open_target(key))
            open_row.addWidget(button)
        layout.addLayout(open_row)
        self.open_review_button = QPushButton(
            self._text("Open Current Review", "Open Current Review")
        )
        self.open_review_button.clicked.connect(self._show_current_review)
        layout.addWidget(self.open_review_button)
        return panel

    def _build_action_bar(self) -> QHBoxLayout:
        actions = QHBoxLayout()
        commands: tuple[tuple[str, Callable[[], None]], ...] = (
            (self._text("批准群組", "Approve Group"), self._approve_group),
            (self._text("拒絕群組", "Reject Group"), self._reject_group),
            (self._text("批准單筆", "Approve Record"), self._approve_record),
            (self._text("拒絕單筆", "Reject Record"), self._reject_record),
            (self._text("略過", "Skip"), self._skip_record),
            (self._text("重新稽核", "Run Audit Again"), self._run_audit_again),
            (self._text("匯出 CSV", "Export CSV"), self._export_csv),
            (self._text("匯出 JSON", "Export JSON"), self._export_json),
            (self._text("套用", "Apply"), self._apply),
            (self._text("復原", "Rollback"), self._rollback),
        )
        for label, command in commands:
            button = QPushButton(label)
            button.clicked.connect(command)
            actions.addWidget(button)
        actions.addStretch()
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        return actions

    @staticmethod
    def _panel() -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(
            "QFrame {background:white;border:1px solid #d8dee6;border-radius:7px;}"
        )
        return panel

    def _load_groups(self) -> None:
        self.group_list.clear()
        for group in self.view_model.groups:
            suffix = "" if group.batch_approvable else self._text("｜需人工", " | Manual")
            item = QListWidgetItem(
                f"{group.root_count if hasattr(group, 'root_count') else group.record_count}  "
                f"{group.title}{suffix}"
            )
            item.setData(GROUP_ID_ROLE, group.group_id)
            self.group_list.addItem(item)
        if self.group_list.count():
            self.group_list.setCurrentRow(0)

    def _on_group_changed(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        self._current_group_id = str(current.data(GROUP_ID_ROLE))
        self._current_page = 0
        self.detail_view.setPlainText(
            self.view_model.group_detail(self._current_group_id)
        )
        self._render_page()

    def _render_page(self) -> None:
        if not self._current_group_id:
            return
        page = self.view_model.records_page(
            self._current_group_id, self._current_page
        )
        self._current_page = page.page
        self.record_table.setRowCount(len(page.records))
        for row_index, record in enumerate(page.records):
            values = (
                record.sample_id,
                record.root_cause,
                record.confidence,
                record.decision,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(RECORD_ID_ROLE, record.record_id)
                self.record_table.setItem(row_index, column, item)
        self.page_label.setText(
            self._text(
                f"第 {page.page + 1}/{page.page_count} 頁，共 {page.total_count} 筆",
                f"Page {page.page + 1}/{page.page_count}, {page.total_count} records",
            )
        )
        self.previous_page_button.setEnabled(page.page > 0)
        self.next_page_button.setEnabled(page.page + 1 < page.page_count)
        if page.records:
            self.record_table.selectRow(0)

    def _move_page(self, offset: int) -> None:
        self._current_page += offset
        self._render_page()

    def _on_record_changed(self) -> None:
        record_id = self._selected_record_id()
        if record_id:
            self.detail_view.setPlainText(self.view_model.record_detail(record_id))

    def _decision_identity(self) -> tuple[str, str]:
        return self.reviewer_edit.text().strip(), self.reason_edit.text().strip()

    def _approve_group(self) -> None:
        self._run_decision(
            lambda reviewer, reason: self.view_model.approve_group(
                self._current_group_id, reviewer, reason
            )
        )

    def _reject_group(self) -> None:
        self._run_decision(
            lambda reviewer, reason: self.view_model.reject_group(
                self._current_group_id, reviewer, reason
            )
        )

    def _approve_record(self) -> None:
        record_id = self._require_selected_record()
        if record_id:
            self._run_decision(
                lambda reviewer, reason: self.view_model.approve_record(
                    record_id, reviewer, reason
                )
            )

    def _reject_record(self) -> None:
        record_id = self._require_selected_record()
        if record_id:
            self._run_decision(
                lambda reviewer, reason: self.view_model.reject_record(
                    record_id, reviewer, reason
                )
            )

    def _skip_record(self) -> None:
        record_id = self._require_selected_record()
        if record_id:
            self._run_decision(
                lambda reviewer, reason: self.view_model.skip_record(
                    record_id, reviewer, reason
                )
            )

    def _run_decision(self, command: Callable[[str, str], None]) -> None:
        reviewer, reason = self._decision_identity()
        try:
            command(reviewer, reason)
        except (CleanupDecisionError, ValueError, OSError) as exc:
            QMessageBox.warning(self, self.windowTitle(), str(exc))
            return
        self._render_page()

    def _selected_record_id(self) -> str:
        selected = self.record_table.selectedItems()
        return str(selected[0].data(RECORD_ID_ROLE)) if selected else ""

    def _require_selected_record(self) -> str:
        record_id = self._selected_record_id()
        if not record_id:
            QMessageBox.information(
                self, self.windowTitle(), self._text("請先選一筆資料。", "Select a record first.")
            )
        return record_id

    def _open_target(self, target: str) -> None:
        record_id = self._require_selected_record()
        if not record_id:
            return
        path_value = self.view_model.open_target(record_id, target)
        path = Path(path_value) if path_value else None
        if path is None or not path.exists():
            QMessageBox.information(
                self,
                self.windowTitle(),
                self._text("此項目沒有可開啟的檔案。", "No file is available for this item."),
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))

    def _show_current_review(self) -> None:
        record_id = self._require_selected_record()
        if not record_id:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self._text("目前覆核資料", "Current Review"))
        configure_responsive_dialog(
            dialog,
            preferred=(760, 560),
            minimum=(480, 360),
            parent=self,
        )
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(self.view_model.current_review_json(record_id))
        view.setReadOnly(True)
        layout.addWidget(view)
        close = QPushButton(self._text("關閉", "Close"))
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        dialog.exec_()

    def _run_audit_again(self) -> None:
        try:
            analysis = self.view_model.run_audit_again()
        except (HistoricalCleanupError, OSError, ValueError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        stats = analysis.planner_statistics
        stale = analysis.manifest_sha256 != self.view_model.manifest_sha256
        message = self._text(
            f"READY {stats['ready_count']}，BLOCKED {stats['blocking_count']}。",
            f"READY {stats['ready_count']}, BLOCKED {stats['blocking_count']}.",
        )
        if stale:
            message += self._text(
                " Manifest 已變更，請關閉並重新開啟助理。",
                " The manifest changed; close and reopen the assistant.",
            )
        QMessageBox.information(self, self.windowTitle(), message)

    def _export_csv(self) -> None:
        self._export("csv")

    def _export_json(self) -> None:
        self._export("json")

    def _export(self, extension: str) -> None:
        destination, _filter = QFileDialog.getSaveFileName(
            self,
            self._text("匯出清理提案", "Export Cleanup Proposal"),
            str(Path(self.view_model.manifest_path).with_suffix(f".cleanup.{extension}")),
            f"{extension.upper()} (*.{extension})",
        )
        if not destination:
            return
        try:
            path = (
                self.view_model.export_csv(destination)
                if extension == "csv"
                else self.view_model.export_json(destination)
            )
        except OSError as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        QMessageBox.information(self, self.windowTitle(), str(path))

    def _apply(self) -> None:
        if QMessageBox.question(
            self,
            self.windowTitle(),
            self._text(
                "只會透過 Phase 1C 套用已批准提案。確定繼續？",
                "Only approved proposals will be applied through Phase 1C. Continue?",
            ),
        ) != QMessageBox.Yes:
            return
        try:
            report = self.view_model.apply()
        except (
            CleanupDecisionError,
            HistoricalCleanupError,
            ReviewRepairError,
            OSError,
            ValueError,
        ) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        after = report["after"]["planner_statistics"]
        QMessageBox.information(
            self,
            self.windowTitle(),
            self._text(
                f"套用完成：READY {after['ready_count']}，BLOCKED {after['blocking_count']}。",
                f"Applied: READY {after['ready_count']}, BLOCKED {after['blocking_count']}.",
            ),
        )

    def _rollback(self) -> None:
        try:
            report = self.view_model.rollback()
        except (
            CleanupDecisionError,
            HistoricalCleanupError,
            ReviewRepairError,
            OSError,
            ValueError,
        ) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        restored = report["restored"]["planner_statistics"]
        QMessageBox.information(
            self,
            self.windowTitle(),
            self._text(
                f"復原完成：READY {restored['ready_count']}，BLOCKED {restored['blocking_count']}。",
                f"Rolled back: READY {restored['ready_count']}, BLOCKED {restored['blocking_count']}.",
            ),
        )

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en
