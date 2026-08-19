"""Read-only operator view of previously submitted training cases."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from app.gui.training_batch_dialog import TrainingBatchDialog
from tools.record_visibility import hide_record
from tools.submission_history import (
    SubmissionHistoryRecord,
    load_submission_entries,
    load_submission_history,
)

HISTORY_RECORD_ROLE = Qt.UserRole
ACTION_LABELS = {
    "portable": ("離線補訓包", "Offline training package"),
    "direct": ("直接訓練", "Direct training"),
    "annotation": ("補標後訓練", "Train after annotation"),
    "color": ("顏色校正", "Color calibration"),
}


class SubmissionHistoryDialog(QDialog):
    """Display append-only submission batches and their individual images."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        language: str = "zh_TW",
        selected_product: str | None = None,
        selected_area: str | None = None,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.data_root = Path(data_root).expanduser().resolve()
        self.language = language
        self.selected_product = str(selected_product or "")
        self.selected_area = str(selected_area or "")
        self.records = load_submission_history(self.data_root)
        self.setWindowTitle(self._text("補訓批次紀錄", "Retraining Batch History"))
        configure_responsive_dialog(
            self,
            preferred=(1250, 760),
            minimum=(720, 480),
            parent=parent,
        )
        self._build_ui()
        self._populate_filters()
        self._apply_filters()

    def _text(self, zh: str, en: str) -> str:
        return zh if self.language.lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel(
            self._text(
                "每個版本保留當次選圖、複核結果與訓練設定；查看照片不會重新加入清單，也不會重複送訓。",
                "Each version preserves its selected images, review results, and training settings. "
                "Viewing images never queues or resubmits them.",
            )
        )
        title.setWordWrap(True)
        title.setStyleSheet(
            "QLabel { background: #243447; color: white; padding: 12px; "
            "font-size: 12pt; font-weight: bold; }"
        )
        layout.addWidget(title)

        filters = QHBoxLayout()
        filters.addWidget(QLabel(self._text("產品", "Product")))
        self.product_filter = QComboBox()
        filters.addWidget(self.product_filter)
        filters.addWidget(QLabel(self._text("工位", "Station")))
        self.area_filter = QComboBox()
        filters.addWidget(self.area_filter)
        filters.addWidget(QLabel(self._text("送出類型", "Submission type")))
        self.action_filter = QComboBox()
        filters.addWidget(self.action_filter)
        filters.addStretch()
        refresh_button = QPushButton(self._text("重新整理", "Refresh"))
        refresh_button.clicked.connect(self._refresh)
        filters.addWidget(refresh_button)
        layout.addLayout(filters)

        headers = [
            self._text("送出時間", "Submitted"),
            self._text("批次版本", "Batch version"),
            self._text("類型", "Type"),
            self._text("產品", "Product"),
            self._text("工位", "Station"),
            self._text("本批照片", "Cases"),
            self._text("直接訓練", "Ready"),
            self._text("待補標", "Pending"),
            self._text("顏色項目", "Color items"),
            self._text("任務編號", "Job ID"),
        ]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_details)
        self.table.itemDoubleClicked.connect(lambda _item: self._open_selected_batch())
        layout.addWidget(self.table, 1)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "QLabel { background: #eef2f6; border: 1px solid #c8d1dc; padding: 8px; }"
        )
        layout.addWidget(self.details_label)

        footer = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-weight: bold;")
        footer.addWidget(self.summary_label)
        footer.addStretch()
        self.clear_button = QPushButton(
            self._text("清除選取紀錄", "Clear selected record")
        )
        self.clear_button.setEnabled(False)
        self.clear_button.setStyleSheet(
            "QPushButton { color:#a61b1b;background:#fff5f5;border:1px solid #d96c6c;"
            "border-radius:4px;padding:6px 10px;font-weight:bold; }"
            "QPushButton:disabled { color:#9aa0a6;background:#f3f4f6;border-color:#d1d5db; }"
        )
        self.clear_button.clicked.connect(self._clear_selected_record)
        footer.addWidget(self.clear_button)
        self.open_button = QPushButton(self._text("查看本批照片", "View batch images"))
        self.open_button.clicked.connect(self._open_selected_batch)
        footer.addWidget(self.open_button)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        for combo in (self.product_filter, self.area_filter, self.action_filter):
            combo.currentIndexChanged.connect(self._apply_filters)

    def _populate_filters(self) -> None:
        self._set_filter(
            self.product_filter,
            sorted({record.product for record in self.records if record.product}),
            self.selected_product,
        )
        self._set_filter(
            self.area_filter,
            sorted({record.area for record in self.records if record.area}),
            self.selected_area,
        )
        self.action_filter.blockSignals(True)
        try:
            self.action_filter.clear()
            self.action_filter.addItem(self._text("全部", "All"), "")
            for action in ("direct", "annotation", "color", "portable"):
                self.action_filter.addItem(self._text(*ACTION_LABELS[action]), action)
        finally:
            self.action_filter.blockSignals(False)

    def _set_filter(self, combo: QComboBox, values: list[str], selected: str) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._text("全部", "All"), "")
            for value in values:
                combo.addItem(value, value)
            selected_index = combo.findData(selected)
            combo.setCurrentIndex(selected_index if selected_index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _refresh(self) -> None:
        product = str(self.product_filter.currentData() or "")
        area = str(self.area_filter.currentData() or "")
        action = str(self.action_filter.currentData() or "")
        self.records = load_submission_history(self.data_root)
        self.selected_product = product
        self.selected_area = area
        self._populate_filters()
        action_index = self.action_filter.findData(action)
        self.action_filter.setCurrentIndex(action_index if action_index >= 0 else 0)
        self._apply_filters()

    def _apply_filters(self, _index: int | None = None) -> None:
        product = str(self.product_filter.currentData() or "")
        area = str(self.area_filter.currentData() or "")
        action = str(self.action_filter.currentData() or "")
        visible = [
            record
            for record in self.records
            if (not product or record.product == product)
            and (not area or record.area == area)
            and (not action or record.action == action)
        ]
        self.table.setRowCount(0)
        for record in visible:
            row_index = self.table.rowCount()
            self.table.insertRow(row_index)
            values = [
                _format_datetime(record.submitted_at),
                record.batch_version or self._text("舊紀錄（未命名）", "Legacy (unnamed)"),
                self._text(*ACTION_LABELS.get(record.action, (record.action, record.action))),
                record.product or "—",
                record.area or "—",
                str(record.case_count),
                str(record.ready_count),
                str(record.pending_count),
                str(record.color_feedback_count),
                record.job_id or "—",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(HISTORY_RECORD_ROLE, record)
                self.table.setItem(row_index, column, item)
        total_cases = sum(record.case_count for record in visible)
        self.summary_label.setText(
            self._text(
                f"共 {len(visible)} 批｜合計 {total_cases} 張",
                f"{len(visible)} batches | {total_cases} cases",
            )
        )
        if visible:
            self.table.selectRow(0)
        else:
            self._update_details()

    def _selected_record(self) -> SubmissionHistoryRecord | None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return None
        item = self.table.item(selected_rows[0].row(), 0)
        record = item.data(HISTORY_RECORD_ROLE) if item else None
        return record if isinstance(record, SubmissionHistoryRecord) else None

    def _update_details(self) -> None:
        record = self._selected_record()
        self.open_button.setEnabled(record is not None and record.case_count > 0)
        self.clear_button.setEnabled(record is not None)
        if record is None:
            self.details_label.setText(self._text("尚無送訓紀錄。", "No submission history."))
            return
        version = record.batch_version or self._text(
            "舊紀錄（未命名）",
            "Legacy (unnamed)",
        )
        training_settings = _format_training_options(
            record.training_options,
            language=self.language,
        )
        self.details_label.setText(
            self._text(
                f"批次版本：{version}｜"
                f"類型：{self._text(*ACTION_LABELS.get(record.action, (record.action, record.action)))}｜"
                f"產品／工位：{record.product or '—'} / {record.area or '—'}｜"
                f"本批：{record.case_count} 張｜任務：{record.job_id or '僅校正資料'}\n"
                f"{training_settings}",
                f"Batch version: {version} | "
                f"Type: {self._text(*ACTION_LABELS.get(record.action, (record.action, record.action)))} | "
                f"Target: {record.product or '—'} / {record.area or '—'} | "
                f"Cases: {record.case_count} | Job: {record.job_id or 'calibration only'}\n"
                f"{training_settings}",
            )
        )

    def _clear_selected_record(self) -> None:
        """Hide one submitted-history entry while preserving all training data."""
        record = self._selected_record()
        if record is None:
            return
        answer = QMessageBox.question(
            self,
            self.windowTitle(),
            self._text(
                f"確定清除這筆已送出紀錄？\n\n{record.product or '—'} / {record.area or '—'}｜"
                f"{record.case_count} 張\n\n訓練圖片、標註、模型與補訓工作不會被刪除。",
                f"Clear this submitted-history entry?\n\n{record.product or '—'} / "
                f"{record.area or '—'} | {record.case_count} cases\n\nTraining images, "
                "labels, models, and retraining jobs will not be deleted.",
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        try:
            hide_record(
                self.data_root,
                "submission_history",
                record.submission_id,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            QMessageBox.critical(self, self.windowTitle(), str(exc))
            return
        self._refresh()

    def _open_selected_batch(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        entries = load_submission_entries(record)
        if not entries:
            QMessageBox.information(
                self,
                self.windowTitle(),
                self._text(
                    "此舊批次保留了數量與任務紀錄，但找不到可顯示的照片明細。",
                    "This legacy batch retains its counts and job record, but no displayable image details remain.",
                ),
            )
            return
        batch_dialog = TrainingBatchDialog(
            entries,
            language=self.language,
            history_mode=True,
            parent=self,
        )
        if record.batch_version:
            batch_dialog.setWindowTitle(
                f"{record.batch_version}｜{batch_dialog.windowTitle()}"
            )
        batch_dialog.exec_()


def _format_datetime(value: datetime | None) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S") if value else "—"


def _format_training_options(
    options: tuple[tuple[str, int | str], ...],
    *,
    language: str,
) -> str:
    """Render the immutable settings saved with one retraining batch."""
    is_chinese = language.lower().startswith("zh")
    if not options:
        return "訓練設定：舊紀錄未保存" if is_chinese else "Training settings: not saved by legacy version"
    values = dict(options)
    position_mode = str(values.get("position_training_mode") or "yolo_only")
    position_enabled = position_mode == "calibrate_validate"
    if is_chinese:
        return (
            f"訓練設定：Epochs {values.get('epochs', '—')}｜"
            f"每張增強 {values.get('augmentations_per_image', '—')}｜"
            f"Batch {values.get('batch', '—')}｜"
            f"影像尺寸 {values.get('imgsz', '—')}｜"
            f"位置檢測 {'啟用' if position_enabled else '停用'}"
        )
    return (
        f"Training settings: epochs {values.get('epochs', '—')} | "
        f"augmentations/image {values.get('augmentations_per_image', '—')} | "
        f"batch {values.get('batch', '—')} | imgsz {values.get('imgsz', '—')} | "
        f"position {'enabled' if position_enabled else 'disabled'}"
    )
