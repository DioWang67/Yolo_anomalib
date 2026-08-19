"""Operator-facing model version inventory and rollback dialog."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
from core.services.model_version_registry import (
    ModelVersionRecord,
    ModelVersionRegistry,
    ModelVersionRegistryError,
)

RECORD_ROLE = Qt.UserRole


class ModelVersionsDialog(QDialog):
    """Show all model versions and let an operator activate a selected one.

    Args:
        registry: Model version registry rooted at the inference models folder.
        language: UI language code.
        selected_product: Product initially selected in the inference window.
        selected_area: Station initially selected in the inference window.
        selected_model_type: Backend initially selected in the inference window.
        is_inspection_running: Callback used to block unsafe live switching.
        on_activated: Callback invoked after a successful atomic switch.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        registry: ModelVersionRegistry,
        *,
        language: str = "zh_TW",
        selected_product: str | None = None,
        selected_area: str | None = None,
        selected_model_type: str | None = None,
        is_inspection_running: Callable[[], bool] | None = None,
        on_activated: Callable[[ModelVersionRecord], None] | None = None,
        is_combination_managed: Callable[[ModelVersionRecord], bool] | None = None,
        on_use_in_combination: Callable[[ModelVersionRecord], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.registry = registry
        self.language = language
        self.initial_filters = (
            selected_product or "",
            selected_area or "",
            selected_model_type or "",
        )
        self.is_inspection_running = is_inspection_running or (lambda: False)
        self.on_activated = on_activated
        self.is_combination_managed = is_combination_managed or (lambda _record: False)
        self.on_use_in_combination = on_use_in_combination
        self.records: list[ModelVersionRecord] = []
        self.setWindowTitle(self._text("模型版本管理", "Model Version Management"))
        configure_responsive_dialog(
            self,
            preferred=(1500, 880),
            minimum=(820, 520),
            parent=parent,
        )
        self._build_ui()
        self.refresh_versions(restore_initial_filters=True)

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel(
            self._text(
                "模型版本與部署紀錄。選取歷史版本後可直接切換；目前檢測中的模型不會被即時替換。",
                "Model versions and deployment history. Select a historical version to activate it safely.",
            )
        )
        title.setWordWrap(True)
        title.setStyleSheet(
            "QLabel { background: #243447; color: white; padding: 12px; "
            "font-size: 13pt; font-weight: bold; }"
        )
        layout.addWidget(title)

        filters = QHBoxLayout()
        filters.addWidget(QLabel(self._text("產品", "Product")))
        self.product_filter = QComboBox()
        filters.addWidget(self.product_filter)
        filters.addWidget(QLabel(self._text("工位", "Station")))
        self.area_filter = QComboBox()
        filters.addWidget(self.area_filter)
        filters.addWidget(QLabel(self._text("模型類型", "Model type")))
        self.type_filter = QComboBox()
        filters.addWidget(self.type_filter)
        filters.addWidget(QLabel(self._text("訓練日期", "Training date")))
        self.date_filter = QComboBox()
        filters.addWidget(self.date_filter)
        filters.addStretch()
        self.refresh_button = QPushButton(self._text("重新整理", "Refresh"))
        self.refresh_button.clicked.connect(self.refresh_versions)
        filters.addWidget(self.refresh_button)
        layout.addLayout(filters)

        headers = [
            self._text("狀態", "Status"),
            self._text("產品", "Product"),
            self._text("工位", "Station"),
            self._text("類型", "Type"),
            self._text("版本", "Version"),
            self._text("訓練／檔案時間", "Training / file time"),
            self._text("部署時間", "Deployed at"),
            self._text("評估結果", "Evaluation"),
            self._text("大小", "Size"),
            self._text("模型檔案", "Artifact"),
            self._text("完整性", "Integrity"),
        ]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._update_selection_details)
        self.table.itemDoubleClicked.connect(lambda _item: self._activate_selected())
        layout.addWidget(self.table, 1)

        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "QLabel { background: #eef2f6; border: 1px solid #c8d1dc; "
            "padding: 8px; min-height: 48px; }"
        )
        layout.addWidget(self.details_label)

        footer = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-weight: bold;")
        footer.addWidget(self.summary_label)
        footer.addStretch()
        self.rollback_button = QPushButton(
            self._text("回復上一個使用版本", "Restore previous active version")
        )
        self.rollback_button.clicked.connect(self._rollback_selected_target)
        footer.addWidget(self.rollback_button)
        self.activate_button = QPushButton(
            self._text("切換至選取版本", "Activate selected version")
        )
        self.activate_button.setMinimumHeight(40)
        self.activate_button.setStyleSheet(
            "QPushButton { background: #1f6f43; color: white; "
            "font-weight: bold; padding: 8px 18px; }"
            "QPushButton:disabled { background: #9aa5b1; }"
        )
        self.activate_button.clicked.connect(self._activate_selected)
        footer.addWidget(self.activate_button)
        close_button = QPushButton(self._text("關閉", "Close"))
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

        for combo in (
            self.product_filter,
            self.area_filter,
            self.type_filter,
            self.date_filter,
        ):
            combo.currentIndexChanged.connect(self._apply_filters)

    def refresh_versions(self, *, restore_initial_filters: bool = False) -> None:
        """Reload model metadata from disk and rebuild all filter values."""
        selected_identity = self._selected_record().identity if self._selected_record() else None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.records = self.registry.list_versions()
        except ModelVersionRegistryError as exc:
            QMessageBox.critical(
                self,
                self._text("模型版本", "Model versions"),
                str(exc),
            )
            self.records = []
        finally:
            QApplication.restoreOverrideCursor()

        current_values = (
            self.initial_filters
            if restore_initial_filters
            else (
                self.product_filter.currentData() or "",
                self.area_filter.currentData() or "",
                self.type_filter.currentData() or "",
            )
        )
        self._replace_filter_items(
            self.product_filter,
            sorted({item.product for item in self.records}),
            current_values[0],
        )
        self._replace_filter_items(
            self.area_filter,
            sorted({item.area for item in self.records}),
            current_values[1],
        )
        self._replace_filter_items(
            self.type_filter,
            sorted({item.model_type for item in self.records}),
            current_values[2],
        )
        dates = sorted(
            {
                item.trained_at.date().isoformat()
                for item in self.records
                if item.trained_at is not None
            },
            reverse=True,
        )
        self._replace_filter_items(self.date_filter, dates, "")
        self._apply_filters(selected_identity=selected_identity)

    def _replace_filter_items(
        self, combo: QComboBox, values: list[str], selected: str
    ) -> None:
        combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem(self._text("全部", "All"), "")
            for value in values:
                combo.addItem(value, value)
            index = combo.findData(selected)
            combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            combo.blockSignals(False)

    def _apply_filters(self, _index=None, *, selected_identity=None) -> None:
        product = str(self.product_filter.currentData() or "")
        area = str(self.area_filter.currentData() or "")
        model_type = str(self.type_filter.currentData() or "")
        trained_date = str(self.date_filter.currentData() or "")
        visible = [
            item
            for item in self.records
            if (not product or item.product == product)
            and (not area or item.area == area)
            and (not model_type or item.model_type == model_type)
            and (
                not trained_date
                or (
                    item.trained_at is not None
                    and item.trained_at.date().isoformat() == trained_date
                )
            )
        ]
        self._populate_table(visible, selected_identity=selected_identity)

    def _populate_table(
        self,
        records: list[ModelVersionRecord],
        *,
        selected_identity=None,
    ) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        selected_row = -1
        for record in records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            status = self._text("使用中", "Active") if record.is_current else self._text("歷史版本", "Historical")
            trained_at = _format_datetime(record.trained_at)
            if record.training_time_inferred and trained_at:
                trained_at += self._text("（檔案時間）", " (file time)")
            integrity = (
                self._text("完整", "Complete")
                if not record.warning
                else record.warning
            )
            values = [
                status,
                record.product,
                record.area,
                record.model_type,
                f"v{record.version}" if record.version != "legacy" else self._text("未標記", "Legacy"),
                trained_at or "—",
                _format_datetime(record.deployed_at or record.activated_at) or "—",
                _format_metrics(record.evaluation_metrics),
                _format_file_size(record.file_size),
                record.weight_path.name,
                integrity,
            ]
            background = QColor("#dff3e4") if record.is_current else QColor("#ffffff")
            if not record.exists:
                background = QColor("#fde2e2")
            elif record.warning and not record.is_current:
                background = QColor("#fff5d6")
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setData(RECORD_ROLE, record)
                cell.setBackground(background)
                cell.setToolTip(str(record.weight_path))
                self.table.setItem(row, column, cell)
            if selected_identity and record.identity == selected_identity:
                selected_row = row
        self.table.setSortingEnabled(True)
        self.summary_label.setText(
            self._text(
                f"共 {len(records)} 個版本；使用中 {sum(item.is_current for item in records)} 個",
                f"{len(records)} versions; {sum(item.is_current for item in records)} active",
            )
        )
        if selected_row >= 0:
            self.table.selectRow(selected_row)
        elif records:
            current_row = next(
                (index for index, item in enumerate(records) if item.is_current), 0
            )
            self.table.selectRow(current_row)
        else:
            self._update_selection_details()

    def _selected_record(self) -> ModelVersionRecord | None:
        selected_rows = self.table.selectionModel().selectedRows() if hasattr(self, "table") else []
        if not selected_rows:
            return None
        item = self.table.item(selected_rows[0].row(), 0)
        value = item.data(RECORD_ROLE) if item else None
        return value if isinstance(value, ModelVersionRecord) else None

    def _update_selection_details(self) -> None:
        record = self._selected_record()
        managed = bool(record and self.is_combination_managed(record))
        enabled = bool(
            record
            and record.exists
            and (managed or not record.is_current)
        )
        self.activate_button.setText(
            self._text("建立檢測組合", "Create inspection combination")
            if managed
            else self._text("切換至選取版本", "Activate selected version")
        )
        self.activate_button.setEnabled(enabled)
        self.rollback_button.setEnabled(record is not None and not managed)
        if record is None:
            self.details_label.setText(self._text("請選取一個版本。", "Select a version."))
            return
        metrics = json.dumps(record.evaluation_metrics, ensure_ascii=False, sort_keys=True)
        metadata_state = (
            self._text("具備配套設定快照", "Version-matched config available")
            if record.has_config_snapshot
            else self._text("舊版紀錄不完整；切換時將保留目前設定", "Legacy metadata; current config will be retained")
        )
        self.details_label.setText(
            self._text("資料集", "Dataset")
            + f": {record.dataset_hash or '—'}    "
            + self._text("訓練設定", "Training config")
            + f": {record.training_config_hash or '—'}\n"
            + self._text("評估指標", "Metrics")
            + f": {metrics if metrics != '{}' else '—'}    {metadata_state}"
            + (
                self._text(
                    "\n此工位使用檢測組合管理；套用時會建立包含此模型的組合版本。",
                    "\nThis station uses inspection combinations; applying this model creates a combination version.",
                )
                if managed
                else ""
            )
        )

    def _activate_selected(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        if record.is_current and not self.is_combination_managed(record):
            return
        self._activate_record(record)

    def _rollback_selected_target(self) -> None:
        selected = self._selected_record()
        if selected is None:
            return
        previous = self.registry.previous_version(
            selected.product, selected.area, selected.model_type
        )
        if previous is None:
            QMessageBox.information(
                self,
                self._text("回復模型", "Restore model"),
                self._text("找不到可回復的上一個版本。", "No previous version is available."),
            )
            return
        self._activate_record(previous)

    def _activate_record(self, record: ModelVersionRecord) -> None:
        if self.is_inspection_running():
            QMessageBox.warning(
                self,
                self._text("無法切換模型", "Cannot switch model"),
                self._text("檢測正在執行，請先停止檢測後再切換模型。", "Stop inspection before switching models."),
            )
            return
        if self.is_combination_managed(record):
            if self.on_use_in_combination is not None:
                self.on_use_in_combination(record)
            return
        target = f"{record.product}/{record.area}/{record.model_type}  v{record.version}"
        warning = ""
        allow_incomplete = not record.has_config_snapshot
        if allow_incomplete:
            warning = self._text(
                "\n\n此舊版本缺少當時的設定快照，將沿用目前設定並只更換權重。切換後必須先做驗證。",
                "\n\nThis legacy version has no matching config snapshot. Current settings will be retained; validate it before production use.",
            )
        answer = QMessageBox.question(
            self,
            self._text("確認模型切換", "Confirm model switch"),
            self._text("確定要切換至：", "Activate: ") + f"\n{target}{warning}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            activated = self.registry.activate(
                record, allow_incomplete=allow_incomplete
            )
            if self.on_activated:
                self.on_activated(activated)
        except ModelVersionRegistryError as exc:
            QMessageBox.critical(
                self,
                self._text("模型切換失敗", "Model switch failed"),
                str(exc),
            )
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.refresh_versions()
        QMessageBox.information(
            self,
            self._text("模型已切換", "Model activated"),
            self._text(
                "模型版本已更新；下一次檢測會載入選取的版本。",
                "The selected version will be loaded by the next inspection.",
            ),
        )


def _format_datetime(value: datetime | None) -> str:
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S") if value else ""


def _format_file_size(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _format_metrics(metrics: dict) -> str:
    if not metrics:
        return "—"
    preferred = ("map50", "mAP50", "precision", "recall", "map50_95", "mAP50-95")
    parts = []
    for key in preferred:
        if key not in metrics:
            continue
        value = metrics[key]
        parts.append(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}")
        if len(parts) == 3:
            break
    if not parts:
        parts = [f"{key}={value}" for key, value in list(metrics.items())[:3]]
    return ", ".join(parts)
