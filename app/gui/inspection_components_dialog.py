"""Unified engineering inventory for model and color component versions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
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
    QWidget,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from core.services.inspection_component_catalog import (
    InspectionComponentCatalog,
    InspectionComponentCatalogError,
    InspectionComponentRecord,
)

COMPONENT_ROLE = Qt.UserRole
_CATEGORY_LABELS = {
    "AI_MODEL": "AI 模型",
    "COLOR_BASE": "顏色基準",
    "COLOR_PROFILE": "顏色方案",
    "COLOR_REVISION": "校正修訂",
}
_STATUS_LABELS = {
    "DEPLOYED": "正式組合使用",
    "DEFAULT": "預設指標",
    "HISTORY": "歷史版本",
    "REVOKED": "已撤銷",
}
_INTEGRITY_LABELS = {
    "VERIFIED": "完整",
    "INCOMPLETE": "缺少設定快照",
    "MISSING": "檔案遺失",
    "WARNING": "需注意",
}


class InspectionComponentsDialog(QDialog):
    """Filter all component versions and route one into combination creation."""

    def __init__(
        self,
        *,
        catalog: InspectionComponentCatalog,
        selected_product: str = "",
        selected_area: str = "",
        selected_inference_type: str = "",
        on_create_combination: (
            Callable[[InspectionComponentRecord], None] | None
        ) = None,
        training_data_root: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog
        # Optional: the training project's data root, used only to look up
        # the photo batch behind a model.
        self.training_data_root = (
            Path(training_data_root) if training_data_root else None
        )
        self.initial_filters = (
            selected_product,
            selected_area,
            selected_inference_type,
        )
        self.on_create_combination = on_create_combination
        self.records: tuple[InspectionComponentRecord, ...] = ()
        self.visible_records: tuple[InspectionComponentRecord, ...] = ()
        self.setWindowTitle("模型與顏色版本")
        configure_responsive_dialog(
            self,
            preferred=(1480, 820),
            minimum=(900, 540),
            parent=parent,
        )
        self._build_ui()
        self.refresh(restore_initial_filters=True)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel(
            "集中查看 AI 模型與顏色設定版本。選取版本後可建立候選檢測組合；"
            "本頁不會直接變更正式檢測。"
        )
        title.setWordWrap(True)
        title.setStyleSheet(
            "background:#243447;color:white;padding:12px;"
            "font-size:12pt;font-weight:600;"
        )
        layout.addWidget(title)

        filters = QHBoxLayout()
        self.product_filter = self._filter(filters, "產品")
        self.area_filter = self._filter(filters, "工位")
        self.inference_filter = self._filter(filters, "推論類型")
        self.category_filter = self._filter(filters, "版本類別")
        self.type_filter = self._filter(filters, "檢測類型")
        filters.addStretch(1)
        refresh_button = QPushButton("重新整理")
        refresh_button.clicked.connect(self.refresh)
        filters.addWidget(refresh_button)
        layout.addLayout(filters)

        headers = (
            "狀態",
            "版本類別",
            "產品",
            "工位",
            "推論類型",
            "檢測項目",
            "版本",
            "建立時間",
            "完整性",
            "來源",
        )
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            9, QHeaderView.Stretch
        )
        self.table.itemSelectionChanged.connect(self._update_details)
        self.table.itemDoubleClicked.connect(
            lambda _item: self._create_combination()
        )
        layout.addWidget(self.table, 1)

        self.details_label = QLabel("請選取一個模型或顏色版本。")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet(
            "background:#eef2f6;border:1px solid #c8d1dc;"
            "padding:8px;min-height:48px;"
        )
        layout.addWidget(self.details_label)

        footer = QHBoxLayout()
        self.summary_label = QLabel()
        footer.addWidget(self.summary_label)
        footer.addStretch(1)
        self.training_images_button = QPushButton("查看訓練照片")
        self.training_images_button.setEnabled(False)
        self.training_images_button.clicked.connect(self._open_training_images)
        footer.addWidget(self.training_images_button)
        self.compose_button = QPushButton("建立檢測組合")
        self.compose_button.setMinimumHeight(40)
        self.compose_button.clicked.connect(self._create_combination)
        footer.addWidget(self.compose_button)
        close_button = QPushButton("關閉")
        close_button.clicked.connect(self.accept)
        footer.addWidget(close_button)
        layout.addLayout(footer)

    @staticmethod
    def _filter(layout: QHBoxLayout, label: str) -> QComboBox:
        layout.addWidget(QLabel(label))
        combo = QComboBox()
        layout.addWidget(combo)
        return combo

    def refresh(self, *, restore_initial_filters: bool = False) -> None:
        try:
            self.records = self.catalog.list_components()
        except InspectionComponentCatalogError as exc:
            self.records = ()
            self.details_label.setText(str(exc))
        selected = self.initial_filters if restore_initial_filters else (
            str(self.product_filter.currentData() or ""),
            str(self.area_filter.currentData() or ""),
            str(self.inference_filter.currentData() or ""),
        )
        self._replace_filter(
            self.product_filter,
            sorted({record.product for record in self.records}),
            selected[0],
        )
        self._replace_filter(
            self.area_filter,
            sorted({record.area for record in self.records}),
            selected[1],
        )
        self._replace_filter(
            self.inference_filter,
            sorted({record.inference_type for record in self.records}),
            selected[2],
        )
        self._replace_filter(
            self.category_filter,
            sorted({record.category for record in self.records}),
            "",
            labels=_CATEGORY_LABELS,
        )
        self._replace_filter(
            self.type_filter,
            sorted({record.component_type for record in self.records}),
            "",
        )
        for combo in (
            self.product_filter,
            self.area_filter,
            self.inference_filter,
            self.category_filter,
            self.type_filter,
        ):
            try:
                combo.currentIndexChanged.disconnect(self._apply_filters)
            except TypeError:
                pass
            combo.currentIndexChanged.connect(self._apply_filters)
        self._apply_filters()

    @staticmethod
    def _replace_filter(
        combo: QComboBox,
        values: list[str],
        selected: str,
        *,
        labels: dict[str, str] | None = None,
    ) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("全部", "")
        for value in values:
            combo.addItem((labels or {}).get(value, value), value)
        index = combo.findData(selected)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _apply_filters(self, _index: int | None = None) -> None:
        values = (
            str(self.product_filter.currentData() or ""),
            str(self.area_filter.currentData() or ""),
            str(self.inference_filter.currentData() or ""),
            str(self.category_filter.currentData() or ""),
            str(self.type_filter.currentData() or ""),
        )
        self.visible_records = tuple(
            record
            for record in self.records
            if (not values[0] or record.product == values[0])
            and (not values[1] or record.area == values[1])
            and (not values[2] or record.inference_type == values[2])
            and (not values[3] or record.category == values[3])
            and (not values[4] or record.component_type == values[4])
        )
        self._populate_table()

    def _populate_table(self) -> None:
        self.table.setRowCount(len(self.visible_records))
        for row, record in enumerate(self.visible_records):
            values = (
                _STATUS_LABELS.get(record.status, record.status),
                _CATEGORY_LABELS.get(record.category, record.category),
                record.product,
                record.area,
                record.inference_type,
                record.component_type,
                record.version,
                record.created_at.replace("T", " ")[:19] or "—",
                _INTEGRITY_LABELS.get(record.integrity, record.integrity),
                str(record.source_path),
            )
            background = (
                QColor("#dff3e4") if record.status == "DEPLOYED" else None
            )
            if record.status == "DEFAULT":
                background = QColor("#eaf3ff")
            if record.status == "REVOKED" or not record.can_compose:
                background = QColor("#fde2e2")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(COMPONENT_ROLE, record)
                if background is not None:
                    item.setBackground(background)
                self.table.setItem(row, column, item)
        self.summary_label.setText(
            f"顯示 {len(self.visible_records)}／{len(self.records)} 個模型與顏色版本"
        )
        if self.visible_records:
            self.table.selectRow(0)
        else:
            self.compose_button.setEnabled(False)
            self.details_label.setText("目前篩選條件下沒有模型或顏色版本。")

    def _selected_record(self) -> InspectionComponentRecord | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        value = item.data(COMPONENT_ROLE) if item else None
        return value if isinstance(value, InspectionComponentRecord) else None

    def _update_details(self) -> None:
        record = self._selected_record()
        if record is None:
            self.compose_button.setEnabled(False)
            self.training_images_button.setEnabled(False)
            return
        self.compose_button.setEnabled(record.can_compose)
        self.training_images_button.setEnabled(record.has_training_provenance)
        self.details_label.setText(
            f"{_CATEGORY_LABELS.get(record.category, record.category)}／"
            f"{record.component_type}／{record.version}\n"
            f"訓練資料：{self._provenance_summary(record)}\n{record.detail}"
        )

    @staticmethod
    def _provenance_summary(record: InspectionComponentRecord) -> str:
        """Describe the training set without overstating what is known.

        Components deployed before provenance recording report "not
        recorded" rather than a hash: their old dataset_hash digests paths
        and mtimes and cannot be resolved back to any set of images.
        """
        if record.category != "AI_MODEL":
            return "不適用"
        if not record.has_training_provenance:
            return "無記錄"
        parts = [record.dataset_id[:12]]
        if record.dataset_image_count:
            parts.append(f"{record.dataset_image_count} 張")
        if record.provenance_is_inferred:
            parts.append("推測")
        return "｜".join(parts)

    def _open_training_images(self) -> None:
        """Show the photo batch behind the selected model.

        The dataset ID is the authoritative identity, but the images
        themselves are only listed in the immutable per-batch manifest that
        submission history keeps, which is keyed by job.
        """
        record = self._selected_record()
        if record is None or not record.has_training_provenance:
            return
        if self.training_data_root is None:
            QMessageBox.information(
                self,
                "訓練照片",
                "此視窗未取得訓練資料位置，無法顯示照片。",
            )
            return
        from app.gui.training_batch_dialog import TrainingBatchDialog
        from tools.submission_history import (
            load_submission_entries,
            load_submission_history,
        )

        try:
            history = load_submission_history(self.training_data_root)
        except (OSError, RuntimeError, ValueError) as exc:
            QMessageBox.warning(self, "訓練照片", f"無法讀取送出歷史：{exc}")
            return
        match = next(
            (
                item
                for item in history
                if record.training_job_id and item.job_id == record.training_job_id
            ),
            None,
        )
        if match is None:
            QMessageBox.information(
                self,
                "訓練照片",
                "這個模型有資料集識別碼，但找不到對應的送出批次；"
                "照片可能已從訓練專案移除。\n"
                f"資料集：{record.dataset_id[:12]}",
            )
            return
        dialog = TrainingBatchDialog(
            load_submission_entries(match),
            history_mode=True,
            parent=self,
        )
        prefix = match.batch_version or record.dataset_id[:12]
        if record.provenance_is_inferred:
            prefix += "（推測）"
        dialog.setWindowTitle(f"{prefix}｜{dialog.windowTitle()}")
        dialog.exec_()

    def _create_combination(self) -> None:
        record = self._selected_record()
        if record is None or not record.can_compose:
            return
        if self.on_create_combination is not None:
            self.on_create_combination(record)
