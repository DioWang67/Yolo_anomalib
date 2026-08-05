"""Read-only drill-down for evidence excluded from color baseline rebuilding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
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

from core.services.color_baseline_evidence import (
    ColorBaselineEvidenceExclusion,
)

_PATH_ROLE = Qt.UserRole
_SOURCE_LABELS = {
    "acceptance": "驗收資料",
    "color_review": "顏色覆核",
    "merged": "跨來源真值",
}


class ColorBaselineExclusionsDialog(QDialog):
    """List excluded evidence with exact lineage and an optional image link."""

    def __init__(
        self,
        exclusions: Sequence[ColorBaselineEvidenceExclusion],
        *,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._exclusions = tuple(exclusions)
        self.setWindowTitle("完整顏色基準：排除照片")
        self.resize(980, 430)

        layout = QVBoxLayout(self)
        self.description_label = QLabel(
            "以下照片未納入本次完整顏色基準；原始照片與人工真值均未修改。"
        )
        self.description_label.setWordWrap(True)
        layout.addWidget(self.description_label)

        self.table = QTableWidget(len(self._exclusions), 4)
        self.table.setHorizontalHeaderLabels(
            ("來源", "Sample ID", "排除原因", "影像檔")
        )
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        for row, exclusion in enumerate(self._exclusions):
            values = (
                _SOURCE_LABELS.get(exclusion.source_kind, exclusion.source_kind),
                exclusion.sample_id or "—",
                exclusion.reason,
                exclusion.image_path or "—",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                tooltip = value
                if column == 0:
                    tooltip = f"{value}\n來源檔：{exclusion.source_manifest or '—'}"
                elif column == 2:
                    tooltip = f"{value}\n內部原因代碼：{exclusion.reason_code}"
                item.setToolTip(tooltip)
                item.setData(_PATH_ROLE, exclusion.image_path)
                self.table.setItem(row, column, item)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.open_image_button = QPushButton("開啟選取照片")
        self.open_image_button.setEnabled(False)
        self.open_image_button.clicked.connect(self._open_selected_image)
        actions.addWidget(self.open_image_button)
        close_button = QPushButton("關閉")
        close_button.clicked.connect(self.accept)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        self.table.itemSelectionChanged.connect(self._update_actions)
        self.table.itemDoubleClicked.connect(
            lambda _item: self._open_selected_image()
        )
        if self._exclusions:
            self.table.selectRow(0)

    def _selected_path(self) -> Path | None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        raw_path = str(item.data(_PATH_ROLE) or "") if item else ""
        return Path(raw_path).expanduser() if raw_path else None

    def _update_actions(self) -> None:
        path = self._selected_path()
        self.open_image_button.setEnabled(
            bool(path and path.is_file() and not path.is_symlink())
        )

    def _open_selected_image(self) -> None:
        path = self._selected_path()
        if path is None or path.is_symlink() or not path.is_file():
            QMessageBox.warning(
                self,
                "無法開啟照片",
                "選取項目沒有可讀取的本機影像檔。",
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.resolve())))
