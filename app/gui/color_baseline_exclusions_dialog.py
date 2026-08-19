"""Read-only drill-down for evidence excluded from color baseline rebuilding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QDesktopServices, QImageReader
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

_SOURCE_LABELS = {
    "acceptance": "驗收資料",
    "color_review": "顏色覆核",
    "merged": "跨來源真值",
}
_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_IMAGE_SOURCE_KINDS = frozenset({"acceptance", "color_review"})


def _trusted_image_path(
    exclusion: ColorBaselineEvidenceExclusion,
) -> Path | None:
    """Resolve one decodable image without escaping its evidence source root."""
    if exclusion.source_kind not in _IMAGE_SOURCE_KINDS:
        return None
    if not exclusion.source_manifest or not exclusion.image_path:
        return None

    try:
        unresolved_manifest = Path(exclusion.source_manifest).expanduser()
        if unresolved_manifest.is_symlink():
            return None
        manifest_path = unresolved_manifest.resolve(strict=True)
        if not manifest_path.is_file():
            return None

        source_root = manifest_path.parent
        unresolved_image = Path(exclusion.image_path).expanduser()
        if not unresolved_image.is_absolute():
            unresolved_image = source_root / unresolved_image
        if unresolved_image.is_symlink():
            return None
        image_path = unresolved_image.resolve(strict=True)
        image_path.relative_to(source_root)
        if image_path.suffix.casefold() not in _IMAGE_SUFFIXES:
            return None
        if not image_path.is_file():
            return None
    except (OSError, RuntimeError, ValueError):
        return None

    reader = QImageReader(str(image_path))
    reader.setDecideFormatFromContent(True)
    return image_path if not reader.read().isNull() else None


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

    def _selected_exclusion(self) -> ColorBaselineEvidenceExclusion | None:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._exclusions):
            return None
        return self._exclusions[row]

    def _selected_path(self) -> Path | None:
        exclusion = self._selected_exclusion()
        return _trusted_image_path(exclusion) if exclusion is not None else None

    def _update_actions(self) -> None:
        self.open_image_button.setEnabled(self._selected_path() is not None)

    def _open_selected_image(self) -> None:
        path = self._selected_path()
        if path is None:
            QMessageBox.warning(
                self,
                "無法開啟照片",
                "選取項目沒有位於來源資料夾內且可驗證的影像檔。",
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            QMessageBox.warning(
                self,
                "無法開啟照片",
                "系統沒有可用的影像檢視器。",
            )
