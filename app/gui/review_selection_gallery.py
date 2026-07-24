"""Reusable thumbnail gallery for choosing failed cases that need review."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.gui.async_image_service import (
    IMAGE_STATUS_CORRUPT,
    IMAGE_STATUS_LOADED,
    IMAGE_STATUS_LOADING,
    IMAGE_STATUS_MISSING,
    IMAGE_STATUS_PERMISSION_DENIED,
    IMAGE_STATUS_UNEXPECTED,
    IMAGE_STATUS_UNSUPPORTED,
    AsyncImageService,
    ImageLoadResult,
)
from app.gui.dialog_geometry import configure_responsive_dialog
from tools.review_workflow import WorkflowState, derive_workflow_state

ROW_INDEX_ROLE = Qt.UserRole
PREVIEW_PATH_ROLE = Qt.UserRole + 1
BASE_TEXT_ROLE = Qt.UserRole + 2
REVIEWED_ROLE = Qt.UserRole + 3
THUMBNAIL_STATE_ROLE = Qt.UserRole + 4
THUMBNAIL_REQUESTED_ROLE = Qt.UserRole + 5
IMAGE_SHA_ROLE = Qt.UserRole + 6
THUMBNAIL_PREFETCH_ITEMS = 16
GALLERY_PAGE_SIZE = 100
ALL_REASON_FILTER = "__all_reasons__"
UNCLASSIFIED_REASON_FILTER = "__unclassified_reason__"
_PLACEHOLDER_ICONS: dict[tuple[str, int, int], QIcon] = {}
UNREVIEWED_WORKFLOW_STATES = frozenset(
    {WorkflowState.NEW, WorkflowState.SELECTED, WorkflowState.IN_REVIEW}
)
SYSTEM_REASON_LABELS = {
    "MISSING": ("缺件", "Missing component"),
    "UNEXPECTED_COMPONENT": ("多餘元件", "Unexpected component"),
    "POSITION_SHIFT": ("位置偏移", "Position shift"),
    "COLOR_MISMATCH": ("顏色不符", "Color mismatch"),
    "SEQUENCE_MISMATCH": ("順序不符", "Sequence mismatch"),
    "BOARD_ALIGNMENT": ("板件定位異常", "Board alignment"),
}
FAILURE_REASON_LABELS = {
    "misclassification": ("誤判／過殺", "Misclassification / false reject"),
    "missed_detection": ("漏判／漏檢", "Missed detection"),
    "new_defect_type": ("新缺陷類型", "New defect type"),
    "wrong_box": ("框的位置或數量錯誤", "Wrong box position or count"),
    "wrong_class": ("類別判定錯誤", "Wrong class"),
    "threshold_not_met": ("閾值未達標", "Threshold not met"),
    "color_issue": ("顏色判定問題", "Color issue"),
    "lighting_issue": ("光源／曝光問題", "Lighting / exposure issue"),
    "other": ("其他人工原因", "Other operator reason"),
}
SKIP_REASON_LABELS = {
    "unjudgeable": ("略過：無法判定", "Skipped: unable to judge"),
    "image_quality": ("略過：圖片品質不良", "Skipped: poor image quality"),
    "image_quality_issue": ("略過：影像品質有問題", "Skipped: image quality issue"),
    "equipment_lighting": ("略過：設備／光源問題", "Skipped: equipment / lighting"),
    "confirmed_failure": ("略過：確認為既有失敗", "Skipped: confirmed failure"),
    "other": ("略過：其他原因", "Skipped: other reason"),
}


class ReviewSelectionGallery(QWidget):
    """Show failure evidence as a dense, checkable thumbnail grid."""

    selection_changed = pyqtSignal(object)

    def __init__(
        self,
        *,
        language: str = "zh_TW",
        selected_only: bool = False,
        image_service: AsyncImageService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.language = language
        self.selected_only = selected_only
        self._source_entries: list[tuple[int, dict[str, str]]] = []
        self._selected_indices: set[int] = set()
        self._entry_indices: set[int] = set()
        self._reason_keys_by_index: dict[int, tuple[str, ...]] = {}
        self._loaded_thumbnail_indices: set[int] = set()
        self._resident_thumbnail_indices: set[int] = set()
        self._generation = 0
        self._render_limit = GALLERY_PAGE_SIZE
        self._owns_image_service = image_service is None
        self.image_service = image_service or AsyncImageService.from_environment(
            parent=self
        )
        self.image_service.result_ready.connect(self._on_image_result)
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.timeout.connect(self._request_visible_thumbnails)
        self._build_ui()

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-size:11pt;font-weight:bold;color:#243447;")
        toolbar.addWidget(self.summary_label)
        toolbar.addStretch()
        self.filter_combo = QComboBox()
        for value, zh_label, en_label in (
            ("all", "全部", "All"),
            ("selected", "已選", "Selected"),
            ("unselected", "未選", "Unselected"),
            ("reviewed", "已複核", "Reviewed"),
            ("unreviewed", "未複核", "Unreviewed"),
        ):
            self.filter_combo.addItem(self._text(zh_label, en_label), value)
        self.filter_combo.currentIndexChanged.connect(self._reset_render_limit)
        self.filter_combo.setVisible(not self.selected_only)
        self.reason_filter_label = QLabel(self._text("原因", "Reason"))
        self.reason_filter_combo = QComboBox()
        self.reason_filter_combo.setMinimumWidth(210)
        self.reason_filter_combo.addItem(
            self._text("全部原因", "All reasons"),
            ALL_REASON_FILTER,
        )
        self.reason_filter_combo.currentIndexChanged.connect(
            self._reset_render_limit
        )
        self.select_all_button = QPushButton(
            self._text("全選目前結果", "Select all current results")
        )
        self.select_all_button.clicked.connect(
            lambda: self._set_all_check_states(Qt.Checked)
        )
        self.clear_button = QPushButton(
            self._text(
                "移除全部已選圖片" if self.selected_only else "清除目前勾選",
                "Remove all selected images"
                if self.selected_only
                else "Clear visible selection",
            )
        )
        self.clear_button.clicked.connect(
            lambda: self._set_all_check_states(Qt.Unchecked)
        )
        self.select_all_button.setVisible(not self.selected_only)
        toolbar.addWidget(self.reason_filter_label)
        toolbar.addWidget(self.reason_filter_combo)
        toolbar.addWidget(self.filter_combo)
        self.load_more_button = QPushButton()
        self.load_more_button.clicked.connect(self._load_more)
        toolbar.addWidget(self.select_all_button)
        toolbar.addWidget(self.clear_button)
        toolbar.addWidget(self.load_more_button)
        layout.addLayout(toolbar)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListView.IconMode)
        self.thumbnail_list.setResizeMode(QListView.Adjust)
        self.thumbnail_list.setMovement(QListView.Static)
        self.thumbnail_list.setIconSize(QSize(235, 155))
        self.thumbnail_list.setGridSize(QSize(268, 244))
        self.thumbnail_list.setSpacing(8)
        self.thumbnail_list.setWordWrap(True)
        self.thumbnail_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.thumbnail_list.setStyleSheet(
            "QListWidget { background:#f6f8fb;border:1px solid #d0d5dd; }"
            "QListWidget::item { border:1px solid #c8d0da;border-radius:8px;"
            "padding:6px;color:#243447; }"
            "QListWidget::item:hover { border:2px solid #5b8fc9; }"
            "QListWidget::item:selected { border:2px solid #2563a6;color:#243447; }"
            "QListWidget::indicator { width:22px;height:22px; }"
        )
        self.thumbnail_list.itemChanged.connect(self._on_item_changed)
        self.thumbnail_list.itemDoubleClicked.connect(self._show_large_preview)
        self.thumbnail_list.verticalScrollBar().valueChanged.connect(
            self._schedule_visible_thumbnails
        )
        self.thumbnail_list.viewport().installEventFilter(self)
        layout.addWidget(self.thumbnail_list, 1)

    def set_entries(
        self,
        entries: list[tuple[int, dict[str, str]]],
        selected_indices: set[int],
    ) -> None:
        """Replace source entries while restoring their persisted selection."""
        self._source_entries = list(entries)
        self._selected_indices = set(selected_indices)
        self._render_limit = GALLERY_PAGE_SIZE
        self._reason_keys_by_index = {
            row_index: record_reason_keys(row)
            for row_index, row in self._source_entries
        }
        self._refresh_reason_filter_options()
        self._render_entries()

    def _refresh_reason_filter_options(self) -> None:
        """Rebuild dynamic reason choices without changing the active choice."""
        selected_reason = str(
            self.reason_filter_combo.currentData() or ALL_REASON_FILTER
        )
        counts: dict[str, int] = {}
        for row_index, _row in self._source_entries:
            keys = self._reason_keys_by_index.get(row_index, ())
            if not keys:
                keys = (UNCLASSIFIED_REASON_FILTER,)
            for key in keys:
                counts[key] = counts.get(key, 0) + 1

        self.reason_filter_combo.blockSignals(True)
        try:
            self.reason_filter_combo.clear()
            self.reason_filter_combo.addItem(
                self._text("全部原因", "All reasons"),
                ALL_REASON_FILTER,
            )
            options = sorted(
                counts,
                key=lambda key: reason_filter_label(
                    key,
                    language=self.language,
                ).casefold(),
            )
            for key in options:
                label = reason_filter_label(key, language=self.language)
                self.reason_filter_combo.addItem(
                    f"{label} ({counts[key]})",
                    key,
                )
            selected_index = self.reason_filter_combo.findData(selected_reason)
            self.reason_filter_combo.setCurrentIndex(
                selected_index if selected_index >= 0 else 0
            )
        finally:
            self.reason_filter_combo.blockSignals(False)

    def _matches_reason_filter(self, row_index: int, reason_value: str) -> bool:
        if reason_value == ALL_REASON_FILTER:
            return True
        reasons = self._reason_keys_by_index.get(row_index, ())
        if reason_value == UNCLASSIFIED_REASON_FILTER:
            return not reasons
        return reason_value in reasons

    def _reset_render_limit(self, _index: int | None = None) -> None:
        self._render_limit = GALLERY_PAGE_SIZE
        self._render_entries()

    def _load_more(self) -> None:
        self._render_limit += GALLERY_PAGE_SIZE
        self._render_entries()

    def _render_entries(self) -> None:
        """Render the current view filter without changing persisted selection."""
        self._generation += 1
        self._loaded_thumbnail_indices.clear()
        self._resident_thumbnail_indices.clear()
        filter_value = (
            "all" if self.selected_only else str(self.filter_combo.currentData() or "all")
        )
        reason_value = str(
            self.reason_filter_combo.currentData() or ALL_REASON_FILTER
        )
        matching_entries = [
            (row_index, row)
            for row_index, row in self._source_entries
            if self._matches_filter(row_index, row, filter_value)
            and self._matches_reason_filter(row_index, reason_value)
        ]
        entries = matching_entries[: self._render_limit]
        self.thumbnail_list.blockSignals(True)
        try:
            self.thumbnail_list.clear()
            self._entry_indices = {
                index for index, _row in matching_entries
            }
            for row_index, row in entries:
                reviewed = record_is_reviewed(row)
                state_label = self._text("已複核", "Reviewed") if reviewed else self._text("待複核", "Pending")
                status = str(row.get("status") or "FAIL").strip().upper()
                classification = _classification_text(row, language=self.language)
                item = QListWidgetItem(
                    f"【{state_label}｜{status}】\n"
                    f"{row.get('timestamp', '')}\n"
                    f"{row.get('product', '')} / {row.get('area', '')}"
                    f"{classification}"
                )
                item.setData(BASE_TEXT_ROLE, item.text())
                item.setData(REVIEWED_ROLE, reviewed)
                item.setData(ROW_INDEX_ROLE, row_index)
                preview_path = best_review_preview_path(row)
                item.setData(PREVIEW_PATH_ROLE, str(preview_path))
                item.setData(IMAGE_SHA_ROLE, _preview_sha(row))
                item.setData(THUMBNAIL_STATE_ROLE, IMAGE_STATUS_LOADING)
                item.setData(THUMBNAIL_REQUESTED_ROLE, False)
                item.setFlags(
                    item.flags()
                    | Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                    | Qt.ItemIsUserCheckable
                )
                item.setCheckState(
                    Qt.Checked if row_index in self._selected_indices else Qt.Unchecked
                )
                item.setBackground(QColor("#eef6ff" if reviewed else "#fff7ed"))
                item.setForeground(QColor("#243447"))
                self._apply_item_check_style(item)
                item.setIcon(
                    _placeholder_icon(
                        IMAGE_STATUS_LOADING,
                        self.thumbnail_list.iconSize(),
                    )
                )
                self._append_thumbnail_tooltip(item)
                self.thumbnail_list.addItem(item)
        finally:
            self.thumbnail_list.blockSignals(False)
        remaining = max(0, len(matching_entries) - len(entries))
        self.load_more_button.setVisible(remaining > 0)
        self.load_more_button.setText(
            self._text(
                f"再載入 {min(GALLERY_PAGE_SIZE, remaining)} 張（尚有 {remaining}）",
                f"Load {min(GALLERY_PAGE_SIZE, remaining)} more ({remaining} remaining)",
            )
        )
        self._update_summary()
        if self.image_service.synchronous:
            self.request_all_thumbnails()
        else:
            self._schedule_visible_thumbnails()

    def _matches_filter(
        self,
        row_index: int,
        row: dict[str, str],
        filter_value: str,
    ) -> bool:
        if filter_value == "selected":
            return row_index in self._selected_indices
        if filter_value == "unselected":
            return row_index not in self._selected_indices
        if filter_value == "reviewed":
            return record_is_reviewed(row)
        if filter_value == "unreviewed":
            return not record_is_reviewed(row)
        return True

    def entry_indices(self) -> set[int]:
        """Return manifest indices currently displayed by the gallery."""
        return set(self._entry_indices)

    def source_entry_indices(self) -> set[int]:
        """Return every entry in the time-range result, including filtered rows."""
        return {index for index, _row in self._source_entries}

    def selected_indices(self) -> set[int]:
        """Return selected indices in the active filter, including lazy pages."""
        return set(self._selected_indices & self._entry_indices)

    def _items(self) -> list[QListWidgetItem]:
        return [
            self.thumbnail_list.item(index)
            for index in range(self.thumbnail_list.count())
        ]

    def _set_all_check_states(self, state: Qt.CheckState) -> None:
        if state == Qt.Checked:
            self._selected_indices.update(self._entry_indices)
        else:
            self._selected_indices.difference_update(self._entry_indices)
        self.thumbnail_list.blockSignals(True)
        try:
            for item in self._items():
                item.setCheckState(state)
                self._apply_item_check_style(item)
        finally:
            self.thumbnail_list.blockSignals(False)
        self._emit_selection_changed()
        self._render_if_selection_filter_active()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        row_index = int(item.data(ROW_INDEX_ROLE))
        checked = item.checkState() == Qt.Checked

        # Styling changes item data (text, brush and font), each of which can
        # emit itemChanged. Block those nested emissions so a connected slot
        # cannot rebuild the list and delete `item` while this handler is active.
        signals_were_blocked = self.thumbnail_list.blockSignals(True)
        try:
            self._apply_item_check_style(item)
        finally:
            self.thumbnail_list.blockSignals(signals_were_blocked)

        if checked:
            self._selected_indices.add(row_index)
        else:
            self._selected_indices.discard(row_index)

        # Do not access `item` after emitting. Persistence validation may refresh
        # the gallery synchronously and invalidate every QListWidgetItem wrapper.
        self._emit_selection_changed()
        self._render_if_selection_filter_active()

    def _render_if_selection_filter_active(self) -> None:
        if not self.selected_only and self.filter_combo.currentData() in {
            "selected",
            "unselected",
        }:
            self._render_entries()

    def _apply_item_check_style(self, item: QListWidgetItem) -> None:
        """Make a persisted checkbox choice obvious without relying on the OS theme."""
        base_text = str(item.data(BASE_TEXT_ROLE) or item.text())
        checked = item.checkState() == Qt.Checked
        if checked:
            prefix = self._text("✓ 已選取", "✓ SELECTED")
        else:
            prefix = self._text("○ 未選取", "○ NOT SELECTED")
        item.setText(f"{prefix}\n{base_text}")
        item.setBackground(
            QColor(
                "#b7ebc6"
                if checked
                else ("#eef6ff" if bool(item.data(REVIEWED_ROLE)) else "#fff7ed")
            )
        )
        item.setForeground(QColor("#123c23" if checked else "#243447"))
        font = item.font()
        font.setBold(checked)
        item.setFont(font)
        if checked:
            tooltip = self._text(
                "已選取：會保留在人工複核清單中。",
                "Selected: this image is retained in the review queue.",
            )
        else:
            tooltip = self._text(
                "未選取：勾選方框即可加入人工複核清單。",
                "Not selected: check the box to add this image to the review queue.",
            )
        item.setToolTip(tooltip)
        self._append_thumbnail_tooltip(item)

    def _emit_selection_changed(self) -> None:
        selected = self.selected_indices()
        self._update_summary()
        self.selection_changed.emit(selected)

    def _update_summary(self) -> None:
        total = len(self._source_entries)
        matching = len(self._entry_indices)
        rendered = self.thumbnail_list.count()
        source_indices = self.source_entry_indices()
        selected = len(self._selected_indices & source_indices)
        self.summary_label.setText(
            self._text(
                (
                    f"已選圖片 {total} 張｜此畫面不顯示未選圖片"
                    if self.selected_only
                    else f"目前結果 {total} 張｜符合 {matching} 張｜"
                    f"已載入 {rendered} 張｜已選 {selected} 張"
                ),
                (
                    f"{total} selected images | Unselected images are hidden"
                    if self.selected_only
                    else f"{total} results | {matching} match | "
                    f"{rendered} loaded | {selected} selected"
                ),
            )
        )
    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API
        if watched is self.thumbnail_list.viewport() and event.type() in {
            QEvent.Resize,
            QEvent.Show,
        }:
            self._schedule_visible_thumbnails()
        return super().eventFilter(watched, event)

    def _schedule_visible_thumbnails(self) -> None:
        self._thumbnail_timer.start(0)

    def _request_visible_thumbnails(self) -> None:
        if not self.image_service.active or self.thumbnail_list.count() == 0:
            return
        visible_positions = self._visible_item_positions()
        if not visible_positions:
            visible_positions = list(range(min(20, self.thumbnail_list.count())))
        visible_set = set(visible_positions)
        first = max(min(visible_positions) - THUMBNAIL_PREFETCH_ITEMS, 0)
        last = min(
            max(visible_positions) + THUMBNAIL_PREFETCH_ITEMS + 1,
            self.thumbnail_list.count(),
        )
        requested_positions = visible_set | set(range(first, last))
        self._resident_thumbnail_indices = {
            int(self.thumbnail_list.item(position).data(ROW_INDEX_ROLE))
            for position in requested_positions
        }
        self._evict_nonresident_icons()
        for position in sorted(
            requested_positions,
            key=lambda value: value not in visible_set,
        ):
            self._request_thumbnail_at(
                position,
                priority=10 if position in visible_set else 0,
            )

    def request_all_thumbnails(self) -> None:
        """Queue all rendered placeholders for deterministic benchmarks and tests."""
        if self.image_service.synchronous:
            self._resident_thumbnail_indices = {
                int(self.thumbnail_list.item(position).data(ROW_INDEX_ROLE))
                for position in range(self.thumbnail_list.count())
            }
        for position in range(self.thumbnail_list.count()):
            self._request_thumbnail_at(position, priority=-1)

    def _request_thumbnail_at(self, position: int, *, priority: int) -> None:
        item = self.thumbnail_list.item(position)
        if item is None or bool(item.data(THUMBNAIL_REQUESTED_ROLE)):
            return
        path = str(item.data(PREVIEW_PATH_ROLE) or "")
        row_index = int(item.data(ROW_INDEX_ROLE))
        token = (id(self), self._generation, row_index, path)
        signals_were_blocked = self.thumbnail_list.blockSignals(True)
        try:
            item.setData(THUMBNAIL_REQUESTED_ROLE, True)
        finally:
            self.thumbnail_list.blockSignals(signals_were_blocked)
        self.image_service.request_image(
            path,
            purpose="thumbnail",
            token=token,
            target_size=self.thumbnail_list.iconSize(),
            image_sha=str(item.data(IMAGE_SHA_ROLE) or ""),
            priority=priority,
        )

    def _visible_item_positions(self) -> list[int]:
        viewport_rect = self.thumbnail_list.viewport().rect()
        return [
            position
            for position in range(self.thumbnail_list.count())
            if self.thumbnail_list.visualItemRect(
                self.thumbnail_list.item(position)
            ).intersects(viewport_rect)
        ]

    def _on_image_result(self, result: ImageLoadResult) -> None:
        if (
            result.purpose != "thumbnail"
            or not isinstance(result.token, tuple)
            or len(result.token) != 4
        ):
            return
        owner, generation, row_index, path = result.token
        if owner != id(self) or generation != self._generation:
            return
        item = self._find_item(row_index=row_index, path=path)
        if item is None:
            return
        self._loaded_thumbnail_indices.add(row_index)
        signals_were_blocked = self.thumbnail_list.blockSignals(True)
        try:
            if row_index not in self._resident_thumbnail_indices:
                item.setData(THUMBNAIL_REQUESTED_ROLE, False)
                item.setData(THUMBNAIL_STATE_ROLE, IMAGE_STATUS_LOADING)
                item.setIcon(
                    _placeholder_icon(
                        IMAGE_STATUS_LOADING,
                        self.thumbnail_list.iconSize(),
                    )
                )
                return
            item.setData(THUMBNAIL_STATE_ROLE, result.status)
            if result.status == IMAGE_STATUS_LOADED and result.image is not None:
                item.setIcon(QIcon(QPixmap.fromImage(result.image)))
            else:
                item.setIcon(
                    _placeholder_icon(result.status, self.thumbnail_list.iconSize())
                )
            self._apply_item_check_style(item)
        finally:
            self.thumbnail_list.blockSignals(signals_were_blocked)

    def _evict_nonresident_icons(self) -> None:
        signals_were_blocked = self.thumbnail_list.blockSignals(True)
        try:
            for position in range(self.thumbnail_list.count()):
                item = self.thumbnail_list.item(position)
                row_index = int(item.data(ROW_INDEX_ROLE))
                if row_index in self._resident_thumbnail_indices:
                    continue
                if str(item.data(THUMBNAIL_STATE_ROLE)) != IMAGE_STATUS_LOADING:
                    item.setData(THUMBNAIL_STATE_ROLE, IMAGE_STATUS_LOADING)
                    item.setData(THUMBNAIL_REQUESTED_ROLE, False)
                    item.setIcon(
                        _placeholder_icon(
                            IMAGE_STATUS_LOADING,
                            self.thumbnail_list.iconSize(),
                        )
                    )
        finally:
            self.thumbnail_list.blockSignals(signals_were_blocked)

    def _find_item(self, *, row_index: int, path: str) -> QListWidgetItem | None:
        for position in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(position)
            if (
                int(item.data(ROW_INDEX_ROLE)) == row_index
                and str(item.data(PREVIEW_PATH_ROLE) or "") == path
            ):
                return item
        return None

    def _append_thumbnail_tooltip(self, item: QListWidgetItem) -> None:
        state = str(item.data(THUMBNAIL_STATE_ROLE) or IMAGE_STATUS_LOADING)
        message = _image_state_text(state, language=self.language)
        if not message:
            return
        base = item.toolTip().split("\nImage: ", 1)[0].split("\n影像：", 1)[0]
        label = "影像：" if str(self.language).lower().startswith("zh") else "Image: "
        item.setToolTip(f"{base}\n{label}{message}")

    @property
    def loaded_thumbnail_count(self) -> int:
        return len(self._loaded_thumbnail_indices)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._generation += 1
        self._thumbnail_timer.stop()
        if self._owns_image_service:
            self.image_service.shutdown()
        super().closeEvent(event)

    def _show_large_preview(self, item: QListWidgetItem) -> None:
        path = Path(str(item.data(PREVIEW_PATH_ROLE) or ""))
        preview = _AsyncLargePreviewDialog(
            path=path,
            title=item.text().replace("\n", " | "),
            image_service=self.image_service,
            language=self.language,
            parent=self,
        )
        preview.exec_()


class _AsyncLargePreviewDialog(QDialog):
    """Modal shell whose image decode remains in the shared bounded service."""

    def __init__(
        self,
        *,
        path: Path,
        title: str,
        image_service: AsyncImageService,
        language: str,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        self._path = str(path)
        self._owner_token = id(self)
        self._image_service = image_service
        self._language = language
        self.setWindowTitle(title)
        configure_responsive_dialog(
            self,
            preferred=(1200, 800),
            minimum=(600, 420),
            parent=parent,
        )
        layout = QVBoxLayout(self)
        self.label = QLabel(
            "圖片載入中…" if language.lower().startswith("zh") else "Loading image…"
        )
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        self._image_service.result_ready.connect(self._on_result)
        self._image_service.request_image(
            path,
            purpose="full",
            token=(self._owner_token, self._path),
            priority=20,
        )

    def _on_result(self, result: ImageLoadResult) -> None:
        if (
            result.purpose != "full"
            or result.token != (self._owner_token, self._path)
        ):
            return
        if result.status == IMAGE_STATUS_LOADED and result.image is not None:
            pixmap = QPixmap.fromImage(result.image)
            self.label.setText("")
            self.label.setPixmap(
                pixmap.scaled(
                    QSize(1160, 740),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
            return
        self.label.setText(_image_state_text(result.status, language=self._language))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._owner_token = -1
        try:
            self._image_service.result_ready.disconnect(self._on_result)
        except TypeError:
            pass
        super().closeEvent(event)


def record_reason_keys(row: dict[str, Any]) -> tuple[str, ...]:
    """Return stable structured reason keys for one legacy or current row."""
    keys: list[str] = []
    failure_category = str(row.get("failure_category") or "").strip().lower()
    if failure_category:
        keys.append(f"failure:{failure_category}")

    for decision_reason in _split_decision_reasons(row.get("decision_reasons")):
        key = f"system:{decision_reason.upper()}"
        if key not in keys:
            keys.append(key)

    skip_reason = str(row.get("skip_reason") or "").strip().lower()
    if skip_reason:
        key = f"skip:{skip_reason}"
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def reason_filter_label(reason_key: str, *, language: str) -> str:
    """Translate a stable reason key while preserving unknown future codes."""
    is_zh = str(language).lower().startswith("zh")
    if reason_key == UNCLASSIFIED_REASON_FILTER:
        return "未標記原因" if is_zh else "Unclassified reason"
    source, separator, value = str(reason_key).partition(":")
    if not separator:
        return str(reason_key)
    labels: tuple[str, str] | None = None
    if source == "system":
        labels = SYSTEM_REASON_LABELS.get(value.upper())
    elif source == "failure":
        labels = FAILURE_REASON_LABELS.get(value.lower())
    elif source == "skip":
        labels = SKIP_REASON_LABELS.get(value.lower())
    if labels is not None:
        return labels[0] if is_zh else labels[1]
    return value if is_zh else value.replace("_", " ").title()


def _split_decision_reasons(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        text = str(value or "").strip()
        if not text:
            return ()
        raw_values: list[Any]
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            raw_values = (
                list(parsed)
                if isinstance(parsed, list)
                else re.split(r"[|,;]", text.strip("[]"))
            )
        else:
            raw_values = re.split(r"[|,;]", text)
    reasons: list[str] = []
    for raw in raw_values:
        reason = str(raw or "").strip().strip("'\"")
        if not reason or reason.upper() in {"NONE", "N/A", "NULL"}:
            continue
        if reason not in reasons:
            reasons.append(reason)
    return tuple(reasons)


def record_is_reviewed(row: dict[str, Any]) -> bool:
    """Group review completion from the Phase 1B derived workflow state."""
    return derive_workflow_state(row) not in UNREVIEWED_WORKFLOW_STATES


def best_review_preview_path(row: dict[str, Any]) -> Path:
    """Return the strongest available visual evidence for overview selection."""
    for key in ("annotated_path", "preprocessed_path", "original_path"):
        path = Path(str(row.get(key) or ""))
        if path.is_file():
            return path
    return Path()


def _classification_text(row: dict[str, Any], *, language: str) -> str:
    """Return a compact independent failure-cause tag for one thumbnail."""
    if str(row.get("failure_category") or "") != "threshold_not_met":
        return ""
    source = str(row.get("failure_source") or "").strip()
    source_labels = {
        "yolo": "YOLO",
        "color": "顏色" if str(language).lower().startswith("zh") else "Color",
    }
    source_label = source_labels.get(source, source or "-")
    category_label = (
        "閾值未達標"
        if str(language).lower().startswith("zh")
        else "Threshold not met"
    )
    return f"\n【{category_label}｜{source_label}】"


def _preview_sha(row: dict[str, Any]) -> str:
    return str(
        row.get("annotated_sha256")
        or row.get("image_sha")
        or row.get("original_sha256")
        or ""
    )


def _placeholder_icon(status: str, size: QSize) -> QIcon:
    cache_key = (status, size.width(), size.height())
    cached = _PLACEHOLDER_ICONS.get(cache_key)
    if cached is not None:
        return cached
    colors = {
        IMAGE_STATUS_LOADING: "#d0d5dd",
        IMAGE_STATUS_MISSING: "#f59e0b",
        IMAGE_STATUS_PERMISSION_DENIED: "#dc2626",
        IMAGE_STATUS_UNSUPPORTED: "#7c3aed",
        IMAGE_STATUS_CORRUPT: "#b91c1c",
        IMAGE_STATUS_UNEXPECTED: "#991b1b",
    }
    pixmap = QPixmap(max(size.width(), 1), max(size.height(), 1))
    pixmap.fill(QColor(colors.get(status, "#d0d5dd")))
    painter = QPainter(pixmap)
    try:
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            pixmap.rect(),
            Qt.AlignCenter,
            status.replace("_", " ").upper(),
        )
    finally:
        painter.end()
    icon = QIcon(pixmap)
    _PLACEHOLDER_ICONS[cache_key] = icon
    return icon


def _image_state_text(status: str, *, language: str) -> str:
    zh = str(language).lower().startswith("zh")
    messages = {
        IMAGE_STATUS_LOADING: ("載入中", "Loading"),
        IMAGE_STATUS_MISSING: ("路徑不存在", "Path missing"),
        IMAGE_STATUS_PERMISSION_DENIED: ("沒有讀取權限", "Permission denied"),
        IMAGE_STATUS_UNSUPPORTED: ("不支援的格式", "Unsupported format"),
        IMAGE_STATUS_CORRUPT: ("圖片損毀", "Corrupt image"),
        IMAGE_STATUS_UNEXPECTED: ("載入錯誤", "Loader error"),
    }
    value = messages.get(status)
    return "" if value is None else value[0 if zh else 1]
