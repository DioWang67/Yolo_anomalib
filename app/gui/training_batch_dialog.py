"""Batch confirmation dialog for inference cases selected for retraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

DIRECT_TRAIN_LABELS = {"confirmed_ng", "verified_empty"}
ANNOTATION_LABELS = {"false_positive", "false_negative", "wrong_class"}
BATCH_ACTION_LABELS = {
    "confirmed_ng": ("辨識結果正確", "Detection is correct"),
    "verified_empty": ("影像中沒有目標", "No target in image"),
    "false_positive": ("框的位置或數量錯誤", "Box position or count is wrong"),
    "false_negative": ("有目標，但系統沒有框", "Target exists but no box was detected"),
    "wrong_class": ("框的類別錯誤", "Box class is wrong"),
    "uncertain": ("舊版未判定（不送訓）", "Legacy undecided (do not train)"),
    "image_quality_issue": (
        "影像過曝／模糊／遮擋",
        "Overexposed, blurred, or obstructed image",
    ),
}

ROW_INDEX_ROLE = Qt.UserRole
PREVIEW_PATH_ROLE = Qt.UserRole + 1
CATEGORY_ROLE = Qt.UserRole + 2


class TrainingBatchDialog(QDialog):
    """Display all reviewed candidates as a selectable thumbnail grid.

    Args:
        entries: Pairs of manifest row index and manifest row data.
        language: UI language code.
        submit_mode: Require at least one selected candidate when true.
        queue_mode: Show separate direct-train and annotation actions.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        entries: list[tuple[int, dict[str, str]]],
        *,
        language: str = "zh_TW",
        submit_mode: bool = False,
        queue_mode: bool = False,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.entries = entries
        self.language = language
        self.submit_mode = submit_mode
        self.queue_mode = queue_mode
        self.selected_action: str | None = None
        self.setWindowTitle(
            self._text(
                "已選擇補訓清單" if self.queue_mode else "產線模型補訓｜送出確認",
                "Selected Retraining Queue"
                if self.queue_mode
                else "Production Retraining | Submission Review",
            )
        )
        self.resize(1500, 900)
        self.setWindowState(self.windowState() | Qt.WindowMaximized)
        self._build_ui()
        self._populate_items()
        self._update_summary()

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel(
            self._text(
                (
                    "這裡保留所有日期已選擇的照片。可直接送出已確認照片，"
                    "或開始補標註；取消勾選可排除照片，雙擊可放大。"
                    if self.queue_mode
                    else "確認本次要送入模型補訓的影像。取消勾選可排除誤入資料；雙擊可放大。"
                ),
                (
                    "This queue keeps selected images from every date. Send confirmed "
                    "images directly or start annotation; uncheck to exclude and "
                    "double-click to enlarge."
                    if self.queue_mode
                    else "Confirm images for model retraining. Uncheck accidental "
                    "entries; double-click to enlarge."
                ),
            )
        )
        title.setWordWrap(True)
        title.setStyleSheet(
            "QLabel { background: #243447; color: white; padding: 12px; "
            "font-size: 13pt; font-weight: bold; }"
        )
        layout.addWidget(title)

        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel(self._text("顯示：", "Show:")))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem(self._text("全部", "All"), "all")
        self.filter_combo.addItem(self._text("可直接訓練", "Direct training"), "direct")
        self.filter_combo.addItem(self._text("需要補標", "Needs annotation"), "annotation")
        self.filter_combo.addItem(self._text("暫不送訓", "On hold"), "hold")
        self.filter_combo.addItem(self._text("已排除", "Excluded"), "excluded")
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_combo)
        toolbar.addStretch()
        include_button = QPushButton(self._text("目前顯示全部加入", "Include visible"))
        include_button.clicked.connect(lambda: self._set_visible_check_state(Qt.Checked))
        exclude_button = QPushButton(self._text("排除目前顯示", "Exclude visible"))
        exclude_button.clicked.connect(
            lambda: self._set_visible_check_state(Qt.Unchecked)
        )
        toolbar.addWidget(include_button)
        toolbar.addWidget(exclude_button)
        layout.addLayout(toolbar)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListView.IconMode)
        self.thumbnail_list.setResizeMode(QListView.Adjust)
        self.thumbnail_list.setMovement(QListView.Static)
        self.thumbnail_list.setIconSize(QSize(245, 165))
        self.thumbnail_list.setGridSize(QSize(280, 235))
        self.thumbnail_list.setSpacing(8)
        self.thumbnail_list.setWordWrap(True)
        self.thumbnail_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.thumbnail_list.itemChanged.connect(self._on_item_changed)
        self.thumbnail_list.itemDoubleClicked.connect(self._show_large_preview)
        layout.addWidget(self.thumbnail_list, 1)

        footer = QHBoxLayout()
        self.summary_label = QLabel()
        self.summary_label.setStyleSheet("font-size: 11pt; font-weight: bold;")
        footer.addWidget(self.summary_label)
        footer.addStretch()
        cancel_button = QPushButton(
            self._text(
                "儲存清單" if self.queue_mode else "返回",
                "Save list" if self.queue_mode else "Back",
            )
        )
        cancel_button.clicked.connect(self.accept if self.queue_mode else self.reject)
        footer.addWidget(cancel_button)
        self.direct_action_button: QPushButton | None = None
        self.annotation_action_button: QPushButton | None = None
        if self.queue_mode:
            self.direct_action_button = QPushButton(
                self._text("送出可直接訓練照片", "Send direct-training images")
            )
            self.annotation_action_button = QPushButton(
                self._text(
                    "開始補標註（完成後自動送訓）",
                    "Annotate, then train automatically",
                )
            )
            for button in (
                self.direct_action_button,
                self.annotation_action_button,
            ):
                button.setMinimumHeight(42)
            self.direct_action_button.setStyleSheet(
                "QPushButton { background: #237a3b; color: white; "
                "font-weight: bold; padding: 8px 18px; }"
            )
            self.annotation_action_button.setStyleSheet(
                "QPushButton { background: #ad641f; color: white; "
                "font-weight: bold; padding: 8px 18px; }"
            )
            self.direct_action_button.clicked.connect(
                lambda: self._confirm_queue_action("direct")
            )
            self.annotation_action_button.clicked.connect(
                lambda: self._confirm_queue_action("annotation")
            )
            footer.addWidget(self.direct_action_button)
            footer.addWidget(self.annotation_action_button)
            self.confirm_button = self.direct_action_button
        else:
            self.confirm_button = QPushButton(
                self._text(
                    "確認送出" if self.submit_mode else "儲存並返回",
                    "Confirm submission" if self.submit_mode else "Save and return",
                )
            )
            self.confirm_button.setMinimumHeight(42)
            self.confirm_button.setStyleSheet(
                "QPushButton { background: #237a3b; color: white; "
                "font-weight: bold; padding: 8px 18px; }"
            )
            self.confirm_button.clicked.connect(self._confirm)
            footer.addWidget(self.confirm_button)
        layout.addLayout(footer)

    def _populate_items(self) -> None:
        self.thumbnail_list.blockSignals(True)
        try:
            for row_index, row in self.entries:
                review_label = str(row.get("review_label") or "")
                if review_label in DIRECT_TRAIN_LABELS:
                    category = "direct"
                elif review_label in ANNOTATION_LABELS:
                    category = "annotation"
                else:
                    category = "hold"
                action_labels = BATCH_ACTION_LABELS.get(
                    review_label, ("尚未確認", "Pending")
                )
                category_labels = {
                    "direct": ("可直接訓練", "Direct training"),
                    "annotation": ("需要補標", "Needs annotation"),
                    "hold": ("暫不送訓", "On hold"),
                }
                category_label = self._text(*category_labels[category])
                item = QListWidgetItem(
                    self._text(*action_labels)
                    + "\n"
                    + category_label
                    + "\n"
                    + str(row.get("timestamp") or "")
                )
                item.setData(ROW_INDEX_ROLE, row_index)
                preview_path = _best_preview_path(row)
                item.setData(PREVIEW_PATH_ROLE, str(preview_path))
                item.setData(CATEGORY_ROLE, category)
                item_flags = item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                if category != "hold":
                    item_flags |= Qt.ItemIsUserCheckable
                item.setFlags(item_flags)
                item.setCheckState(
                    Qt.Unchecked
                    if category == "hold"
                    or str(row.get("training_selected") or "1") == "0"
                    else Qt.Checked
                )
                if category == "hold":
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                background_colors = {
                    "direct": "#e8f5e9",
                    "annotation": "#fff3e0",
                    "hold": "#ffebee",
                }
                item.setBackground(QColor(background_colors[category]))
                pixmap = QPixmap(str(preview_path)) if preview_path.is_file() else QPixmap()
                if not pixmap.isNull():
                    item.setIcon(
                        QIcon(
                            pixmap.scaled(
                                self.thumbnail_list.iconSize(),
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation,
                            )
                        )
                    )
                else:
                    item.setToolTip(self._text("影像不存在", "Image unavailable"))
                self.thumbnail_list.addItem(item)
        finally:
            self.thumbnail_list.blockSignals(False)

    def selected_indices(self) -> set[int]:
        """Return manifest indices currently included in the training batch."""
        return {
            int(item.data(ROW_INDEX_ROLE))
            for item in self._items()
            if item.checkState() == Qt.Checked
            and item.data(CATEGORY_ROLE) != "hold"
        }

    def _items(self) -> list[QListWidgetItem]:
        return [
            self.thumbnail_list.item(index)
            for index in range(self.thumbnail_list.count())
        ]

    def _apply_filter(self) -> None:
        mode = str(self.filter_combo.currentData())
        for item in self._items():
            if mode == "all":
                hidden = False
            elif mode == "excluded":
                hidden = item.checkState() != Qt.Unchecked
            else:
                hidden = str(item.data(CATEGORY_ROLE)) != mode
            item.setHidden(hidden)

    def _set_visible_check_state(self, state: Qt.CheckState) -> None:
        self.thumbnail_list.blockSignals(True)
        try:
            for item in self._items():
                if not item.isHidden() and item.data(CATEGORY_ROLE) != "hold":
                    item.setCheckState(state)
        finally:
            self.thumbnail_list.blockSignals(False)
        self._update_summary()
        self._apply_filter()

    def _on_item_changed(self, _item: QListWidgetItem) -> None:
        self._update_summary()
        if self.filter_combo.currentData() == "excluded":
            self._apply_filter()

    def _update_summary(self) -> None:
        selected = self.selected_indices()
        direct = sum(
            item.checkState() == Qt.Checked and item.data(CATEGORY_ROLE) == "direct"
            for item in self._items()
        )
        annotation = len(selected) - direct
        hold = sum(item.data(CATEGORY_ROLE) == "hold" for item in self._items())
        excluded = self.thumbnail_list.count() - len(selected) - hold
        self.summary_label.setText(
            self._text(
                f"本次補訓 {len(selected)} 張｜直接訓練 {direct} 張｜"
                f"需要補標 {annotation} 張｜暫不送訓 {hold} 張｜已排除 {excluded} 張",
                f"Selected {len(selected)} | Direct {direct} | "
                f"Annotation {annotation} | On hold {hold} | Excluded {excluded}",
            )
        )
        if self.direct_action_button is not None:
            self.direct_action_button.setEnabled(direct > 0)
        if self.annotation_action_button is not None:
            self.annotation_action_button.setEnabled(annotation > 0)

    def _confirm(self) -> None:
        if self.submit_mode and not self.selected_indices():
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "目前沒有選擇任何補訓影像。請至少勾選一張，或返回繼續複核。",
                    "No training image is selected. Include at least one or go back.",
                ),
            )
            return
        self.accept()

    def action_selected_indices(self) -> set[int]:
        """Return checked indices belonging to the selected queue action."""
        if self.selected_action not in {"direct", "annotation"}:
            return set()
        return {
            int(item.data(ROW_INDEX_ROLE))
            for item in self._items()
            if item.checkState() == Qt.Checked
            and item.data(CATEGORY_ROLE) == self.selected_action
        }

    def _confirm_queue_action(self, action: str) -> None:
        self.selected_action = action
        if not self.action_selected_indices():
            self.selected_action = None
            QMessageBox.warning(
                self,
                self.windowTitle(),
                self._text(
                    "此分類目前沒有已勾選照片。",
                    "No checked image is available in this category.",
                ),
            )
            return
        self.accept()

    def _show_large_preview(self, item: QListWidgetItem) -> None:
        path = Path(str(item.data(PREVIEW_PATH_ROLE) or ""))
        pixmap = QPixmap(str(path)) if path.is_file() else QPixmap()
        if pixmap.isNull():
            return
        preview = QDialog(self)
        preview.setWindowTitle(item.text().replace("\n", " | "))
        preview.resize(1200, 800)
        layout = QVBoxLayout(preview)
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setPixmap(
            pixmap.scaled(QSize(1160, 740), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        layout.addWidget(label)
        preview.exec_()


def _best_preview_path(row: dict[str, str]) -> Path:
    """Return the best available inference evidence image for a grid item."""
    for key in ("annotated_path", "preprocessed_path", "original_path"):
        path = Path(str(row.get(key) or ""))
        if path.is_file():
            return path
    return Path()
