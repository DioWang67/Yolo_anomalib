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
    QFrame,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from app.gui.dialog_geometry import configure_responsive_dialog
from tools.review_routing import action_route

DIRECT_TRAIN_LABELS = {"confirmed_ng", "verified_empty"}
ANNOTATION_LABELS = {"false_positive", "false_negative", "wrong_box", "wrong_class"}
COLOR_REVIEW_LABELS = {"color_confirmed_ng", "color_false_reject"}
BATCH_ACTION_LABELS = {
    "confirmed_ng": ("確認 NG（AI 判定正確）", "Confirmed NG (AI verdict correct)"),
    "confirmed_ok": ("確認 OK（AI 判定正確）", "Confirmed OK (AI verdict correct)"),
    "verified_empty": ("實際無目標（負樣本）", "No target present (negative sample)"),
    "false_positive": ("實際 OK（AI 過殺）", "Actually OK (AI overkill)"),
    "false_negative": ("實際 NG（AI 漏檢）", "Actually NG (AI missed it)"),
    "wrong_box": ("NG：框的位置／數量需修正", "NG: box position or count needs correction"),
    "wrong_class": ("NG：框的類別需修正", "NG: box class needs correction"),
    "uncertain": ("舊版未判定（不送訓）", "Legacy undecided (do not train)"),
    "image_quality_issue": (
        "圖片無法判定（不採用）",
        "Image cannot be judged (exclude)",
    ),
    "color_confirmed_ng": (
        "顏色確實 NG（顏色判定正確）",
        "Color is truly NG (color verdict correct)",
    ),
    "color_false_reject": (
        "顏色其實 OK（門檻過嚴）",
        "Color is actually OK (threshold too strict)",
    ),
}

ROW_INDEX_ROLE = Qt.UserRole
PREVIEW_PATH_ROLE = Qt.UserRole + 1
CATEGORY_ROLE = Qt.UserRole + 2
BASE_TEXT_ROLE = Qt.UserRole + 3


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
        history_mode: bool = False,
        parent: Any = None,
    ) -> None:
        super().__init__(parent)
        self.entries = entries
        self.language = language
        self.submit_mode = submit_mode
        self.queue_mode = queue_mode
        self.history_mode = history_mode
        if self.queue_mode and self.history_mode:
            raise ValueError("queue_mode and history_mode are mutually exclusive")
        self.selected_action: str | None = None
        self.setWindowTitle(
            self._text(
                (
                    "已送訓照片（唯讀）"
                    if self.history_mode
                    else (
                        "補訓資料中心｜確認與送出"
                        if self.queue_mode
                        else "產線模型補訓｜送出確認"
                    )
                ),
                (
                    "Submitted Images (Read-only)"
                    if self.history_mode
                    else (
                        "Selected Retraining Queue"
                        if self.queue_mode
                        else "Production Retraining | Submission Review"
                    )
                ),
            )
        )
        configure_responsive_dialog(
            self,
            preferred=(1400, 880),
            minimum=(820, 560),
            parent=parent,
        )
        self._build_ui()
        self._populate_items()
        self._update_summary()

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        self.direct_action_button: QPushButton | None = None
        self.annotation_action_button: QPushButton | None = None
        self.color_action_button: QPushButton | None = None
        self.portable_action_button: QPushButton | None = None
        self.confirm_button: QPushButton | None = None
        self.remove_selected_button: QPushButton | None = None
        self.route_count_labels: dict[str, QLabel] = {}
        title = QLabel(
            self._text(
                (
                    "這是已送出批次的唯讀照片；雙擊可放大，不會重新送訓。"
                    if self.history_mode
                    else (
                        "第 3 步：確認待送照片。先勾選要處理的影像，再從下方選擇一條正確路徑。"
                        if self.queue_mode
                        else "確認本次要送入模型補訓的影像。取消勾選可排除誤入資料；雙擊可放大。"
                    )
                ),
                (
                    "This is a read-only submitted batch. Double-click to enlarge; it cannot be resubmitted."
                    if self.history_mode
                    else (
                        "Step 3: confirm queued images, then choose exactly one processing route below."
                        if self.queue_mode
                        else "Confirm images for model retraining. Uncheck accidental "
                        "entries; double-click to enlarge."
                    )
                ),
            )
        )
        title.setWordWrap(True)
        title.setStyleSheet(
            "QLabel { background: #20354a; color: white; padding: 14px 16px; "
            "border-radius: 8px; font-size: 14pt; font-weight: bold; }"
        )
        layout.addWidget(title)
        if self.queue_mode:
            layout.addWidget(self._build_workflow_strip())

        content = QHBoxLayout()
        content.setSpacing(14)
        gallery_panel = QFrame()
        gallery_panel.setObjectName("GalleryPanel")
        gallery_panel.setStyleSheet(
            "QFrame#GalleryPanel { background: white; border: 1px solid #d8dee6; "
            "border-radius: 8px; }"
        )
        gallery_layout = QVBoxLayout(gallery_panel)
        gallery_layout.setContentsMargins(12, 12, 12, 12)
        gallery_layout.setSpacing(10)

        toolbar = QHBoxLayout()
        filter_label = QLabel(self._text("查看分類", "Category"))
        filter_label.setStyleSheet("font-weight: bold; color: #344054;")
        toolbar.addWidget(filter_label)
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(190)
        self.filter_combo.addItem(self._text("全部", "All"), "all")
        self.filter_combo.addItem(self._text("可直接訓練", "Direct training"), "direct")
        self.filter_combo.addItem(self._text("需要補標", "Needs annotation"), "annotation")
        self.filter_combo.addItem(self._text("顏色校正", "Color calibration"), "color")
        self.filter_combo.addItem(self._text("暫不送訓", "On hold"), "hold")
        if not self.history_mode:
            self.filter_combo.addItem(self._text("已排除", "Excluded"), "excluded")
        self.filter_combo.currentIndexChanged.connect(self._apply_filter)
        toolbar.addWidget(self.filter_combo)
        toolbar.addStretch()
        if not self.history_mode:
            include_button = QPushButton(
                self._text("全選這一類", "Select this category")
            )
            include_button.clicked.connect(lambda: self._set_visible_check_state(Qt.Checked))
            exclude_button = QPushButton(
                self._text("全部取消", "Clear this category")
            )
            exclude_button.clicked.connect(
                lambda: self._set_visible_check_state(Qt.Unchecked)
            )
            toolbar.addWidget(include_button)
            toolbar.addWidget(exclude_button)
        gallery_layout.addLayout(toolbar)

        if self.queue_mode:
            selection_hint = QLabel(
                self._text(
                    "✓ 勾選代表保留在待送清單；點選照片後可從右側移除。雙擊照片可放大。",
                    "Checked items stay in the queue. Select images to remove them from the right panel; double-click to enlarge.",
                )
            )
            selection_hint.setWordWrap(True)
            selection_hint.setStyleSheet(
                "QLabel { background:#eef6ff; color:#174a7e; padding:8px 10px; "
                "border:1px solid #b8d3ee; border-radius:5px; }"
            )
            gallery_layout.addWidget(selection_hint)

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListView.IconMode)
        self.thumbnail_list.setResizeMode(QListView.Adjust)
        self.thumbnail_list.setMovement(QListView.Static)
        self.thumbnail_list.setIconSize(QSize(245, 165))
        self.thumbnail_list.setGridSize(QSize(280, 260))
        self.thumbnail_list.setSpacing(8)
        self.thumbnail_list.setWordWrap(True)
        self.thumbnail_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
            if self.queue_mode
            else QAbstractItemView.SingleSelection
        )
        self.thumbnail_list.itemChanged.connect(self._on_item_changed)
        self.thumbnail_list.itemSelectionChanged.connect(
            self._update_remove_button
        )
        self.thumbnail_list.itemDoubleClicked.connect(self._show_large_preview)
        gallery_layout.addWidget(self.thumbnail_list, 1)
        content.addWidget(gallery_panel, 1)

        if self.queue_mode:
            route_scroll = QScrollArea()
            route_scroll.setObjectName("RouteActionScroll")
            route_scroll.setWidgetResizable(True)
            route_scroll.setFrameShape(QFrame.NoFrame)
            route_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            route_scroll.setWidget(self._build_route_action_panel())
            content.addWidget(route_scroll)

        layout.addLayout(content, 1)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "QLabel { background:#f5f7fa; color:#344054; padding:9px 12px; "
            "border:1px solid #d8dee6; border-radius:6px; "
            "font-size:10.5pt; font-weight:bold; }"
        )
        if self.queue_mode:
            self.route_summary_layout.insertWidget(2, self.summary_label)
            self.confirm_button = self.direct_action_button
            return

        footer = QHBoxLayout()
        footer.addWidget(self.summary_label, 1)
        close_button = QPushButton(
            self._text(
                (
                    "關閉"
                    if self.history_mode
                    else ("儲存待送清單並返回" if self.queue_mode else "返回")
                ),
                (
                    "Close"
                    if self.history_mode
                    else ("Save queue and return" if self.queue_mode else "Back")
                ),
            )
        )
        close_button.clicked.connect(
            self.accept if self.queue_mode or self.history_mode else self.reject
        )
        footer.addWidget(close_button)
        if not self.history_mode:
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

    def _build_workflow_strip(self) -> QFrame:
        panel = QFrame()
        panel.setStyleSheet(
            "QFrame { background: #f5f7fa; border: 1px solid #d8dee6; border-radius: 7px; }"
        )
        steps = QHBoxLayout(panel)
        steps.setContentsMargins(10, 8, 10, 8)
        labels = (
            ("1", self._text("資料已複核", "Data reviewed"), "done"),
            ("2", self._text("確認並送出", "Confirm and submit"), "active"),
            ("3", self._text("補標、訓練與驗證", "Correct, train and validate"), "next"),
        )
        for number, name, state in labels:
            label = QLabel(f"{number}  {name}")
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(34)
            styles = {
                "done": "background:#e7f4ea;color:#216e39;border:1px solid #9bc9a7;",
                "active": "background:#2563a6;color:white;border:1px solid #1f5188;font-weight:bold;",
                "next": "background:white;color:#697586;border:1px solid #d8dee6;",
            }
            label.setStyleSheet(styles[state] + "border-radius:5px;padding:4px;")
            steps.addWidget(label, 1)
        return panel

    def _build_route_action_panel(self) -> QFrame:
        panel = QFrame()
        panel.setMinimumWidth(330)
        panel.setMaximumWidth(380)
        panel.setStyleSheet(
            "QFrame#RoutePanel { background:#f8fafc; border:1px solid #d8dee6; "
            "border-radius:8px; }"
        )
        panel.setObjectName("RoutePanel")
        cards = QVBoxLayout(panel)
        cards.setContentsMargins(14, 14, 14, 14)
        cards.setSpacing(10)
        self.route_summary_layout = cards

        heading = QLabel(self._text("這批要怎麼處理？", "How should this batch proceed?"))
        heading.setStyleSheet("font-size:14pt;font-weight:bold;color:#20354a;border:0;")
        guidance = QLabel(
            self._text(
                "系統已依判定結果分好類。每個按鈕只會處理該分類中已勾選的照片，不會混送。",
                "Cases are already grouped by verdict. Each action processes only checked images in that category.",
            )
        )
        guidance.setWordWrap(True)
        guidance.setStyleSheet("color:#5b6573;border:0;")
        cards.addWidget(heading)
        cards.addWidget(guidance)
        specs = (
            (
                "direct",
                self._text("框與類別已正確", "Boxes and classes are correct"),
                self._text("不需修改標註，直接加入 YOLO 補訓。", "No label edits; send directly to YOLO."),
                self._text("直接送入 YOLO 補訓", "Send directly to YOLO"),
                "#237a3b",
            ),
            (
                "annotation",
                self._text("框或類別需要修正", "Boxes or classes need correction"),
                self._text("先開啟補標工具；完成後自動送入 YOLO。", "Annotate first; YOLO starts after completion."),
                self._text("開啟補標工具", "Open annotation"),
                "#ad641f",
            ),
            (
                "color",
                self._text("只有顏色判定需校正", "Only color verdict needs calibration"),
                self._text("保留目前框，只更新顏色統計與門檻。", "Keep boxes; update color statistics and thresholds."),
                self._text("送出顏色校正資料", "Submit color calibration data"),
                "#00796b",
            ),
        )
        for route, title, description, button_text, color in specs:
            card, button = self._route_card(
                route, title, description, button_text, color
            )
            cards.addWidget(card, 1)
            if route == "direct":
                self.direct_action_button = button
            elif route == "annotation":
                self.annotation_action_button = button
            else:
                self.color_action_button = button
        cards.addStretch()

        secondary_title = QLabel(self._text("其他動作", "Other actions"))
        secondary_title.setStyleSheet("font-weight:bold;color:#344054;border:0;")
        cards.addWidget(secondary_title)

        self.portable_action_button = QPushButton(
            self._text("匯出離線補訓包", "Export offline training package")
        )
        self.portable_action_button.setMinimumHeight(38)
        self.portable_action_button.setStyleSheet(
            "QPushButton { background:white;color:#6f42c1;font-weight:bold;"
            "border:1px solid #8b63d2;padding:7px 12px;border-radius:5px; }"
            "QPushButton:disabled { color:#9aa0a6;border-color:#d1d5db; }"
        )
        self.portable_action_button.clicked.connect(
            lambda: self._confirm_queue_action("portable")
        )
        cards.addWidget(self.portable_action_button)

        self.remove_selected_button = QPushButton(
            self._text("移除目前點選的照片", "Remove selected images")
        )
        self.remove_selected_button.setEnabled(False)
        self.remove_selected_button.setToolTip(
            self._text(
                "只移除待送狀態，不刪除原圖、複核答案或已送出紀錄。可用 Ctrl／Shift 選取多張。",
                "Removes only the pending state; images, review verdicts, and history remain. Use Ctrl/Shift for multiple selection.",
            )
        )
        self.remove_selected_button.setStyleSheet(
            "QPushButton { color:#a61b1b;background:white;border:1px solid #d96c6c;"
            "border-radius:5px;padding:7px 12px;font-weight:bold; }"
            "QPushButton:disabled { color:#9aa0a6;background:#f3f4f6;border-color:#d1d5db; }"
        )
        self.remove_selected_button.clicked.connect(self._remove_selected_from_queue)
        cards.addWidget(self.remove_selected_button)

        save_button = QPushButton(self._text("儲存待送清單並返回", "Save queue and return"))
        save_button.setMinimumHeight(38)
        save_button.clicked.connect(self.accept)
        cards.addWidget(save_button)
        return panel

    def _route_card(
        self,
        route: str,
        title: str,
        description: str,
        button_text: str,
        color: str,
    ) -> tuple[QFrame, QPushButton]:
        card = QFrame()
        card.setObjectName(f"RouteCard_{route}")
        card.setStyleSheet(
            f"QFrame#{card.objectName()} {{ background:white; border:2px solid {color}; "
            "border-radius:6px; }}"
        )
        content = QVBoxLayout(card)
        content.setContentsMargins(12, 9, 12, 10)
        heading = QLabel(title)
        heading.setStyleSheet(f"color:{color};font-size:11pt;font-weight:bold;border:0;")
        count = QLabel(self._text("已勾選 0 張", "0 selected"))
        count.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        count.setStyleSheet("color:#243447;font-size:10pt;font-weight:bold;border:0;")
        detail = QLabel(description)
        detail.setWordWrap(True)
        detail.setStyleSheet("color:#5b6573;border:0;")
        button = QPushButton(button_text)
        button.setMinimumHeight(40)
        button.setStyleSheet(
            f"QPushButton {{ background:{color};color:white;border:0;border-radius:5px;"
            "font-weight:bold;padding:8px 14px; }"
            "QPushButton:disabled { background:#aab2bd;color:#eef1f4; }"
        )
        button.clicked.connect(lambda _checked=False, value=route: self._confirm_queue_action(value))
        heading_row = QHBoxLayout()
        heading_row.setContentsMargins(0, 0, 0, 0)
        heading_row.addWidget(heading, 1)
        heading_row.addWidget(count)
        content.addLayout(heading_row)
        content.addWidget(detail)
        content.addWidget(button)
        self.route_count_labels[route] = count
        return card, button

    def _populate_items(self) -> None:
        self.thumbnail_list.blockSignals(True)
        try:
            for row_index, row in self.entries:
                review_label = str(row.get("review_label") or "")
                route = action_route(row)
                if route == "color":
                    category = "color"
                elif route == "both":
                    category = "annotation"
                elif review_label in DIRECT_TRAIN_LABELS:
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
                    "color": ("顏色校正", "Color calibration"),
                    "hold": ("暫不送訓", "On hold"),
                }
                category_label = self._text(*category_labels[category])
                item = QListWidgetItem(
                    f"【{category_label}】"
                    + "\n"
                    + self._text(*action_labels)
                    + "\n"
                    + str(row.get("timestamp") or "")
                )
                item.setData(BASE_TEXT_ROLE, item.text())
                item.setData(ROW_INDEX_ROLE, row_index)
                preview_path = _best_preview_path(row)
                item.setData(PREVIEW_PATH_ROLE, str(preview_path))
                item.setData(CATEGORY_ROLE, category)
                item_flags = item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable
                if category != "hold" and not self.history_mode:
                    item_flags |= Qt.ItemIsUserCheckable
                item.setFlags(item_flags)
                if not self.history_mode:
                    item.setCheckState(
                        Qt.Unchecked
                        if category == "hold"
                        or str(row.get("training_selected") or "1") == "0"
                        else Qt.Checked
                    )
                if category == "hold" or self.history_mode:
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable)
                background_colors = {
                    "direct": "#e8f5e9",
                    "annotation": "#fff3e0",
                    "color": "#e0f2f1",
                    "hold": "#ffebee",
                }
                item.setBackground(QColor(background_colors[category]))
                self._apply_item_check_style(item)
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
                    self._apply_item_check_style(item)
        finally:
            self.thumbnail_list.blockSignals(False)
        self._update_summary()
        self._apply_filter()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        self._apply_item_check_style(item)
        self._update_summary()
        if self.filter_combo.currentData() == "excluded":
            self._apply_filter()

    def _update_remove_button(self) -> None:
        if self.remove_selected_button is not None:
            self.remove_selected_button.setEnabled(
                bool(self.thumbnail_list.selectedItems())
            )

    def _remove_selected_from_queue(self) -> None:
        """Uncheck selected queue rows and return so the parent persists it."""
        selected_items = self.thumbnail_list.selectedItems()
        if not selected_items:
            QMessageBox.information(
                self,
                self.windowTitle(),
                self._text(
                    "請先點選要從待送清單移除的照片。",
                    "Select one or more cases to remove from the pending queue.",
                ),
            )
            return
        self.thumbnail_list.blockSignals(True)
        try:
            for item in selected_items:
                if item.data(CATEGORY_ROLE) != "hold":
                    item.setCheckState(Qt.Unchecked)
                    self._apply_item_check_style(item)
        finally:
            self.thumbnail_list.blockSignals(False)
        self.selected_action = None
        self._update_summary()
        self.accept()

    def _apply_item_check_style(self, item: QListWidgetItem) -> None:
        """Show queue inclusion with a strong badge and color, independent of focus."""
        if self.history_mode:
            return
        base_text = str(item.data(BASE_TEXT_ROLE) or item.text())
        checked = item.checkState() == Qt.Checked
        category = str(item.data(CATEGORY_ROLE) or "hold")
        prefix = self._text("✓ 已加入待送", "✓ INCLUDED") if checked else self._text("○ 未加入待送", "○ NOT INCLUDED")
        item.setText(f"{prefix}\n{base_text}")
        base_colors = {
            "direct": "#e8f5e9",
            "annotation": "#fff3e0",
            "color": "#e0f2f1",
            "hold": "#ffebee",
        }
        item.setBackground(QColor("#9fe0b2" if checked else base_colors.get(category, "#f5f7fa")))
        item.setForeground(QColor("#123c23" if checked else "#243447"))
        font = item.font()
        font.setBold(checked)
        item.setFont(font)

    def _update_summary(self) -> None:
        selected = self.selected_indices()
        direct = sum(
            item.checkState() == Qt.Checked and item.data(CATEGORY_ROLE) == "direct"
            for item in self._items()
        )
        color = sum(
            item.checkState() == Qt.Checked and item.data(CATEGORY_ROLE) == "color"
            for item in self._items()
        )
        annotation = len(selected) - direct - color
        hold = sum(item.data(CATEGORY_ROLE) == "hold" for item in self._items())
        excluded = self.thumbnail_list.count() - len(selected) - hold
        self.summary_label.setText(
            self._history_summary()
            if self.history_mode
            else self._text(
                f"已選擇 {len(selected)} 張｜直接訓練 {direct} 張｜"
                f"需要補標 {annotation} 張｜顏色校正 {color} 張｜"
                f"暫不送訓 {hold} 張｜已排除 {excluded} 張",
                f"Selected {len(selected)} | Direct {direct} | "
                f"Annotation {annotation} | Color {color} | "
                f"On hold {hold} | Excluded {excluded}",
            )
        )
        if self.direct_action_button is not None:
            self.direct_action_button.setEnabled(direct > 0)
            self.direct_action_button.setText(
                self._text(
                    f"直接送入 YOLO 補訓（{direct}）",
                    f"Send directly to YOLO ({direct})",
                )
            )
        if self.annotation_action_button is not None:
            self.annotation_action_button.setEnabled(annotation > 0)
            self.annotation_action_button.setText(
                self._text(
                    f"開啟補標工具（{annotation}）",
                    f"Open annotation ({annotation})",
                )
            )
        if self.color_action_button is not None:
            self.color_action_button.setEnabled(color > 0)
            self.color_action_button.setText(
                self._text(
                    f"送出顏色校正資料（{color}）",
                    f"Submit color calibration data ({color})",
                )
            )
        if self.portable_action_button is not None:
            portable_count = direct + annotation
            self.portable_action_button.setEnabled(portable_count > 0)
            self.portable_action_button.setText(
                self._text(
                    f"匯出離線補訓包（{portable_count}）",
                    f"Export offline training package ({portable_count})",
                )
            )
        selected_counts = {
            "direct": direct,
            "annotation": annotation,
            "color": color,
        }
        for route, label in self.route_count_labels.items():
            count = selected_counts[route]
            label.setText(
                self._text(f"已勾選 {count} 張", f"{count} selected")
            )
        category_totals = {
            category: sum(
                item.data(CATEGORY_ROLE) == category for item in self._items()
            )
            for category in ("direct", "annotation", "color", "hold")
        }
        filter_labels = {
            "all": self._text(
                f"全部（{self.thumbnail_list.count()}）",
                f"All ({self.thumbnail_list.count()})",
            ),
            "direct": self._text(
                f"可直接訓練（{category_totals['direct']}）",
                f"Direct training ({category_totals['direct']})",
            ),
            "annotation": self._text(
                f"需要補標（{category_totals['annotation']}）",
                f"Needs annotation ({category_totals['annotation']})",
            ),
            "color": self._text(
                f"顏色校正（{category_totals['color']}）",
                f"Color calibration ({category_totals['color']})",
            ),
            "hold": self._text(
                f"暫不送訓（{category_totals['hold']}）",
                f"On hold ({category_totals['hold']})",
            ),
            "excluded": self._text(
                f"已排除（{excluded}）", f"Excluded ({excluded})"
            ),
        }
        for mode, text in filter_labels.items():
            index = self.filter_combo.findData(mode)
            if index >= 0:
                self.filter_combo.setItemText(index, text)

    def _history_summary(self) -> str:
        counts = {"direct": 0, "annotation": 0, "color": 0, "hold": 0}
        for item in self._items():
            category = str(item.data(CATEGORY_ROLE) or "hold")
            counts[category if category in counts else "hold"] += 1
        return self._text(
            f"本批共 {self.thumbnail_list.count()} 張｜直接訓練 {counts['direct']} 張｜"
            f"補標後訓練 {counts['annotation']} 張｜顏色校正 {counts['color']} 張｜"
            f"未採用 {counts['hold']} 張",
            f"{self.thumbnail_list.count()} cases | Direct {counts['direct']} | "
            f"Annotated {counts['annotation']} | Color {counts['color']} | "
            f"Not used {counts['hold']}",
        )

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
        if self.selected_action not in {"direct", "annotation", "color", "portable"}:
            return set()
        if self.selected_action == "portable":
            return {
                int(item.data(ROW_INDEX_ROLE))
                for item in self._items()
                if item.checkState() == Qt.Checked
                and item.data(CATEGORY_ROLE) in {"direct", "annotation"}
            }
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
        configure_responsive_dialog(
            preview,
            preferred=(1200, 800),
            minimum=(600, 420),
            parent=self,
        )
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
