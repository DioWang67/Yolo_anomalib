"""Passive Phase 2B review workspace components and display-only view models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTextEdit,
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
from tools.review_workflow import (
    WorkflowState,
    blocking_violations,
    derive_review_semantics,
    derive_workflow_state,
    validate_record_consistency,
)

ROW_INDEX_ROLE = Qt.UserRole
STATUS_GROUP_ROLE = Qt.UserRole + 1
PREVIEW_PATH_ROLE = Qt.UserRole + 2
THUMBNAIL_STATE_ROLE = Qt.UserRole + 3
THUMBNAIL_REQUESTED_ROLE = Qt.UserRole + 4
IMAGE_SHA_ROLE = Qt.UserRole + 5
THUMBNAIL_PREFETCH_ITEMS = 12
_PLACEHOLDER_ICONS: dict[tuple[str, int, int], QIcon] = {}
PENDING_STATES = frozenset(
    {WorkflowState.NEW, WorkflowState.SELECTED, WorkflowState.IN_REVIEW}
)


@dataclass(frozen=True)
class ReviewListItemViewModel:
    row_index: int
    title: str
    subtitle: str
    preview_path: str
    status_group: str
    status_label: str
    image_sha: str = ""


@dataclass(frozen=True)
class ReviewDetailsViewModel:
    ai_summary: str
    metadata_summary: str
    semantics_summary: str


def build_review_list_item(
    row_index: int,
    record: dict[str, str],
    *,
    language: str,
) -> ReviewListItemViewModel:
    """Build one display item exclusively from Phase 1B-derived state."""
    state = derive_workflow_state(record)
    violations = blocking_violations(validate_record_consistency(record))
    zh = str(language).lower().startswith("zh")
    if state == WorkflowState.ERROR or violations:
        status_group = "error"
        status_label = "有錯誤" if zh else "Error"
    elif state in PENDING_STATES:
        status_group = "pending"
        status_label = "未完成" if zh else "Pending"
    else:
        status_group = "completed"
        status_label = "已完成" if zh else "Completed"
    preview = _best_preview(record)
    return ReviewListItemViewModel(
        row_index=row_index,
        title=(
            f"{record.get('product', '')} / {record.get('area', '')}"
            f"  [{status_label}]"
        ),
        subtitle=str(record.get("timestamp") or ""),
        preview_path=str(preview),
        status_group=status_group,
        status_label=status_label,
        image_sha=str(
            record.get("annotated_sha256")
            or record.get("image_sha")
            or record.get("original_sha256")
            or ""
        ),
    )


def build_review_details(
    record: dict[str, str],
    *,
    language: str,
) -> ReviewDetailsViewModel:
    """Build AI, equipment, and typed semantic summaries for the right panel."""
    zh = str(language).lower().startswith("zh")
    detections = _parse_detections(record)
    detection_lines: list[str] = []
    for index, detection in enumerate(detections, start=1):
        class_name = str(
            _first_present(
                detection.get("class"),
                detection.get("class_name"),
                detection.get("class_id"),
            )
        )
        confidence = _format_confidence(
            _first_present(detection.get("confidence"), detection.get("conf"))
        )
        bbox = detection.get("bbox")
        bbox_text = json.dumps(bbox, ensure_ascii=False) if isinstance(bbox, list) else "-"
        detection_lines.append(
            f"#{index}  class={class_name}  confidence={confidence}  bbox={bbox_text}"
        )
    if not detection_lines:
        detection_lines.append("沒有結構化偵測資料" if zh else "No structured detections")
    model_version = str(record.get("model_version") or record.get("weights") or "-")
    ai_title = "AI 預測" if zh else "AI prediction"
    ai_summary = f"{ai_title} ({len(detections)})\n" + "\n".join(detection_lines)
    ai_summary += f"\n{'模型版本' if zh else 'Model'}: {model_version}"

    metadata_labels = (
        ("產品" if zh else "Product", "product"),
        ("區域" if zh else "Area", "area"),
        ("站別" if zh else "Station", "station_id"),
        ("機台" if zh else "Machine", "machine_id"),
        ("相機" if zh else "Camera", "camera_id"),
        ("工單" if zh else "Work order", "work_order"),
    )
    metadata_summary = "\n".join(
        f"{label}: {record.get(field) or '-'}" for label, field in metadata_labels
    )
    semantics = derive_review_semantics(record)
    semantics_summary = "\n".join(
        (
            f"product verdict: {semantics.product_verdict.value}",
            f"AI correctness: {semantics.ai_correctness.value}",
            f"annotation validity: {semantics.annotation_validity.value}",
            f"required action: {semantics.required_action.value}",
        )
    )
    return ReviewDetailsViewModel(ai_summary, metadata_summary, semantics_summary)


class ReviewThumbnailPanel(QWidget):
    """Display prebuilt review items without interpreting legacy fields."""

    current_changed = pyqtSignal(int)

    def __init__(
        self,
        *,
        language: str = "zh_TW",
        image_service: AsyncImageService | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.language = language
        self._items: tuple[ReviewListItemViewModel, ...] = ()
        self._generation = 0
        self._resident_thumbnail_indices: set[int] = set()
        self._owns_image_service = image_service is None
        self.image_service = image_service or AsyncImageService.from_environment(
            parent=self
        )
        self.image_service.result_ready.connect(self._on_image_result)
        self._thumbnail_timer = QTimer(self)
        self._thumbnail_timer.setSingleShot(True)
        self._thumbnail_timer.timeout.connect(self._request_visible_thumbnails)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.filter_combo = QComboBox()
        for value, zh_label, en_label in (
            ("all", "全部", "All"),
            ("pending", "未完成", "Pending"),
            ("completed", "已完成", "Completed"),
            ("error", "有錯誤", "Errors"),
        ):
            self.filter_combo.addItem(self._text(zh_label, en_label), value)
        self.filter_combo.currentIndexChanged.connect(self._render)
        layout.addWidget(self.filter_combo)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.setIconSize(QSize(96, 64))
        self.list_widget.currentItemChanged.connect(self._emit_current)
        self.list_widget.verticalScrollBar().valueChanged.connect(
            self._schedule_visible_thumbnails
        )
        self.list_widget.viewport().installEventFilter(self)
        layout.addWidget(self.list_widget, 1)

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    def set_items(
        self,
        items: list[ReviewListItemViewModel],
        *,
        current_index: int | None,
    ) -> None:
        incoming = tuple(items)
        if incoming == self._items:
            self.set_current_index(current_index)
            return
        self._items = incoming
        self._render(current_index=current_index)

    def _render(
        self,
        _combo_index: int | None = None,
        *,
        current_index: int | None = None,
    ) -> None:
        self._generation += 1
        self._resident_thumbnail_indices.clear()
        if current_index is None:
            current_item = self.list_widget.currentItem()
            current_index = (
                int(current_item.data(ROW_INDEX_ROLE)) if current_item else None
            )
        selected_filter = str(self.filter_combo.currentData() or "all")
        self.list_widget.blockSignals(True)
        try:
            self.list_widget.clear()
            selected_item: QListWidgetItem | None = None
            first_item: QListWidgetItem | None = None
            for model in self._items:
                if selected_filter != "all" and model.status_group != selected_filter:
                    continue
                item = QListWidgetItem(f"{model.title}\n{model.subtitle}")
                item.setData(ROW_INDEX_ROLE, model.row_index)
                item.setData(STATUS_GROUP_ROLE, model.status_group)
                item.setData(PREVIEW_PATH_ROLE, model.preview_path)
                item.setData(IMAGE_SHA_ROLE, model.image_sha)
                item.setData(THUMBNAIL_STATE_ROLE, IMAGE_STATUS_LOADING)
                item.setData(THUMBNAIL_REQUESTED_ROLE, False)
                item.setIcon(
                    _placeholder_icon(
                        IMAGE_STATUS_LOADING,
                        self.list_widget.iconSize(),
                    )
                )
                if model.status_group == "completed":
                    item.setBackground(Qt.green)
                elif model.status_group == "error":
                    item.setBackground(Qt.red)
                self.list_widget.addItem(item)
                if first_item is None:
                    first_item = item
                if model.row_index == current_index:
                    selected_item = item
            self.list_widget.setCurrentItem(selected_item or first_item)
        finally:
            self.list_widget.blockSignals(False)
        if selected_item is None and first_item is not None and current_index is not None:
            self.current_changed.emit(int(first_item.data(ROW_INDEX_ROLE)))
        if self.image_service.synchronous:
            self._resident_thumbnail_indices = {
                int(self.list_widget.item(position).data(ROW_INDEX_ROLE))
                for position in range(self.list_widget.count())
            }
            for position in range(self.list_widget.count()):
                self._request_thumbnail_at(position, priority=0)
        else:
            self._schedule_visible_thumbnails()

    def set_current_index(self, row_index: int | None) -> None:
        if row_index is None:
            self.list_widget.clearSelection()
            return
        for position in range(self.list_widget.count()):
            item = self.list_widget.item(position)
            if int(item.data(ROW_INDEX_ROLE)) == row_index:
                signals_were_blocked = self.list_widget.blockSignals(True)
                try:
                    self.list_widget.setCurrentItem(item)
                finally:
                    self.list_widget.blockSignals(signals_were_blocked)
                return

    def update_item(self, model: ReviewListItemViewModel) -> None:
        items = list(self._items)
        for position, current in enumerate(items):
            if current.row_index == model.row_index:
                if current == model:
                    return
                items[position] = model
                self._items = tuple(items)
                self._render(current_index=model.row_index)
                return

    def _emit_current(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        if current is not None:
            self.current_changed.emit(int(current.data(ROW_INDEX_ROLE)))

    def displayed_indices(self) -> set[int]:
        return {
            int(self.list_widget.item(index).data(ROW_INDEX_ROLE))
            for index in range(self.list_widget.count())
        }

    def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802 - Qt API
        if sip.isdeleted(self) or sip.isdeleted(self.list_widget):
            return False
        if watched is self.list_widget.viewport() and event.type() in {
            QEvent.Resize,
            QEvent.Show,
        }:
            self._schedule_visible_thumbnails()
        return super().eventFilter(watched, event)

    def _schedule_visible_thumbnails(self) -> None:
        if sip.isdeleted(self) or sip.isdeleted(self.list_widget):
            return
        self._thumbnail_timer.start(0)

    def _request_visible_thumbnails(self) -> None:
        if sip.isdeleted(self) or sip.isdeleted(self.list_widget):
            return
        if not self.image_service.active or self.list_widget.count() == 0:
            return
        viewport_rect = self.list_widget.viewport().rect()
        visible = [
            position
            for position in range(self.list_widget.count())
            if self.list_widget.visualItemRect(
                self.list_widget.item(position)
            ).intersects(viewport_rect)
        ]
        if not visible:
            visible = list(range(min(20, self.list_widget.count())))
        visible_set = set(visible)
        first = max(min(visible) - THUMBNAIL_PREFETCH_ITEMS, 0)
        last = min(
            max(visible) + THUMBNAIL_PREFETCH_ITEMS + 1,
            self.list_widget.count(),
        )
        positions = visible_set | set(range(first, last))
        self._resident_thumbnail_indices = {
            int(self.list_widget.item(position).data(ROW_INDEX_ROLE))
            for position in positions
        }
        self._evict_nonresident_icons()
        for position in sorted(positions, key=lambda value: value not in visible_set):
            self._request_thumbnail_at(
                position,
                priority=10 if position in visible_set else 0,
            )

    def _request_thumbnail_at(self, position: int, *, priority: int) -> None:
        if sip.isdeleted(self) or sip.isdeleted(self.list_widget):
            return
        item = self.list_widget.item(position)
        if item is None or bool(item.data(THUMBNAIL_REQUESTED_ROLE)):
            return
        path = str(item.data(PREVIEW_PATH_ROLE) or "")
        row_index = int(item.data(ROW_INDEX_ROLE))
        token = (id(self), self._generation, row_index, path)
        signals_were_blocked = self.list_widget.blockSignals(True)
        try:
            item.setData(THUMBNAIL_REQUESTED_ROLE, True)
        finally:
            self.list_widget.blockSignals(signals_were_blocked)
        self.image_service.request_image(
            path,
            purpose="thumbnail",
            token=token,
            target_size=self.list_widget.iconSize(),
            image_sha=str(item.data(IMAGE_SHA_ROLE) or ""),
            priority=priority,
        )

    def _on_image_result(self, result: ImageLoadResult) -> None:
        if sip.isdeleted(self) or sip.isdeleted(self.list_widget):
            return
        if (
            result.purpose != "thumbnail"
            or not isinstance(result.token, tuple)
            or len(result.token) != 4
        ):
            return
        owner, generation, row_index, path = result.token
        if owner != id(self) or generation != self._generation:
            return
        for position in range(self.list_widget.count()):
            item = self.list_widget.item(position)
            if (
                int(item.data(ROW_INDEX_ROLE)) != row_index
                or str(item.data(PREVIEW_PATH_ROLE) or "") != path
            ):
                continue
            if row_index not in self._resident_thumbnail_indices:
                item.setData(THUMBNAIL_REQUESTED_ROLE, False)
                item.setData(THUMBNAIL_STATE_ROLE, IMAGE_STATUS_LOADING)
                item.setIcon(
                    _placeholder_icon(
                        IMAGE_STATUS_LOADING,
                        self.list_widget.iconSize(),
                    )
                )
                return
            item.setData(THUMBNAIL_STATE_ROLE, result.status)
            if result.status == IMAGE_STATUS_LOADED and result.image is not None:
                item.setIcon(QIcon(QPixmap.fromImage(result.image)))
            else:
                item.setIcon(_placeholder_icon(result.status, self.list_widget.iconSize()))
            item.setToolTip(
                _image_error_text(
                    result.status,
                    path=result.path,
                    sample_id=str(row_index),
                    language=self.language,
                )
            )
            return

    def _evict_nonresident_icons(self) -> None:
        for position in range(self.list_widget.count()):
            item = self.list_widget.item(position)
            row_index = int(item.data(ROW_INDEX_ROLE))
            if row_index in self._resident_thumbnail_indices:
                continue
            if str(item.data(THUMBNAIL_STATE_ROLE)) != IMAGE_STATUS_LOADING:
                item.setData(THUMBNAIL_STATE_ROLE, IMAGE_STATUS_LOADING)
                item.setData(THUMBNAIL_REQUESTED_ROLE, False)
                item.setIcon(
                    _placeholder_icon(
                        IMAGE_STATUS_LOADING,
                        self.list_widget.iconSize(),
                    )
                )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._generation += 1
        self._thumbnail_timer.stop()
        if self._owns_image_service:
            self.image_service.shutdown()
        super().closeEvent(event)


class ReviewImageViewer(QWidget):
    """Asynchronous full-image viewer with stale-result protection and zoom."""

    mode_changed = pyqtSignal(str)
    state_changed = pyqtSignal(str)

    def __init__(
        self,
        *,
        language: str = "zh_TW",
        image_service: AsyncImageService | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.language = language
        self._paths = {"original": Path(), "overlay": Path()}
        self._mode = "overlay"
        self._pixmap = QPixmap()
        self._zoom = 1.0
        self._fit = True
        self.image_state = "missing"
        self._sample_id = "-"
        self._generation = 0
        self._owns_image_service = image_service is None
        self.image_service = image_service or AsyncImageService.from_environment(
            parent=self
        )
        self.image_service.result_ready.connect(self._on_image_result)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QGridLayout()
        toolbar.setHorizontalSpacing(6)
        toolbar.setVerticalSpacing(4)
        self.original_button = QPushButton(self._text("原圖", "Original"))
        self.overlay_button = QPushButton(self._text("AI 疊圖", "AI overlay"))
        self.original_button.setCheckable(True)
        self.overlay_button.setCheckable(True)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.addButton(self.original_button)
        group.addButton(self.overlay_button)
        self.overlay_button.setChecked(True)
        self.original_button.clicked.connect(lambda: self.set_mode("original"))
        self.overlay_button.clicked.connect(lambda: self.set_mode("overlay"))
        self.fit_button = QPushButton(self._text("符合視窗", "Fit"))
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.fit_button.clicked.connect(self.fit_to_window)
        self.zoom_out_button.clicked.connect(self.zoom_out)
        self.zoom_in_button.clicked.connect(self.zoom_in)
        for position, button in enumerate((
            self.original_button,
            self.overlay_button,
            self.fit_button,
            self.zoom_out_button,
            self.zoom_in_button,
        )):
            row, column = divmod(position, 3)
            toolbar.addWidget(button, row, column)
        toolbar.setColumnStretch(3, 1)
        layout.addLayout(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(240, 180)
        self.image_label.setStyleSheet("background:#202124;color:white;padding:12px;")
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area, 1)

    def _text(self, zh: str, en: str) -> str:
        return zh if str(self.language).lower().startswith("zh") else en

    @property
    def mode(self) -> str:
        return self._mode

    def set_images(
        self,
        *,
        original_path: Any,
        overlay_path: Any,
        sample_id: str = "-",
    ) -> None:
        self._paths = {
            "original": Path(str(original_path or "")),
            "overlay": Path(str(overlay_path or "")),
        }
        self._sample_id = str(sample_id or "-")
        self._load_current()

    def set_mode(self, mode: str) -> None:
        if mode not in {"original", "overlay"}:
            raise ValueError(f"Unsupported image mode: {mode}")
        self._mode = mode
        self.original_button.setChecked(mode == "original")
        self.overlay_button.setChecked(mode == "overlay")
        self._load_current()
        self.mode_changed.emit(mode)

    def toggle_mode(self) -> None:
        self.set_mode("original" if self._mode == "overlay" else "overlay")

    def fit_to_window(self) -> None:
        self._fit = True
        self._render()

    def zoom_in(self) -> None:
        self._fit = False
        self._zoom = min(self._zoom * 1.25, 8.0)
        self._render()

    def zoom_out(self) -> None:
        self._fit = False
        self._zoom = max(self._zoom / 1.25, 0.1)
        self._render()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._fit:
            self._render()

    def _load_current(self) -> None:
        path = self._paths[self._mode]
        self._generation += 1
        generation = self._generation
        self._pixmap = QPixmap()
        self.image_state = IMAGE_STATUS_LOADING
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText(
            self._text(
                f"圖片載入中…\n樣本：{self._sample_id}",
                f"Loading image…\nSample: {self._sample_id}",
            )
        )
        self.state_changed.emit(self.image_state)
        self.image_service.request_image(
            path,
            purpose="full",
            token=(id(self), generation, self._mode, str(path)),
            priority=20,
        )

    def _on_image_result(self, result: ImageLoadResult) -> None:
        if (
            result.purpose != "full"
            or not isinstance(result.token, tuple)
            or len(result.token) != 4
        ):
            return
        owner, generation, mode, path = result.token
        if (
            owner != id(self)
            or generation != self._generation
            or mode != self._mode
            or path != str(self._paths[self._mode])
        ):
            return
        self.image_state = result.status
        self.image_label.setPixmap(QPixmap())
        if result.status == IMAGE_STATUS_LOADED and result.image is not None:
            self._pixmap = QPixmap.fromImage(result.image)
            self._zoom = 1.0
            self._render()
        else:
            self._pixmap = QPixmap()
            self.image_label.setText(
                _image_error_text(
                    result.status,
                    path=result.path,
                    sample_id=self._sample_id,
                    language=self.language,
                )
            )
        self.state_changed.emit(self.image_state)

    def _render(self) -> None:
        if self._pixmap.isNull():
            return
        if self._fit:
            viewport = self.scroll_area.viewport().size()
            target = QSize(max(viewport.width() - 8, 1), max(viewport.height() - 8, 1))
            rendered = self._pixmap.scaled(
                target,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        else:
            rendered = self._pixmap.scaled(
                self._pixmap.size() * self._zoom,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.image_label.setText("")
        self.image_label.setPixmap(rendered)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._generation += 1
        self._pixmap = QPixmap()
        if self._owns_image_service:
            self.image_service.shutdown()
        else:
            self.image_service.release_full_cache()
        super().closeEvent(event)


def is_text_input_focus(widget: QWidget | None) -> bool:
    """Prevent letter shortcuts while the operator is editing text or a combo."""
    return isinstance(
        widget,
        (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox),
    )


def _parse_detections(record: dict[str, str]) -> list[dict[str, Any]]:
    try:
        value = json.loads(str(record.get("detections_json") or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _format_confidence(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "-"


def _first_present(*values: Any) -> Any:
    return next(
        (value for value in values if value is not None and str(value) != ""),
        "-",
    )


def _best_preview(record: dict[str, str]) -> Path:
    for field in ("annotated_path", "preprocessed_path", "original_path"):
        path = Path(str(record.get(field) or ""))
        if path.is_file():
            return path
    return Path(str(record.get("original_path") or ""))


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
        painter.drawText(pixmap.rect(), Qt.AlignCenter, status.replace("_", " "))
    finally:
        painter.end()
    icon = QIcon(pixmap)
    _PLACEHOLDER_ICONS[cache_key] = icon
    return icon


def _image_error_text(
    status: str,
    *,
    path: str,
    sample_id: str,
    language: str,
) -> str:
    zh = str(language).lower().startswith("zh")
    messages = {
        IMAGE_STATUS_LOADING: ("圖片載入中", "Loading image"),
        IMAGE_STATUS_MISSING: ("圖片路徑不存在", "Image path does not exist"),
        IMAGE_STATUS_PERMISSION_DENIED: ("沒有圖片讀取權限", "Permission denied"),
        IMAGE_STATUS_UNSUPPORTED: ("圖片格式不支援", "Unsupported image format"),
        IMAGE_STATUS_CORRUPT: ("圖片損毀或無法解碼", "Image is corrupt or unreadable"),
        IMAGE_STATUS_UNEXPECTED: ("圖片載入發生非預期錯誤", "Unexpected image loader error"),
    }
    message = messages.get(status, ("圖片無法讀取", "Image unavailable"))[
        0 if zh else 1
    ]
    sample_label = "樣本" if zh else "Sample"
    path_label = "路徑" if zh else "Path"
    return f"{message}\n{sample_label}: {sample_id}\n{path_label}: {path}"
