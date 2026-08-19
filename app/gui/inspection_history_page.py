"""Full-width, read-only inspection history page."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QDate, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import normalize_language, tr
from app.gui.widgets import ImageViewer
from core.services.inspection_excel_report import (
    InspectionExcelReport,
    InspectionExcelReportCancelled,
    InspectionExcelReportError,
    InspectionExcelReportService,
)
from core.services.inspection_history import (
    HISTORY_TIMEZONE,
    InspectionHistoryError,
    InspectionHistoryFilters,
    InspectionHistoryRecord,
    InspectionHistoryService,
    InspectionHistorySnapshot,
    InspectionTimeRange,
    local_day_range,
    parse_inspection_timestamp,
    rolling_days_range,
)

HISTORY_REFRESH_DEBOUNCE_MS = 750
_PERIOD_TODAY = "TODAY"
_PERIOD_SEVEN_DAYS = "SEVEN_DAYS"
_PERIOD_ALL = "ALL"
_PERIOD_CUSTOM = "CUSTOM"

_STATUS_COLORS = {
    "PASS": QColor("#dcfce7"),
    "FAIL": QColor("#fee2e2"),
    "DETECTION_FAIL": QColor("#fee2e2"),
    "NG": QColor("#fee2e2"),
    "ERROR": QColor("#fef3c7"),
    "INFERENCE_ERROR": QColor("#fef3c7"),
}

_REASON_TRANSLATION_KEYS = {
    "MISSING": "history_reason_missing",
    "WRONG_COMPONENT": "history_reason_wrong_component",
    "POSITION_SHIFT": "history_reason_position_shift",
    "BOARD_ALIGNMENT": "history_reason_board_alignment",
    "UNEXPECTED_COMPONENT": "history_reason_unexpected_component",
    "LOW_CONFIDENCE": "history_reason_low_confidence",
    "COLOR_MISMATCH": "history_reason_color_mismatch",
    "SEQUENCE_MISMATCH": "history_reason_sequence_mismatch",
    "ANOMALY_DETECTED": "history_reason_anomaly_detected",
    "INFERENCE_ERROR": "history_reason_inference_error",
}

_ACTIVE_HISTORY_WORKERS: set[QThread] = set()


class _HistoryPreview(ImageViewer):
    """Read-only history preview that emits only for a loaded image."""

    activated = pyqtSignal()

    def mouseReleaseEvent(self, event) -> None:
        super().mouseReleaseEvent(event)
        pixmap = self.pixmap()
        if (
            event.button() == Qt.LeftButton
            and pixmap is not None
            and not pixmap.isNull()
        ):
            self.activated.emit()


class _HistoryImageDialog(QDialog):
    """Minimal modal viewer for one saved inspection image."""

    def __init__(
        self,
        image_path: Path,
        *,
        language: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._language = normalize_language(language)
        self.setWindowTitle(tr(self._language, "history_image_title"))
        self.setModal(True)
        self.resize(960, 680)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.viewer = ImageViewer(image_path.name)
        self.viewer.set_language(self._language)
        self.viewer.setMinimumSize(720, 520)
        self.viewer.set_image(str(image_path))
        layout.addWidget(self.viewer, 1)

        close_button = QPushButton(
            tr(self._language, "history_image_close"),
            self,
        )
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, 0, Qt.AlignRight)


@dataclass(frozen=True)
class _HistoryLoadRequest:
    generation: int
    filters: InspectionHistoryFilters
    time_range: InspectionTimeRange | None
    product: str
    station: str
    status: str
    period: str


class _HistoryLoadWorker(QThread):
    loaded = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(
        self,
        service: InspectionHistoryService,
        request: _HistoryLoadRequest,
    ) -> None:
        super().__init__()
        self.service = service
        self.request = request

    def run(self) -> None:
        try:
            snapshot = self.service.load(self.request.filters)
        except (InspectionHistoryError, ValueError) as exc:
            self.failed.emit(self.request.generation, str(exc))
            return
        self.loaded.emit(self.request.generation, snapshot)


class _HistoryExportWorker(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    progress = pyqtSignal(int, int)

    def __init__(
        self,
        service: InspectionExcelReportService,
        destination: Path,
        filters: InspectionHistoryFilters,
        language: str,
    ) -> None:
        super().__init__()
        self.service = service
        self.destination = destination
        self.filters = filters
        self.language = language
        self._cancel_event = Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def run(self) -> None:
        try:
            result = self.service.export(
                self.destination,
                self.filters,
                language=self.language,
                progress_callback=self.progress.emit,
                is_cancelled=self._cancel_event.is_set,
            )
        except InspectionExcelReportCancelled:
            self.cancelled.emit()
            return
        except (
            InspectionExcelReportError,
            OSError,
            ValueError,
        ) as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(result)


class InspectionHistoryPage(QWidget):
    """Present bounded history queries without exposing database mutation."""

    back_to_inspection_requested = pyqtSignal()

    def __init__(
        self,
        database_path: str | Path,
        *,
        language: str,
        service: InspectionHistoryService | None = None,
        report_service: InspectionExcelReportService | None = None,
        now_provider: Callable[[], datetime] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("InspectionHistoryPage")
        self._language = normalize_language(language)
        self._database_path = Path(database_path)
        self._service = service or InspectionHistoryService(database_path)
        self._report_service = report_service or InspectionExcelReportService(
            database_path
        )
        self._now_provider = now_provider or (
            lambda: datetime.now(HISTORY_TIMEZONE)
        )
        self._records: tuple[InspectionHistoryRecord, ...] = ()
        self._selected_preview_path = ""
        self._dirty = True
        self._preferred_target = ("", "")
        self._summary_total = 0
        self._sync_enabled = False
        self._load_error = False
        self._active_time_range: InspectionTimeRange | None = None
        self._active_product = ""
        self._active_station = ""
        self._active_status = ""
        self._active_period = _PERIOD_TODAY
        self._page_offset = 0
        self._filtered_total = 0
        self._load_generation = 0
        self._active_worker: _HistoryLoadWorker | None = None
        self._export_worker: _HistoryExportWorker | None = None
        self._pending_request: _HistoryLoadRequest | None = None
        self._closed = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(HISTORY_REFRESH_DEBOUNCE_MS)
        self._refresh_timer.timeout.connect(self._refresh_if_dirty)

        self.setStyleSheet(
            "QWidget#InspectionHistoryPage {"
            "background:#f3f6fa;"
            "font-family:'Segoe UI','Microsoft JhengHei';font-size:10pt;}"
            "QFrame#historyHeader,QFrame#historyFilterCard,"
            "QFrame#historySummaryCard,QFrame#historyDetailCard {"
            "background:white;border:1px solid #dce3ec;border-radius:9px;}"
            "QPushButton#historyBackButton {"
            "background:transparent;color:#34506b;border:0;padding:8px 12px;"
            "font-weight:600;}"
            "QPushButton#historyBackButton:hover {"
            "background:#edf3f8;border-radius:6px;}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(8)
        root.addWidget(self._build_header())
        root.addWidget(self._build_filter_card())
        self.summary_scope_label = QLabel(self)
        self.summary_scope_label.setWordWrap(True)
        self.summary_scope_label.setStyleSheet(
            "background:#eef4fa;color:#334e68;border:1px solid #d4e1ec;"
            "border-radius:7px;padding:7px 10px;font-size:9pt;"
        )
        root.addWidget(self.summary_scope_label)
        root.addLayout(self._build_summary_row())
        root.addWidget(self._build_content(), 1)
        self.set_language(self._language)

    def _build_header(self) -> QFrame:
        header = QFrame(self)
        header.setObjectName("historyHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(8, 6, 12, 6)
        layout.setSpacing(12)

        self.back_button = QPushButton(header)
        self.back_button.setObjectName("historyBackButton")
        self.back_button.clicked.connect(self.back_to_inspection_requested.emit)
        layout.addWidget(self.back_button)

        divider = QFrame(header)
        divider.setFrameShape(QFrame.VLine)
        divider.setStyleSheet("color:#dce3ec;")
        layout.addWidget(divider)

        title_column = QVBoxLayout()
        title_column.setSpacing(2)
        self.title_label = QLabel(header)
        self.title_label.setStyleSheet(
            "font-size:14pt;font-weight:700;color:#1f3347;border:0;"
        )
        title_column.addWidget(self.title_label)
        self.hint_label = QLabel(header)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            "color:#5b6b7c;font-size:9pt;border:0;"
        )
        title_column.addWidget(self.hint_label)
        layout.addLayout(title_column, 1)

        self.scope_badge = QLabel(header)
        self.scope_badge.setAlignment(Qt.AlignCenter)
        self.scope_badge.setStyleSheet(
            "background:#eef6ff;color:#245b8f;border:1px solid #bdd7ee;"
            "border-radius:11px;padding:4px 10px;font-weight:600;"
        )
        layout.addWidget(self.scope_badge)
        self.sync_badge = QLabel(header)
        self.sync_badge.setAlignment(Qt.AlignCenter)
        self.sync_badge.setStyleSheet(
            "background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;"
            "border-radius:11px;padding:4px 10px;font-weight:600;"
        )
        layout.addWidget(self.sync_badge)
        self.previous_page_button = QPushButton(header)
        self.previous_page_button.clicked.connect(self._show_previous_page)
        layout.addWidget(self.previous_page_button)
        self.page_label = QLabel(header)
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setMinimumWidth(90)
        layout.addWidget(self.page_label)
        self.next_page_button = QPushButton(header)
        self.next_page_button.clicked.connect(self._show_next_page)
        layout.addWidget(self.next_page_button)
        return header

    def _build_filter_card(self) -> QFrame:
        card = QFrame(self)
        card.setObjectName("historyFilterCard")
        layout = QGridLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        self.product_label = QLabel(card)
        self.product_combo = QComboBox(card)
        self.product_combo.setMinimumWidth(140)
        self.product_combo.currentIndexChanged.connect(
            self._on_product_filter_changed
        )
        layout.addWidget(self.product_label, 0, 0)
        layout.addWidget(self.product_combo, 0, 1)

        self.station_label = QLabel(card)
        self.station_combo = QComboBox(card)
        self.station_combo.setMinimumWidth(110)
        self.station_combo.currentIndexChanged.connect(
            self._on_filter_changed
        )
        layout.addWidget(self.station_label, 0, 2)
        layout.addWidget(self.station_combo, 0, 3)

        self.status_label = QLabel(card)
        self.status_combo = QComboBox(card)
        self.status_combo.setMinimumWidth(120)
        self.status_combo.currentIndexChanged.connect(
            self._on_filter_changed
        )
        layout.addWidget(self.status_label, 0, 4)
        layout.addWidget(self.status_combo, 0, 5)

        self.limit_label = QLabel(card)
        self.limit_combo = QComboBox(card)
        for limit in (100, 250, 500):
            self.limit_combo.addItem(str(limit), limit)
        self.limit_combo.currentIndexChanged.connect(self._on_filter_changed)
        layout.addWidget(self.limit_label, 0, 6)
        layout.addWidget(self.limit_combo, 0, 7)

        self.period_label = QLabel(card)
        self.period_combo = QComboBox(card)
        self.period_combo.setMinimumWidth(140)
        self.period_combo.currentIndexChanged.connect(
            self._on_period_changed
        )
        layout.addWidget(self.period_label, 1, 0)
        layout.addWidget(self.period_combo, 1, 1)

        initial_now = self._now_provider()
        if not isinstance(initial_now, datetime) or initial_now.tzinfo is None:
            raise ValueError(
                "Inspection history clock must provide timezone-aware time."
            )
        local_today = initial_now.astimezone(HISTORY_TIMEZONE).date()
        today = QDate(
            local_today.year,
            local_today.month,
            local_today.day,
        )
        self.custom_start_label = QLabel(card)
        self.custom_start_date = QDateEdit(today.addDays(-7), card)
        self.custom_start_date.setCalendarPopup(True)
        self.custom_start_date.setDisplayFormat("yyyy-MM-dd")
        self.custom_start_date.setMinimumDate(QDate(2000, 1, 1))
        self.custom_start_date.setMaximumDate(today.addYears(20))
        self.custom_start_date.dateChanged.connect(self._on_filter_changed)
        layout.addWidget(self.custom_start_label, 1, 2)
        layout.addWidget(self.custom_start_date, 1, 3)

        self.custom_end_label = QLabel(card)
        self.custom_end_date = QDateEdit(today, card)
        self.custom_end_date.setCalendarPopup(True)
        self.custom_end_date.setDisplayFormat("yyyy-MM-dd")
        self.custom_end_date.setMinimumDate(QDate(2000, 1, 1))
        self.custom_end_date.setMaximumDate(today.addYears(20))
        self.custom_end_date.dateChanged.connect(self._on_filter_changed)
        layout.addWidget(self.custom_end_label, 1, 4)
        layout.addWidget(self.custom_end_date, 1, 5)

        self.refresh_button = QPushButton(card)
        self.refresh_button.setObjectName("secondaryAction")
        self.refresh_button.clicked.connect(self.refresh)
        layout.addWidget(self.refresh_button, 1, 7)
        self.export_button = QPushButton(card)
        self.export_button.setObjectName("primaryAction")
        self.export_button.clicked.connect(self._on_export_button_clicked)
        layout.addWidget(self.export_button, 1, 6)
        self._set_custom_date_visibility(False)
        return card

    def _build_summary_row(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setSpacing(10)
        self.total_value, self.total_caption = self._summary_card(layout)
        self.pass_value, self.pass_caption = self._summary_card(layout)
        self.fail_value, self.fail_caption = self._summary_card(layout)
        self.error_value, self.error_caption = self._summary_card(layout)
        self.yield_value, self.yield_caption = self._summary_card(layout)
        return layout

    def _summary_card(
        self,
        parent_layout: QHBoxLayout,
    ) -> tuple[QLabel, QLabel]:
        card = QFrame(self)
        card.setObjectName("historySummaryCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 5, 12, 5)
        layout.setSpacing(1)
        value = QLabel("0", card)
        value.setAlignment(Qt.AlignCenter)
        value.setStyleSheet(
            "font-size:15pt;font-weight:700;color:#1f3347;border:0;"
        )
        caption = QLabel(card)
        caption.setAlignment(Qt.AlignCenter)
        caption.setStyleSheet("color:#617286;font-size:9pt;border:0;")
        layout.addWidget(value)
        layout.addWidget(caption)
        parent_layout.addWidget(card, 1)
        return value, caption

    def _build_content(self) -> QSplitter:
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        self.table = QTableWidget(0, 8, splitter)
        self.table.setObjectName("inspectionHistoryTable")
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        vertical_header = self.table.verticalHeader()
        assert vertical_header is not None
        vertical_header.setVisible(False)
        header = self.table.horizontalHeader()
        assert header is not None
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._show_selected_record)
        splitter.addWidget(self.table)

        detail = QFrame(splitter)
        detail.setObjectName("historyDetailCard")
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(14, 12, 14, 12)
        detail_layout.setSpacing(8)
        self.detail_title = QLabel(detail)
        self.detail_title.setStyleSheet(
            "font-size:12pt;font-weight:700;color:#1f3347;border:0;"
        )
        detail_layout.addWidget(self.detail_title)

        self.preview = _HistoryPreview()
        self.preview.setObjectName("inspectionHistoryPreview")
        self.preview.setMinimumSize(300, 160)
        self.preview.setMaximumHeight(260)
        self.preview.setCursor(Qt.PointingHandCursor)
        self.preview.activated.connect(self._open_large_preview)
        detail_layout.addWidget(self.preview)

        self.detail_grid = QGridLayout()
        self.detail_grid.setHorizontalSpacing(12)
        self.detail_grid.setVerticalSpacing(3)
        self._detail_labels: dict[str, tuple[QLabel, QLabel]] = {}
        for row, key in enumerate(
            (
                "status",
                "target",
                "timestamp",
                "detector",
                "latency",
                "model",
                "equipment",
                "review",
                "reason",
            )
        ):
            name = QLabel(detail)
            name.setStyleSheet(
                "color:#617286;font-weight:600;border:0;font-size:9pt;"
            )
            value = QLabel(detail)
            value.setWordWrap(True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            value.setStyleSheet(
                "color:#243b53;border:0;font-size:9pt;"
            )
            self.detail_grid.addWidget(name, row, 0, Qt.AlignTop)
            self.detail_grid.addWidget(value, row, 1)
            self._detail_labels[key] = (name, value)
        self.detail_grid.setColumnStretch(1, 1)
        detail_layout.addLayout(self.detail_grid)
        detail_layout.addStretch()
        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([850, 470])
        return splitter

    def show_for_target(self, product: str, station: str) -> None:
        self._preferred_target = (
            str(product).strip(),
            str(station).strip(),
        )
        self._page_offset = 0
        self._refresh_timer.stop()
        self.refresh()

    def set_database_context(
        self,
        database_path: str | Path,
        *,
        sync_enabled: bool,
    ) -> None:
        """Switch to the configured Result directory while the page is idle."""
        path = Path(database_path)
        if path != self._database_path:
            if self.is_loading or (
                self._export_worker is not None
                and self._export_worker.isRunning()
            ):
                raise RuntimeError(
                    "Inspection history database cannot change during an "
                    "active read or export."
                )
            self._database_path = path
            self._service = InspectionHistoryService(path)
            self._report_service = InspectionExcelReportService(path)
            self._dirty = True
        self._sync_enabled = bool(sync_enabled)
        self._update_sync_badge()

    def mark_dirty(self) -> None:
        """Coalesce rapid storage events into one GUI refresh."""
        self._dirty = True
        if self.isVisible():
            self._refresh_timer.start()

    def _refresh_if_dirty(self) -> None:
        if self._dirty and self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        """Queue one bounded SQLite snapshot without blocking the GUI thread."""
        if self._closed:
            return
        self._refresh_timer.stop()
        limit = int(self.limit_combo.currentData() or 100)
        try:
            filters, time_range, product, station, status, period = (
                self._collect_filter_selection(
                    limit=limit,
                    offset=self._page_offset,
                )
            )
        except ValueError as exc:
            self._load_generation += 1
            self._pending_request = None
            self._render_error(str(exc))
            self._update_page_controls()
            return

        self._load_generation += 1
        request = _HistoryLoadRequest(
            generation=self._load_generation,
            filters=filters,
            time_range=time_range,
            product=product,
            station=station,
            status=status,
            period=period,
        )
        if self._active_worker is not None and self._active_worker.isRunning():
            self._pending_request = request
            self._render_loading()
            return
        self._start_load(request)

    def _collect_filter_selection(
        self,
        *,
        limit: int,
        offset: int,
    ) -> tuple[
        InspectionHistoryFilters,
        InspectionTimeRange | None,
        str,
        str,
        str,
        str,
    ]:
        product = self._selected_data(
            self.product_combo,
            fallback=self._preferred_target[0],
        )
        station = self._selected_data(
            self.station_combo,
            fallback=self._preferred_target[1],
        )
        status = self._selected_data(self.status_combo)
        period = self._selected_data(
            self.period_combo,
            fallback=_PERIOD_TODAY,
        )
        time_range = self._resolve_time_range(period)
        filters = InspectionHistoryFilters(
            product=product,
            station=station,
            status=status,
            limit=limit,
            offset=offset,
            started_at=(
                time_range.started_at if time_range is not None else None
            ),
            ended_at=(
                time_range.ended_at if time_range is not None else None
            ),
        ).normalized()
        return filters, time_range, product, station, status, period

    def _export_excel(self) -> None:
        if self._closed:
            return
        try:
            filters, _, _, _, _, _ = self._collect_filter_selection(
                limit=500,
                offset=0,
            )
        except ValueError as exc:
            QMessageBox.warning(
                self,
                tr(self._language, "history_export_error_title"),
                tr(self._language, "history_filter_error").format(error=exc),
            )
            return
        timestamp = self._now_provider().astimezone(HISTORY_TIMEZONE).strftime(
            "%Y%m%d_%H%M%S"
        )
        default_path = self._database_path.parent / (
            f"inspection_report_{timestamp}.xlsx"
        )
        destination, _ = QFileDialog.getSaveFileName(
            self,
            tr(self._language, "history_export_dialog_title"),
            str(default_path),
            "Excel Workbook (*.xlsx)",
        )
        if not destination:
            return
        destination_path = Path(destination)
        if destination_path.suffix.lower() != ".xlsx":
            destination_path = destination_path.with_suffix(".xlsx")

        worker = _HistoryExportWorker(
            self._report_service,
            destination_path,
            filters,
            self._language,
        )
        self._export_worker = worker
        _ACTIVE_HISTORY_WORKERS.add(worker)
        worker.succeeded.connect(self._on_export_succeeded)
        worker.failed.connect(self._on_export_failed)
        worker.cancelled.connect(self._on_export_cancelled)
        worker.progress.connect(self._on_export_progress)
        worker.finished.connect(lambda: self._on_export_finished(worker))
        self.export_button.setEnabled(True)
        self.export_button.setText(
            tr(self._language, "history_export_cancel")
        )
        worker.start()

    def _on_export_button_clicked(self) -> None:
        worker = self._export_worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            self.export_button.setEnabled(False)
            self.export_button.setText(
                tr(self._language, "history_export_cancelling")
            )
            return
        self._export_excel()

    def _on_export_progress(self, current: int, total: int) -> None:
        if self._closed or total <= 0:
            return
        percent = min(100, max(0, round((current / total) * 100)))
        self.export_button.setText(
            tr(self._language, "history_export_cancel_progress").format(
                percent=percent
            )
        )

    def _on_export_succeeded(self, result: InspectionExcelReport) -> None:
        if self._closed:
            return
        answer = QMessageBox.question(
            self,
            tr(self._language, "history_export_success_title"),
            tr(self._language, "history_export_success").format(
                count=f"{result.record_count:,}",
                path=result.path,
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(result.path.parent))
            )

    def _on_export_cancelled(self) -> None:
        if self._closed:
            return
        self.summary_scope_label.setText(
            tr(self._language, "history_export_cancelled")
        )

    def _on_export_failed(self, message: str) -> None:
        if self._closed:
            return
        QMessageBox.critical(
            self,
            tr(self._language, "history_export_error_title"),
            tr(self._language, "history_export_error").format(error=message),
        )

    def _on_export_finished(self, worker: _HistoryExportWorker) -> None:
        _ACTIVE_HISTORY_WORKERS.discard(worker)
        worker.deleteLater()
        if self._export_worker is worker:
            self._export_worker = None
        if not self._closed:
            self.export_button.setEnabled(True)
            self.export_button.setText(
                tr(self._language, "history_export_excel")
            )

    def _start_load(self, request: _HistoryLoadRequest) -> None:
        self._pending_request = None
        worker = _HistoryLoadWorker(self._service, request)
        self._active_worker = worker
        _ACTIVE_HISTORY_WORKERS.add(worker)
        worker.loaded.connect(self._on_history_loaded)
        worker.failed.connect(self._on_history_failed)
        worker.finished.connect(lambda: self._on_load_finished(worker))
        self._render_loading()
        worker.start()

    def _on_history_loaded(
        self,
        generation: int,
        snapshot: InspectionHistorySnapshot,
    ) -> None:
        if self._closed or generation != self._load_generation:
            return
        request = (
            self._active_worker.request
            if self._active_worker is not None
            else None
        )
        if request is None or request.generation != generation:
            return
        if self._page_offset >= snapshot.filtered_total and self._page_offset:
            self._page_offset = max(
                0,
                ((max(0, snapshot.filtered_total - 1)) // request.filters.limit)
                * request.filters.limit,
            )
            self.refresh()
            return

        self._dirty = False
        self._preferred_target = ("", "")
        self._active_time_range = request.time_range
        self._active_product = request.product
        self._active_station = request.station
        self._active_status = request.status
        self._active_period = request.period
        self._filtered_total = snapshot.filtered_total
        self._rebuild_target_filters(
            snapshot,
            selected_product=request.product,
            selected_station=request.station,
        )
        self._render_snapshot(snapshot)
        self._update_sync_badge(snapshot)
        self._update_page_controls()

    def _on_history_failed(self, generation: int, message: str) -> None:
        if self._closed or generation != self._load_generation:
            return
        self._filtered_total = 0
        self._render_error(message)
        self._update_page_controls()

    def _on_load_finished(self, worker: _HistoryLoadWorker) -> None:
        _ACTIVE_HISTORY_WORKERS.discard(worker)
        worker.deleteLater()
        if self._active_worker is worker:
            self._active_worker = None
        if self._closed:
            return
        pending = self._pending_request
        if pending is not None:
            self._start_load(pending)
            return
        self.refresh_button.setEnabled(True)
        self._update_page_controls()

    def _render_loading(self) -> None:
        self.refresh_button.setEnabled(False)
        self.previous_page_button.setEnabled(False)
        self.next_page_button.setEnabled(False)
        self.summary_scope_label.setText(
            tr(self._language, "history_loading")
        )

    @property
    def is_loading(self) -> bool:
        return (
            self._active_worker is not None
            and self._active_worker.isRunning()
        ) or self._pending_request is not None

    def shutdown(self) -> None:
        """Ignore late worker results while allowing SQLite reads to finish."""
        self._closed = True
        self._load_generation += 1
        self._pending_request = None
        worker = self._active_worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
        export_worker = self._export_worker
        if export_worker is not None and export_worker.isRunning():
            export_worker.cancel()
            export_worker.requestInterruption()

    def _show_previous_page(self) -> None:
        limit = int(self.limit_combo.currentData() or 100)
        self._page_offset = max(0, self._page_offset - limit)
        self.refresh()

    def _show_next_page(self) -> None:
        limit = int(self.limit_combo.currentData() or 100)
        if self._page_offset + limit >= self._filtered_total:
            return
        self._page_offset += limit
        self.refresh()

    def _update_page_controls(self) -> None:
        limit = int(self.limit_combo.currentData() or 100)
        total_pages = max(1, (self._filtered_total + limit - 1) // limit)
        current_page = min(total_pages, (self._page_offset // limit) + 1)
        self.page_label.setText(
            tr(self._language, "history_page_number").format(
                current=current_page,
                total=total_pages,
            )
        )
        self.previous_page_button.setEnabled(
            not self.is_loading and self._page_offset > 0
        )
        self.next_page_button.setEnabled(
            not self.is_loading
            and self._page_offset + limit < self._filtered_total
        )

    def _resolve_time_range(
        self,
        period: str,
    ) -> InspectionTimeRange | None:
        now = self._now_provider()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ValueError(
                "Inspection history clock must provide timezone-aware time."
            )
        if period == _PERIOD_ALL:
            return None
        if period == _PERIOD_TODAY:
            return local_day_range(
                now.astimezone(HISTORY_TIMEZONE).date()
            )
        if period == _PERIOD_SEVEN_DAYS:
            return rolling_days_range(now, days=7)
        if period == _PERIOD_CUSTOM:
            start_day = self.custom_start_date.date().toPyDate()
            end_day = self.custom_end_date.date().toPyDate()
            start_range = local_day_range(start_day)
            end_range = local_day_range(end_day)
            return InspectionTimeRange(
                started_at=start_range.started_at,
                ended_at=end_range.ended_at,
            )
        raise ValueError(f"Unsupported inspection period: {period}")

    @staticmethod
    def _selected_data(combo: QComboBox, *, fallback: str = "") -> str:
        if combo.count() == 0:
            return fallback
        return str(combo.currentData() or "").strip()

    def _rebuild_target_filters(
        self,
        snapshot: InspectionHistorySnapshot,
        *,
        selected_product: str,
        selected_station: str,
    ) -> None:
        self.product_combo.blockSignals(True)
        self.station_combo.blockSignals(True)
        try:
            products = list(snapshot.products)
            if selected_product and selected_product not in products:
                products.append(selected_product)
            stations = list(snapshot.stations)
            if selected_station and selected_station not in stations:
                stations.append(selected_station)
            self._fill_filter_combo(
                self.product_combo,
                products,
                selected_product,
            )
            self._fill_filter_combo(
                self.station_combo,
                stations,
                selected_station,
            )
        finally:
            self.product_combo.blockSignals(False)
            self.station_combo.blockSignals(False)

    def _fill_filter_combo(
        self,
        combo: QComboBox,
        values: list[str],
        selected: str,
    ) -> None:
        combo.clear()
        combo.addItem(tr(self._language, "history_all"), "")
        for value in sorted(set(values), key=str.casefold):
            if value:
                combo.addItem(value, value)
        index = combo.findData(selected)
        combo.setCurrentIndex(max(0, index))

    def _render_snapshot(self, snapshot: InspectionHistorySnapshot) -> None:
        self._records = snapshot.records
        summary = snapshot.summary
        self._summary_total = summary.total
        self._load_error = False
        self.total_value.setText(f"{summary.total:,}")
        self.pass_value.setText(f"{summary.passed:,}")
        self.fail_value.setText(f"{summary.failed:,}")
        self.error_value.setText(f"{summary.errors:,}")
        self.yield_value.setText(
            "--"
            if summary.yield_rate is None
            else f"{summary.yield_rate:.1f}%"
        )
        self.scope_badge.setText(
            tr(self._language, "history_records_shown").format(
                shown=len(snapshot.records),
                total=snapshot.filtered_total,
            )
        )
        self.scope_badge.setToolTip("")
        self._update_summary_scope_label()

        self.table.setRowCount(len(snapshot.records))
        for row_index, record in enumerate(snapshot.records):
            localized_reason = _localized_reasons(
                record.reason_codes,
                self._language,
            )
            values = (
                _display_timestamp(record.timestamp),
                _display_status(record.status),
                record.product or "--",
                record.station or "--",
                record.detector or "--",
                _display_latency(record.inference_time),
                localized_reason or "--",
                record.model_version or "--",
            )
            background = _STATUS_COLORS.get(record.status.upper())
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, row_index)
                if column == 6 and record.reason:
                    item.setToolTip(record.reason)
                if background is not None:
                    item.setBackground(background)
                self.table.setItem(row_index, column, item)

        if snapshot.records:
            self.table.selectRow(0)
        else:
            self._clear_selected_record()

    def _render_error(self, message: str) -> None:
        self._dirty = True
        self._records = ()
        self._summary_total = 0
        self._load_error = True
        for value_label in (
            self.total_value,
            self.pass_value,
            self.fail_value,
            self.error_value,
            self.yield_value,
        ):
            value_label.setText("--")
        self.table.setRowCount(0)
        self._clear_selected_record()
        self.scope_badge.setText(tr(self._language, "history_load_error"))
        self.scope_badge.setToolTip(message)
        self.summary_scope_label.setText(
            tr(self._language, "history_filter_error").format(error=message)
        )

    def _update_sync_badge(
        self,
        snapshot: InspectionHistorySnapshot | None = None,
    ) -> None:
        if not self._sync_enabled:
            self.sync_badge.setText(
                tr(self._language, "history_sync_disabled")
            )
            self.sync_badge.setStyleSheet(
                "background:#f1f5f9;color:#475569;border:1px solid #cbd5e1;"
                "border-radius:11px;padding:4px 10px;font-weight:600;"
            )
            return
        if snapshot is None:
            self.sync_badge.setText(
                tr(self._language, "history_sync_checking")
            )
            return
        sync = snapshot.sync
        if sync.dead:
            key = "history_sync_failed"
            value = sync.dead
            colors = ("#fee2e2", "#991b1b", "#fecaca")
        elif sync.pending or sync.inflight:
            key = "history_sync_pending"
            value = sync.pending + sync.inflight
            colors = ("#fef3c7", "#92400e", "#fde68a")
        else:
            key = "history_sync_current"
            value = sync.synced
            colors = ("#dcfce7", "#166534", "#bbf7d0")
        self.sync_badge.setText(
            tr(self._language, key).format(count=f"{value:,}")
        )
        background, foreground, border = colors
        self.sync_badge.setStyleSheet(
            f"background:{background};color:{foreground};"
            f"border:1px solid {border};border-radius:11px;"
            "padding:4px 10px;font-weight:600;"
        )

    def _show_selected_record(self) -> None:
        selected = self.table.selectedItems()
        if not selected:
            self._clear_selected_record()
            return
        first = self.table.item(selected[0].row(), 0)
        if first is None:
            self._clear_selected_record()
            return
        row_index = first.data(Qt.UserRole)
        if type(row_index) is not int or not 0 <= row_index < len(self._records):
            self._clear_selected_record()
            return
        record = self._records[row_index]
        if record.preview_path and Path(record.preview_path).is_file():
            self._selected_preview_path = record.preview_path
            self.preview.set_image(record.preview_path)
        else:
            self._selected_preview_path = ""
            self.preview.clear()
        self._set_detail_values(record)

    def _clear_selected_record(self) -> None:
        self._selected_preview_path = ""
        self.preview.clear()
        self._set_detail_values(None)

    def _open_large_preview(self) -> None:
        image_path = Path(self._selected_preview_path)
        if not self._selected_preview_path or not image_path.is_file():
            QMessageBox.warning(
                self,
                tr(self._language, "history_image_unavailable_title"),
                tr(self._language, "history_image_unavailable"),
            )
            return
        _HistoryImageDialog(
            image_path,
            language=self._language,
            parent=self,
        ).exec_()

    def _set_detail_values(
        self,
        record: InspectionHistoryRecord | None,
    ) -> None:
        values = {
            "status": "--",
            "target": "--",
            "timestamp": "--",
            "detector": "--",
            "latency": "--",
            "model": "--",
            "equipment": "--",
            "review": "--",
            "reason": "--",
        }
        if record is not None:
            equipment = " / ".join(
                value
                for value in (
                    record.machine_id,
                    record.work_order,
                    record.camera_id,
                )
                if value
            )
            review = " / ".join(
                value
                for value in (
                    record.review_outcome,
                    record.failure_category,
                )
                if value
            )
            localized_reason = _localized_reasons(
                record.reason_codes,
                self._language,
            )
            values.update(
                {
                    "status": _display_status(record.status),
                    "target": f"{record.product or '--'} / {record.station or '--'}",
                    "timestamp": _display_timestamp(record.timestamp),
                    "detector": record.detector or "--",
                    "latency": _display_latency(record.inference_time),
                    "model": record.model_version or "--",
                    "equipment": equipment or "--",
                    "review": review or "--",
                    "reason": localized_reason or "--",
                }
            )
        for key, value in values.items():
            self._detail_labels[key][1].setText(value)
        self._detail_labels["reason"][1].setToolTip(
            record.reason if record is not None else ""
        )

    def _update_summary_scope_label(self) -> None:
        target_parts = [
            value
            for value in (self._active_product, self._active_station)
            if value
        ]
        target = (
            " / ".join(target_parts)
            if target_parts
            else tr(self._language, "history_all_targets")
        )
        status = _status_filter_label(
            self._active_status,
            self._language,
        )
        period = self._period_scope_text()
        self.summary_scope_label.setText(
            tr(self._language, "history_scope_summary").format(
                target=target,
                period=period,
                status=status,
            )
        )

    def _period_scope_text(self) -> str:
        time_range = self._active_time_range
        if self._active_period == _PERIOD_ALL or time_range is None:
            return str(tr(self._language, "history_period_all"))
        start_text = time_range.started_at.strftime("%Y-%m-%d %H:%M")
        end_text = time_range.ended_at.strftime("%Y-%m-%d %H:%M")
        if self._active_period == _PERIOD_TODAY:
            return str(
                tr(self._language, "history_scope_today").format(
                    date=time_range.started_at.strftime("%Y-%m-%d")
                )
            )
        if self._active_period == _PERIOD_SEVEN_DAYS:
            return str(
                tr(self._language, "history_scope_range").format(
                    label=tr(self._language, "history_period_seven_days"),
                    start=start_text,
                    end=end_text,
                )
            )
        inclusive_end = time_range.ended_at - timedelta(days=1)
        return str(
            tr(self._language, "history_scope_custom").format(
                start=time_range.started_at.strftime("%Y-%m-%d"),
                end=inclusive_end.strftime("%Y-%m-%d"),
            )
        )

    def _on_filter_changed(self) -> None:
        self._preferred_target = ("", "")
        self._page_offset = 0
        self.refresh()

    def _on_period_changed(self) -> None:
        self._preferred_target = ("", "")
        self._page_offset = 0
        period = self._selected_data(
            self.period_combo,
            fallback=_PERIOD_TODAY,
        )
        self._set_custom_date_visibility(period == _PERIOD_CUSTOM)
        self.refresh()

    def _set_custom_date_visibility(self, visible: bool) -> None:
        for widget in (
            self.custom_start_label,
            self.custom_start_date,
            self.custom_end_label,
            self.custom_end_date,
        ):
            widget.setVisible(visible)

    def _on_product_filter_changed(self) -> None:
        """A station from the previous product must not leak into a new scope."""
        self._preferred_target = ("", "")
        self._page_offset = 0
        self.station_combo.blockSignals(True)
        try:
            if self.station_combo.count():
                self.station_combo.setCurrentIndex(0)
        finally:
            self.station_combo.blockSignals(False)
        self.refresh()

    def _rebuild_status_filter(self) -> None:
        selected = self._selected_data(self.status_combo)
        self.status_combo.blockSignals(True)
        try:
            self.status_combo.clear()
            for key, translation_key in (
                ("", "history_all"),
                ("PASS", "history_status_pass"),
                ("FAIL", "history_status_fail"),
                ("ERROR", "history_status_error"),
            ):
                self.status_combo.addItem(
                    tr(self._language, translation_key),
                    key,
                )
            self.status_combo.setCurrentIndex(
                max(0, self.status_combo.findData(selected))
            )
        finally:
            self.status_combo.blockSignals(False)

    def _rebuild_period_filter(self) -> None:
        selected = self._selected_data(
            self.period_combo,
            fallback=_PERIOD_TODAY,
        )
        self.period_combo.blockSignals(True)
        try:
            self.period_combo.clear()
            for key, translation_key in (
                (_PERIOD_TODAY, "history_period_today"),
                (_PERIOD_SEVEN_DAYS, "history_period_seven_days"),
                (_PERIOD_ALL, "history_period_all"),
                (_PERIOD_CUSTOM, "history_period_custom"),
            ):
                self.period_combo.addItem(
                    tr(self._language, translation_key),
                    key,
                )
            self.period_combo.setCurrentIndex(
                max(0, self.period_combo.findData(selected))
            )
        finally:
            self.period_combo.blockSignals(False)
        self._set_custom_date_visibility(
            self._selected_data(self.period_combo) == _PERIOD_CUSTOM
        )

    def set_language(self, language: str) -> None:
        self._language = normalize_language(language)
        self.back_button.setText(tr(self._language, "history_back"))
        self.title_label.setText(tr(self._language, "history_page_title"))
        self.hint_label.setText(tr(self._language, "history_page_hint"))
        self._update_sync_badge()
        self.product_label.setText(tr(self._language, "product"))
        self.station_label.setText(tr(self._language, "area"))
        self.status_label.setText(tr(self._language, "status"))
        self.limit_label.setText(tr(self._language, "history_limit"))
        self.period_label.setText(tr(self._language, "history_period"))
        self.custom_start_label.setText(
            tr(self._language, "history_custom_start")
        )
        self.custom_end_label.setText(
            tr(self._language, "history_custom_end")
        )
        self.refresh_button.setText(tr(self._language, "history_refresh"))
        self.export_button.setText(
            tr(
                self._language,
                (
                    "history_export_cancel"
                    if self._export_worker is not None
                    and self._export_worker.isRunning()
                    else "history_export_excel"
                ),
            )
        )
        self.previous_page_button.setText(
            tr(self._language, "history_previous_page")
        )
        self.next_page_button.setText(tr(self._language, "history_next_page"))
        self._update_page_controls()
        self.total_caption.setText(tr(self._language, "history_total"))
        self.pass_caption.setText(tr(self._language, "history_pass"))
        self.fail_caption.setText(tr(self._language, "history_fail"))
        self.error_caption.setText(tr(self._language, "history_error"))
        self.yield_caption.setText(tr(self._language, "history_yield"))
        self.table.setHorizontalHeaderLabels(
            [
                tr(self._language, "history_col_time"),
                tr(self._language, "status"),
                tr(self._language, "product"),
                tr(self._language, "area"),
                tr(self._language, "type"),
                tr(self._language, "latency"),
                tr(self._language, "error_reason"),
                tr(self._language, "model_file"),
            ]
        )
        self.detail_title.setText(tr(self._language, "history_detail_title"))
        self.preview.set_language(self._language)
        self.preview.set_title(tr(self._language, "history_preview"))
        self.preview.setToolTip(
            tr(self._language, "history_preview_click_hint")
        )
        detail_keys = {
            "status": "status",
            "target": "product_area",
            "timestamp": "history_col_time",
            "detector": "type",
            "latency": "latency",
            "model": "model_file",
            "equipment": "history_equipment",
            "review": "history_review",
            "reason": "error_reason",
        }
        for key, translation_key in detail_keys.items():
            self._detail_labels[key][0].setText(
                tr(self._language, translation_key)
            )
        for row_index, record in enumerate(self._records):
            reason_item = self.table.item(row_index, 6)
            if reason_item is not None:
                reason_item.setText(
                    _localized_reasons(
                        record.reason_codes,
                        self._language,
                    )
                    or "--"
                )
        if self.table.selectedItems():
            self._show_selected_record()
        self._rebuild_status_filter()
        self._rebuild_period_filter()
        if self.product_combo.count():
            product = self._selected_data(self.product_combo)
            station = self._selected_data(self.station_combo)
            products = [
                str(self.product_combo.itemData(index) or "")
                for index in range(1, self.product_combo.count())
            ]
            stations = [
                str(self.station_combo.itemData(index) or "")
                for index in range(1, self.station_combo.count())
            ]
            self.product_combo.blockSignals(True)
            self.station_combo.blockSignals(True)
            try:
                self._fill_filter_combo(
                    self.product_combo,
                    products,
                    product,
                )
                self._fill_filter_combo(
                    self.station_combo,
                    stations,
                    station,
                )
            finally:
                self.product_combo.blockSignals(False)
                self.station_combo.blockSignals(False)
        if self._load_error:
            self.scope_badge.setText(tr(self._language, "history_load_error"))
            self.summary_scope_label.setText(
                tr(self._language, "history_filter_error").format(
                    error=self.scope_badge.toolTip()
                )
            )
        elif self._records or self._summary_total:
            self.scope_badge.setText(
                tr(self._language, "history_records_shown").format(
                    shown=len(self._records),
                    total=f"{self._filtered_total:,}",
                )
            )
            self._update_summary_scope_label()
        else:
            self.scope_badge.setText(tr(self._language, "history_empty"))
            self._update_summary_scope_label()


def _display_status(status: str) -> str:
    normalized = str(status or "").strip().upper()
    if normalized == "DETECTION_FAIL":
        return "NG"
    if normalized == "INFERENCE_ERROR":
        return "ERROR"
    return normalized or "--"


def _display_latency(seconds: float | None) -> str:
    return "--" if seconds is None else f"{seconds * 1000.0:.1f} ms"


def _display_timestamp(value: str) -> str:
    parsed = parse_inspection_timestamp(value)
    if parsed is None:
        return str(value or "").replace("T", " ")[:19] or "--"
    return str(parsed.strftime("%Y-%m-%d %H:%M:%S"))


def _localized_reasons(
    reason_codes: tuple[str, ...],
    language: str,
) -> str:
    localized: list[str] = []
    normalized_language = normalize_language(language)
    for raw_code in reason_codes:
        code = str(raw_code).strip()
        translation_key = _REASON_TRANSLATION_KEYS.get(code.upper())
        localized.append(
            tr(normalized_language, translation_key)
            if translation_key is not None
            else code
        )
    return "、".join(value for value in localized if value)


def _status_filter_label(status: str, language: str) -> str:
    key = {
        "": "history_all",
        "PASS": "history_status_pass",
        "FAIL": "history_status_fail",
        "ERROR": "history_status_error",
    }.get(status, "history_all")
    return str(tr(normalize_language(language), key))
