from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import app.gui.inspection_history_page as history_page_module
from app.gui.inspection_history_page import InspectionHistoryPage
from core.services.inspection_history import (
    HISTORY_TIMEZONE,
    InspectionHistoryError,
    InspectionHistoryFilters,
    InspectionHistoryService,
    parse_inspection_timestamp,
)
from core.services.inspection_repository import InspectionRepository

_FIXED_NOW = datetime(
    2026,
    7,
    28,
    18,
    30,
    tzinfo=HISTORY_TIMEZONE,
)


def _page(database_path: Path, *, language: str = "zh") -> InspectionHistoryPage:
    return InspectionHistoryPage(
        database_path,
        language=language,
        now_provider=lambda: _FIXED_NOW,
    )


def _wait_page(page: InspectionHistoryPage, qtbot) -> None:
    qtbot.waitUntil(lambda: not page.is_loading, timeout=3000)


def _insert_record(
    repository: InspectionRepository,
    root: Path,
    *,
    name: str,
    timestamp: str,
    status: str,
    product: str = "Cable1",
    station: str = "A",
    reason: object = None,
) -> None:
    repository.upsert_snapshot(
        {
            "timestamp": timestamp,
            "status": status,
            "detector": "yolo",
            "product": product,
            "equipment": {
                "station": station,
                "machine_id": "M-01",
                "work_order": "WO-007",
                "camera_id": "CAM-A",
            },
            "model_info": {"model_version": "v3"},
            "inference_time": 0.123,
            "fail_reasons": reason or [],
            "artifacts": {
                "original_path": str(root / f"{name}-original.jpg"),
                "annotated_path": str(root / f"{name}-annotated.jpg"),
            },
        },
        snapshot_path=root / f"{name}.json",
    )


def _populated_database(tmp_path: Path) -> Path:
    database_path = tmp_path / "Result" / "inspection_records.sqlite3"
    repository = InspectionRepository(database_path)
    _insert_record(
        repository,
        tmp_path,
        name="pass-a",
        timestamp="2026-07-28T08:00:00+00:00",
        status="PASS",
    )
    _insert_record(
        repository,
        tmp_path,
        name="fail-a",
        timestamp="2026-07-28T09:00:00+00:00",
        status="DETECTION_FAIL",
        reason=[{"code": "POSITION_SHIFT"}],
    )
    _insert_record(
        repository,
        tmp_path,
        name="error-b",
        timestamp="2026-07-28T10:00:00+00:00",
        status="INFERENCE_ERROR",
        station="B",
        reason=["MODEL_TIMEOUT"],
    )
    _insert_record(
        repository,
        tmp_path,
        name="other-product",
        timestamp="2026-07-28T11:00:00+00:00",
        status="PASS",
        product="Cable2",
    )
    return database_path


def test_history_service_filters_records_but_summarizes_target_scope(
    tmp_path: Path,
) -> None:
    database_path = _populated_database(tmp_path)
    service = InspectionHistoryService(database_path)

    snapshot = service.load(
        InspectionHistoryFilters(
            product="Cable1",
            station="A",
            status="FAIL",
            limit=100,
        )
    )

    assert snapshot.summary.total == 2
    assert snapshot.summary.passed == 1
    assert snapshot.summary.failed == 1
    assert snapshot.summary.errors == 0
    assert snapshot.summary.yield_rate == 50.0
    assert len(snapshot.records) == 1
    assert snapshot.records[0].status == "DETECTION_FAIL"
    assert snapshot.records[0].reason == "POSITION_SHIFT"
    assert snapshot.products == ("Cable1", "Cable2")
    assert snapshot.stations == ("A", "B")
    assert snapshot.sync.pending == 4


def test_history_service_returns_empty_without_creating_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing" / "inspection_records.sqlite3"

    snapshot = InspectionHistoryService(database_path).load()

    assert snapshot.records == ()
    assert snapshot.summary.total == 0
    assert database_path.exists() is False


@pytest.mark.parametrize(
    "filters",
    (
        InspectionHistoryFilters(status="unknown"),
        InspectionHistoryFilters(limit=0),
        InspectionHistoryFilters(limit=501),
        InspectionHistoryFilters(limit=True),
        InspectionHistoryFilters(offset=-1),
        InspectionHistoryFilters(offset=True),
        InspectionHistoryFilters(
            started_at=datetime(2026, 7, 28),
            ended_at=datetime(2026, 7, 29),
        ),
        InspectionHistoryFilters(
            started_at=_FIXED_NOW,
            ended_at=_FIXED_NOW,
        ),
    ),
)
def test_history_service_rejects_invalid_query_boundaries(
    tmp_path: Path,
    filters: InspectionHistoryFilters,
) -> None:
    with pytest.raises(ValueError):
        InspectionHistoryService(tmp_path / "records.sqlite3").load(filters)


def test_history_service_reports_corrupt_database(tmp_path: Path) -> None:
    database_path = tmp_path / "inspection_records.sqlite3"
    database_path.write_text("not a sqlite database", encoding="utf-8")

    with pytest.raises(InspectionHistoryError):
        InspectionHistoryService(database_path).load()


def test_history_service_filters_aware_and_legacy_local_timestamps(
    tmp_path: Path,
) -> None:
    database_path = _populated_database(tmp_path)
    repository = InspectionRepository(database_path)
    _insert_record(
        repository,
        tmp_path,
        name="legacy-local",
        timestamp="2026-07-28T17:30:00",
        status="PASS",
    )
    started_at = datetime(
        2026,
        7,
        28,
        17,
        0,
        tzinfo=HISTORY_TIMEZONE,
    )
    ended_at = datetime(
        2026,
        7,
        28,
        18,
        30,
        tzinfo=HISTORY_TIMEZONE,
    )

    snapshot = InspectionHistoryService(database_path).load(
        InspectionHistoryFilters(
            product="Cable1",
            started_at=started_at,
            ended_at=ended_at,
        )
    )

    assert snapshot.summary.total == 3
    assert {
        record.timestamp for record in snapshot.records
    } == {
        "2026-07-28T09:00:00+00:00",
        "2026-07-28T10:00:00+00:00",
        "2026-07-28T17:30:00",
    }


def test_history_service_returns_stable_bounded_pages(tmp_path: Path) -> None:
    database_path = _populated_database(tmp_path)
    service = InspectionHistoryService(database_path)

    first = service.load(InspectionHistoryFilters(limit=2, offset=0))
    second = service.load(InspectionHistoryFilters(limit=2, offset=2))

    assert first.filtered_total == 4
    assert second.filtered_total == 4
    assert len(first.records) == 2
    assert len(second.records) == 2
    assert {
        record.inspection_id for record in first.records
    }.isdisjoint(record.inspection_id for record in second.records)


def test_timestamp_parser_converts_aware_values_and_preserves_local_legacy() -> None:
    aware = parse_inspection_timestamp("2026-07-28T05:00:00Z")
    legacy = parse_inspection_timestamp("2026-07-28T13:00:00")

    assert aware is not None
    assert aware.strftime("%Y-%m-%d %H:%M %z") == "2026-07-28 13:00 +0800"
    assert legacy is not None
    assert legacy.strftime("%Y-%m-%d %H:%M %z") == "2026-07-28 13:00 +0800"


def test_history_page_shows_target_summary_and_status_details(
    tmp_path: Path,
    qtbot,
) -> None:
    database_path = _populated_database(tmp_path)
    page = _page(database_path)
    qtbot.addWidget(page)
    page.show()

    page.show_for_target("Cable1", "A")
    _wait_page(page, qtbot)

    assert page.product_combo.currentData() == "Cable1"
    assert page.station_combo.currentData() == "A"
    assert page.total_value.text() == "2"
    assert page.pass_value.text() == "1"
    assert page.fail_value.text() == "1"
    assert page.yield_value.text() == "50.0%"
    assert page.table.rowCount() == 2
    assert page.table.item(0, 1).text() == "NG"
    assert page.table.item(0, 0).text() == "2026-07-28 17:00:00"
    assert page._detail_labels["reason"][1].text() == "位置偏移"
    assert page._detail_labels["reason"][1].toolTip() == "POSITION_SHIFT"
    assert "今日" in page.summary_scope_label.text()
    assert "不包含推論錯誤" in page.summary_scope_label.text()

    fail_index = page.status_combo.findData("FAIL")
    page.status_combo.setCurrentIndex(fail_index)
    _wait_page(page, qtbot)

    assert page.table.rowCount() == 1
    assert page.table.item(0, 1).text() == "NG"
    assert page.table.item(0, 0).data(Qt.UserRole) == 0
    assert "表格狀態：NG / 檢測失敗" in page.summary_scope_label.text()


def test_history_preview_click_opens_saved_image(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    image_path = tmp_path / "saved-result.png"
    image = QImage(80, 60, QImage.Format_RGB32)
    image.fill(Qt.black)
    assert image.save(str(image_path))

    page = _page(tmp_path / "records.sqlite3")
    qtbot.addWidget(page)
    page.show()
    page._selected_preview_path = str(image_path)
    page.preview.set_image(str(image_path))
    qtbot.waitUntil(
        lambda: page.preview.pixmap() is not None
        and not page.preview.pixmap().isNull(),
        timeout=3000,
    )

    opened: list[tuple[Path, str]] = []

    class FakePreviewDialog:
        def __init__(
            self,
            selected_path: Path,
            *,
            language: str,
            parent,
        ) -> None:
            assert parent is page
            opened.append((selected_path, language))

        def exec_(self) -> int:
            return 0

    monkeypatch.setattr(
        history_page_module,
        "_HistoryImageDialog",
        FakePreviewDialog,
    )

    qtbot.mouseClick(page.preview, Qt.LeftButton)

    assert opened == [(image_path, "zh")]
    assert page.preview.toolTip() == "點擊預覽可放大查看"


def test_history_page_preserves_filter_data_when_language_changes(
    tmp_path: Path,
    qtbot,
) -> None:
    database_path = _populated_database(tmp_path)
    page = _page(database_path)
    qtbot.addWidget(page)
    page.show_for_target("Cable1", "A")
    _wait_page(page, qtbot)
    page.status_combo.setCurrentIndex(page.status_combo.findData("FAIL"))
    _wait_page(page, qtbot)

    page.set_language("en")

    assert page.product_combo.currentData() == "Cable1"
    assert page.station_combo.currentData() == "A"
    assert page.status_combo.currentData() == "FAIL"
    assert page.table.rowCount() == 1
    assert page.table.item(0, 6).text() == "Position shift"
    assert page._detail_labels["reason"][1].text() == "Position shift"


def test_history_page_shows_company_sync_queue_status(
    tmp_path: Path,
    qtbot,
) -> None:
    database_path = _populated_database(tmp_path)
    page = _page(database_path)
    qtbot.addWidget(page)
    page.set_database_context(database_path, sync_enabled=True)
    page.show()
    page.show_for_target("Cable1", "A")
    _wait_page(page, qtbot)

    assert "4" in page.sync_badge.text()
    assert "sync" not in page.sync_badge.text().lower()


def test_history_page_clears_station_scope_when_product_changes(
    tmp_path: Path,
    qtbot,
) -> None:
    database_path = _populated_database(tmp_path)
    page = _page(database_path)
    qtbot.addWidget(page)
    page.show_for_target("Cable1", "B")
    _wait_page(page, qtbot)

    page.product_combo.setCurrentIndex(
        page.product_combo.findData("Cable2")
    )
    _wait_page(page, qtbot)

    assert page.product_combo.currentData() == "Cable2"
    assert page.station_combo.currentData() == ""
    assert page.total_value.text() == "1"
    assert page.table.rowCount() == 1


def test_history_page_custom_dates_are_inclusive(
    tmp_path: Path,
    qtbot,
) -> None:
    database_path = _populated_database(tmp_path)
    page = _page(database_path)
    qtbot.addWidget(page)
    page.show_for_target("Cable1", "")
    _wait_page(page, qtbot)

    page.period_combo.setCurrentIndex(
        page.period_combo.findData("CUSTOM")
    )
    page.custom_start_date.setDate(QDate(2026, 7, 28))
    page.custom_end_date.setDate(QDate(2026, 7, 28))

    page.show()
    _wait_page(page, qtbot)
    assert page.custom_start_date.isVisibleTo(page) is True
    assert page.total_value.text() == "3"
    assert "2026-07-28 ～ 2026-07-28" in page.summary_scope_label.text()

    page.custom_start_date.setDate(QDate(2026, 7, 29))
    _wait_page(page, qtbot)

    assert page.total_value.text() == "--"
    assert "篩選或資料錯誤" in page.summary_scope_label.text()


def test_history_page_coalesces_rapid_storage_refreshes(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    database_path = _populated_database(tmp_path)
    page = _page(database_path)
    qtbot.addWidget(page)
    page.show()
    page.show_for_target("Cable1", "A")
    _wait_page(page, qtbot)
    page._refresh_timer.setInterval(20)
    original_load = page._service.load
    load_calls = []

    def tracked_load(filters):
        load_calls.append(filters)
        return original_load(filters)

    monkeypatch.setattr(page._service, "load", tracked_load)

    page.mark_dirty()
    page.mark_dirty()
    page.mark_dirty()

    qtbot.waitUntil(
        lambda: len(load_calls) == 1 and not page.is_loading,
        timeout=1000,
    )
    qtbot.wait(50)
    assert len(load_calls) == 1


def test_history_page_exports_current_filters_without_blocking_gui(
    tmp_path: Path,
    qtbot,
    monkeypatch,
) -> None:
    database_path = _populated_database(tmp_path)
    destination = tmp_path / "operator-report.xlsx"
    page = _page(database_path)
    qtbot.addWidget(page)
    page.show()
    page.show_for_target("Cable1", "A")
    _wait_page(page, qtbot)
    page.status_combo.setCurrentIndex(page.status_combo.findData("FAIL"))
    _wait_page(page, qtbot)
    messages = []
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(destination), ""),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (
            messages.append(args[2]) or QMessageBox.No
        ),
    )

    page.export_button.click()

    qtbot.waitUntil(
        lambda: page._export_worker is None,
        timeout=5000,
    )
    assert destination.is_file()
    assert page.export_button.isEnabled()
    assert messages
    assert "1" in messages[0]
