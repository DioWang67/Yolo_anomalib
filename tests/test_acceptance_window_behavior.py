from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtGui import QPixmap

import app.acceptance.window as acceptance_window
from core.services.model_acceptance import (
    AcceptanceComparison,
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceRecord,
    AcceptanceSnapshot,
)


def _record(
    sample_id: str,
    *,
    truth: str = "",
    machine: str = "",
    review_status: str = "pending",
    machine_reasons: str = "",
) -> AcceptanceRecord:
    return AcceptanceRecord(
        sample_id=sample_id,
        image_path=f"images/{sample_id}.png",
        image_sha256=sample_id,
        product="Cable1",
        area="A",
        expected_verdict=truth,
        review_status=review_status,
        machine_status=machine,
        machine_reasons=machine_reasons,
        model_version="v1",
        runtime_config_sha256="a" * 64,
        color_model_sha256="b" * 64,
        latency_ms="12.5",
    )


class _Catalog:
    def __init__(self, _root: Path) -> None:
        pass

    def products(self) -> list[str]:
        return ["Cable1"]

    def areas(self, product: str) -> list[str]:
        return ["A"] if product == "Cable1" else []

    def inference_types(self, product: str, area: str) -> list[str]:
        return ["yolo"] if (product, area) == ("Cable1", "A") else []


class _Repository:
    def __init__(self, root: Path, records: tuple[AcceptanceRecord, ...]) -> None:
        self.root = root
        self._records = records
        self.saved: list[AcceptanceInferenceOutcome] = []
        self.confirmed: list[tuple[str, str]] = []
        self.imported: list[Path] = []
        self.backups: list[Path] = []
        self.compared: list[AcceptanceSnapshot] = []
        self.raise_on: str | None = None
        self._snapshots: tuple[AcceptanceSnapshot, ...] = ()

    def records(self) -> tuple[AcceptanceRecord, ...]:
        return self._records

    def image_file(self, record: AcceptanceRecord) -> Path:
        return self.root / record.image_path

    def import_images(self, paths, **_kwargs) -> tuple[AcceptanceRecord, ...]:
        if self.raise_on == "import":
            raise AcceptanceDataError("unsafe import")
        self.imported = list(paths)
        return self._records[:1]

    def save_inference(self, outcome: AcceptanceInferenceOutcome) -> None:
        if self.raise_on == "save":
            raise AcceptanceDataError("save failed")
        self.saved.append(outcome)

    def confirm(self, sample_id: str, *, verdict: str, **_kwargs) -> AcceptanceRecord:
        if self.raise_on == "confirm":
            raise AcceptanceDataError("reason required")
        self.confirmed.append((sample_id, verdict))
        updated = tuple(
            replace(record, expected_verdict=verdict, review_status="confirmed")
            if record.sample_id == sample_id
            else record
            for record in self._records
        )
        self._records = updated
        return next(record for record in updated if record.sample_id == sample_id)

    def create_snapshot(self, *, label: str) -> AcceptanceSnapshot:
        if self.raise_on == "snapshot":
            raise AcceptanceDataError("snapshot failed")
        snapshot = AcceptanceSnapshot(
            snapshot_id=label,
            root=self.root / label,
            manifest_path=self.root / label / "manifest.csv",
            summary_path=self.root / label / "summary.json",
        )
        self._snapshots = (snapshot,)
        return snapshot

    def snapshots(self) -> tuple[AcceptanceSnapshot, ...]:
        return self._snapshots

    def compare_with_snapshot(self, snapshot: AcceptanceSnapshot) -> AcceptanceComparison:
        self.compared.append(snapshot)
        return AcceptanceComparison(
            snapshot_id=snapshot.snapshot_id,
            baseline_version="v1",
            current_version="v2",
            common_samples=2,
            improved=1,
            regressed=0,
            unchanged_correct=1,
            unchanged_incorrect=0,
            baseline_false_positives=1,
            current_false_positives=0,
            baseline_false_negatives=0,
            current_false_negatives=0,
            changed_sample_ids=("sample-1",),
        )

    def export_backup_zip(self, destination: Path) -> Path:
        if self.raise_on == "backup":
            raise OSError("disk full")
        self.backups.append(destination)
        return destination


@pytest.fixture
def acceptance_ui(monkeypatch, tmp_path: Path, qtbot):
    records = (
        _record("pending"),
        _record("fp", truth="OK", machine="NG", review_status="confirmed"),
        _record(
            "fn",
            truth="NG",
            machine="OK",
            review_status="confirmed",
            machine_reasons="COLOR_MISMATCH",
        ),
        _record("error", truth="NG", machine="ERROR", review_status="confirmed"),
    )
    repository = _Repository(tmp_path / "acceptance", records)
    paths = SimpleNamespace(models=tmp_path / "models", acceptance=tmp_path / "acceptance")
    monkeypatch.setattr(acceptance_window, "load_station_data_paths", lambda _root: paths)
    monkeypatch.setattr(acceptance_window, "ModelCatalog", _Catalog)
    monkeypatch.setattr(acceptance_window, "AcceptanceRepository", lambda _root: repository)
    window = acceptance_window.ModelAcceptanceWindow(project_root=tmp_path)
    qtbot.addWidget(window)
    return window, repository


def test_acceptance_window_builds_catalog_and_filters_records(acceptance_ui) -> None:
    window, _repository = acceptance_ui

    assert window.product_combo.currentText() == "Cable1"
    assert window.area_combo.currentText() == "A"
    assert window.type_combo.currentText() == "yolo"
    assert window.sample_list.count() == 4

    expected_counts = {
        "all": 4,
        "mismatch": 2,
        "false_positive": 1,
        "false_negative": 1,
        "color_mismatch": 1,
        "pending": 1,
        "error": 1,
    }
    for filter_value, expected in expected_counts.items():
        window.filter_combo.setCurrentIndex(window.filter_combo.findData(filter_value))
        assert window.sample_list.count() == expected

    window._records = ()
    window._render_records()
    assert window.sample_list.count() == 0
    assert window._selected_record() is None


def test_acceptance_window_renders_record_and_color_evidence(acceptance_ui) -> None:
    window, _repository = acceptance_ui
    record = replace(
        window._records[1],
        expected_reasons="MISSING|OTHER",
        defect_class="connector",
        notes="fixture checked",
        error="backend warning",
        color_details_json=(
            '[{"index": 1, "detector_class": "LED", "predicted_color": "Red", '
            '"diff": 0.5, "threshold": 0.2, "is_ok": false}, "ignored"]'
        ),
    )

    window._render_record_detail(record)

    assert window._reason_checks["MISSING"].isChecked()
    assert window.defect_class_edit.text() == "connector"
    assert window.notes_edit.toPlainText() == "fixture checked"
    assert window.color_table.rowCount() == 1
    assert window.color_table.item(0, 5).text() == "FAIL"
    assert "backend warning" in window.detail_label.text()

    window._render_color_details(replace(record, color_details_json="not-json"))
    assert window.color_table.rowCount() == 0
    window._render_color_details(replace(record, color_details_json='{"not": "a list"}'))
    assert window.color_table.rowCount() == 0


def test_import_actions_handle_selection_success_and_domain_error(
    acceptance_ui,
    monkeypatch,
    tmp_path: Path,
) -> None:
    window, repository = acceptance_ui
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    nested = tmp_path / "folder"
    nested.mkdir()
    (nested / "one.jpg").write_bytes(b"image")
    (nested / "ignore.txt").write_text("not an image", encoding="utf-8")

    monkeypatch.setattr(
        acceptance_window.QFileDialog,
        "getOpenFileNames",
        lambda *_args, **_kwargs: ([str(source)], "Images"),
    )
    window._add_files()
    assert repository.imported == [source]

    monkeypatch.setattr(
        acceptance_window.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: str(nested),
    )
    window._add_folder()
    assert repository.imported == [nested / "one.jpg"]

    monkeypatch.setattr(
        acceptance_window.QFileDialog,
        "getExistingDirectory",
        lambda *_args, **_kwargs: "",
    )
    window._add_folder()

    errors: list[str] = []
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "critical",
        lambda _parent, _title, message: errors.append(message),
    )
    repository.raise_on = "import"
    window._import_paths((source,))
    assert errors == ["unsafe import"]

    window.repository = None
    window._import_paths((source,))


def test_run_actions_matrix_snapshot_and_comparison(
    acceptance_ui,
    monkeypatch,
) -> None:
    window, repository = acceptance_ui
    started: list[tuple[AcceptanceRecord, ...]] = []
    monkeypatch.setattr(window, "_start_inference", started.append)

    window.sample_list.setCurrentRow(0)
    window._run_selected()
    window._run_pending()
    window._run_all()
    assert [len(records) for records in started] == [1, 1, 4]

    messages: list[tuple[str, str]] = []
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "information",
        lambda _parent, title, message: messages.append((title, message)),
    )
    monkeypatch.setattr(acceptance_window.QMessageBox, "warning", lambda *_args: None)
    executed: list[bool] = []

    class _Matrix:
        def __init__(self, **_kwargs) -> None:
            pass

        def exec_(self) -> None:
            executed.append(True)

    monkeypatch.setattr(acceptance_window, "AcceptanceMatrixDialog", _Matrix)
    window._open_matrix()
    assert executed == [True]

    saved_repository = window.repository
    window.repository = None
    window._open_matrix()
    window._create_snapshot()
    window._compare_latest_snapshot()
    window.repository = saved_repository

    window._records = (window._records[0],)
    window._open_matrix()
    assert messages

    window._records = repository.records()
    window.type_combo.clear()
    window._open_matrix()
    window.type_combo.addItem("yolo")

    window._create_snapshot()
    assert repository.snapshots()
    messages.clear()
    window._compare_latest_snapshot()
    assert repository.compared == [repository.snapshots()[-1]]
    assert len(messages) == 1
    _title, comparison_message = messages[0]
    assert repository.snapshots()[-1].snapshot_id in comparison_message
    assert "v1" in comparison_message
    assert "v2" in comparison_message
    assert comparison_message.splitlines()[-1].endswith("1")
    assert "sample-1" not in comparison_message

    repository._snapshots = ()
    window._compare_latest_snapshot()
    repository.raise_on = "snapshot"
    critical: list[str] = []
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "critical",
        lambda _parent, _title, message: critical.append(message),
    )
    window._create_snapshot()
    assert critical == ["snapshot failed"]


def test_backup_and_inference_lifecycle_are_bounded(
    acceptance_ui,
    monkeypatch,
    tmp_path: Path,
) -> None:
    window, repository = acceptance_ui
    archive = tmp_path / "backup.zip"
    monkeypatch.setattr(acceptance_window.QMessageBox, "information", lambda *_args: None)
    monkeypatch.setattr(acceptance_window.QMessageBox, "critical", lambda *_args: None)
    monkeypatch.setattr(acceptance_window.QMessageBox, "warning", lambda *_args: None)
    monkeypatch.setattr(
        acceptance_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(archive), "ZIP"),
    )
    monkeypatch.setattr(acceptance_window.BackupWorker, "start", lambda _self: None)
    monkeypatch.setattr(
        acceptance_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: ("", "ZIP"),
    )
    window._export_backup()
    monkeypatch.setattr(
        acceptance_window.QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(archive), "ZIP"),
    )
    window._export_backup()
    assert window._backup_worker is not None
    window._backup_completed(str(archive))
    window._backup_failed("disk full")
    window._backup_finished()
    assert window._backup_worker is None

    monkeypatch.setattr(acceptance_window.InferenceBatchWorker, "start", lambda _self: None)
    window._start_inference(())
    window.type_combo.clear()
    window._start_inference(window._records[:1])
    window.type_combo.addItem("yolo")
    window._start_inference(window._records[:2])
    assert window._worker is not None
    assert window.progress.maximum() == 2
    window._inference_progress(1, 2)
    assert window.progress.value() == 1

    outcome = AcceptanceInferenceOutcome(
        sample_id="pending",
        machine_status="OK",
        machine_reasons=(),
        model_version="v2",
        model_sha256="c" * 64,
        inference_at="2026-08-11T00:00:00Z",
        latency_ms=4.0,
        error="",
        annotated_frame=np.zeros((3, 4, 3), dtype=np.uint8),
    )
    window._inference_ready(object())
    saved_repository = window.repository
    window.repository = None
    window._inference_ready(outcome)
    window.repository = saved_repository
    window._inference_ready(outcome)
    assert repository.saved == [outcome]
    assert "pending" in window._annotated_pixmaps

    repository.raise_on = "save"
    failures: list[str] = []
    monkeypatch.setattr(window, "_inference_failed", failures.append)
    window._inference_ready(outcome)
    assert failures == ["save failed"]
    repository.raise_on = None

    window._close_when_finished = True
    close = Mock()
    monkeypatch.setattr(window, "close", close)
    window._inference_finished()
    close.assert_called_once()
    assert window._worker is None


def test_confirmation_summary_close_guards_and_helpers(
    acceptance_ui,
    monkeypatch,
) -> None:
    window, repository = acceptance_ui
    window.sample_list.setCurrentRow(0)
    window._reason_checks["MISSING"].setChecked(True)
    window.reviewer_edit.setText("operator")
    window._confirm_ng()
    assert repository.confirmed[-1] == ("pending", "NG")
    window._confirm_ok()
    assert repository.confirmed[-1][1] == "OK"

    repository.raise_on = "confirm"
    warnings: list[str] = []
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "warning",
        lambda _parent, _title, message: warnings.append(message),
    )
    window._confirm("NG")
    assert warnings == ["reason required"]
    window._update_summary()
    assert "TP" in window.summary_label.text()

    class _Event:
        def __init__(self) -> None:
            self.accepted = False
            self.ignored = False

        def accept(self) -> None:
            self.accepted = True

        def ignore(self) -> None:
            self.ignored = True

    worker = SimpleNamespace(isRunning=lambda: True, requestInterruption=Mock())
    window._worker = worker
    event = _Event()
    window.closeEvent(event)
    assert event.ignored
    worker.requestInterruption.assert_called_once()

    window._worker = None
    window._backup_worker = SimpleNamespace(isRunning=lambda: True)
    event = _Event()
    window.closeEvent(event)
    assert event.ignored

    window._backup_worker = None
    event = _Event()
    window.closeEvent(event)
    assert event.accepted

    assert acceptance_window._format_rate(None)
    assert acceptance_window._format_rate(0.25) == "25.00%"
    assert acceptance_window._format_number("1.25") == "1.2500"
    assert acceptance_window._format_number(object())
    assert not acceptance_window._frame_to_pixmap(np.zeros((2, 3), dtype=np.uint8)).isNull()
    assert not acceptance_window._frame_to_pixmap(np.zeros((2, 3, 3), dtype=np.uint8)).isNull()
    assert acceptance_window._record_matches_filter(window._records[0], "unknown")


def test_workers_emit_results_failures_and_release_resources(monkeypatch, tmp_path: Path) -> None:
    record = _record("sample")
    repository = _Repository(tmp_path, (record,))
    outcome = AcceptanceInferenceOutcome(
        sample_id=record.sample_id,
        machine_status="OK",
        machine_reasons=(),
        model_version="v1",
        model_sha256="hash",
        inference_at="now",
        latency_ms=1.0,
        error="",
    )
    closed: list[bool] = []

    class _Service:
        def __init__(self, **_kwargs) -> None:
            pass

        def infer(self, *_args, **_kwargs) -> AcceptanceInferenceOutcome:
            return outcome

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(acceptance_window, "AcceptanceInferenceService", _Service)
    worker = acceptance_window.InferenceBatchWorker(
        project_root=tmp_path,
        repository=repository,
        records=(record,),
        inference_type="yolo",
    )
    outcomes: list[AcceptanceInferenceOutcome] = []
    progress: list[tuple[int, int]] = []
    worker.outcome_ready.connect(outcomes.append)
    worker.progress_changed.connect(lambda current, total: progress.append((current, total)))
    worker.run()
    assert outcomes == [outcome]
    assert progress == [(1, 1)]
    assert closed == [True]

    class _FailingService:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("backend unavailable")

    monkeypatch.setattr(acceptance_window, "AcceptanceInferenceService", _FailingService)
    failures: list[str] = []
    worker.failed.connect(failures.append)
    worker.run()
    assert failures[-1] == "backend unavailable"

    backup = acceptance_window.BackupWorker(repository, tmp_path / "backup.zip")
    completed: list[str] = []
    backup.completed.connect(completed.append)
    backup.run()
    assert completed == [str(tmp_path / "backup.zip")]

    repository.raise_on = "backup"
    backup_failures: list[str] = []
    backup.failed.connect(backup_failures.append)
    backup.run()
    assert backup_failures == ["disk full"]


def test_scaled_image_label_handles_empty_and_source_pixmaps(qtbot) -> None:
    label = acceptance_window.ScaledImageLabel("empty")
    qtbot.addWidget(label)
    label.set_source(None)
    assert label.pixmap() is not None and label.pixmap().isNull()
    pixmap = QPixmap(20, 10)
    pixmap.fill()
    label.set_source(pixmap)
    assert label.pixmap() is not None and not label.pixmap().isNull()
