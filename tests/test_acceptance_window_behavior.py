from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PyQt5.QtGui import QPixmap

import app.acceptance.window as acceptance_window
from core.services.acceptance_matrix import (
    AcceptanceColorVariant,
    ColorVariantDiscovery,
    ColorVariantExclusion,
)
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
        self.root.mkdir(parents=True, exist_ok=True)
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

    def verified_image_file(self, record: AcceptanceRecord) -> Path:
        return self.image_file(record)

    def manifest_sha256(self) -> str:
        return "a" * 64

    def import_images(self, paths, **_kwargs) -> tuple[AcceptanceRecord, ...]:
        if self.raise_on == "import":
            raise AcceptanceDataError("unsafe import")
        self.imported = list(paths)
        return self._records[:1]

    def save_inference(self, outcome: AcceptanceInferenceOutcome) -> None:
        if self.raise_on == "save":
            raise AcceptanceDataError("save failed")
        self.saved.append(outcome)

    def save_inference_batch(
        self,
        outcomes,
        *,
        run_id,
        artifact_bundle,
        expected_manifest_sha256,
    ):
        if self.raise_on == "save":
            raise AcceptanceDataError("save failed")
        assert expected_manifest_sha256 == "a" * 64
        self.saved.extend(outcomes)
        by_id = {outcome.sample_id: outcome for outcome in outcomes}
        self._records = tuple(
            replace(
                record,
                machine_status=by_id[record.sample_id].machine_status,
                acceptance_run_id=run_id,
                artifact_bundle_sha256=artifact_bundle.bundle_sha256,
            )
            if record.sample_id in by_id
            else record
            for record in self._records
        )
        return self._records, "b" * 64

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

    def create_snapshot(
        self,
        *,
        label: str,
        require_completed_run: bool = False,
    ) -> AcceptanceSnapshot:
        assert require_completed_run
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
    station_root = tmp_path / "models" / "Cable1" / "A" / "yolo"
    station_root.mkdir(parents=True)
    weight_path = station_root / "model.onnx"
    weight_path.write_bytes(b"model")
    (station_root / "config.yaml").write_text(
        f"weights: {weight_path.as_posix()}\n",
        encoding="utf-8",
    )
    (tmp_path / "config.yaml").write_text(
        f"weights: {weight_path.as_posix()}\ndevice: cpu\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        models=tmp_path / "models",
        acceptance=tmp_path / "acceptance",
        color_revisions=tmp_path / ".color_revisions",
        color_baselines=tmp_path / ".color_baselines",
        color_profiles=tmp_path / ".color_profiles",
    )
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
    assert [len(records) for records in started] == [1, 2, 4]

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
    # These records were inferred under no artifact bundle, so a partial run
    # asks before discarding them. Answering is not what this test is about.
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: acceptance_window.QMessageBox.Yes,
    )
    window._start_inference(())
    window.type_combo.clear()
    window._start_inference(window._records[:1])
    window.type_combo.addItem("yolo")
    window._start_inference(window._records[:1])
    assert window._worker is not None
    assert window.progress.maximum() == 1
    window._inference_progress(1, 1)
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
    assert repository.saved == []
    assert "pending" in window._annotated_pixmaps

    window._close_when_finished = True
    close = Mock()
    monkeypatch.setattr(window, "close", close)
    window._inference_finished()
    assert repository.saved == [outcome]
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
    models_root = tmp_path / "models"
    station_root = models_root / "Cable1" / "A" / "yolo"
    station_root.mkdir(parents=True)
    weight_path = station_root / "model.onnx"
    config_path = station_root / "config.yaml"
    global_config = tmp_path / "config.yaml"
    weight_path.write_bytes(b"model")
    config_path.write_text("weights: model.onnx\n", encoding="utf-8")
    global_config.write_text("device: cpu\n", encoding="utf-8")
    bundle = acceptance_window.build_acceptance_artifact_bundle(
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="v1",
        global_config_path=global_config,
        model_config_path=config_path,
        models_root=models_root,
        model_weight_path=weight_path,
        color_revision_overrides={"black-scope": "black-v1.0.2"},
    )
    outcome = AcceptanceInferenceOutcome(
        sample_id=record.sample_id,
        machine_status="OK",
        machine_reasons=(),
        model_version="v1",
        model_sha256=bundle.model_weight.sha256,
        runtime_config_sha256=bundle.model_config.sha256,
        inference_at="now",
        latency_ms=1.0,
        error="",
    )
    closed: list[bool] = []
    service_kwargs: dict[str, object] = {}

    class _Service:
        def __init__(self, **kwargs) -> None:
            service_kwargs.update(kwargs)

        def infer(self, *_args, **_kwargs) -> AcceptanceInferenceOutcome:
            return outcome

        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr(acceptance_window, "AcceptanceInferenceService", _Service)
    worker = acceptance_window.InferenceBatchWorker(
        project_root=tmp_path,
        models_root=models_root,
        repository=repository,
        records=(record,),
        inference_type="yolo",
        artifact_bundle=bundle,
    )
    outcomes: list[AcceptanceInferenceOutcome] = []
    progress: list[tuple[int, int]] = []
    worker.outcome_ready.connect(outcomes.append)
    worker.progress_changed.connect(lambda current, total: progress.append((current, total)))
    worker.run()
    assert outcomes == [outcome]
    assert progress == [(1, 1)]
    assert closed == [True]
    assert service_kwargs["color_revision_overrides"] == {
        "black-scope": "black-v1.0.2"
    }
    assert service_kwargs["include_active_color_revisions"] is False

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
    # No image is shown, and the placeholder explains the absence rather than
    # leaving an empty panel.
    assert label.pixmap() is None or label.pixmap().isNull()
    assert label.text() == "empty"
    pixmap = QPixmap(20, 10)
    pixmap.fill()
    label.set_source(pixmap)
    assert label.pixmap() is not None and not label.pixmap().isNull()


def test_color_model_selector_lists_usable_models_and_blocks_unusable_ones(
    acceptance_ui,
    monkeypatch,
) -> None:
    """The scope's stored color models are offered; withheld ones are inert.

    An unusable entry stays visible so its absence is explained, but selecting
    it must be impossible: it would otherwise reach inference and decide a
    verdict from an incompatible baseline.
    """
    window, _repository = acceptance_ui
    usable = AcceptanceColorVariant(
        variant_id="color-base-usable",
        label="完整顏色基準 / color-base-usable（READY）",
        color_model_path=Path("color_stats.json"),
    )
    monkeypatch.setattr(
        acceptance_window,
        "discover_color_variants",
        lambda *_args, **_kwargs: ColorVariantDiscovery(
            variants=(usable,),
            exclusions=(
                ColorVariantExclusion(
                    label="完整顏色基準 / color-base-old",
                    reason="演算法 stats-robust-v1 不相容",
                ),
            ),
        ),
    )

    window._reload_color_models()

    assert window.color_model_combo.count() == 3
    assert window.color_model_combo.itemText(0) == "目前正式設定（不覆寫）"
    assert window.color_model_combo.itemData(1) is usable
    assert "無法使用" in window.color_model_combo.itemText(2)
    combo_model = window.color_model_combo.model()
    assert combo_model.item(1).isEnabled()
    assert not combo_model.item(2).isEnabled()

    window.color_model_combo.setCurrentIndex(0)
    assert window._selected_color_variant() is None
    window.color_model_combo.setCurrentIndex(1)
    assert window._selected_color_variant() is usable


def test_unreadable_color_store_is_not_shown_as_an_empty_scope(
    acceptance_ui,
    monkeypatch,
) -> None:
    """A store that cannot be read must not look like a scope with no models."""
    window, _repository = acceptance_ui

    def _explode(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(acceptance_window, "discover_color_variants", _explode)

    window._reload_color_models()

    assert window.color_model_combo.count() == 2
    assert "無法讀取顏色模型" in window.color_model_combo.itemText(1)
    assert not window.color_model_combo.model().item(1).isEnabled()


def test_selected_color_model_reaches_inference_with_its_model_config(
    acceptance_ui,
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A color override must travel with the version-matched model config.

    The service stages a temporary config pointing at the chosen color model,
    so sending the color model alone would abort the run.
    """
    window, _repository = acceptance_ui
    config_path = tmp_path / "models" / "Cable1" / "A" / "yolo" / "config.yaml"
    weight_path = config_path.parent / "model.onnx"
    color_path = tmp_path / "color_stats.json"
    color_path.write_text('{"summary": {}}', encoding="utf-8")
    variant = AcceptanceColorVariant(
        variant_id="color-base-usable",
        label="完整顏色基準",
        color_model_path=color_path,
        revision_overrides=(("black-scope", "black-v1.0.2"),),
        include_active_revisions=False,
    )
    monkeypatch.setattr(
        acceptance_window,
        "discover_color_variants",
        lambda *_args, **_kwargs: ColorVariantDiscovery(
            variants=(variant,)
        ),
    )
    identity = SimpleNamespace(version="v1.0.6")
    monkeypatch.setattr(
        acceptance_window,
        "build_model_variant",
        lambda *_args, **_kwargs: SimpleNamespace(
            config_path=config_path,
            weight_path=weight_path,
            identity=identity,
        ),
    )
    window._reload_color_models()
    window.color_model_combo.setCurrentIndex(1)

    captured: dict[str, object] = {}

    class _Worker:
        outcome_ready = progress_changed = failed = finished = None

        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            for signal in ("outcome_ready", "progress_changed", "failed", "finished"):
                setattr(self, signal, SimpleNamespace(connect=lambda _slot: None))

        def start(self) -> None:
            pass

        def isRunning(self) -> bool:
            return False

    monkeypatch.setattr(acceptance_window, "InferenceBatchWorker", _Worker)
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: acceptance_window.QMessageBox.Yes,
    )

    window._start_inference(window._records[:1])

    bundle = captured["artifact_bundle"]
    assert bundle.color_model.path == variant.color_model_path
    assert bundle.model_config.path == config_path
    assert bundle.model_weight.path == weight_path
    assert dict(bundle.color_revision_overrides) == {
        "black-scope": "black-v1.0.2"
    }
    assert bundle.include_active_color_revisions is False


def test_outcome_refreshes_one_row_without_rebuilding_the_list(
    acceptance_ui,
    monkeypatch,
) -> None:
    """Watching a batch must not cost one full list rebuild per sample."""
    window, _repository = acceptance_ui
    window._active_run = SimpleNamespace(sample_ids=("pending",))
    window._run_repository = SimpleNamespace(append_outcome=lambda *_args: None)
    rebuilds: list[str] = []
    monkeypatch.setattr(
        window,
        "_render_records",
        lambda **kwargs: rebuilds.append(str(kwargs.get("select_id", ""))),
    )
    outcome = AcceptanceInferenceOutcome(
        sample_id="pending",
        machine_status="NG",
        machine_reasons=("MISSING",),
        model_version="v1",
        model_sha256="c" * 64,
        inference_at="2026-08-14T00:00:00Z",
        latency_ms=4.0,
        error="",
    )

    window._inference_ready(outcome)

    assert rebuilds == []
    row = next(
        index
        for index in range(window.sample_list.count())
        if window.sample_list.item(index).data(acceptance_window.SAMPLE_ID_ROLE)
        == "pending"
    )
    assert "模型 NG" in window.sample_list.item(row).text()
    assert window._records[0].machine_status == "NG"
    assert window._visible_records[row].machine_status == "NG"


def test_outcome_that_leaves_the_active_filter_falls_back_to_a_rebuild(
    acceptance_ui,
    monkeypatch,
) -> None:
    """An in-place repaint cannot move rows, so a filter change must rebuild.

    Under the 待確認 filter an inferred sample keeps its place, but under 推論
    錯誤 a sample that stops being an error has to leave the list entirely --
    every row after it shifts, which one repaint cannot express.
    """
    window, _repository = acceptance_ui
    window.filter_combo.setCurrentIndex(window.filter_combo.findData("error"))
    window._active_run = SimpleNamespace(sample_ids=("error",))
    window._run_repository = SimpleNamespace(append_outcome=lambda *_args: None)
    assert [record.sample_id for record in window._visible_records] == ["error"]
    rebuilds: list[str] = []
    monkeypatch.setattr(
        window,
        "_render_records",
        lambda **kwargs: rebuilds.append(str(kwargs.get("select_id", ""))),
    )

    window._inference_ready(
        AcceptanceInferenceOutcome(
            sample_id="error",
            machine_status="NG",
            machine_reasons=(),
            model_version="v1",
            model_sha256="c" * 64,
            inference_at="2026-08-14T00:00:00Z",
            latency_ms=4.0,
            error="",
        )
    )

    assert rebuilds == ["error"]


def test_partial_run_asks_before_discarding_other_bundle_results(
    acceptance_ui,
    monkeypatch,
) -> None:
    """Declining must leave the manifest and the run store untouched.

    Committing a partial run clears results produced by a different artifact
    combination, because a snapshot may only mix results from one. That is the
    operator's call, so the run must not even be opened until they make it.
    """
    window, _repository = acceptance_ui
    asked: list[str] = []
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "question",
        lambda _parent, _title, message, *_args: (
            asked.append(message) or acceptance_window.QMessageBox.No
        ),
    )
    begun: list[object] = []
    monkeypatch.setattr(
        window._run_repository,
        "begin",
        lambda **kwargs: begun.append(kwargs),
    )

    class _Worker:
        def __init__(self, **_kwargs) -> None:
            raise AssertionError("a declined run must not start inference")

    monkeypatch.setattr(acceptance_window, "InferenceBatchWorker", _Worker)

    window._start_inference(window._records[:1])

    assert len(asked) == 1
    # Three of the four fixture records carry a result from no bundle at all.
    assert "3 張" in asked[0]
    assert begun == []
    assert window._worker is None
    assert window._active_run is None


def test_same_bundle_rerun_discards_nothing_and_asks_nothing(
    acceptance_ui,
    monkeypatch,
) -> None:
    """Retrying samples under the identical bundle is not a destructive act."""
    window, _repository = acceptance_ui
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: pytest.fail(
            "a same-bundle rerun must not prompt"
        ),
    )
    bundle = window._artifact_bundle_for_run(window.type_combo.currentText())
    window._records = tuple(
        replace(record, artifact_bundle_sha256=bundle.bundle_sha256)
        if record.machine_status
        else record
        for record in window._records
    )

    assert window._confirm_discarded_results(window._records[:1], bundle) is True


def test_default_color_selection_leaves_inference_untouched(
    acceptance_ui,
    monkeypatch,
) -> None:
    """Opening the tool and pressing 推論 must behave exactly as it always did."""
    window, _repository = acceptance_ui
    window.color_model_combo.setCurrentIndex(0)
    captured: dict[str, object] = {}
    critical: list[str] = []
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "critical",
        lambda _parent, _title, message: critical.append(message),
    )

    class _Worker:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            for signal in ("outcome_ready", "progress_changed", "failed", "finished"):
                setattr(self, signal, SimpleNamespace(connect=lambda _slot: None))

        def start(self) -> None:
            pass

        def isRunning(self) -> bool:
            return False

    monkeypatch.setattr(acceptance_window, "InferenceBatchWorker", _Worker)
    monkeypatch.setattr(
        acceptance_window.QMessageBox,
        "question",
        lambda *_args, **_kwargs: acceptance_window.QMessageBox.Yes,
    )

    window._start_inference(window._records[:1])

    assert critical == []
    bundle = captured["artifact_bundle"]
    assert bundle.color_model is None
    assert bundle.model_config.path.name == "config.yaml"
    assert dict(bundle.color_revision_overrides) == {}
    assert bundle.include_active_color_revisions is False


def test_annotated_previews_are_bounded_and_downscaled(
    acceptance_ui,
    monkeypatch,
) -> None:
    """A long batch must not accumulate full-resolution previews.

    Station images are large enough that caching every one at source
    resolution exhausted the graphics heap and killed the process without a
    traceback, so both the per-entry size and the entry count are capped.
    """
    window, _repository = acceptance_ui
    tall = acceptance_window.ANNOTATED_PREVIEW_MAX_EDGE * 2
    frame = np.zeros((tall, tall, 3), dtype=np.uint8)

    for index in range(acceptance_window.ANNOTATED_PREVIEW_CACHE_SIZE + 5):
        window._cache_annotated_preview(f"sample-{index}", frame)

    assert (
        len(window._annotated_pixmaps)
        == acceptance_window.ANNOTATED_PREVIEW_CACHE_SIZE
    )
    cached = window._annotated_pixmaps["sample-5"]
    assert max(cached.width(), cached.height()) == (
        acceptance_window.ANNOTATED_PREVIEW_MAX_EDGE
    )
    # The oldest entries were evicted, the most recent survive.
    assert "sample-0" not in window._annotated_pixmaps
    assert "sample-28" in window._annotated_pixmaps


def test_viewing_a_preview_protects_it_from_eviction(
    acceptance_ui,
) -> None:
    """Eviction follows use, not insertion, so the open record is not dropped."""
    window, _repository = acceptance_ui
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    cap = acceptance_window.ANNOTATED_PREVIEW_CACHE_SIZE
    for index in range(cap):
        window._cache_annotated_preview(f"sample-{index}", frame)

    assert window._annotated_preview("sample-0") is not None
    window._cache_annotated_preview("sample-new", frame)

    assert "sample-0" in window._annotated_pixmaps
    assert "sample-1" not in window._annotated_pixmaps


def test_evicted_preview_does_not_claim_the_record_was_never_inferred(
    acceptance_ui,
) -> None:
    """An inferred record with no cached preview must not read as un-inferred."""
    window, _repository = acceptance_ui
    label = acceptance_window.ScaledImageLabel("尚未執行推論")

    label.set_source(None, empty_text="標註圖預覽已釋出，重新推論此張即可再次檢視")
    assert "已釋出" in label.text()

    label.set_source(None)
    assert label.text() == "尚未執行推論"
