from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtGui import QCloseEvent

import app.gui.color_baseline_rebuild_dialog as rebuild_dialog
from core.services.color_baseline_evidence import ColorBaselineEvidenceExclusion
from core.services.color_baseline_recalibration import (
    ColorBaselineCancelled,
    ColorBaselineError,
)
from core.services.model_version_registry import ModelVersionRecord


def _model(tmp_path: Path, *, with_config: bool = True) -> ModelVersionRecord:
    config = tmp_path / "v1-config.yaml"
    if with_config:
        config.write_text("enable_color_check: true", encoding="utf-8")
    return ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="v1",
        weight_path=tmp_path / "best.pt",
        is_current=False,
        trained_at=None,
        deployed_at=None,
        activated_at=None,
        training_time_inferred=False,
        weight_sha256="weight-sha",
        config_snapshot_path=config if with_config else None,
    )


class _EvidenceProvider:
    snapshot = None

    def __init__(self, *, product: str, area: str, model_type: str) -> None:
        self.product = product
        self.area = area
        self.model_type = model_type

    def collect(self, **_kwargs):
        return self.snapshot


class _Service:
    closed = False

    def __init__(self, **_kwargs) -> None:
        pass

    def close(self) -> None:
        type(self).closed = True


def _snapshot(tmp_path: Path):
    sample = SimpleNamespace(
        sample_id="sample-1",
        source_kind="acceptance",
        source_manifest="manifest.csv",
        image_path=tmp_path / "sample.png",
        image_sha256="image-sha",
    )
    return SimpleNamespace(
        samples=(sample,),
        excluded_samples=(),
        selected_count=1,
        selected_acceptance_count=1,
        selected_feedback_count=0,
        duplicate_count=0,
        confirmed_ng_count=0,
        to_report_dict=lambda: {"selected_count": 1},
    )


def _configure_worker_dependencies(monkeypatch, tmp_path: Path) -> None:
    paths = SimpleNamespace(
        acceptance=tmp_path / "acceptance",
        models=tmp_path / "models",
        color_baselines=tmp_path / "baselines",
    )
    monkeypatch.setattr(rebuild_dialog, "load_station_data_paths", lambda _root: paths)
    monkeypatch.setattr(
        rebuild_dialog,
        "load_workspace_paths",
        lambda _root: SimpleNamespace(training_data=tmp_path / "training_data"),
    )


def test_rebuild_worker_emits_evidence_candidate_and_closes_service(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_worker_dependencies(monkeypatch, tmp_path)
    snapshot = _snapshot(tmp_path)
    monkeypatch.setattr(_EvidenceProvider, "snapshot", snapshot)
    monkeypatch.setattr(_Service, "closed", False)
    build = SimpleNamespace(outlier_filter="filter-report", color_reports=("green",))
    candidate = SimpleNamespace(display_version="baseline-v1")
    monkeypatch.setattr(rebuild_dialog, "ColorBaselineEvidenceProvider", _EvidenceProvider)
    monkeypatch.setattr(rebuild_dialog, "AcceptanceRepository", lambda _root: object())
    monkeypatch.setattr(rebuild_dialog, "AcceptanceInferenceService", _Service)
    monkeypatch.setattr(rebuild_dialog, "_resolve_color_model", lambda *_args: tmp_path / "color.json")
    monkeypatch.setattr(rebuild_dialog, "collect_color_baseline_evidence", lambda **_kwargs: ("evidence",))
    monkeypatch.setattr(
        rebuild_dialog,
        "StatsColorBaselineRebuilder",
        lambda: SimpleNamespace(build=lambda **_kwargs: build),
    )
    monkeypatch.setattr(
        rebuild_dialog,
        "ColorBaselineCandidateStore",
        lambda _root: SimpleNamespace(commit=lambda **_kwargs: candidate),
    )
    worker = rebuild_dialog.ColorBaselineRebuildWorker(
        project_root=tmp_path,
        product="Cable1",
        area="A",
        inference_type="fusion",
        model=_model(tmp_path),
    )
    phases: list[str] = []
    evidence: list[object] = []
    outliers: list[object] = []
    completed: list[tuple[object, object]] = []
    worker.phase_changed.connect(phases.append)
    worker.evidence_ready.connect(evidence.append)
    worker.outlier_filter_ready.connect(outliers.append)
    worker.completed.connect(lambda created, reports: completed.append((created, reports)))

    worker.run()

    assert len(phases) == 3
    assert evidence == [snapshot]
    assert outliers == ["filter-report"]
    assert completed == [(candidate, ("green",))]
    assert _Service.closed


def test_rebuild_worker_reports_validation_error_and_cancellation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _configure_worker_dependencies(monkeypatch, tmp_path)
    worker = rebuild_dialog.ColorBaselineRebuildWorker(
        project_root=tmp_path,
        product="Cable1",
        area="A",
        inference_type="yolo",
        model=_model(tmp_path, with_config=False),
    )
    failures: list[str] = []
    worker.failed.connect(failures.append)
    worker.run()
    assert "config" in failures[0]

    model = _model(tmp_path)
    monkeypatch.setattr(_EvidenceProvider, "snapshot", _snapshot(tmp_path))
    monkeypatch.setattr(_Service, "closed", False)
    monkeypatch.setattr(rebuild_dialog, "ColorBaselineEvidenceProvider", _EvidenceProvider)
    monkeypatch.setattr(rebuild_dialog, "AcceptanceRepository", lambda _root: object())
    monkeypatch.setattr(rebuild_dialog, "AcceptanceInferenceService", _Service)
    monkeypatch.setattr(rebuild_dialog, "_resolve_color_model", lambda *_args: tmp_path / "color.json")
    monkeypatch.setattr(
        rebuild_dialog,
        "collect_color_baseline_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(ColorBaselineCancelled()),
    )
    worker = rebuild_dialog.ColorBaselineRebuildWorker(
        project_root=tmp_path,
        product="Cable1",
        area="A",
        inference_type="yolo",
        model=model,
    )
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.run()
    assert cancelled == [True]


def test_rebuild_dialog_presents_progress_evidence_outliers_and_completion(
    monkeypatch,
    tmp_path: Path,
    qtbot,
) -> None:
    _configure_worker_dependencies(monkeypatch, tmp_path)
    dialog = rebuild_dialog.ColorBaselineRebuildDialog(
        project_root=tmp_path,
        product="Cable1",
        area="A",
        inference_type="yolo",
        model=_model(tmp_path),
    )
    qtbot.addWidget(dialog)
    monkeypatch.setattr(dialog._worker, "start", Mock())
    dialog._start()
    dialog._worker.start.assert_called_once()
    assert not dialog.start_button.isEnabled()

    dialog._update_progress(2, 5, "sample-2")
    assert dialog.progress.maximum() == 5
    assert "sample-2" in dialog.progress.format()
    dialog._update_progress(0, 0, "preparing")
    assert dialog.progress.maximum() == 1

    initial = ColorBaselineEvidenceExclusion(
        sample_id="unsafe",
        source_kind="feedback",
        source_manifest="feedback.csv",
        image_path="bad.png",
        image_sha256="bad",
        reason_code="UNSAFE_PATH",
        reason="unsafe",
    )
    sample = _snapshot(tmp_path).samples[0]
    snapshot = SimpleNamespace(
        samples=(sample,),
        excluded_samples=(initial,),
        selected_count=1,
        selected_acceptance_count=1,
        selected_feedback_count=0,
        duplicate_count=1,
        confirmed_ng_count=2,
    )
    dialog._show_evidence_summary(snapshot)
    assert dialog.excluded_evidence_button.isVisibleTo(dialog)
    assert "已排除 1" in dialog.evidence_label.text()

    report = SimpleNamespace(excluded_sample_ids=("unsafe", "sample-1", "unknown"), excluded_count=2)
    dialog._show_outlier_summary(report)
    assert len(dialog._excluded_evidence) == 3
    assert "離群照片 2" in dialog.evidence_label.text()

    shown: list[tuple[object, ...]] = []

    class _ExclusionsDialog:
        def __init__(self, exclusions, **_kwargs) -> None:
            shown.append(tuple(exclusions))

        def exec_(self) -> None:
            pass

    monkeypatch.setattr(rebuild_dialog, "ColorBaselineExclusionsDialog", _ExclusionsDialog)
    dialog._show_excluded_evidence()
    assert len(shown[0]) == 3
    dialog._excluded_evidence = ()
    dialog._show_excluded_evidence()

    candidate = SimpleNamespace(
        colors={"Green": object()},
        created_at="2026-08-11T00:00:00Z",
        status="READY",
        display_version="baseline-v1",
    )
    reports = (
        SimpleNamespace(
            color="Green",
            state="REBUILT",
            total_crops=20,
            training_crops=15,
            holdout_crops=5,
            previous_accuracy=None,
            candidate_accuracy=0.95,
            note="",
        ),
    )
    created: list[object] = []
    dialog.candidate_created.connect(created.append)
    dialog._completed(candidate, reports)
    assert dialog.result_table.rowCount() == 1
    assert dialog.result_table.item(0, 1).text() == "已重建"
    assert dialog.next_step_label.isVisibleTo(dialog)
    assert created == [candidate]


def test_rebuild_dialog_failure_cancel_and_close_guards(
    monkeypatch,
    tmp_path: Path,
    qtbot,
) -> None:
    _configure_worker_dependencies(monkeypatch, tmp_path)
    dialog = rebuild_dialog.ColorBaselineRebuildDialog(
        project_root=tmp_path,
        product="Cable1",
        area="A",
        inference_type="yolo",
        model=_model(tmp_path),
    )
    qtbot.addWidget(dialog)
    monkeypatch.setattr(rebuild_dialog.QMessageBox, "critical", lambda *_args: None)
    dialog._failed("build failed")
    assert dialog._finished
    assert dialog.phase_label.text() == "重建失敗"
    dialog._cancelled()
    assert "已取消" in dialog.phase_label.text()

    monkeypatch.setattr(dialog._worker, "isRunning", lambda: True)
    interruption = Mock()
    monkeypatch.setattr(dialog._worker, "requestInterruption", interruption)
    dialog._cancel_or_close()
    interruption.assert_called_once()
    assert not dialog.cancel_button.isEnabled()

    event = QCloseEvent()
    dialog.closeEvent(event)
    assert not event.isAccepted()

    monkeypatch.setattr(dialog._worker, "isRunning", lambda: False)
    event = QCloseEvent()
    dialog.closeEvent(event)
    assert event.isAccepted()
    close = Mock()
    monkeypatch.setattr(dialog, "close", close)
    dialog._cancel_or_close()
    close.assert_called_once()


def test_color_model_resolution_hash_and_display_helpers(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    color_model = tmp_path / "color.json"
    color_model.write_text("{}", encoding="utf-8")

    with pytest.raises(ColorBaselineError, match="不存在"):
        rebuild_dialog._resolve_color_model(tmp_path, config)
    config.write_text("enable_color_check: false", encoding="utf-8")
    with pytest.raises(ColorBaselineError, match="未啟用"):
        rebuild_dialog._resolve_color_model(tmp_path, config)
    config.write_text("enable_color_check: true\ncolor_checker_type: learned", encoding="utf-8")
    with pytest.raises(ColorBaselineError, match="Stats Color"):
        rebuild_dialog._resolve_color_model(tmp_path, config)
    config.write_text("enable_color_check: true\ncolor_checker_type: stats", encoding="utf-8")
    with pytest.raises(ColorBaselineError, match="color_model_path"):
        rebuild_dialog._resolve_color_model(tmp_path, config)
    config.write_text(
        "enable_color_check: true\ncolor_checker_type: stats\ncolor_model_path: color.json",
        encoding="utf-8",
    )
    assert rebuild_dialog._resolve_color_model(tmp_path, config) == color_model

    config.write_text(
        "enable_color_check: true\ncolor_checker_type: stats\ncolor_model_path: missing.json",
        encoding="utf-8",
    )
    with pytest.raises(ColorBaselineError, match="找不到"):
        rebuild_dialog._resolve_color_model(tmp_path, config)

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"payload")
    assert rebuild_dialog._sha256_file(payload) == hashlib.sha256(b"payload").hexdigest()
    assert rebuild_dialog._percent(None) == "—"
    assert rebuild_dialog._percent(0.5) == "50.0%"
    assert rebuild_dialog._state_label("PRESERVED_INSUFFICIENT") == "證據不足，沿用舊值"
    assert rebuild_dialog._state_label("custom") == "custom"
