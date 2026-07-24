from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from app.gui.processing_batch_dialog import (
    PROCESSING_DATASET_DRY_RUN_ENV,
    PROCESSING_DATASET_STEP_ENV,
    PROCESSING_EXECUTION_FRAMEWORK_ENV,
    PROCESSING_PIPELINE_ENV,
    processing_dataset_dry_run_enabled,
    processing_dataset_step_enabled,
    processing_execution_framework_enabled,
    processing_pipeline_enabled,
)
from tools.dataset_preparation import (
    DatasetPreparationError,
    DatasetPreparationErrorCode,
    DatasetPreparationService,
    DatasetPreparationStatus,
    capture_dataset_source_snapshot,
)
from tools.dataset_preparation_step import DatasetPreparationStep
from tools.processing_engine_factory import build_processing_execution_engine
from tools.processing_execution import (
    BlockedStep,
    CancellationToken,
    ExcludedStep,
    NoOpAnnotationStep,
    NoOpColorStep,
    ProcessingCancelledError,
    ProcessingExecutionEngine,
    ProcessingStepRegistry,
)
from tools.processing_pipeline import ProcessingPlanner, record_sha256
from tools.processing_plan_validation import ProcessingPlanValidator, ProcessingValidationContext
from tools.processing_reports import ProcessingReportStatus, SampleProcessingStatus
from tools.processing_run_store import ProcessingRunStore

NOW = datetime(2026, 7, 21, 8, 0, tzinfo=timezone.utc)


class Ids:
    def __init__(self):
        self._value = 0

    def __call__(self):
        self._value += 1
        return f"run-{self._value}"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready(tmp_path: Path, sample_id: str, *, label_text="0 0.5 0.5 0.25 0.25\n", color=None):
    image = tmp_path / f"{sample_id}.png"
    label = tmp_path / f"{sample_id}.txt"
    Image.new("RGB", (8, 8), color or (20, 40, 60)).save(image)
    label.write_text(label_text, encoding="utf-8")
    return {
        "sample_id": sample_id,
        "review_selected": "1",
        "review_outcome": "fail",
        "review_label": "confirmed_ng",
        "failure_category": "threshold_not_met",
        "skip_reason": "",
        "product_verdict": "ng",
        "detection_verdict": "correct",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
        "annotation_status": "verified_annotation",
        "source_image": str(image),
        "output_label": str(label),
        "class_names_json": '["defect"]',
        "product": "Cable1",
        "area": "A",
    }


def _plan(*records):
    plan = ProcessingPlanner().create_plan(
        list(enumerate(records)), operator="operator-a", created_at=NOW
    )
    return replace(plan, plan_id="plan-3c1")


def _service(tmp_path, **kwargs):
    return DatasetPreparationService(
        artifact_root=tmp_path / ".processing_runs" / "artifacts",
        clock=lambda: NOW,
        **kwargs,
    )


def _prepare(service, plan, *, report_id="report-1", dry_run=True, token=None):
    return service.prepare(
        plan,
        plan.records,
        report_id=report_id,
        dry_run=dry_run,
        cancellation=token or CancellationToken(clock=lambda: NOW),
    )


def test_dry_run_validates_and_hashes_without_creating_dataset(tmp_path):
    plan = _plan(_ready(tmp_path, "one"), _ready(tmp_path, "two", color=(1, 2, 3)))

    result = _prepare(_service(tmp_path), plan)

    assert result.status == DatasetPreparationStatus.DRY_RUN_VALIDATED
    assert result.dataset_id == f"dataset-{result.dataset_hash[:16]}"
    assert result.accepted_count == 2
    assert result.artifact_path is None
    assert result.preparation_report_path.is_file()
    assert not (tmp_path / ".processing_runs" / "artifacts" / "report-1" / "dataset").exists()
    assert not any((tmp_path / ".processing_runs").rglob("images"))


def test_actual_preparation_creates_one_verified_atomic_dataset(tmp_path):
    plan = _plan(*[_ready(tmp_path, f"sample-{index}", color=(index, 0, 0)) for index in range(5)])

    result = _prepare(_service(tmp_path), plan, dry_run=False)

    assert result.status == DatasetPreparationStatus.COMMITTED
    assert result.artifact_path == tmp_path / ".processing_runs" / "artifacts" / "report-1" / "dataset"
    assert result.accepted_count == 5
    assert sum(result.split_counts.values()) == 5
    assert (result.artifact_path / "dataset.yaml").is_file()
    assert (result.artifact_path / "dataset_metadata.json").is_file()
    assert (result.artifact_path / "source_sample_manifest.json").is_file()
    assert (result.artifact_path / "split_manifest.json").is_file()
    assert (result.artifact_path / "deduplication_audit.json").is_file()
    assert (result.artifact_path / "checksums.json").is_file()
    assert len(list(result.artifact_path.glob("images/*/*"))) == 5
    assert len(list(result.artifact_path.glob("labels/*/*"))) == 5
    assert "Training not started" not in result.preparation_report_path.read_text(encoding="utf-8")
    assert "training was not started" in result.preparation_report_path.read_text(encoding="utf-8")


def test_same_inputs_have_stable_hash_and_second_run_reuses_content(tmp_path):
    plan = _plan(*[_ready(tmp_path, f"stable-{index}", color=(index, 2, 3)) for index in range(3)])
    service = _service(tmp_path)

    first = _prepare(service, plan, report_id="first", dry_run=False)
    second = _prepare(service, plan, report_id="second", dry_run=False)

    assert second.dataset_hash == first.dataset_hash
    assert second.dataset_id == first.dataset_id
    assert second.status == DatasetPreparationStatus.REUSED
    assert second.artifact_path == first.artifact_path


def test_label_change_changes_dataset_hash(tmp_path):
    row = _ready(tmp_path, "changed")
    first = _prepare(_service(tmp_path), _plan(row), report_id="before")
    Path(row["output_label"]).write_text("0 0.4 0.5 0.25 0.25\n", encoding="utf-8")
    second = _prepare(_service(tmp_path), _plan(row), report_id="after")
    assert first.dataset_hash != second.dataset_hash


def test_external_source_change_after_plan_snapshot_is_stale(tmp_path):
    row = _ready(tmp_path, "captured")
    plan = _plan(row)
    service = DatasetPreparationService(
        artifact_root=tmp_path / ".processing_runs" / "artifacts",
        source_snapshot=capture_dataset_source_snapshot(plan.records),
        clock=lambda: NOW,
    )
    Image.new("RGB", (8, 8), (200, 100, 50)).save(row["source_image"])

    with pytest.raises(DatasetPreparationError) as caught:
        _prepare(service, plan)

    assert caught.value.code == DatasetPreparationErrorCode.SOURCE_IMAGE_STALE


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda row: Path(row["source_image"]).unlink(), DatasetPreparationErrorCode.SOURCE_IMAGE_MISSING),
        (lambda row: row.update(image_sha256="0" * 64), DatasetPreparationErrorCode.SOURCE_IMAGE_STALE),
        (lambda row: Path(row["output_label"]).unlink(), DatasetPreparationErrorCode.LABEL_MISSING),
        (lambda row: row.update(label_sha256="0" * 64), DatasetPreparationErrorCode.LABEL_STALE),
        (lambda row: Path(row["output_label"]).write_text("bad\n"), DatasetPreparationErrorCode.LABEL_PARSE_ERROR),
        (lambda row: Path(row["output_label"]).write_text("2 0.5 0.5 0.2 0.2\n"), DatasetPreparationErrorCode.CLASS_ID_INVALID),
        (lambda row: Path(row["output_label"]).write_text("0 2 0.5 0.2 0.2\n"), DatasetPreparationErrorCode.BBOX_INVALID),
        (lambda row: row.update(annotation_status="needs_reannotation"), DatasetPreparationErrorCode.REANNOTATION_REQUIRED),
    ],
)
def test_invalid_source_contract_fails_closed_and_cleans_staging(tmp_path, mutation, code):
    row = _ready(tmp_path, "invalid")
    mutation(row)
    plan = _plan(row)

    with pytest.raises(DatasetPreparationError) as caught:
        _prepare(_service(tmp_path), plan, dry_run=False)

    assert caught.value.code == code
    assert not (tmp_path / ".processing_runs" / "artifacts" / "report-1" / "dataset").exists()
    assert not (tmp_path / ".processing_runs" / "artifacts" / "report-1" / ".dataset-staging").exists()


def test_identical_image_with_conflicting_human_labels_is_blocked(tmp_path):
    first = _ready(tmp_path, "human-a")
    second = _ready(tmp_path, "human-b", label_text="0 0.3 0.3 0.2 0.2\n")
    second["source_image"] = first["source_image"]

    with pytest.raises(DatasetPreparationError) as caught:
        _prepare(_service(tmp_path), _plan(first, second))

    assert caught.value.code == DatasetPreparationErrorCode.ANNOTATION_CONFLICT


def test_identical_image_and_label_is_deduplicated_with_audit(tmp_path):
    first = _ready(tmp_path, "duplicate-a")
    second = _ready(tmp_path, "duplicate-b")
    second["source_image"] = first["source_image"]
    second["output_label"] = first["output_label"]

    result = _prepare(_service(tmp_path), _plan(first, second))

    assert result.accepted_count == 1
    assert len(result.deduplication_records) == 1
    assert result.deduplication_records[0].image_sha256 == _sha(Path(first["source_image"]))


def test_snapshot_and_verified_empty_use_phase1a_conversion(tmp_path):
    snapshot = _ready(tmp_path, "snapshot")
    snapshot.pop("source_image")
    snapshot.pop("output_label")
    snapshot.pop("annotation_status")
    snapshot["original_path"] = str(tmp_path / "snapshot.png")
    snapshot["preprocessed_path"] = snapshot["original_path"]
    snapshot["detections_json"] = json.dumps(
        [{"class_id": 0, "confidence": 0.9, "bbox": [1, 1, 6, 6]}]
    )
    empty = _ready(tmp_path, "empty", color=(90, 80, 70))
    empty.pop("source_image")
    empty.pop("output_label")
    empty["annotation_status"] = "verified_empty"
    empty["review_label"] = "verified_empty"
    empty["original_path"] = str(tmp_path / "empty.png")

    result = _prepare(_service(tmp_path), _plan(snapshot, empty))

    assert result.accepted_count == 2
    assert set(result.sample_splits) == {"snapshot", "empty"}


class CancelAfter:
    def __init__(self, allowed_checks: int):
        self.allowed_checks = allowed_checks
        self.checks = 0

    def raise_if_cancelled(self):
        self.checks += 1
        if self.checks > self.allowed_checks:
            raise ProcessingCancelledError("controlled cancellation")


def test_cancellation_before_commit_removes_staging_and_final(tmp_path):
    plan = _plan(*[_ready(tmp_path, f"cancel-{index}", color=(index, 4, 5)) for index in range(4)])

    with pytest.raises(ProcessingCancelledError):
        _prepare(
            _service(tmp_path),
            plan,
            dry_run=False,
            token=CancelAfter(3),
        )

    run_root = tmp_path / ".processing_runs" / "artifacts" / "report-1"
    assert not (run_root / ".dataset-staging").exists()
    assert not (run_root / "dataset").exists()


def test_unsafe_report_identifier_is_rejected_before_write(tmp_path):
    plan = _plan(_ready(tmp_path, "safe"))
    with pytest.raises(DatasetPreparationError) as caught:
        _prepare(_service(tmp_path), plan, report_id="../escape")
    assert caught.value.code == DatasetPreparationErrorCode.DATASET_COMMIT_FAILED


def test_commit_failure_is_retryable_and_leaves_no_final_dataset(tmp_path):
    def fail_commit(_source, _destination):
        raise PermissionError("locked")

    plan = _plan(_ready(tmp_path, "locked"))
    service = _service(tmp_path, commit_directory=fail_commit)

    with pytest.raises(DatasetPreparationError) as caught:
        _prepare(service, plan, dry_run=False)

    assert caught.value.code == DatasetPreparationErrorCode.DATASET_COMMIT_FAILED
    assert caught.value.retryable is True
    assert not (tmp_path / ".processing_runs" / "artifacts" / "report-1" / "dataset").exists()


def _engine(tmp_path, plan, *, dry_run):
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    service = DatasetPreparationService(artifact_root=store.artifacts_dir, clock=lambda: NOW)
    registry = ProcessingStepRegistry(
        (NoOpAnnotationStep(), NoOpColorStep(), BlockedStep(), ExcludedStep()),
        batch_steps=(DatasetPreparationStep(service),),
    )
    context = ProcessingValidationContext(
        current_manifest_sha=plan.source_manifest_sha,
        current_record_hashes={record.sample_id: record_sha256(record.fields) for record in plan.records},
        artifact_root=str(store.artifacts_dir),
    )
    return ProcessingExecutionEngine(
        validator=ProcessingPlanValidator(clock=lambda: NOW),
        context_provider=lambda _plan: context,
        store=store,
        step_registry=registry,
        clock=lambda: NOW,
        id_generator=Ids(),
        dry_run=dry_run,
    )


def test_engine_passes_complete_ready_batch_and_maps_one_dataset_to_all_samples(tmp_path):
    plan = _plan(*[_ready(tmp_path, f"batch-{index}", color=(index, 3, 4)) for index in range(3)])

    outcome = _engine(tmp_path, plan, dry_run=False).execute(plan)

    assert outcome.report.status == ProcessingReportStatus.COMPLETED
    assert outcome.report.summary.succeeded == 3
    assert {item.status for item in outcome.report.sample_results} == {SampleProcessingStatus.SUCCESS}
    assert {item.action for item in outcome.report.sample_results} == {"DATASET_PREPARED"}
    assert len({item.metadata["dataset_id"] for item in outcome.report.sample_results}) == 1
    assert all(item.metadata["training_started"] is False for item in outcome.report.sample_results)
    assert len(outcome.report.artifacts) == 2  # immutable plan + one batch preparation report


def test_engine_dry_run_never_claims_dataset_success(tmp_path):
    plan = _plan(_ready(tmp_path, "preview"))
    outcome = _engine(tmp_path, plan, dry_run=True).execute(plan)
    result = outcome.report.sample_results[0]
    assert result.status == SampleProcessingStatus.DEFERRED
    assert result.action == "dataset_dry_run_validated"
    assert result.metadata["dataset_path"] == ""


def test_batch_failure_marks_every_ready_sample_failed_and_creates_no_dataset(tmp_path):
    valid = _ready(tmp_path, "valid")
    invalid = _ready(tmp_path, "bad")
    Path(invalid["output_label"]).write_text("bad\n", encoding="utf-8")
    plan = _plan(valid, invalid)

    outcome = _engine(tmp_path, plan, dry_run=False).execute(plan)

    assert outcome.report.summary.failed == 2
    assert all(item.status == SampleProcessingStatus.FAILED for item in outcome.report.sample_results)
    assert {item.error_code for item in outcome.report.sample_results} == {"LABEL_PARSE_ERROR"}
    assert not any((tmp_path / ".processing_runs" / "artifacts").glob("*/dataset"))


def test_feature_flag_matrix_preserves_legacy_and_phase3b_fallbacks():
    values = {
        PROCESSING_PIPELINE_ENV: "1",
        PROCESSING_EXECUTION_FRAMEWORK_ENV: "1",
        PROCESSING_DATASET_STEP_ENV: "1",
    }
    assert processing_pipeline_enabled(values)
    assert processing_execution_framework_enabled(values)
    assert processing_dataset_step_enabled(values)
    assert not processing_dataset_step_enabled({})
    assert processing_dataset_dry_run_enabled(
        {PROCESSING_DATASET_DRY_RUN_ENV: "1"}
    )
    assert not processing_dataset_dry_run_enabled({})


@pytest.mark.parametrize(
    ("dataset_enabled", "expected_status"),
    [
        (False, SampleProcessingStatus.DEFERRED),
        (True, SampleProcessingStatus.SUCCESS),
    ],
)
def test_engine_factory_keeps_noop_fallback_and_enables_only_dataset_batch(
    tmp_path, dataset_enabled, expected_status
):
    plan = _plan(_ready(tmp_path, "factory"))
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    validator = ProcessingPlanValidator(clock=lambda: NOW)
    context = ProcessingValidationContext(
        current_manifest_sha=plan.source_manifest_sha,
        current_record_hashes={
            record.sample_id: record_sha256(record.fields) for record in plan.records
        },
        artifact_root=str(store.artifacts_dir),
    )
    engine = build_processing_execution_engine(
        plan=plan,
        manifest_path=tmp_path / "review.csv",
        store=store,
        validator=validator,
        context_provider=lambda _plan: context,
        dataset_step_enabled=dataset_enabled,
    )

    result = engine.execute(plan).report.sample_results[0]

    assert result.status == expected_status
    assert result.step_id == (
        "dataset_preparation" if dataset_enabled else "noop_ready"
    )


def test_ui_modules_do_not_import_dataset_business_rules():
    import app.gui.dataset_preparation_view_model as view_model
    import app.gui.processing_batch_dialog as dialog
    import app.gui.review_cases_dialog as review_dialog

    presentation = inspect.getsource(dialog) + inspect.getsource(view_model)
    review_host = inspect.getsource(review_dialog)
    assert "export_review_dataset" not in presentation
    assert "select_canonical_ready_items" not in presentation + review_host
    assert "DatasetPreparationService" not in presentation + review_host
