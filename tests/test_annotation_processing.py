from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from tools.annotation_packages import (
    AnnotationPackageError,
    AnnotationPackageService,
    AnnotationPackageStatus,
    load_annotation_package,
    resolve_package_member,
)
from tools.annotation_resume import AnnotationResumeService
from tools.annotation_revisions import AnnotationRevisionStore
from tools.annotation_tool_launcher import (
    AnnotationToolLaunchStatus,
    ManualAnnotationToolLauncher,
    SubprocessAnnotationToolLauncher,
)
from tools.annotation_validation import (
    AnnotationErrorCode,
    AnnotationOperation,
    normalized_label_sha256,
    parse_yolo_label,
    validate_annotation_revision,
)
from tools.processing_engine_factory import build_processing_execution_engine
from tools.processing_execution import CancellationToken, ProcessingCancelledError
from tools.processing_pipeline import ProcessingPlanner, RoutingDecisionType, record_sha256
from tools.processing_plan_validation import ProcessingPlanValidator, ProcessingValidationContext
from tools.processing_reports import ProcessingReportStatus
from tools.processing_run_store import ProcessingRunStore

NOW = datetime(2026, 7, 21, 3, 0, tzinfo=timezone.utc)


class Ids:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return f"id-{self.value}"


def _record(tmp_path: Path, sample_id: str, label: str = "wrong_box") -> dict[str, str]:
    image = tmp_path / f"{sample_id}.png"
    marker = hashlib.sha256(sample_id.encode("utf-8")).digest()[0]
    Image.new("RGB", (100, 100), (marker, 4, 6)).save(image)
    detection = [{"class_id": 0, "class_name": "defect", "confidence": 0.9, "bbox": [25, 25, 75, 75]}]
    detection_verdict = {"wrong_box": "wrong_box", "wrong_class": "wrong_class", "false_negative": "missed"}[label]
    return {
        "sample_id": sample_id, "review_selected": "1", "review_outcome": "fail",
        "review_label": label, "failure_category": label, "skip_reason": "",
        "product_verdict": "ng", "detection_verdict": detection_verdict,
        "color_verdict": "not_applicable", "action_route": "yolo",
        "training_selected": "1", "original_path": str(image),
        "preprocessed_path": str(image), "detections_json": json.dumps(detection),
        "class_names_json": '["defect", "other"]', "product": "Cable1", "area": "A",
    }


def _plan(*records, source_sha="a" * 64):
    plan = ProcessingPlanner().create_plan(
        list(enumerate(records)), operator="operator", source_manifest_sha=source_sha,
        created_at=NOW,
    )
    return replace(plan, plan_id="plan-annotation")


def _ready_record(sample_id: str) -> dict[str, str]:
    return {
        "sample_id": sample_id, "review_selected": "1", "review_outcome": "fail",
        "review_label": "confirmed_ng", "failure_category": "threshold_not_met",
        "skip_reason": "", "product_verdict": "ng", "detection_verdict": "correct",
        "color_verdict": "not_applicable", "action_route": "yolo", "training_selected": "1",
    }


def _excluded_record(sample_id: str) -> dict[str, str]:
    return {
        "sample_id": sample_id, "review_selected": "1", "review_outcome": "skip",
        "review_label": "image_quality_issue", "failure_category": "", "skip_reason": "image_quality_issue",
        "product_verdict": "unjudgeable", "detection_verdict": "unjudgeable",
        "color_verdict": "unjudgeable", "action_route": "none", "training_selected": "0",
    }


def _package(tmp_path: Path, *records):
    plan = _plan(*records)
    routed = tuple(
        item for item in plan.routing_decisions
        if item.decision in {RoutingDecisionType.NEEDS_ANNOTATION, RoutingDecisionType.NEEDS_CLASS_FIX}
    )
    service = AnnotationPackageService(
        artifact_root=tmp_path / ".processing_runs" / "artifacts",
        clock=lambda: NOW, id_generator=lambda: "package-1",
    )
    package = service.create(
        plan, tuple(plan.records[item.source_index] for item in routed), routed,
        report_id="report-1", cancellation=CancellationToken(clock=lambda: NOW),
    )
    return plan, package


def test_yolo_validation_accepts_normalized_bbox_and_rejects_invalid_values():
    boxes, errors, _messages = parse_yolo_label("0 0.5 0.5 0.2 0.2\n", class_count=1)
    assert len(boxes) == 1 and not errors
    _boxes, errors, _messages = parse_yolo_label("2 nan 0.5 2 0.2\n", class_count=1)
    assert AnnotationErrorCode.CLASS_ID_INVALID in errors
    assert AnnotationErrorCode.BBOX_INVALID in errors


def test_class_only_geometry_change_requires_explicit_escalation():
    result = validate_annotation_revision(
        parent_label_text="0 0.5 0.5 0.2 0.2", working_label_text="1 0.6 0.5 0.2 0.2",
        class_count=2, requested_operation=AnnotationOperation.FIX_CLASS_ONLY,
        allow_empty=False, revision_reason="correct class",
    )
    assert AnnotationErrorCode.CLASS_FIX_GEOMETRY_CHANGED in result.errors
    escalated = validate_annotation_revision(
        parent_label_text="0 0.5 0.5 0.2 0.2", working_label_text="1 0.6 0.5 0.2 0.2",
        class_count=2, requested_operation=AnnotationOperation.FIX_CLASS_ONLY,
        allow_empty=False, revision_reason="correct class", escalation_reason="box also wrong",
    )
    assert escalated.valid
    assert escalated.actual_operation == AnnotationOperation.FIX_BOUNDING_BOX


def test_no_effective_change_and_missing_reason_are_blocking():
    result = validate_annotation_revision(
        parent_label_text="0 0.5 0.5 0.2 0.2", working_label_text=" 0 0.5 0.5 0.2 0.2 \n",
        class_count=1, requested_operation=AnnotationOperation.FIX_BOUNDING_BOX,
        allow_empty=False, revision_reason="",
    )
    assert set(result.errors) >= {
        AnnotationErrorCode.NO_EFFECTIVE_CHANGE,
        AnnotationErrorCode.REVISION_REASON_REQUIRED,
    }


def test_empty_label_policy_and_duplicate_diagnostic_are_explicit():
    allowed = validate_annotation_revision(
        parent_label_text="0 0.5 0.5 0.2 0.2", working_label_text="",
        class_count=1, requested_operation=AnnotationOperation.REVIEW_EMPTY_LABEL,
        allow_empty=True, revision_reason="confirmed false positive",
    )
    assert allowed.valid
    rejected = validate_annotation_revision(
        parent_label_text="0 0.5 0.5 0.2 0.2", working_label_text="",
        class_count=1, requested_operation=AnnotationOperation.FIX_BOUNDING_BOX,
        allow_empty=False, revision_reason="remove",
    )
    assert AnnotationErrorCode.EMPTY_LABEL_NOT_ALLOWED in rejected.errors
    duplicate = validate_annotation_revision(
        parent_label_text="0 0.5 0.5 0.2 0.2",
        working_label_text="0 0.5 0.5 0.3 0.3\n0 0.5 0.5 0.3 0.3",
        class_count=1, requested_operation=AnnotationOperation.FIX_BOUNDING_BOX,
        allow_empty=False, revision_reason="test duplicate",
    )
    assert duplicate.valid
    assert duplicate.warnings[0].startswith("duplicate_bbox:")


def test_package_contains_required_immutable_and_working_artifacts(tmp_path):
    _plan_value, package = _package(tmp_path, _record(tmp_path, "box"), _record(tmp_path, "class", "wrong_class"))
    loaded = load_annotation_package(package.package_path)
    assert loaded.status == AnnotationPackageStatus.WAITING_FOR_OPERATOR
    assert {item.requested_operation for item in loaded.items} == {
        AnnotationOperation.FIX_BOUNDING_BOX, AnnotationOperation.FIX_CLASS_ONLY,
    }
    for name in ("package.json", "class_mapping.json", "instructions.json", "validation_report.json", "completion.json", "checksums.json"):
        assert (package.root / name).is_file()
    assert len(list((package.root / "working_labels").glob("*.txt"))) == 2


def test_package_cancellation_before_commit_leaves_no_package(tmp_path):
    record = _record(tmp_path, "cancel")
    plan = _plan(record)
    token = CancellationToken(clock=lambda: NOW)
    token.request_cancel("test")
    service = AnnotationPackageService(artifact_root=tmp_path / "artifacts")
    with pytest.raises(ProcessingCancelledError):
        service.create(plan, plan.records, plan.routing_decisions, report_id="report", cancellation=token)
    assert not list((tmp_path / "artifacts" / "report" / "annotation").glob("package-*"))


def test_package_id_collision_never_overwrites_existing_package(tmp_path):
    row = _record(tmp_path, "collision")
    plan, package = _package(tmp_path, row)
    before = package.package_path.read_bytes()
    service = AnnotationPackageService(
        artifact_root=tmp_path / ".processing_runs" / "artifacts",
        id_generator=lambda: "package-1",
    )
    with pytest.raises(AnnotationPackageError) as captured:
        service.create(plan, plan.records, plan.routing_decisions, report_id="report-1", cancellation=CancellationToken())
    assert captured.value.code == "ANNOTATION_PACKAGE_EXISTS"
    assert package.package_path.read_bytes() == before


def test_package_tamper_and_working_path_escape_are_rejected(tmp_path):
    _plan_value, package = _package(tmp_path, _record(tmp_path, "tamper"))
    with pytest.raises(AnnotationPackageError) as captured:
        resolve_package_member(package, "../outside.txt")
    assert captured.value.code == "WORKING_PATH_ESCAPE"
    package.package_path.write_text("{}", encoding="utf-8")
    with pytest.raises(AnnotationPackageError) as captured:
        load_annotation_package(package.package_path)
    assert captured.value.code == "ANNOTATION_PACKAGE_STALE"


def test_class_mapping_conflict_fails_closed(tmp_path):
    first = _record(tmp_path, "one")
    second = _record(tmp_path, "two")
    second["class_names_json"] = '["different"]'
    plan = _plan(first, second)
    service = AnnotationPackageService(artifact_root=tmp_path / "artifacts")
    with pytest.raises(AnnotationPackageError, match="exact class mapping"):
        service.create(plan, plan.records, plan.routing_decisions, report_id="report", cancellation=CancellationToken())


def test_identical_image_records_require_explicit_canonical_selection(tmp_path):
    first = _record(tmp_path, "duplicate-one")
    second = _record(tmp_path, "duplicate-two")
    Path(second["original_path"]).write_bytes(Path(first["original_path"]).read_bytes())
    plan = _plan(first, second)
    service = AnnotationPackageService(artifact_root=tmp_path / "artifacts")
    with pytest.raises(AnnotationPackageError) as captured:
        service.create(plan, plan.records, plan.routing_decisions, report_id="report", cancellation=CancellationToken())
    assert captured.value.code == "CANONICAL_CONFLICT"
    assert not list((tmp_path / "artifacts").rglob("package.json"))


def test_manual_and_subprocess_launchers_never_wait(tmp_path):
    _plan_value, package = _package(tmp_path, _record(tmp_path, "launch"))
    assert ManualAnnotationToolLauncher().launch(package).status == AnnotationToolLaunchStatus.MANUAL
    executable = tmp_path / "tool.exe"
    executable.write_bytes(b"fake")
    calls = []
    process = type("Process", (), {"pid": 42})()
    launcher = SubprocessAnnotationToolLauncher(executable, popen=lambda argv, **kwargs: calls.append((argv, kwargs)) or process)
    result = launcher.launch(package)
    assert result.status == AnnotationToolLaunchStatus.LAUNCHED
    assert calls[0][1]["shell"] is False


def test_resume_commits_revision_proposal_and_follow_up_plan(tmp_path):
    row = _record(tmp_path, "resume")
    plan, package = _package(tmp_path, row)
    working = package.root / package.items[0].working_label
    working.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    manifest.write_text("placeholder", encoding="utf-8")
    store = AnnotationRevisionStore(
        root=tmp_path / ".annotation_revisions", source_manifest=manifest,
        clock=lambda: NOW, id_generator=Ids(),
    )
    service = AnnotationResumeService(
        revision_store=store, clock=lambda: NOW, id_generator=Ids(),
    )
    result = service.resume(
        package.package_path, plan, current_entries=((0, row),),
        current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"resume": "box corrected"},
    )
    assert result.status == AnnotationPackageStatus.COMPLETED
    assert len(result.successful_revisions) == 1
    revision = result.successful_revisions[0]
    assert revision.label_path.read_text(encoding="utf-8").startswith("0 ")
    assert result.record_update_proposals[0]["manifest_modified"] is False
    assert result.follow_up_plan is not None
    assert result.follow_up_plan.plan_id != plan.plan_id
    assert result.follow_up_plan.retry_source_plan_id == plan.plan_id
    assert result.follow_up_plan.retry_source_report_id == result.completion_id
    assert result.follow_up_plan.annotation_source_package_id == package.package_id
    assert result.follow_up_plan.annotation_revision_ids == (revision.revision_id,)
    assert result.follow_up_plan.routing_decisions[0].decision == RoutingDecisionType.READY_FOR_DATASET


def test_resume_partial_failure_commits_only_valid_item(tmp_path):
    first, second = _record(tmp_path, "good"), _record(tmp_path, "bad")
    plan, package = _package(tmp_path, first, second)
    (package.root / package.items[0].working_label).write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    (package.root / package.items[1].working_label).write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    manifest.write_text("x", encoding="utf-8")
    service = AnnotationResumeService(
        revision_store=AnnotationRevisionStore(root=tmp_path / "revisions", source_manifest=manifest, clock=lambda: NOW, id_generator=Ids()),
        clock=lambda: NOW, id_generator=Ids(),
    )
    result = service.resume(
        package.package_path, plan, current_entries=((0, first), (1, second)),
        current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"good": "fixed", "bad": "fixed"}, build_follow_up=False,
    )
    assert result.status == AnnotationPackageStatus.PARTIAL_FAILURE
    assert [value.sample_id for value in result.successful_revisions] == ["good"]
    assert result.failures[0].error_codes == (AnnotationErrorCode.CLASS_ID_INVALID.value,)
    (package.root / package.items[1].working_label).write_text(
        "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    fixed = service.resume(
        package.package_path, plan, current_entries=((0, first), (1, second)),
        current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"good": "fixed", "bad": "fixed after validation"},
        build_follow_up=False,
    )
    assert fixed.status == AnnotationPackageStatus.COMPLETED
    assert [value.sample_id for value in fixed.successful_revisions] == ["bad"]
    assert len(list((tmp_path / "revisions").rglob("revision.json"))) == 2


def test_resume_rejects_stale_manifest_image_parent_and_path_escape(tmp_path):
    row = _record(tmp_path, "stale")
    plan, package = _package(tmp_path, row)
    manifest = tmp_path / "review.csv"
    manifest.write_text("x", encoding="utf-8")
    service = AnnotationResumeService(
        revision_store=AnnotationRevisionStore(root=tmp_path / "revisions", source_manifest=manifest),
    )
    result = service.resume(
        package.package_path, plan, current_entries=((0, row),),
        current_manifest_sha="b" * 64, revision_reasons={"stale": "fixed"},
    )
    assert result.failures[0].error_codes == (AnnotationErrorCode.ANNOTATION_PACKAGE_STALE.value,)


def test_revision_revocation_is_append_only_and_canonical_reference_is_readable(tmp_path):
    row = _record(tmp_path, "revoke")
    plan, package = _package(tmp_path, row)
    working = package.root / package.items[0].working_label
    working.write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    manifest.write_text("x", encoding="utf-8")
    store = AnnotationRevisionStore(root=tmp_path / "revisions", source_manifest=manifest, clock=lambda: NOW, id_generator=Ids())
    resume_service = AnnotationResumeService(
        revision_store=store, clock=lambda: NOW, id_generator=Ids()
    )
    result = resume_service.resume(
        package.package_path, plan, current_entries=((0, row),), current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"revoke": "fixed"}, build_follow_up=False,
    )
    revision = result.successful_revisions[0]
    old_label = revision.label_path.read_bytes()
    from tools.export_review_dataset import load_approved_annotation_selections
    active = load_approved_annotation_selections(manifest)
    assert active[revision.image_sha256]["mode"] == "new_annotation_revision"
    revocation = store.revoke(revision, operator="operator", reason="bad review")
    assert revocation.is_file() and store.is_revoked(revision)
    assert revision.label_path.read_bytes() == old_label
    assert load_approved_annotation_selections(manifest) == {}
    with pytest.raises(AnnotationPackageError) as captured:
        resume_service._verify_canonical_revisions((revision,))
    assert captured.value.code == AnnotationErrorCode.CANONICAL_CONFLICT.value


def test_annotation_feature_flag_is_opt_in_and_independent():
    from app.gui.processing_batch_dialog import processing_annotation_step_enabled
    assert processing_annotation_step_enabled({"YOLO_PROCESSING_ANNOTATION_STEP": "1"})
    assert processing_annotation_step_enabled({"YOLO_PROCESSING_ANNOTATION_STEP": "true"})
    assert not processing_annotation_step_enabled({})
    assert not processing_annotation_step_enabled({"YOLO_PROCESSING_DATASET_STEP": "1"})


def test_engine_creates_one_annotation_package_and_does_not_start_training(tmp_path):
    row = _record(tmp_path, "engine")
    plan = _plan(row)
    manifest = tmp_path / "review.csv"
    manifest.write_text("unchanged", encoding="utf-8")
    source_before = manifest.read_bytes()
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    context = ProcessingValidationContext(
        current_manifest_sha=plan.source_manifest_sha,
        current_record_hashes={"engine": record_sha256(plan.records[0].fields)},
        artifact_root=store.artifacts_dir,
    )
    engine = build_processing_execution_engine(
        plan=plan, manifest_path=manifest, store=store,
        validator=ProcessingPlanValidator(clock=lambda: NOW),
        context_provider=lambda _plan_value: context,
        dataset_step_enabled=False, annotation_step_enabled=True,
    )
    outcome = engine.execute(plan)
    assert outcome.report.status == ProcessingReportStatus.COMPLETED_WITH_WARNINGS
    result = outcome.report.sample_results[0]
    assert result.action == "ANNOTATION_PACKAGE_CREATED"
    assert result.metadata["requires_resume"] is True
    assert result.metadata["training_started"] is False
    assert Path(result.metadata["package_path"]).is_file()
    assert manifest.read_bytes() == source_before


def test_validation_failure_creates_no_annotation_package(tmp_path):
    row = _record(tmp_path, "blocked-before-package")
    plan = _plan(row)
    manifest = tmp_path / "review.csv"
    manifest.write_text("unchanged", encoding="utf-8")
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    stale = ProcessingValidationContext(
        current_manifest_sha="b" * 64,
        current_record_hashes={row["sample_id"]: record_sha256(plan.records[0].fields)},
        artifact_root=store.artifacts_dir,
    )
    engine = build_processing_execution_engine(
        plan=plan, manifest_path=manifest, store=store,
        validator=ProcessingPlanValidator(clock=lambda: NOW), context_provider=lambda _plan_value: stale,
        dataset_step_enabled=False, annotation_step_enabled=True,
    )
    outcome = engine.execute(plan)
    assert outcome.report.status == ProcessingReportStatus.VALIDATION_FAILED
    assert not list(store.artifacts_dir.rglob("package.json"))


def test_controlled_four_record_fixture_routes_and_packages_only_annotation_items(tmp_path):
    box = _record(tmp_path, "fixture-box")
    class_fix = _record(tmp_path, "fixture-class", "wrong_class")
    rows = (box, class_fix, _ready_record("fixture-ready"), _excluded_record("fixture-excluded"))
    plan = _plan(*rows)
    assert (
        plan.statistics.ready_count,
        plan.statistics.annotation_count,
        plan.statistics.excluded_count,
    ) == (1, 2, 1)
    manifest = tmp_path / "review.csv"
    manifest.write_text("fixture", encoding="utf-8")
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    context = ProcessingValidationContext(
        current_manifest_sha=plan.source_manifest_sha,
        current_record_hashes={record.sample_id: record_sha256(record.fields) for record in plan.records},
        artifact_root=store.artifacts_dir,
    )
    outcome = build_processing_execution_engine(
        plan=plan, manifest_path=manifest, store=store,
        validator=ProcessingPlanValidator(clock=lambda: NOW), context_provider=lambda _value: context,
        dataset_step_enabled=False, annotation_step_enabled=True,
    ).execute(plan)
    package_paths = list(store.artifacts_dir.rglob("package.json"))
    assert len(package_paths) == 1
    package = load_annotation_package(package_paths[0])
    assert [item.sample_id for item in package.items] == ["fixture-box", "fixture-class"]
    assert all(
        result.metadata.get("training_started") is not True
        for result in outcome.report.sample_results
    )


def test_resume_is_idempotent_for_already_committed_items(tmp_path):
    row = _record(tmp_path, "repeat")
    plan, package = _package(tmp_path, row)
    (package.root / package.items[0].working_label).write_text("0 0.5 0.5 0.3 0.3\n", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    manifest.write_text("x", encoding="utf-8")
    store = AnnotationRevisionStore(root=tmp_path / "revisions", source_manifest=manifest, clock=lambda: NOW, id_generator=Ids())
    service = AnnotationResumeService(revision_store=store, clock=lambda: NOW, id_generator=Ids())
    first = service.resume(
        package.package_path, plan, current_entries=((0, row),), current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"repeat": "fixed"}, build_follow_up=False,
    )
    second = service.resume(
        package.package_path, plan, current_entries=((0, row),), current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"repeat": "fixed"}, build_follow_up=False,
    )
    assert first.status == second.status == AnnotationPackageStatus.COMPLETED
    assert len(list((tmp_path / "revisions").rglob("revision.json"))) == 1

    (package.root / package.items[0].working_label).write_text(
        "0 0.5 0.5 0.4 0.4\n", encoding="utf-8"
    )
    changed = service.resume(
        package.package_path, plan, current_entries=((0, row),), current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={"repeat": "second correction"}, build_follow_up=False,
    )
    assert changed.status == AnnotationPackageStatus.COMPLETED
    assert len(list((tmp_path / "revisions").rglob("revision.json"))) == 2
    from tools.export_review_dataset import load_approved_annotation_selections
    active = load_approved_annotation_selections(manifest)
    assert active[next(iter(active))]["label_sha256_by_sample"]["repeat"] == normalized_label_sha256(
        "0 0.5 0.5 0.4 0.4\n"
    )


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("image", AnnotationErrorCode.IMAGE_STALE.value),
        ("parent", AnnotationErrorCode.PARENT_LABEL_STALE.value),
        ("missing", AnnotationErrorCode.LABEL_MISSING.value),
    ],
)
def test_resume_stale_evidence_codes_are_stable(tmp_path, mutation, expected):
    row = _record(tmp_path, f"stale-{mutation}")
    plan, package = _package(tmp_path, row)
    item = package.items[0]
    if mutation == "image":
        (package.root / item.source_image).write_bytes(b"changed")
    elif mutation == "parent":
        (package.root / item.original_label).write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    else:
        (package.root / item.working_label).unlink()
    manifest = tmp_path / "review.csv"
    manifest.write_text("x", encoding="utf-8")
    result = AnnotationResumeService(
        revision_store=AnnotationRevisionStore(root=tmp_path / "revisions", source_manifest=manifest)
    ).resume(
        package.package_path, plan, current_entries=((0, row),), current_manifest_sha=plan.source_manifest_sha,
        revision_reasons={row["sample_id"]: "fixed"}, build_follow_up=False,
    )
    assert expected in result.failures[0].error_codes
