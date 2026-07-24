from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.gui.processing_batch_dialog import processing_color_step_enabled
from core.services.color_override_loader import ColorOverrideLoader
from tools.color_calibration_packages import (
    ColorCalibrationPackageService,
    load_color_calibration_package,
)
from tools.color_calibration_resume import (
    ColorCalibrationApprovalService,
    ColorCalibrationResumeService,
)
from tools.color_calibration_service import (
    CalibrationPolicy,
    ColorCalibrationError,
    ColorCalibrationScope,
    ColorCalibrationService,
    ColorEvidence,
    PictureToolThresholdBackend,
    sha256_file,
)
from tools.color_calibration_step import ColorCalibrationPreparationStep
from tools.color_configuration_resolver import ColorConfigurationResolver
from tools.color_configuration_revisions import ColorConfigurationRevisionStore
from tools.processing_events import InMemoryEventSink, ProcessingEventPublisher, ProcessingEventType
from tools.processing_execution import (
    BatchExecutionContext,
    BatchProcessingStep,
    BatchStepResult,
    BlockedStep,
    CancellationToken,
    ExcludedStep,
    NoOpReadyStep,
    ProcessingExecutionEngine,
    ProcessingStepRegistry,
)
from tools.processing_pipeline import ProcessingPlanner, RoutingDecisionType, record_sha256
from tools.processing_plan_validation import ProcessingPlanValidator, ProcessingValidationContext
from tools.processing_reports import ProcessingStepStatus, SampleProcessingStatus, StepResult
from tools.processing_run_store import ProcessingRunStore

NOW = datetime(2026, 7, 21, 4, 0, tzinfo=timezone.utc)


class FakeBackend:
    def recommend(self, samples, policy):
        scope = samples[0].scope
        return {
            "status": "ready",
            "product": scope.product,
            "area": scope.area,
            "model_type": scope.model_type,
            "checker_type": scope.checker_type,
            "threshold_key": scope.threshold_key,
            "current_public_threshold": 0.5,
            "suggested_public_threshold": 0.7,
            "current_config_value": 0.5,
            "suggested_config_value": 0.3,
            "reasons": [],
        }


class SequenceIds:
    def __init__(self, prefix="id"):
        self.value = 0
        self.prefix = prefix

    def __call__(self):
        self.value += 1
        return f"{self.prefix}-{self.value}"


def _color_record(sample_id: str, *, actual_ok: bool, diff: float, area="A", route="color"):
    label = "color_false_reject" if actual_ok else "color_confirmed_ng"
    return {
        "sample_id": sample_id,
        "review_selected": "1",
        "review_outcome": "fail",
        "review_label": label,
        "failure_category": "color_issue",
        "skip_reason": "",
        "product_verdict": "ok" if actual_ok else "ng",
        "detection_verdict": "correct",
        "color_verdict": "actually_ok" if actual_ok else "confirmed_ng",
        "action_route": route,
        "training_selected": "1",
        "product": "Cable1",
        "area": area,
        "detector": "yolo",
        "color_checker_type": "stats",
        "color_result_json": json.dumps({
            "is_ok": False,
            "items": [{
                "index": 0,
                "is_ok": False,
                "class_name": "red",
                "best_color": "red",
                "diff": diff,
                "threshold": 0.5,
            }],
        }),
    }


def _excluded_record():
    return {
        "sample_id": "excluded", "review_selected": "1", "review_outcome": "skip",
        "review_label": "image_quality_issue", "failure_category": "", "skip_reason": "image_quality_issue",
        "product_verdict": "unjudgeable", "detection_verdict": "unjudgeable",
        "color_verdict": "unjudgeable", "action_route": "none", "training_selected": "0",
    }


def _ready_record():
    return {
        "sample_id": "ready", "review_selected": "1", "review_outcome": "fail",
        "review_label": "confirmed_ng", "failure_category": "threshold_not_met", "skip_reason": "",
        "product_verdict": "ng", "detection_verdict": "correct", "color_verdict": "not_applicable",
        "action_route": "yolo", "training_selected": "1",
    }


def _fixture(tmp_path: Path):
    models = tmp_path / "models"
    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("color_checker_type: stats\ncolor_threshold_overrides:\n  red: 0.5\n", encoding="utf-8")
    other = models / "Cable1" / "B" / "yolo" / "config.yaml"
    other.parent.mkdir(parents=True)
    other.write_text("color_checker_type: stats\n", encoding="utf-8")
    records = [
        _color_record("ok-1", actual_ok=True, diff=0.6),
        _color_record("ng-1", actual_ok=False, diff=0.8),
        _color_record("ng-2", actual_ok=False, diff=0.9),
        _color_record("other-scope", actual_ok=True, diff=0.6, area="B"),
        _ready_record(),
        _excluded_record(),
    ]
    plan = ProcessingPlanner().create_plan(
        list(enumerate(records)), operator="creator", source_manifest_sha="a" * 64, created_at=NOW,
    )
    plan = replace(plan, plan_id="plan-color")
    service = ColorCalibrationService(
        models_root=models,
        backend=FakeBackend(),
        policy=CalibrationPolicy(minimum_total=1, minimum_ok=1, minimum_ng=1),
    )
    package_service = ColorCalibrationPackageService(
        artifact_root=tmp_path / ".processing_runs" / "artifacts",
        calibration_service=service,
        clock=lambda: NOW,
        id_generator=lambda: "package-1",
    )
    package = package_service.create(
        plan, plan.records, plan.routing_decisions,
        report_id="report-1", cancellation=CancellationToken(clock=lambda: NOW),
    )
    return models, records, plan, package


def test_scope_is_exact_and_missing_values_fail_closed():
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "red")
    assert scope.key == "Cable1/A/yolo/stats/red"
    assert len(scope.scope_hash) == 24
    with pytest.raises(ColorCalibrationError, match="incomplete") as caught:
        ColorCalibrationScope("Cable1", "", "yolo", "stats", "red")
    assert caught.value.code == "CALIBRATION_SCOPE_MISSING"


def test_existing_backend_exposes_deterministic_pure_recommendation():
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "red")
    evidence = (
        ColorEvidence(scope, "ok", "0", 0.6, 0.5, True, "threshold"),
        ColorEvidence(scope, "ng", "0", 0.8, 0.5, False, "threshold"),
    )
    backend = PictureToolThresholdBackend()
    policy = CalibrationPolicy(minimum_total=2, minimum_ok=1, minimum_ng=1)
    first = backend.recommend(evidence, policy)
    second = backend.recommend(tuple(reversed(evidence)), policy)
    assert first == second
    assert first["status"] == "ready"


def test_package_contains_required_immutable_artifacts_and_two_scopes(tmp_path):
    _models, _records, _plan, package = _fixture(tmp_path)
    assert len(package.scopes) == 2
    for relative in (
        "package.json", "scopes.json", "source_samples.json", "metrics.json",
        "approval.json", "completion.json", "validation_report.json", "checksums.json",
    ):
        assert (package.root / relative).is_file()
    loaded = load_color_calibration_package(package.package_path)
    assert loaded.package_id == package.package_id
    assert all(gate.passed for gate in loaded.gates)


def test_package_source_snapshot_excludes_ready_and_excluded(tmp_path):
    _models, _records, _plan, package = _fixture(tmp_path)
    assert set(package.sample_ids) == {"ok-1", "ng-1", "ng-2", "other-scope"}


def test_package_tamper_is_detected(tmp_path):
    _models, _records, _plan, package = _fixture(tmp_path)
    (package.root / "metrics.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ColorCalibrationError) as caught:
        load_color_calibration_package(package.package_path)
    assert caught.value.code == "COLOR_PACKAGE_SHA_MISMATCH"


def test_preview_reports_correction_without_regression(tmp_path):
    _models, _records, _plan, package = _fixture(tmp_path)
    red = next(value for value in package.previews if value.scope.area == "A")
    assert red.metrics["corrected_count"] == 1
    assert red.metrics["regressed_count"] == 0
    assert red.metrics["after"]["false_negative"] == 0


def test_approval_requires_named_reason_and_separate_reviewer(tmp_path):
    _models, _records, _plan, package = _fixture(tmp_path)
    service = ColorCalibrationApprovalService(clock=lambda: NOW)
    scope_hash = package.scopes[0].scope_hash
    with pytest.raises(ColorCalibrationError) as caught:
        service.decide(package.package_path, scope_hash, approved=True, reviewer="creator", reason="looks safe")
    assert caught.value.code == "COLOR_SELF_APPROVAL_FORBIDDEN"
    with pytest.raises(ColorCalibrationError):
        service.decide(package.package_path, scope_hash, approved=True, reviewer="reviewer", reason="")


def test_partial_scope_approval_and_rejection_are_recorded(tmp_path):
    _models, _records, _plan, package = _fixture(tmp_path)
    service = ColorCalibrationApprovalService(clock=lambda: NOW)
    first, second = package.scopes
    service.decide(package.package_path, first.scope_hash, approved=True, reviewer="reviewer", reason="validated")
    service.decide(package.package_path, second.scope_hash, approved=False, reviewer="reviewer", reason="more data needed")
    payload = json.loads((package.root / "approval.json").read_text(encoding="utf-8"))
    assert payload["decisions"][first.scope_hash]["decision"] == "APPROVED"
    assert payload["decisions"][second.scope_hash]["decision"] == "REJECTED"
    event_types = {
        json.loads(path.read_text(encoding="utf-8"))["event_type"]
        for path in (package.root / "events").glob("*.json")
    }
    assert event_types == {"COLOR_SCOPE_APPROVED", "COLOR_SCOPE_REJECTED"}


def test_pending_scope_cannot_apply(tmp_path):
    models, records, plan, package = _fixture(tmp_path)
    resolver = ColorConfigurationResolver(models_root=models, revisions_root=tmp_path / ".color_revisions")
    service = ColorCalibrationResumeService(
        revision_store=resolver.revision_store, planner=ProcessingPlanner(),
        current_config_resolver=lambda scope: (resolver.resolve(scope).source_path, resolver.resolve(scope).config_sha256),
    )
    with pytest.raises(ColorCalibrationError) as caught:
        service.resume(package.package_path, plan, current_entries=list(enumerate(records)), current_manifest_sha="a" * 64, operator="operator")
    assert caught.value.code == "COLOR_APPROVAL_PENDING"


def test_resume_honors_cooperative_cancellation_before_commit(tmp_path):
    models, records, plan, package = _fixture(tmp_path)
    approvals = ColorCalibrationApprovalService(clock=lambda: NOW)
    for scope in package.scopes:
        approvals.decide(
            package.package_path,
            scope.scope_hash,
            approved=True,
            reviewer="reviewer",
            reason="validated",
        )
    resolver = ColorConfigurationResolver(
        models_root=models, revisions_root=tmp_path / ".color_revisions"
    )
    resume = ColorCalibrationResumeService(
        revision_store=resolver.revision_store,
        planner=ProcessingPlanner(),
        current_config_resolver=lambda scope: (
            resolver.resolve(scope).source_path,
            resolver.resolve(scope).config_sha256,
        ),
    )
    cancellation = CancellationToken(clock=lambda: NOW)
    cancellation.request_cancel("operator_requested")
    outcome = resume.resume(
        package.package_path,
        plan,
        current_entries=list(enumerate(records)),
        current_manifest_sha="a" * 64,
        operator="operator",
        cancellation_token=cancellation,
    )
    assert not outcome.revision_ids
    assert {item["code"] for item in outcome.failures} == {
        "COLOR_CALIBRATION_CANCELLED"
    }
    assert not (tmp_path / ".color_revisions" / "active").exists()


def test_approved_revision_activation_and_follow_up_replan(tmp_path):
    models, records, plan, package = _fixture(tmp_path)
    approvals = ColorCalibrationApprovalService(clock=lambda: NOW)
    first, second = package.scopes
    approvals.decide(package.package_path, first.scope_hash, approved=True, reviewer="reviewer", reason="validated")
    approvals.decide(package.package_path, second.scope_hash, approved=False, reviewer="reviewer", reason="different station")
    resolver = ColorConfigurationResolver(models_root=models, revisions_root=tmp_path / ".color_revisions")
    resume = ColorCalibrationResumeService(
        revision_store=resolver.revision_store, planner=ProcessingPlanner(),
        current_config_resolver=lambda scope: (resolver.resolve(scope).source_path, resolver.resolve(scope).config_sha256),
        clock=lambda: NOW, id_generator=lambda: "completion-1",
    )
    completion = resume.resume(
        package.package_path, plan, current_entries=list(enumerate(records)),
        current_manifest_sha="a" * 64, operator="operator",
    )
    assert completion.status == "PARTIAL_SUCCESS"
    assert len(completion.revision_ids) == 1
    assert resolver.revision_store.active_pointer_path(first).is_file()
    assert not resolver.revision_store.active_pointer_path(second).exists()
    assert completion.follow_up_plan is not None
    assert completion.follow_up_plan.color_source_package_id == package.package_id
    assert completion.follow_up_plan.statistics.color_count == 4
    assert completion.follow_up_plan.statistics.ready_count == 1
    assert any(
        json.loads(path.read_text(encoding="utf-8"))["event_type"]
        == "COLOR_PACKAGE_PARTIAL"
        for path in (package.root / "events").glob("*.json")
    )


def test_stale_current_config_prevents_commit_and_activation(tmp_path):
    models, records, plan, package = _fixture(tmp_path)
    approvals = ColorCalibrationApprovalService(clock=lambda: NOW)
    for scope in package.scopes:
        approvals.decide(package.package_path, scope.scope_hash, approved=True, reviewer="reviewer", reason="validated")
    (models / "Cable1" / "A" / "yolo" / "config.yaml").write_text("changed: true\n", encoding="utf-8")
    resolver = ColorConfigurationResolver(models_root=models, revisions_root=tmp_path / ".color_revisions")
    resume = ColorCalibrationResumeService(
        revision_store=resolver.revision_store, planner=ProcessingPlanner(),
        current_config_resolver=lambda scope: (resolver.resolve(scope).source_path, resolver.resolve(scope).config_sha256),
    )
    outcome = resume.resume(package.package_path, plan, current_entries=list(enumerate(records)), current_manifest_sha="a" * 64, operator="operator")
    assert "CURRENT_CONFIG_STALE" in {item["code"] for item in outcome.failures}
    assert not resolver.revision_store.active_pointer_path(package.scopes[0]).exists()


def test_activation_failure_preserves_previous_pointer(tmp_path):
    models, _records, _plan, package = _fixture(tmp_path)
    scope = package.scopes[0]
    base_sha = sha256_file(models / "Cable1" / scope.area / "yolo" / "config.yaml")
    ids = SequenceIds("revision")
    healthy = ColorConfigurationRevisionStore(root=tmp_path / ".color_revisions", id_generator=ids, clock=lambda: NOW)
    proposed = json.loads((package.root / "proposed_configs" / f"{scope.scope_hash}.json").read_text(encoding="utf-8"))
    revision = healthy.commit(package, scope, operator="reviewer", reason="safe", proposal_sha256="p", preview_sha256="v", proposed_config=proposed, metrics={}, parent_config_sha256=base_sha)
    pointer = healthy.activate(revision, operator="reviewer", reason="safe", expected_current_sha256=base_sha)
    original = pointer.read_bytes()
    failing = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions", id_generator=SequenceIds("failed"), clock=lambda: NOW,
        replace_file=lambda _source, _target: (_ for _ in ()).throw(OSError("injected")),
    )
    with pytest.raises(ColorCalibrationError) as caught:
        failing._atomic_pointer_write(pointer, {"replacement": True})
    assert caught.value.code == "COLOR_ACTIVATION_FAILED"
    assert pointer.read_bytes() == original


def test_revoke_is_append_only_and_active_revision_cannot_be_revoked(tmp_path):
    models, _records, _plan, package = _fixture(tmp_path)
    scope = package.scopes[0]
    base_sha = sha256_file(models / "Cable1" / scope.area / "yolo" / "config.yaml")
    store = ColorConfigurationRevisionStore(root=tmp_path / ".color_revisions", id_generator=SequenceIds(), clock=lambda: NOW)
    proposed = json.loads((package.root / "proposed_configs" / f"{scope.scope_hash}.json").read_text(encoding="utf-8"))
    revision = store.commit(package, scope, operator="reviewer", reason="safe", proposal_sha256="p", preview_sha256="v", proposed_config=proposed, metrics={}, parent_config_sha256=base_sha)
    store.activate(revision, operator="reviewer", reason="safe", expected_current_sha256=base_sha)
    with pytest.raises(ColorCalibrationError) as caught:
        store.revoke(revision, operator="reviewer", reason="bad")
    assert caught.value.code == "COLOR_ACTIVE_REVISION_CANNOT_REVOKE"


def test_runtime_resolver_loads_active_threshold_override(tmp_path):
    models, _records, _plan, package = _fixture(tmp_path)
    scope = package.scopes[0]
    base_sha = sha256_file(models / "Cable1" / scope.area / "yolo" / "config.yaml")
    store = ColorConfigurationRevisionStore(root=tmp_path / ".color_revisions", id_generator=SequenceIds(), clock=lambda: NOW)
    proposed = json.loads((package.root / "proposed_configs" / f"{scope.scope_hash}.json").read_text(encoding="utf-8"))
    revision = store.commit(package, scope, operator="reviewer", reason="safe", proposal_sha256="p", preview_sha256="v", proposed_config=proposed, metrics={}, parent_config_sha256=base_sha)
    store.activate(revision, operator="reviewer", reason="safe", expected_current_sha256=base_sha)
    resolver = ColorConfigurationResolver(models_root=models, revisions_root=tmp_path / ".color_revisions")
    overrides, global_value, revision_ids = resolver.active_overrides(product="Cable1", area=scope.area, model_type="yolo", checker_type="stats")
    assert overrides[scope.threshold_key] == pytest.approx(0.3)
    assert global_value is None
    assert revision_ids == (revision.revision_id,)


def test_color_override_loader_merges_revision_without_schema_change(tmp_path):
    models, _records, _plan, package = _fixture(tmp_path)
    scope = package.scopes[0]
    resolver = ColorConfigurationResolver(models_root=models, revisions_root=tmp_path / ".color_revisions")
    base_sha = sha256_file(models / "Cable1" / scope.area / "yolo" / "config.yaml")
    proposed = json.loads((package.root / "proposed_configs" / f"{scope.scope_hash}.json").read_text(encoding="utf-8"))
    revision = resolver.revision_store.commit(package, scope, operator="reviewer", reason="safe", proposal_sha256="p", preview_sha256="v", proposed_config=proposed, metrics={}, parent_config_sha256=base_sha)
    resolver.revision_store.activate(revision, operator="reviewer", reason="safe", expected_current_sha256=base_sha)
    config = type("Config", (), {"color_checker_type": "stats", "color_threshold_overrides": None, "color_rules_overrides": None, "color_decision_tuning": None})()
    logger = type("Logger", (), {"warning": lambda *args: None})()
    loader = ColorOverrideLoader(models, revision_resolver=resolver)
    overrides, _rules, _tuning = loader.load(
        config, "Cable1", scope.area, "yolo", logger
    )
    assert overrides[scope.threshold_key] == pytest.approx(0.3)
    assert loader.last_active_revision_ids == (revision.revision_id,)


@pytest.mark.parametrize(
    ("environment", "enabled"),
    [({}, False), ({"YOLO_PROCESSING_COLOR_STEP": "0"}, False), ({"YOLO_PROCESSING_COLOR_STEP": "1"}, True), ({"YOLO_PROCESSING_COLOR_STEP": "true"}, True)],
)
def test_color_feature_flag_is_independent(environment, enabled):
    assert processing_color_step_enabled(environment) is enabled


def test_planner_keeps_additional_color_semantics():
    record = _color_record("both", actual_ok=True, diff=0.6, route="both")
    record["detection_verdict"] = "wrong_box"
    plan = ProcessingPlanner().create_plan([(0, record)], operator="operator")
    assert plan.routing_decisions[0].decision == RoutingDecisionType.NEEDS_ANNOTATION
    assert plan.routing_decisions[0].additional_decisions == (RoutingDecisionType.NEEDS_COLOR_CALIBRATION,)


class FakeAnnotationBatch(BatchProcessingStep):
    step_id = "fake_annotation_batch"
    supported_routing = frozenset({RoutingDecisionType.NEEDS_ANNOTATION})

    def execute_batch(self, context: BatchExecutionContext) -> BatchStepResult:
        return BatchStepResult({
            record.sample_id: StepResult(
                status=ProcessingStepStatus.DEFERRED,
                action="ANNOTATION_PACKAGE_CREATED",
                metadata={"annotation_package": "fake"},
            )
            for record in context.records
        })

    def describe(self) -> str:
        return "fake annotation package"


def test_color_step_returns_deferred_and_emits_gate_events(tmp_path):
    _models, _records, plan, package = _fixture(tmp_path)
    service = ColorCalibrationPackageService(
        artifact_root=tmp_path / "step-artifacts",
        calibration_service=ColorCalibrationService(
            models_root=tmp_path / "models", backend=FakeBackend(),
            policy=CalibrationPolicy(minimum_total=1, minimum_ok=1, minimum_ng=1),
        ),
        clock=lambda: NOW, id_generator=lambda: "step-package",
    )
    step = ColorCalibrationPreparationStep(service)
    sink = InMemoryEventSink()
    publisher = ProcessingEventPublisher(
        plan_id=plan.plan_id, report_id="step-report", sink=sink,
        clock=lambda: NOW, id_generator=SequenceIds("event"),
    )
    from tools.processing_execution import ExecutionContext
    decision = next(item for item in plan.routing_decisions if item.decision == RoutingDecisionType.NEEDS_COLOR_CALIBRATION)
    record = next(item for item in plan.records if item.sample_id == decision.sample_id)
    result = step.execute(ExecutionContext(
        plan=plan, report_id="step-report", record=record, decision=decision,
        event_sink=publisher, cancellation_token=CancellationToken(clock=lambda: NOW),
        artifact_root=tmp_path / "step-artifacts", working_directory=tmp_path / "work", dry_run=False,
    ))
    assert result.status == ProcessingStepStatus.DEFERRED
    assert result.action == "COLOR_CALIBRATION_PROPOSED"
    assert result.metadata["requires_approval"] is True
    assert ProcessingEventType.COLOR_PACKAGE_CREATED in {item.event_type for item in sink.events}
    assert ProcessingEventType.COLOR_GATE_PASSED in {item.event_type for item in sink.events}


def test_additional_color_executes_after_batch_annotation_without_overwrite(tmp_path):
    models = tmp_path / "models"
    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("color_checker_type: stats\n", encoding="utf-8")
    record = _color_record("both", actual_ok=True, diff=0.6, route="both")
    record["detection_verdict"] = "wrong_box"
    plan = ProcessingPlanner().create_plan([(0, record)], operator="creator", source_manifest_sha="b" * 64, created_at=NOW)
    plan = replace(plan, plan_id="combined-plan")
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    package_service = ColorCalibrationPackageService(
        artifact_root=store.artifacts_dir,
        calibration_service=ColorCalibrationService(models_root=models, backend=FakeBackend(), policy=CalibrationPolicy(minimum_total=1, minimum_ok=1, minimum_ng=1)),
        clock=lambda: NOW, id_generator=lambda: "combined-package",
    )
    registry = ProcessingStepRegistry(
        (NoOpReadyStep(), ColorCalibrationPreparationStep(package_service), BlockedStep(), ExcludedStep()),
        batch_steps=(FakeAnnotationBatch(),),
    )
    context = ProcessingValidationContext(
        current_manifest_sha=plan.source_manifest_sha,
        current_record_hashes={item.sample_id: record_sha256(item.fields) for item in plan.records},
        artifact_root=store.artifacts_dir,
    )
    engine = ProcessingExecutionEngine(
        validator=ProcessingPlanValidator(clock=lambda: NOW), context_provider=lambda _plan: context,
        store=store, step_registry=registry, clock=lambda: NOW, id_generator=SequenceIds("run"), dry_run=False,
    )
    result = engine.execute(plan).report.sample_results[0]
    assert result.status == SampleProcessingStatus.DEFERRED
    assert "ANNOTATION_PACKAGE_CREATED" in result.action
    assert "COLOR_CALIBRATION_PROPOSED" in result.action
    assert result.metadata["annotation_package"] == "fake"
    assert result.metadata["requires_approval"] is True


class StatusBackend:
    def __init__(self, status: str, proposed=0.7):
        self.status = status
        self.proposed = proposed

    def recommend(self, samples, policy):
        return {
            "status": self.status,
            "current_public_threshold": 0.5,
            "suggested_public_threshold": self.proposed,
            "current_config_value": 0.5,
            "suggested_config_value": 1.0 - self.proposed,
            "reasons": ["policy result"],
        }


class FailingBackend:
    def recommend(self, samples, policy):
        raise ValueError("injected backend failure")


@pytest.mark.parametrize(
    ("backend_status", "proposal_status", "gate_code"),
    [
        ("insufficient_data", "INSUFFICIENT_DATA", "PROPOSAL_INSUFFICIENT_DATA"),
        ("blocked_by_safety_policy", "MANUAL_REVIEW_REQUIRED", "PROPOSAL_MANUAL_REVIEW_REQUIRED"),
        ("no_change", "NO_CHANGE", "PROPOSAL_NO_CHANGE"),
        ("unknown", "INVALID_INPUT", "PROPOSAL_INVALID_INPUT"),
    ],
)
def test_non_proposed_statuses_are_not_approvable(tmp_path, backend_status, proposal_status, gate_code):
    models = tmp_path / "models"
    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("color_checker_type: stats\n", encoding="utf-8")
    records = [_color_record("ok", actual_ok=True, diff=0.6), _color_record("ng", actual_ok=False, diff=0.8)]
    plan = ProcessingPlanner().create_plan(list(enumerate(records)), operator="creator")
    service = ColorCalibrationService(models_root=models, backend=StatusBackend(backend_status))
    evidence, _diagnostics = service.collect_evidence(plan.records)
    proposal, _preview, gate = service.build(evidence[0].scope, evidence)
    assert proposal.status.value == proposal_status
    assert gate.approval_allowed is False
    assert gate_code in gate.blocking_issues


def test_regression_gate_lists_sample_ids(tmp_path):
    models = tmp_path / "models"
    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("color_checker_type: stats\n", encoding="utf-8")
    records = [_color_record("ok", actual_ok=True, diff=0.6), _color_record("critical-ng", actual_ok=False, diff=0.8)]
    plan = ProcessingPlanner().create_plan(list(enumerate(records)), operator="creator")
    service = ColorCalibrationService(models_root=models, backend=StatusBackend("ready", proposed=0.9))
    evidence, _diagnostics = service.collect_evidence(plan.records)
    _proposal, _preview, gate = service.build(evidence[0].scope, evidence)
    assert gate.passed is False
    assert "REGRESSION_POLICY_FAILED" in gate.blocking_issues
    assert "critical-ng" in gate.regression_samples


def test_backend_failure_becomes_blocked_failed_proposal(tmp_path):
    models = tmp_path / "models"
    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("color_checker_type: stats\n", encoding="utf-8")
    records = [_color_record("ok", actual_ok=True, diff=0.6)]
    plan = ProcessingPlanner().create_plan(list(enumerate(records)), operator="creator")
    service = ColorCalibrationService(models_root=models, backend=FailingBackend())
    evidence, _diagnostics = service.collect_evidence(plan.records)
    proposal, _preview, gate = service.build(evidence[0].scope, evidence)
    assert proposal.status.value == "FAILED"
    assert gate.approval_allowed is False
    assert "PROPOSAL_FAILED" in gate.blocking_issues


def test_revision_commit_is_idempotent_and_rollback_is_new_activation(tmp_path):
    models, _records, _plan, package = _fixture(tmp_path)
    scope = package.scopes[0]
    base_sha = sha256_file(models / "Cable1" / scope.area / "yolo" / "config.yaml")
    store = ColorConfigurationRevisionStore(root=tmp_path / ".color_revisions", id_generator=SequenceIds(), clock=lambda: NOW)
    proposed = json.loads((package.root / "proposed_configs" / f"{scope.scope_hash}.json").read_text(encoding="utf-8"))
    first = store.commit(package, scope, operator="reviewer", reason="first", proposal_sha256="p1", preview_sha256="v1", proposed_config=proposed, metrics={}, parent_config_sha256=base_sha)
    duplicate = store.commit(package, scope, operator="reviewer", reason="first", proposal_sha256="p1", preview_sha256="v1", proposed_config=proposed, metrics={}, parent_config_sha256=base_sha)
    assert duplicate.revision_id == first.revision_id
    store.activate(first, operator="reviewer", reason="first", expected_current_sha256=base_sha)
    second_config = dict(proposed)
    second_config["config_value"] = 0.2
    second_config["public_threshold"] = 0.8
    second = store.commit(
        package, scope, operator="reviewer", reason="second", proposal_sha256="p2", preview_sha256="v2",
        proposed_config=second_config, metrics={}, parent_revision_id=first.revision_id,
        parent_config_sha256=first.new_config_sha256,
    )
    store.activate(second, operator="reviewer", reason="second", expected_current_sha256=first.new_config_sha256)
    store.rollback(scope, first.revision_id, operator="reviewer", reason="regression observed")
    assert store.read_active_pointer(scope)["revision_id"] == first.revision_id
    revocation = store.revoke(second, operator="reviewer", reason="regressed")
    assert revocation.is_file()
    assert second.config_path.is_file()
    assert store.is_revoked(second)
    assert any((first.root / "activation_events").glob("*.json"))
