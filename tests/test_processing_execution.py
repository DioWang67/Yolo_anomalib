import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tools.processing_events import (
    CompositeEventSink,
    ProcessingEventSink,
    ProcessingEventSinkError,
    ProcessingEventType,
)
from tools.processing_execution import (
    BlockedStep,
    CancellationToken,
    ExcludedStep,
    FatalProcessingStepError,
    NoOpAnnotationStep,
    NoOpColorStep,
    NoRetryableSamplesError,
    ProcessingExecutionEngine,
    ProcessingExecutionPersistenceError,
    ProcessingStep,
    ProcessingStepRegistry,
    RetryPlanValidationError,
    build_retry_plan,
)
from tools.processing_pipeline import (
    ProcessingPlanner,
    RoutingDecisionType,
    record_sha256,
)
from tools.processing_plan_validation import (
    ProcessingPlanValidator,
    ProcessingValidationContext,
)
from tools.processing_reports import (
    ProcessingReportStatus,
    ProcessingStepStatus,
    SampleProcessingStatus,
    StepResult,
)
from tools.processing_run_store import (
    ProcessingPersistenceError,
    ProcessingRunStore,
)

NOW = datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)


class DeterministicIds:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return f"id-{self.value:04d}"


def _record(sample_id, label="confirmed_ng", **updates):
    records = {
        "confirmed_ng": {
            "review_outcome": "fail",
            "failure_category": "threshold_not_met",
            "skip_reason": "",
            "product_verdict": "ng",
            "detection_verdict": "correct",
            "color_verdict": "not_applicable",
            "action_route": "yolo",
            "training_selected": "1",
        },
        "wrong_box": {
            "review_outcome": "fail",
            "failure_category": "wrong_box",
            "skip_reason": "",
            "product_verdict": "ng",
            "detection_verdict": "wrong_box",
            "color_verdict": "not_applicable",
            "action_route": "yolo",
            "training_selected": "1",
        },
        "color_false_reject": {
            "review_outcome": "fail",
            "failure_category": "color_issue",
            "skip_reason": "",
            "product_verdict": "ok",
            "detection_verdict": "correct",
            "color_verdict": "actually_ok",
            "action_route": "color",
            "training_selected": "1",
        },
        "image_quality_issue": {
            "review_outcome": "skip",
            "failure_category": "",
            "skip_reason": "image_quality_issue",
            "product_verdict": "unjudgeable",
            "detection_verdict": "unjudgeable",
            "color_verdict": "unjudgeable",
            "action_route": "none",
            "training_selected": "0",
        },
    }
    record = {
        "sample_id": sample_id,
        "review_selected": "1",
        "review_label": label,
        **records[label],
    }
    record.update(updates)
    return record


def _plan(*records):
    plan = ProcessingPlanner().create_plan(
        list(enumerate(records)),
        operator="operator-a",
        created_at=NOW,
    )
    return replace(plan, plan_id="plan-1")


def _context(plan, **updates):
    values = {
        "current_manifest_sha": plan.source_manifest_sha,
        "current_record_hashes": {
            record.sample_id: record_sha256(record.fields) for record in plan.records
        },
        "artifact_root": ".processing_runs/artifacts",
    }
    values.update(updates)
    return ProcessingValidationContext(**values)


def _engine(tmp_path, plan, *, registry=None, sink_factory=None, store=None):
    store = store or ProcessingRunStore(tmp_path / ".processing_runs")
    engine = ProcessingExecutionEngine(
        validator=ProcessingPlanValidator(clock=lambda: NOW),
        context_provider=lambda current_plan: _context(current_plan),
        store=store,
        step_registry=registry,
        clock=lambda: NOW,
        id_generator=DeterministicIds(),
        event_sink_factory=sink_factory,
    )
    return engine, store


def test_valid_plan_creates_deferred_report_and_audit_artifacts(tmp_path):
    plan = _plan(
        _record("ready"),
        _record("annotation", "wrong_box"),
        _record("color", "color_false_reject"),
        _record("excluded", "image_quality_issue"),
    )
    engine, store = _engine(tmp_path, plan)

    outcome = engine.execute(plan)

    assert outcome.report.status == ProcessingReportStatus.COMPLETED_WITH_WARNINGS
    assert [result.status for result in outcome.report.sample_results] == [
        SampleProcessingStatus.DEFERRED,
        SampleProcessingStatus.DEFERRED,
        SampleProcessingStatus.DEFERRED,
        SampleProcessingStatus.SKIPPED,
    ]
    assert outcome.report.summary.total == 4
    assert outcome.report.summary.deferred == 3
    assert outcome.report.summary.excluded == 1
    assert outcome.plan_document.path.is_file()
    assert outcome.report_document.path.is_file()
    assert outcome.event_path.is_file()
    assert outcome.report.events_reference.sha256
    assert (store.root / "latest.json").is_file()
    assert not any(store.artifacts_dir.rglob("*"))


def test_noop_steps_never_claim_success(tmp_path):
    plan = _plan(_record("ready"), _record("annotation", "wrong_box"))
    engine, _store = _engine(tmp_path, plan)

    report = engine.execute(plan).report

    assert report.summary.succeeded == 0
    assert all(
        result.status != SampleProcessingStatus.SUCCESS
        for result in report.sample_results
    )


def test_blocking_validation_persists_validation_failed_without_steps(tmp_path):
    plan = _plan(_record("blocked", review_selected="0"))
    engine, _store = _engine(tmp_path, plan)

    outcome = engine.execute(plan)

    assert outcome.report.status == ProcessingReportStatus.VALIDATION_FAILED
    assert outcome.report.sample_results[0].status == SampleProcessingStatus.BLOCKED
    assert not any(
        event.event_type == ProcessingEventType.STEP_STARTED
        for event in outcome.events
    )
    assert outcome.report_document.path.is_file()


def test_additional_decision_executes_both_deferred_steps(tmp_path):
    plan = _plan(
        _record(
            "both",
            "color_false_reject",
            detection_verdict="wrong_box",
            action_route="both",
        )
    )
    engine, _store = _engine(tmp_path, plan)

    outcome = engine.execute(plan)

    result = outcome.report.sample_results[0]
    assert result.status == SampleProcessingStatus.DEFERRED
    assert result.step_id == "noop_annotation + noop_color"
    assert result.additional_routing == (
        RoutingDecisionType.NEEDS_COLOR_CALIBRATION,
    )
    assert sum(
        event.event_type == ProcessingEventType.STEP_STARTED
        for event in outcome.events
    ) == 2


class ControlledReadyStep(ProcessingStep):
    step_id = "controlled_ready"
    supported_routing = frozenset({RoutingDecisionType.READY_FOR_DATASET})

    def __init__(self, behavior):
        self.behavior = behavior

    def execute(self, context):
        return self.behavior(context)

    def describe(self):
        return "Controlled test step."


def _registry(ready_step):
    return ProcessingStepRegistry(
        (
            ready_step,
            NoOpAnnotationStep(),
            NoOpColorStep(),
            BlockedStep(),
            ExcludedStep(),
        )
    )


def test_one_sample_failure_does_not_stop_the_next_sample(tmp_path):
    def behavior(context):
        if context.record.sample_id == "bad":
            return StepResult(
                ProcessingStepStatus.FAILED,
                "controlled_failure",
                error_code="controlled",
                message="failed",
                retryable=True,
            )
        return StepResult(
            ProcessingStepStatus.DEFERRED,
            "would_prepare_dataset",
            message="deferred",
        )

    plan = _plan(_record("bad"), _record("good"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(behavior)),
    )

    report = engine.execute(plan).report

    assert report.status == ProcessingReportStatus.PARTIAL_FAILURE
    assert [result.status for result in report.sample_results] == [
        SampleProcessingStatus.FAILED,
        SampleProcessingStatus.DEFERRED,
    ]


def test_unexpected_step_exception_becomes_retryable_failed_result(tmp_path):
    def behavior(_context):
        raise RuntimeError("controlled unexpected failure")

    plan = _plan(_record("bad"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(behavior)),
    )

    report = engine.execute(plan).report

    result = report.sample_results[0]
    assert report.status == ProcessingReportStatus.FAILED
    assert result.error_code == "unexpected_step_error"
    assert result.retryable is True


def test_explicit_fatal_error_interrupts_remaining_samples(tmp_path):
    def behavior(context):
        if context.record.sample_id == "fatal":
            raise FatalProcessingStepError("stop now")
        return StepResult(ProcessingStepStatus.DEFERRED, "deferred")

    plan = _plan(_record("fatal"), _record("later"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(behavior)),
    )

    report = engine.execute(plan).report

    assert report.status == ProcessingReportStatus.INTERRUPTED
    assert report.sample_results[0].status == SampleProcessingStatus.FAILED
    assert report.sample_results[1].status == SampleProcessingStatus.CANCELLED
    assert report.sample_results[1].metadata["not_started"] is True


def test_cancellation_before_first_sample_is_cooperative_and_audited(tmp_path):
    plan = _plan(_record("one"), _record("two"))
    engine, _store = _engine(tmp_path, plan)
    token = CancellationToken(clock=lambda: NOW)
    token.request_cancel("operator")

    outcome = engine.execute(plan, cancellation_token=token)

    assert outcome.report.status == ProcessingReportStatus.CANCELLED
    assert all(
        result.status == SampleProcessingStatus.CANCELLED
        for result in outcome.report.sample_results
    )
    event_types = {event.event_type for event in outcome.events}
    assert ProcessingEventType.EXECUTION_CANCEL_REQUESTED in event_types
    assert ProcessingEventType.EXECUTION_CANCELLED in event_types


def test_cancellation_inside_step_stops_after_safe_boundary(tmp_path):
    def behavior(context):
        context.cancellation_token.request_cancel("inside-step")
        return StepResult(ProcessingStepStatus.DEFERRED, "deferred")

    plan = _plan(_record("current"), _record("later"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(behavior)),
    )

    outcome = engine.execute(plan)
    report = outcome.report

    assert report.status == ProcessingReportStatus.CANCELLED
    assert report.sample_results[0].status == SampleProcessingStatus.CANCELLED
    assert report.sample_results[0].metadata["not_started"] is False
    assert report.sample_results[1].metadata["not_started"] is True
    assert ProcessingEventType.EXECUTION_CANCEL_REQUESTED in {
        event.event_type for event in outcome.events
    }


class FailingSink(ProcessingEventSink):
    def emit(self, event):
        raise ProcessingEventSinkError(f"sink failed {event.sequence}")


def test_sink_failure_is_report_warning_not_execution_failure(tmp_path):
    plan = _plan(_record("ready"))

    def sink_factory(memory, persistent):
        return CompositeEventSink((memory, persistent, FailingSink()))

    engine, _store = _engine(tmp_path, plan, sink_factory=sink_factory)

    outcome = engine.execute(plan)

    assert outcome.report.status == ProcessingReportStatus.COMPLETED_WITH_WARNINGS
    assert any("event_sink_failure" in warning for warning in outcome.report.warnings)


def test_report_persistence_failure_never_returns_completed(tmp_path, monkeypatch):
    plan = _plan(_record("ready"))
    engine, store = _engine(tmp_path, plan)

    def fail_report(_report):
        raise ProcessingPersistenceError("simulated report write failure")

    monkeypatch.setattr(store, "persist_report", fail_report)

    with pytest.raises(ProcessingExecutionPersistenceError) as captured:
        engine.execute(plan)

    assert captured.value.report.status == ProcessingReportStatus.FAILED
    assert all(path.exists() for path in captured.value.partial_artifacts)


def test_retry_plan_contains_only_retryable_failure_and_is_idempotent(tmp_path):
    def behavior(context):
        if context.record.sample_id == "retry-me":
            return StepResult(
                ProcessingStepStatus.FAILED,
                "failed",
                error_code="temporary",
                retryable=True,
            )
        return StepResult(ProcessingStepStatus.DEFERRED, "deferred")

    plan = _plan(_record("retry-me"), _record("deferred"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(behavior)),
    )
    report = engine.execute(plan).report
    validator = ProcessingPlanValidator(clock=lambda: NOW)

    first = build_retry_plan(
        report,
        plan,
        _context(plan),
        validator=validator,
        clock=lambda: NOW,
    )
    second = build_retry_plan(
        report,
        plan,
        _context(plan),
        validator=validator,
        clock=lambda: NOW,
    )

    assert first == second
    assert [record.sample_id for record in first.records] == ["retry-me"]
    assert first.retry_source_plan_id == plan.plan_id
    assert first.retry_source_report_id == report.report_id
    assert first.attempt == 2


def test_retry_deferred_requires_explicit_opt_in(tmp_path):
    plan = _plan(_record("deferred"))
    engine, _store = _engine(tmp_path, plan)
    report = engine.execute(plan).report
    validator = ProcessingPlanValidator(clock=lambda: NOW)

    with pytest.raises(NoRetryableSamplesError):
        build_retry_plan(report, plan, _context(plan), validator=validator)

    retry = build_retry_plan(
        report,
        plan,
        _context(plan),
        validator=validator,
        include_deferred=True,
        clock=lambda: NOW,
    )
    assert retry.sample_count == 1


def test_retry_plan_rejects_stale_source(tmp_path):
    def failure(_context):
        return StepResult(
            ProcessingStepStatus.FAILED,
            "failed",
            error_code="temporary",
            retryable=True,
        )

    plan = _plan(_record("retry-me"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(failure)),
    )
    report = engine.execute(plan).report

    with pytest.raises(RetryPlanValidationError):
        build_retry_plan(
            report,
            plan,
            _context(plan, current_manifest_sha="f" * 64),
            validator=ProcessingPlanValidator(clock=lambda: NOW),
            clock=lambda: NOW,
        )


def test_execution_does_not_mutate_immutable_plan(tmp_path):
    plan = _plan(_record("ready"))
    before = plan
    engine, _store = _engine(tmp_path, plan)

    engine.execute(plan)

    assert plan == before
    assert plan.records[0].fields["review_label"] == "confirmed_ng"


def test_missing_registered_step_becomes_diagnostic_failed_result(tmp_path):
    plan = _plan(_record("ready"))
    registry = ProcessingStepRegistry(
        (NoOpAnnotationStep(), NoOpColorStep(), BlockedStep(), ExcludedStep())
    )
    engine, _store = _engine(tmp_path, plan, registry=registry)

    report = engine.execute(plan).report

    assert report.status == ProcessingReportStatus.FAILED
    assert report.sample_results[0].error_code == "step_not_registered"


def test_150_sample_dry_run_is_deterministic_and_summary_matches(tmp_path):
    plan = _plan(*(_record(f"sample-{index:03d}") for index in range(150)))
    engine, _store = _engine(tmp_path, plan)

    outcome = engine.execute(plan)

    assert outcome.report.summary.total == 150
    assert outcome.report.summary.deferred == 150
    assert [result.sample_id for result in outcome.report.sample_results] == [
        f"sample-{index:03d}" for index in range(150)
    ]
    assert [event.sequence for event in outcome.events] == list(
        range(1, len(outcome.events) + 1)
    )


def test_event_jsonl_matches_in_memory_event_count_and_source_sha(tmp_path):
    plan = _plan(_record("ready"))
    engine, _store = _engine(tmp_path, plan)

    outcome = engine.execute(plan)

    assert len(outcome.event_path.read_text(encoding="utf-8").splitlines()) == len(
        outcome.events
    )
    assert outcome.report.source_manifest_sha == plan.source_manifest_sha
    latest = json.loads(
        (_store.root / "latest.json").read_text(encoding="utf-8")
    )
    assert latest["report_sha256"] == outcome.report_document.sha256


def test_cancelled_not_started_samples_can_build_retry_plan(tmp_path):
    plan = _plan(_record("one"), _record("two"))
    engine, _store = _engine(tmp_path, plan)
    token = CancellationToken(clock=lambda: NOW)
    token.request_cancel("before-run")
    report = engine.execute(plan, cancellation_token=token).report

    retry = build_retry_plan(
        report,
        plan,
        _context(plan),
        validator=ProcessingPlanValidator(clock=lambda: NOW),
        clock=lambda: NOW,
    )

    assert {record.sample_id for record in retry.records} == {"one", "two"}


def test_blocked_and_excluded_samples_never_enter_retry_plan(tmp_path):
    plan = _plan(
        _record("blocked", review_selected="0"),
        _record("excluded", "image_quality_issue"),
    )
    engine, _store = _engine(tmp_path, plan)
    report = engine.execute(plan).report

    with pytest.raises(NoRetryableSamplesError):
        build_retry_plan(
            report,
            plan,
            _context(plan),
            validator=ProcessingPlanValidator(clock=lambda: NOW),
        )


def test_non_retryable_failed_sample_is_not_retried(tmp_path):
    def failure(_context):
        return StepResult(
            ProcessingStepStatus.FAILED,
            "failed",
            error_code="permanent",
            retryable=False,
        )

    plan = _plan(_record("permanent"))
    engine, _store = _engine(
        tmp_path,
        plan,
        registry=_registry(ControlledReadyStep(failure)),
    )
    report = engine.execute(plan).report

    with pytest.raises(NoRetryableSamplesError):
        build_retry_plan(
            report,
            plan,
            _context(plan),
            validator=ProcessingPlanValidator(clock=lambda: NOW),
        )
