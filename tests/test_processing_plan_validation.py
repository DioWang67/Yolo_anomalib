import csv
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from tools.processing_pipeline import (
    ExecutionMode,
    ProcessingPlanner,
    ProcessingStatistics,
    RoutingDecisionType,
    record_sha256,
    sha256_file,
)
from tools.processing_plan_validation import (
    ProcessingPlanValidator,
    ProcessingValidationContext,
    build_validation_context_from_manifest,
)

NOW = datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)


def _record(sample_id="sample-1", label="confirmed_ng", **updates):
    values = {
        "sample_id": sample_id,
        "review_selected": "1",
        "review_outcome": "fail",
        "review_label": label,
        "failure_category": "threshold_not_met",
        "skip_reason": "",
        "product_verdict": "ng",
        "detection_verdict": "correct",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }
    if label == "image_quality_issue":
        values.update(
            review_outcome="skip",
            failure_category="",
            skip_reason="image_quality_issue",
            product_verdict="unjudgeable",
            detection_verdict="unjudgeable",
            color_verdict="unjudgeable",
            action_route="none",
            training_selected="0",
        )
    values.update(updates)
    return values


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


def _codes(result):
    return {issue.code for issue in result.blocking_issues}


def test_valid_processing_plan():
    plan = _plan(_record())

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert result.valid is True
    assert result.executable_sample_ids == ("sample-1",)
    assert result.skipped_sample_ids == ()


def test_empty_plan_is_blocked():
    plan = _plan()

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert "empty_plan" in _codes(result)


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (lambda plan: replace(plan, sample_count=2), "sample_count_mismatch"),
        (lambda plan: replace(plan, routing_decisions=()), "missing_routing_decision"),
        (
            lambda plan: replace(
                plan,
                statistics=ProcessingStatistics(99, 0, 0, 0, 0),
            ),
            "statistics_mismatch",
        ),
        (lambda plan: replace(plan, execution_mode="UNKNOWN"), "unsupported_execution_mode"),
        (lambda plan: replace(plan, operator=""), "invalid_operator"),
        (lambda plan: replace(plan, schema_version=999), "unsupported_plan_schema"),
        (lambda plan: replace(plan, routing_code_version="old"), "stale_routing_code"),
        (lambda plan: replace(plan, source_manifest_sha="bad"), "invalid_source_manifest_sha"),
        (lambda plan: replace(plan, review_revision=""), "invalid_review_revision"),
    ],
)
def test_invalid_plan_header_and_shape(mutator, expected_code):
    original = _plan(_record())
    plan = mutator(original)

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(original))

    assert result.valid is False
    assert expected_code in _codes(result)


def test_duplicate_sample_and_source_index_are_blocked():
    original = _plan(_record("one"), _record("two"))
    duplicate_record = replace(original.records[1], sample_id="one", source_index=0)
    plan = replace(original, records=(original.records[0], duplicate_record))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(original))

    assert {"duplicate_sample_id", "duplicate_source_index"}.issubset(_codes(result))


def test_unknown_and_invalid_additional_routing_are_blocked():
    original = _plan(_record())
    invalid_decision = replace(
        original.routing_decisions[0],
        decision="UNKNOWN",
        additional_decisions=(RoutingDecisionType.EXCLUDED,),
    )
    plan = replace(original, routing_decisions=(invalid_decision,))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(original))

    assert "unknown_primary_routing" in _codes(result)
    assert "unknown_additional_routing" in _codes(result)


def test_additional_routing_is_only_color_after_annotation_or_class_fix():
    original = _plan(_record())
    invalid_decision = replace(
        original.routing_decisions[0],
        additional_decisions=(RoutingDecisionType.NEEDS_COLOR_CALIBRATION,),
    )
    plan = replace(original, routing_decisions=(invalid_decision,))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert "unknown_additional_routing" in _codes(result)


def test_stale_manifest_and_record_are_blocked():
    plan = _plan(_record())
    context = _context(
        plan,
        current_manifest_sha="f" * 64,
        current_record_hashes={"sample-1": "0" * 64},
    )

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, context)

    assert {"stale_manifest", "stale_record"}.issubset(_codes(result))


def test_missing_current_record_is_stale():
    plan = _plan(_record())

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(
        plan,
        _context(plan, current_record_hashes={"different": "0" * 64}),
    )

    assert "stale_record_missing" in _codes(result)


def test_blocked_sample_prevents_execution():
    plan = _plan(_record(review_selected="0"))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert result.valid is False
    assert "plan_contains_blocking_sample" in _codes(result)


def test_excluded_sample_is_valid_but_not_executable():
    plan = _plan(_record(label="image_quality_issue"))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert result.valid is True
    assert result.executable_sample_ids == ()
    assert result.skipped_sample_ids == ("sample-1",)


def test_auto_deploy_requires_explicit_policy():
    original = _plan(_record())
    plan = replace(original, execution_mode=ExecutionMode.AUTO_DEPLOY_AFTER_GATE)

    denied = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))
    allowed = ProcessingPlanValidator(clock=lambda: NOW).validate(
        plan,
        _context(plan, deployment_confirmation_allowed=True),
    )

    assert "deployment_confirmation_required" in _codes(denied)
    assert allowed.valid is True


def test_artifact_path_outside_root_is_blocked():
    plan = _plan(_record(processing_artifact_path="../../escape.txt"))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert "artifact_path_outside_root" in _codes(result)


def test_mutable_plan_container_is_blocked():
    original = _plan(_record())
    plan = replace(original, records=list(original.records))

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(original))

    assert "mutable_plan_snapshot" in _codes(result)


def test_retry_attempt_requires_auditable_references():
    original = _plan(_record())
    plan = replace(original, attempt=2)

    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, _context(plan))

    assert "retry_reference_missing" in _codes(result)


def test_manifest_context_uses_current_sha_and_record_hashes(tmp_path):
    source = _record()
    manifest = tmp_path / "review.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source))
        writer.writeheader()
        writer.writerow(source)
    plan = ProcessingPlanner().create_plan(
        [(0, source)],
        operator="operator-a",
        created_at=NOW,
        source_manifest_sha=sha256_file(manifest),
    )
    plan = replace(plan, plan_id="plan-1")

    context = build_validation_context_from_manifest(
        manifest,
        plan,
        artifact_root=tmp_path / ".processing_runs" / "artifacts",
    )
    result = ProcessingPlanValidator(clock=lambda: NOW).validate(plan, context)

    assert result.valid is True
    assert context.current_manifest_sha == plan.source_manifest_sha
    assert context.current_record_hashes["sample-1"] == record_sha256(
        plan.records[0].fields
    )
