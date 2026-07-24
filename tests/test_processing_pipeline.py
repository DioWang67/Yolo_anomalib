from __future__ import annotations

import inspect

import pytest

from tools.processing_pipeline import (
    ExecutionEngine,
    ExecutionMode,
    ExecutionResult,
    Phase3AExecutionEngine,
    ProcessingExecutionUnavailableError,
    ProcessingPlanner,
    RoutingDecisionType,
)


def _record(sample_id: str, label: str = "confirmed_ng", **updates: str) -> dict[str, str]:
    rows = {
        "confirmed_ng": {
            "review_outcome": "fail",
            "failure_category": "threshold_not_met",
            "product_verdict": "ng",
            "detection_verdict": "correct",
            "color_verdict": "not_applicable",
            "action_route": "yolo",
            "training_selected": "1",
        },
        "wrong_box": {
            "review_outcome": "fail",
            "failure_category": "wrong_box",
            "product_verdict": "ng",
            "detection_verdict": "wrong_box",
            "color_verdict": "not_applicable",
            "action_route": "yolo",
            "training_selected": "1",
        },
        "wrong_class": {
            "review_outcome": "fail",
            "failure_category": "wrong_class",
            "product_verdict": "ng",
            "detection_verdict": "wrong_class",
            "color_verdict": "not_applicable",
            "action_route": "yolo",
            "training_selected": "1",
        },
        "color_false_reject": {
            "review_outcome": "fail",
            "failure_category": "color_issue",
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
        "skip_reason": "",
        **rows[label],
    }
    record.update(updates)
    return record


def _plan(*records: dict[str, str]):
    return ProcessingPlanner().create_plan(
        list(enumerate(records)),
        operator="operator-a",
    )


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_record("ready"), RoutingDecisionType.READY_FOR_DATASET),
        (_record("annotation", "wrong_box"), RoutingDecisionType.NEEDS_ANNOTATION),
        (_record("class", "wrong_class"), RoutingDecisionType.NEEDS_CLASS_FIX),
        (
            _record("color", "color_false_reject"),
            RoutingDecisionType.NEEDS_COLOR_CALIBRATION,
        ),
        (
            _record("blocked", review_selected="0"),
            RoutingDecisionType.BLOCKED,
        ),
        (_record("excluded", "image_quality_issue"), RoutingDecisionType.EXCLUDED),
    ],
)
def test_planner_routes_phase_1b_semantics(record, expected):
    plan = _plan(record)

    assert plan.routing_decisions[0].decision == expected


def test_planner_routes_unfinished_review_to_manual_blocking():
    record = {
        "sample_id": "new",
        "review_selected": "1",
        "training_selected": "1",
    }

    plan = _plan(record)

    assert plan.routing_decisions[0].decision == RoutingDecisionType.MANUAL_REVIEW_REQUIRED
    assert plan.blocking_items[0].violation_code == "manual_review_required"


def test_both_route_keeps_annotation_and_color_steps():
    record = _record(
        "both",
        "color_false_reject",
        detection_verdict="wrong_box",
        action_route="both",
    )

    plan = _plan(record)
    decision = plan.routing_decisions[0]

    assert decision.decision == RoutingDecisionType.NEEDS_ANNOTATION
    assert decision.additional_decisions == (
        RoutingDecisionType.NEEDS_COLOR_CALIBRATION,
    )
    assert plan.statistics.annotation_count == 1
    assert plan.statistics.color_count == 1


def test_planner_analyzes_150_samples_and_counts_all_routes():
    records: list[dict[str, str]] = []
    records.extend(_record(f"ready-{index}") for index in range(100))
    records.extend(_record(f"box-{index}", "wrong_box") for index in range(20))
    records.extend(_record(f"class-{index}", "wrong_class") for index in range(10))
    records.extend(
        _record(f"color-{index}", "color_false_reject") for index in range(10)
    )
    records.extend(
        _record(f"blocked-{index}", review_selected="0") for index in range(5)
    )
    records.extend(
        _record(f"excluded-{index}", "image_quality_issue") for index in range(5)
    )

    plan = _plan(*records)

    assert plan.sample_count == 150
    assert plan.statistics.ready_count == 100
    assert plan.statistics.annotation_count == 30
    assert plan.statistics.color_count == 10
    assert plan.statistics.blocking_count == 5
    assert plan.statistics.excluded_count == 5
    assert len(plan.routing_decisions) == 150


def test_blocking_items_keep_sample_reason_and_phase_1b_violation_code():
    plan = _plan(_record("blocked-sample", review_selected="0"))

    assert len(plan.blocking_items) == 1
    item = plan.blocking_items[0]
    assert item.sample_id == "blocked-sample"
    assert item.violation_code == "review_without_selection"
    assert "review_selected" in item.reason


def test_plan_records_are_immutable_snapshots():
    source = _record("immutable")
    plan = _plan(source)
    source["review_label"] = "wrong_box"

    assert plan.records[0].fields["review_label"] == "confirmed_ng"
    with pytest.raises(TypeError):
        plan.records[0].fields["review_label"] = "wrong_box"


def test_execution_modes_are_supported_and_default_is_prepare_and_train():
    default_plan = _plan(_record("default"))
    auto_plan = ProcessingPlanner().create_plan(
        [(0, _record("auto"))],
        operator="operator-a",
        execution_mode=ExecutionMode.AUTO_DEPLOY_AFTER_GATE,
    )

    assert default_plan.execution_mode == ExecutionMode.PREPARE_AND_TRAIN
    assert {mode.value for mode in ExecutionMode} == {
        "PREPARE_ONLY",
        "PREPARE_AND_TRAIN",
        "AUTO_DEPLOY_AFTER_GATE",
    }
    assert any(
        warning.code == "manual_deployment_confirmation_required"
        for warning in auto_plan.warnings
    )


def test_planner_has_no_qt_or_gui_dependency():
    import tools.processing_pipeline as module

    source = inspect.getsource(module)
    assert "PyQt" not in source
    assert "app.gui" not in source


class RecordingEngine(ExecutionEngine):
    def __init__(self) -> None:
        self.received = None

    def execute(self, plan):
        self.received = plan
        return ExecutionResult(plan.plan_id, True, "accepted")


def test_execution_engine_contract_receives_the_complete_plan():
    plan = _plan(_record("ready-a"), _record("ready-b"))
    engine = RecordingEngine()

    result = engine.execute(plan)

    assert engine.received is plan
    assert engine.received.sample_count == 2
    assert result.accepted is True


def test_phase3a_engine_never_starts_downstream_execution():
    plan = _plan(_record("ready"))

    with pytest.raises(ProcessingExecutionUnavailableError, match=plan.plan_id):
        Phase3AExecutionEngine().execute(plan)
