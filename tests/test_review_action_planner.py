import pytest

from tools.review_action_planner import (
    DETECTION_ABSENT,
    DETECTION_PRESENT,
    DETECTION_UNKNOWN,
    SKIP_CONFIRMED_FAILURE,
    SKIP_EQUIPMENT_LIGHTING,
    SKIP_IMAGE_QUALITY,
    SKIP_OTHER,
    SKIP_UNJUDGEABLE,
    detection_state,
    plan_fail,
    plan_pass,
    plan_skip,
    skip_custom_note_from_record,
    skip_ui_reason_from_record,
)
from tools.review_classification import (
    COLOR_ISSUE,
    LIGHTING_ISSUE,
    MISCLASSIFICATION,
    MISSED_DETECTION,
    OTHER_FAILURE,
    WRONG_BOX,
    WRONG_CLASS,
)
from tools.review_workflow import blocking_violations, validate_record_consistency


def _record(**overrides):
    return {
        "sample_id": "sample-1",
        "status": "FAIL",
        "detector": "yolo",
        "review_selected": "1",
        "review_outcome": "",
        "review_label": "",
        "training_selected": "0",
        "detections_json": "[]",
        "detected_box_count": "0",
        "detection_evidence_source": "snapshot",
        **overrides,
    }


def _assert_consistent(plan, record):
    assert not blocking_violations(
        validate_record_consistency(plan.proposed_record(record))
    )


def test_pass_plan_uses_phase1b_mapping_and_is_consistent():
    record = _record()

    plan = plan_pass(record)

    assert plan.review_label == "false_positive"
    assert plan.updates["review_outcome"] == "pass"
    assert plan.updates["training_selected"] == "1"
    _assert_consistent(plan, record)


def test_combined_color_and_non_color_pass_preserves_existing_product_verdict():
    record = _record(
        decision_reasons="COLOR_MISMATCH|MISSING",
        color_failure_count="1",
    )

    plan = plan_pass(record)

    assert plan.review_label == "color_false_reject"
    assert plan.updates["product_verdict"] == "ng"
    assert plan.updates["action_route"] == "color"
    _assert_consistent(plan, record)


def test_pass_snapshot_status_maps_to_confirmed_ok():
    plan = plan_pass(_record(status="PASS"))

    assert plan.review_label == "confirmed_ok"
    assert plan.updates["training_selected"] == "0"


@pytest.mark.parametrize(
    ("category", "expected_label", "training_selected"),
    [
        (MISSED_DETECTION, "false_negative", "1"),
        (WRONG_BOX, "wrong_box", "1"),
        (LIGHTING_ISSUE, "image_quality_issue", "0"),
    ],
)
def test_fail_plan_preserves_existing_routing_results(
    category, expected_label, training_selected
):
    record = _record()

    plan = plan_fail(record, category=category)

    assert plan.review_label == expected_label
    assert plan.updates["training_selected"] == training_selected
    assert plan.semantics.required_action.value in {
        "annotation",
        "exclude",
    }
    _assert_consistent(plan, record)


@pytest.mark.parametrize(
    ("ui_reason", "legacy_reason"),
    [
        (SKIP_UNJUDGEABLE, "image_quality_issue"),
        (SKIP_IMAGE_QUALITY, "image_quality_issue"),
        (SKIP_EQUIPMENT_LIGHTING, "image_quality_issue"),
        (SKIP_CONFIRMED_FAILURE, "confirmed_failure"),
        (SKIP_OTHER, "image_quality_issue"),
    ],
)
def test_five_skip_reasons_map_to_unchanged_legacy_enum(ui_reason, legacy_reason):
    record = _record()
    note = "operator note" if ui_reason == SKIP_OTHER else ""

    plan = plan_skip(record, ui_reason=ui_reason, note=note)
    proposed = plan.proposed_record(record)

    assert plan.updates["skip_reason"] == legacy_reason
    assert plan.updates["training_selected"] == "0"
    assert skip_ui_reason_from_record(proposed) == ui_reason
    assert skip_custom_note_from_record(proposed) == note
    _assert_consistent(plan, record)


def test_other_skip_reason_requires_note():
    with pytest.raises(ValueError, match="custom skip note"):
        plan_skip(_record(), ui_reason=SKIP_OTHER)


def test_unknown_skip_reason_is_rejected():
    with pytest.raises(ValueError, match="Unsupported skip reason"):
        plan_skip(_record(), ui_reason="future_value")


def test_legacy_skip_reason_without_ui_note_has_stable_fallback():
    assert (
        skip_ui_reason_from_record({"skip_reason": "confirmed_failure"})
        == SKIP_CONFIRMED_FAILURE
    )
    assert skip_ui_reason_from_record({}) == SKIP_IMAGE_QUALITY
    assert skip_custom_note_from_record({}) == ""


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"detections_json": '[{"bbox":[1,2,3,4]}]'}, DETECTION_PRESENT),
        ({"detections_json": "invalid", "detection_evidence_source": "unknown"}, DETECTION_UNKNOWN),
        ({"detected_box_count": "2"}, DETECTION_PRESENT),
        ({"detection_evidence_source": "saved_crops", "detected_box_count": ""}, DETECTION_UNKNOWN),
        ({"detection_evidence_source": "saved_crops", "detected_box_count": "1"}, DETECTION_PRESENT),
        ({"detection_evidence_source": "snapshot", "detected_box_count": "0"}, DETECTION_ABSENT),
        ({"detection_evidence_source": "legacy", "detected_box_count": ""}, DETECTION_ABSENT),
    ],
)
def test_detection_state_uses_only_persisted_structured_evidence(overrides, expected):
    assert detection_state(_record(**overrides)) == expected


@pytest.mark.parametrize(
    ("category", "expected_label"),
    [
        (MISCLASSIFICATION, "false_positive"),
        (WRONG_CLASS, "wrong_class"),
        (OTHER_FAILURE, "false_negative"),
    ],
)
def test_remaining_fail_categories_keep_legacy_labels(category, expected_label):
    plan = plan_fail(
        _record(),
        category=category,
        note="required note" if category == OTHER_FAILURE else "",
    )

    assert plan.review_label == expected_label


def test_combined_color_fail_preserves_non_color_product_verdict():
    plan = plan_fail(
        _record(
            decision_reasons="COLOR_MISMATCH|MISSING",
            color_failure_count="1",
        ),
        category=COLOR_ISSUE,
    )

    assert plan.review_label == "color_false_reject"
    assert plan.updates["product_verdict"] == "ng"
