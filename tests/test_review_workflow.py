from copy import deepcopy

import pytest

from tools.review_routing import action_route
from tools.review_workflow import (
    AICorrectness,
    AnnotationValidity,
    ProductVerdict,
    RequiredAction,
    ReviewWorkflowValidationError,
    WorkflowState,
    blocking_violations,
    derive_review_semantics,
    derive_workflow_state,
    ensure_record_consistent,
    map_review_action_to_legacy_fields,
    validate_record_consistency,
    validate_transition,
)


def _reviewed(**overrides):
    record = {
        "sample_id": "sample-1",
        "review_selected": "1",
        "review_outcome": "fail",
        "review_label": "false_negative",
        "failure_category": "missed_detection",
        "skip_reason": "",
        "product_verdict": "ng",
        "detection_verdict": "missed",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }
    record.update(overrides)
    return record


def _codes(record):
    return {
        violation.code
        for violation in blocking_violations(validate_record_consistency(record))
    }


def test_new_state_is_derived_without_mutation():
    record = {"review_selected": "0", "training_selected": "1"}
    original = deepcopy(record)

    assert derive_workflow_state(record) == WorkflowState.NEW
    assert record == original


def test_selected_state_is_derived():
    assert derive_workflow_state(
        {"review_selected": "1", "training_selected": "0"}
    ) == WorkflowState.SELECTED


@pytest.mark.parametrize(
    ("record", "product", "correctness", "annotation", "required"),
    [
        (
            _reviewed(
                review_outcome="pass",
                review_label="false_positive",
                failure_category="",
                product_verdict="ok",
                detection_verdict="false_positive",
            ),
            ProductVerdict.OK,
            AICorrectness.FALSE_POSITIVE,
            AnnotationValidity.NEEDS_BOX_FIX,
            RequiredAction.ANNOTATION,
        ),
        (
            _reviewed(),
            ProductVerdict.NG,
            AICorrectness.FALSE_NEGATIVE,
            AnnotationValidity.MISSING,
            RequiredAction.ANNOTATION,
        ),
        (
            _reviewed(
                review_outcome="skip",
                review_label="confirmed_ng",
                failure_category="",
                skip_reason="confirmed_failure",
                detection_verdict="correct",
                action_route="none",
                training_selected="0",
            ),
            ProductVerdict.NG,
            AICorrectness.CORRECT,
            AnnotationValidity.VERIFIED,
            RequiredAction.EXCLUDE,
        ),
    ],
)
def test_legacy_pass_fail_skip_map_to_separated_semantics(
    record, product, correctness, annotation, required
):
    semantics = derive_review_semantics(record)

    assert semantics.product_verdict == product
    assert semantics.ai_correctness == correctness
    assert semantics.annotation_validity == annotation
    assert semantics.required_action == required


def test_false_positive_is_a_legal_annotation_route():
    record = _reviewed(
        review_outcome="pass",
        review_label="false_positive",
        failure_category="",
        product_verdict="ok",
        detection_verdict="false_positive",
    )

    assert blocking_violations(validate_record_consistency(record)) == ()
    assert derive_workflow_state(record) == WorkflowState.NEEDS_FIX


def test_false_negative_is_a_legal_annotation_route():
    record = _reviewed()

    assert blocking_violations(validate_record_consistency(record)) == ()
    assert derive_workflow_state(record) == WorkflowState.NEEDS_FIX


def test_wrong_box_derives_needs_fix():
    record = _reviewed(
        review_label="wrong_box",
        failure_category="wrong_box",
        detection_verdict="wrong_box",
    )

    assert derive_review_semantics(record).annotation_validity == (
        AnnotationValidity.NEEDS_BOX_FIX
    )
    assert derive_workflow_state(record) == WorkflowState.NEEDS_FIX


def test_color_only_derives_color_calibration_and_keeps_route():
    record = _reviewed(
        review_outcome="fail",
        review_label="color_false_reject",
        failure_category="color_issue",
        product_verdict="ok",
        detection_verdict="correct",
        color_verdict="actually_ok",
        action_route="color",
    )

    assert derive_review_semantics(record).required_action == (
        RequiredAction.COLOR_CALIBRATION
    )
    assert derive_workflow_state(record) == WorkflowState.READY
    assert blocking_violations(validate_record_consistency(record)) == ()


def test_position_false_reject_derives_position_calibration():
    record = _reviewed(
        review_outcome="pass",
        review_label="position_false_reject",
        failure_category="",
        product_verdict="ok",
        detection_verdict="correct",
        action_route="position",
    )

    semantics = derive_review_semantics(record)

    assert semantics.product_verdict == ProductVerdict.OK
    assert semantics.ai_correctness == AICorrectness.CORRECT
    assert semantics.annotation_validity == AnnotationValidity.VERIFIED
    assert semantics.required_action == RequiredAction.POSITION_CALIBRATION
    assert derive_workflow_state(record) == WorkflowState.READY
    assert blocking_violations(validate_record_consistency(record)) == ()


def test_position_feedback_cannot_be_sent_to_yolo():
    record = _reviewed(
        review_outcome="pass",
        review_label="position_false_reject",
        failure_category="",
        product_verdict="ok",
        detection_verdict="correct",
        action_route="yolo",
    )

    assert "position_feedback_wrong_route" in _codes(record)


def test_skip_cannot_be_training_selected():
    record = _reviewed(
        review_outcome="skip",
        review_label="confirmed_ng",
        failure_category="",
        skip_reason="confirmed_failure",
        detection_verdict="correct",
        action_route="none",
        training_selected="1",
    )

    assert "skip_selected_for_training" in _codes(record)


def test_missing_annotation_cannot_enter_direct_training():
    record = _reviewed(action_route="none")

    assert "missing_annotation_direct_training" in _codes(record)


def test_unjudgeable_ai_evidence_cannot_claim_definite_correctness():
    record = _reviewed(
        review_label="confirmed_ng",
        detection_verdict="unjudgeable",
    )

    assert "insufficient_ai_evidence_with_definite_correctness" in _codes(record)


def test_product_ok_cannot_be_confirmed_ng():
    record = _reviewed(
        review_label="confirmed_ng",
        product_verdict="ok",
        detection_verdict="correct",
    )

    assert "product_ok_confirmed_ng" in _codes(record)


def test_false_positive_and_false_negative_cannot_coexist():
    record = _reviewed(
        review_label="false_positive",
        product_verdict="ok",
        detection_verdict="missed",
    )

    assert "false_positive_false_negative_conflict" in _codes(record)


def test_wrong_box_cannot_use_non_annotation_route():
    record = _reviewed(
        review_label="wrong_box",
        failure_category="wrong_box",
        detection_verdict="wrong_box",
        action_route="none",
    )

    assert "annotation_fix_without_correction_route" in _codes(record)


def test_color_only_cannot_use_yolo_route():
    record = _reviewed(
        review_label="color_false_reject",
        failure_category="color_issue",
        product_verdict="ok",
        detection_verdict="correct",
        color_verdict="actually_ok",
        action_route="yolo",
    )

    assert "color_only_sent_to_yolo" in _codes(record)


def test_excluded_record_cannot_enter_handoff():
    record = _reviewed(
        review_outcome="skip",
        review_label="confirmed_ng",
        failure_category="",
        skip_reason="confirmed_failure",
        detection_verdict="correct",
        action_route="none",
        training_selected="0",
        handoff_selected="1",
    )

    assert "excluded_record_in_handoff" in _codes(record)


def test_unreviewed_record_cannot_be_explicitly_ready():
    record = {
        "sample_id": "new",
        "review_selected": "1",
        "training_selected": "1",
        "workflow_state": "READY",
    }

    assert "unreviewed_explicit_ready" in _codes(record)


def test_submitted_review_cannot_change_without_revision_reason():
    before = {**_reviewed(), "submission_status": "submitted"}
    after = {**before, "review_label": "wrong_box"}

    assert {
        violation.code for violation in validate_transition(before, after)
    } == {"submitted_review_modified_without_revision"}
    assert validate_transition(before, {**after, "revision_reason": "OP correction"}) == ()


def test_completed_cannot_reenter_processing_without_retry_reason():
    before = {"job_id": "job-1", "job_status": "deployed"}
    after = {"job_id": "job-1", "job_status": "training"}

    assert {
        violation.code for violation in validate_transition(before, after)
    } == {"completed_reentered_processing_without_retry"}


def test_legacy_inconsistency_is_diagnostic_only_and_not_rewritten():
    record = {"review_label": "uncertain", "training_selected": "0"}
    original = deepcopy(record)

    issues = validate_record_consistency(record)

    assert record == original
    assert {issue.code for issue in issues} >= {
        "legacy_review_selection_missing",
        "legacy_review_outcome_missing",
        "legacy_uncertain_label",
    }
    assert all(not issue.blocking for issue in issues)


def test_new_inconsistent_record_is_blocked():
    record = _reviewed(review_selected="0")

    with pytest.raises(ReviewWorkflowValidationError, match="review_without_selection"):
        ensure_record_consistent(record)


def test_domain_semantics_can_map_back_to_legacy_columns():
    source = _reviewed(
        review_outcome="pass",
        review_label="false_positive",
        failure_category="",
        product_verdict="ok",
        detection_verdict="false_positive",
    )
    semantics = derive_review_semantics(source)

    mapped = map_review_action_to_legacy_fields(
        semantics,
        review_outcome="pass",
        training_selected=True,
    )

    assert mapped == {
        "review_selected": "1",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "failure_category": "",
        "skip_reason": "",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }


@pytest.mark.parametrize(
    "label",
    [
        "confirmed_ng",
        "verified_empty",
        "false_positive",
        "false_negative",
        "wrong_box",
        "wrong_class",
        "color_confirmed_ng",
        "color_false_reject",
        "position_false_reject",
    ],
)
def test_existing_legal_routing_result_is_unchanged(label):
    mapped = map_review_action_to_legacy_fields(label)

    assert action_route(mapped) == mapped["action_route"]
