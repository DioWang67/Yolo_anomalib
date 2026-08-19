"""Pure compatibility domain for legacy review fields and workflow validation.

The workflow state is deliberately derived, never persisted.  This module has no
Qt, SQLite, or filesystem dependency so every save boundary can use the same
rules without changing the existing storage contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ProductVerdict(str, Enum):
    OK = "ok"
    NG = "ng"
    UNKNOWN = "unknown"


class AICorrectness(str, Enum):
    CORRECT = "correct"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    WRONG_CLASS = "wrong_class"
    UNKNOWN = "unknown"


class AnnotationValidity(str, Enum):
    VERIFIED = "verified"
    NEEDS_BOX_FIX = "needs_box_fix"
    NEEDS_CLASS_FIX = "needs_class_fix"
    MISSING = "missing"
    UNUSABLE = "unusable"
    UNKNOWN = "unknown"


class RequiredAction(str, Enum):
    NONE = "none"
    ANNOTATION = "annotation"
    CLASS_FIX = "class_fix"
    COLOR_CALIBRATION = "color_calibration"
    POSITION_CALIBRATION = "position_calibration"
    EXCLUDE = "exclude"
    MANUAL_REVIEW = "manual_review"


class WorkflowState(str, Enum):
    NEW = "NEW"
    SELECTED = "SELECTED"
    IN_REVIEW = "IN_REVIEW"
    REVIEWED = "REVIEWED"
    NEEDS_FIX = "NEEDS_FIX"
    READY = "READY"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    EXCLUDED = "EXCLUDED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ReviewSemantics:
    """Separated domain meaning derived from the legacy compatibility columns."""

    product_verdict: ProductVerdict
    ai_correctness: AICorrectness
    annotation_validity: AnnotationValidity
    required_action: RequiredAction
    legacy_review_label: str = ""
    legacy_action_route: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            key: value.value if isinstance(value, Enum) else str(value)
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True)
class WorkflowViolation:
    """One stable diagnostic emitted by consistency or transition validation."""

    code: str
    message: str
    suggestion: str
    blocking: bool = True


class ReviewWorkflowValidationError(ValueError):
    """Raised when a new record mutation violates the compatibility contract."""

    def __init__(
        self,
        record: Mapping[str, Any],
        violations: tuple[WorkflowViolation, ...],
    ) -> None:
        self.record = dict(record)
        self.violations = violations
        sample_id = record_identity(record)
        details = "; ".join(
            f"{violation.code}: {violation.message}"
            for violation in violations
            if violation.blocking
        )
        super().__init__(f"Invalid review record {sample_id}: {details}")


VALID_REVIEW_OUTCOMES = frozenset({"pass", "fail", "skip"})
VALID_REVIEW_LABELS = frozenset(
    {
        "confirmed_ng",
        "confirmed_ok",
        "verified_empty",
        "false_positive",
        "false_negative",
        "wrong_box",
        "wrong_class",
        "image_quality_issue",
        "color_confirmed_ng",
        "color_false_reject",
        "position_false_reject",
        "uncertain",
    }
)
VALID_FAILURE_CATEGORIES = frozenset(
    {
        "threshold_not_met",
        "misclassification",
        "missed_detection",
        "new_defect_type",
        "wrong_box",
        "wrong_class",
        "color_issue",
        "lighting_issue",
        "other",
    }
)
VALID_SKIP_REASONS = frozenset({"image_quality_issue", "confirmed_failure"})
VALID_PRODUCT_VERDICTS = frozenset({"ok", "ng", "unjudgeable"})
VALID_DETECTION_VERDICTS = frozenset(
    {
        "not_applicable",
        "correct",
        "false_positive",
        "missed",
        "false_negative",
        "wrong_box",
        "wrong_class",
        "unjudgeable",
    }
)
VALID_COLOR_VERDICTS = frozenset(
    {"not_applicable", "confirmed_ng", "actually_ok", "unjudgeable"}
)
VALID_ACTION_ROUTES = frozenset({"none", "yolo", "color", "position", "both"})
DIRECT_TRAIN_LABELS = frozenset(
    {"confirmed_ng", "verified_empty", "position_false_reject"}
)
CORRECTION_LABELS = frozenset(
    {"false_positive", "false_negative", "wrong_box", "wrong_class"}
)
COLOR_LABELS = frozenset({"color_confirmed_ng", "color_false_reject"})
POSITION_LABELS = frozenset({"position_false_reject"})
EXCLUDED_LABELS = frozenset({"image_quality_issue"})
PROCESSING_JOB_STATES = frozenset(
    {"preparing_dataset", "training", "evaluating", "deploying", "processing"}
)
QUEUED_JOB_STATES = frozenset(
    {"queued", "waiting_feedback", "waiting_annotation", "submitted"}
)
ERROR_JOB_STATES = frozenset({"failed", "error", "invalid"})
COMPLETED_JOB_STATES = frozenset({"deployed", "completed"})
VALID_JOB_STATUSES = (
    PROCESSING_JOB_STATES
    | QUEUED_JOB_STATES
    | ERROR_JOB_STATES
    | COMPLETED_JOB_STATES
    | {"cancelled", "unknown"}
)

REVIEW_CORE_FIELDS = frozenset(
    {
        "review_outcome",
        "review_label",
        "failure_category",
        "failure_source",
        "failure_note",
        "skip_reason",
        "product_verdict",
        "detection_verdict",
        "color_verdict",
        "action_route",
    }
)


def derive_review_semantics(record: Mapping[str, Any]) -> ReviewSemantics:
    """Derive separated domain meaning without mutating a legacy record."""
    label = _text(record.get("review_label"))
    route = _text(record.get("action_route")) or _legacy_route(label, record)
    outcome = _text(record.get("review_outcome"))
    product = _derive_product_verdict(record, label)
    ai_correctness = _derive_ai_correctness(record, label)
    annotation = _derive_annotation_validity(record, label)
    required_action = _derive_required_action(
        record,
        label=label,
        route=route,
        outcome=outcome,
        annotation=annotation,
    )
    return ReviewSemantics(
        product_verdict=product,
        ai_correctness=ai_correctness,
        annotation_validity=annotation,
        required_action=required_action,
        legacy_review_label=label,
        legacy_action_route=route,
    )


def derive_workflow_state(record: Mapping[str, Any]) -> WorkflowState:
    """Derive one compact state from review fields and transient lifecycle context."""
    job_status = _job_status(record)
    if job_status in ERROR_JOB_STATES:
        return WorkflowState.ERROR
    if job_status in COMPLETED_JOB_STATES:
        return WorkflowState.COMPLETED
    if job_status == "cancelled":
        return WorkflowState.EXCLUDED
    if job_status in PROCESSING_JOB_STATES:
        return WorkflowState.PROCESSING
    if job_status in QUEUED_JOB_STATES:
        return WorkflowState.QUEUED

    semantics = derive_review_semantics(record)
    label = _text(record.get("review_label"))
    outcome = _text(record.get("review_outcome"))
    reviewed = bool(label or outcome)
    if semantics.required_action == RequiredAction.EXCLUDE:
        return WorkflowState.EXCLUDED
    if reviewed and semantics.required_action in {
        RequiredAction.ANNOTATION,
        RequiredAction.CLASS_FIX,
        RequiredAction.MANUAL_REVIEW,
    }:
        return WorkflowState.NEEDS_FIX
    if reviewed and _bool(record.get("training_selected")):
        return WorkflowState.READY
    if reviewed:
        return WorkflowState.REVIEWED
    if _has_partial_review(record):
        return WorkflowState.IN_REVIEW
    if _bool(record.get("review_selected")):
        return WorkflowState.SELECTED
    return WorkflowState.NEW


def validate_record_consistency(
    record: Mapping[str, Any],
) -> tuple[WorkflowViolation, ...]:
    """Return all contradictions; callers decide whether warnings are blocking."""
    issues: list[WorkflowViolation] = []
    label = _text(record.get("review_label"))
    outcome = _text(record.get("review_outcome"))
    category = _text(record.get("failure_category"))
    skip_reason = _text(record.get("skip_reason"))
    product = _text(record.get("product_verdict"))
    detection = _text(record.get("detection_verdict"))
    color = _text(record.get("color_verdict"))
    route = _text(record.get("action_route"))
    reviewed = bool(label or outcome)
    review_selected_value = _text(record.get("review_selected"))
    selected_for_review = _bool(review_selected_value)
    selected_for_training = _bool(record.get("training_selected"))
    handoff_selected = _bool(record.get("handoff_selected"))
    semantics = derive_review_semantics(record)
    job_status = _job_status(record)
    effective_route = semantics.legacy_action_route

    _validate_known_value(issues, "review_outcome", outcome, VALID_REVIEW_OUTCOMES)
    _validate_known_value(issues, "review_label", label, VALID_REVIEW_LABELS)
    _validate_known_value(
        issues, "failure_category", category, VALID_FAILURE_CATEGORIES
    )
    _validate_known_value(issues, "skip_reason", skip_reason, VALID_SKIP_REASONS)
    _validate_known_value(
        issues, "product_verdict", product, VALID_PRODUCT_VERDICTS
    )
    _validate_known_value(
        issues, "detection_verdict", detection, VALID_DETECTION_VERDICTS
    )
    _validate_known_value(issues, "color_verdict", color, VALID_COLOR_VERDICTS)
    _validate_known_value(issues, "action_route", route, VALID_ACTION_ROUTES)
    _validate_known_value(issues, "job_status", job_status, VALID_JOB_STATUSES)
    for field in ("review_selected", "training_selected"):
        value = _text(record.get(field))
        if value and value not in {"0", "1", "false", "true"}:
            issues.append(
                WorkflowViolation(
                    code=f"invalid_{field}",
                    message=f"{field} must be a Boolean compatibility value, got {value!r}",
                    suggestion=f"Review and explicitly set {field} to 0 or 1.",
                )
            )

    if reviewed and review_selected_value in {"0", "false"}:
        issues.append(
            WorkflowViolation(
                code="review_without_selection",
                message="A human review result exists while review_selected is false",
                suggestion="Confirm this record belonged to the review scope before revising it.",
            )
        )
    elif reviewed and not selected_for_review:
        issues.append(
            WorkflowViolation(
                code="legacy_review_selection_missing",
                message="A reviewed legacy record has no review_selected value",
                suggestion="Keep the row readable; set review_selected on its next explicit revision.",
                blocking=False,
            )
        )
    if label and not outcome:
        issues.append(
            WorkflowViolation(
                code="legacy_review_outcome_missing",
                message="review_label exists but review_outcome is empty",
                suggestion="Keep the legacy row unchanged; set an explicit outcome on its next revision.",
                blocking=False,
            )
        )
    if label == "uncertain":
        issues.append(
            WorkflowViolation(
                code="legacy_uncertain_label",
                message="uncertain is a readable legacy value but is not a current review action",
                suggestion="Manually review the case before adding it to any new queue.",
                blocking=False,
            )
        )
    if outcome == "fail" and not category:
        issues.append(
            WorkflowViolation(
                code="fail_without_category",
                message="A fail outcome requires a failure_category",
                suggestion="Choose a failure category or provide the custom other reason.",
            )
        )
    if outcome == "skip" and not skip_reason:
        issues.append(
            WorkflowViolation(
                code="skip_without_reason",
                message="A skip outcome requires skip_reason",
                suggestion="Choose image_quality_issue or confirmed_failure.",
            )
        )
    if outcome != "skip" and skip_reason:
        issues.append(
            WorkflowViolation(
                code="skip_reason_without_skip",
                message="skip_reason is only valid when review_outcome is skip",
                suggestion="Clear skip_reason or change the outcome to skip.",
            )
        )
    if outcome == "skip" and selected_for_training:
        issues.append(
            WorkflowViolation(
                code="skip_selected_for_training",
                message="A skipped record cannot be selected for training",
                suggestion="Set training_selected to 0.",
            )
        )
    if product == "ok" and label == "confirmed_ng":
        issues.append(
            WorkflowViolation(
                code="product_ok_confirmed_ng",
                message="product_verdict=ok contradicts review_label=confirmed_ng",
                suggestion="Use false_positive/confirmed_ok or correct product_verdict.",
            )
        )
    if _has_false_positive_negative_conflict(label, detection):
        issues.append(
            WorkflowViolation(
                code="false_positive_false_negative_conflict",
                message="The record simultaneously represents false positive and false negative",
                suggestion="Choose the single AI error that matches the physical evidence.",
            )
        )
    definite_ai_labels = {
        "confirmed_ng",
        "confirmed_ok",
        "verified_empty",
        "false_positive",
        "false_negative",
        "wrong_class",
        "color_confirmed_ng",
        "color_false_reject",
        "position_false_reject",
    }
    if detection == "unjudgeable" and (
        semantics.ai_correctness != AICorrectness.UNKNOWN
        or label in definite_ai_labels
    ):
        issues.append(
            WorkflowViolation(
                code="insufficient_ai_evidence_with_definite_correctness",
                message="Unjudgeable detection evidence cannot produce definite AI correctness",
                suggestion="Keep ai correctness unknown and route to manual review/exclusion.",
            )
        )
    if (
        semantics.annotation_validity
        in {AnnotationValidity.NEEDS_BOX_FIX, AnnotationValidity.NEEDS_CLASS_FIX}
        and effective_route not in {"yolo", "both"}
    ):
        issues.append(
            WorkflowViolation(
                code="annotation_fix_without_correction_route",
                message="Annotation needs correction but action_route does not include YOLO annotation",
                suggestion="Use yolo/both or exclude the record.",
            )
        )
    if (
        semantics.annotation_validity == AnnotationValidity.MISSING
        and selected_for_training
        and effective_route not in {"yolo", "both"}
    ):
        issues.append(
            WorkflowViolation(
                code="missing_annotation_direct_training",
                message="A missing annotation cannot enter direct training",
                suggestion="Route it to annotation first or set training_selected to 0.",
            )
        )
    color_only = label in COLOR_LABELS and detection in {"", "correct", "not_applicable"}
    if color_only and effective_route in {"yolo", "both"}:
        issues.append(
            WorkflowViolation(
                code="color_only_sent_to_yolo",
                message="A color-only issue cannot use the YOLO annotation route",
                suggestion="Use action_route=color.",
            )
        )
    if label in POSITION_LABELS and effective_route != "position":
        issues.append(
            WorkflowViolation(
                code="position_feedback_wrong_route",
                message="Position calibration feedback must use action_route=position",
                suggestion="Use action_route=position.",
            )
        )
    if handoff_selected and not reviewed:
        issues.append(
            WorkflowViolation(
                code="unreviewed_ready_for_handoff",
                message="An unreviewed record cannot be marked ready for handoff",
                suggestion="Complete human review before creating a handoff.",
            )
        )
    if handoff_selected and semantics.required_action == RequiredAction.EXCLUDE:
        issues.append(
            WorkflowViolation(
                code="excluded_record_in_handoff",
                message="An excluded record cannot be included in a handoff",
                suggestion="Remove it from the selected handoff rows.",
            )
        )
    if handoff_selected and semantics.required_action == RequiredAction.MANUAL_REVIEW:
        issues.append(
            WorkflowViolation(
                code="manual_review_record_in_handoff",
                message="A record still requiring manual review cannot enter handoff",
                suggestion="Complete an explicit current review first.",
            )
        )
    explicit_state = _text(record.get("workflow_state")).upper()
    if explicit_state == WorkflowState.READY.value and not reviewed:
        issues.append(
            WorkflowViolation(
                code="unreviewed_explicit_ready",
                message="An unreviewed record cannot be READY",
                suggestion="Remove the ready marker and complete review first.",
            )
        )
    return _deduplicate_violations(issues)


def validate_transition(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[WorkflowViolation, ...]:
    """Validate lifecycle direction and immutable submitted-review semantics."""
    issues: list[WorkflowViolation] = []
    before_state = derive_workflow_state(before)
    after_state = derive_workflow_state(after)
    revision_reason = _text(after.get("revision_reason")) or _text(
        after.get("retry_reason")
    )
    if (
        before_state == WorkflowState.COMPLETED
        and after_state == WorkflowState.PROCESSING
        and not revision_reason
    ):
        issues.append(
            WorkflowViolation(
                code="completed_reentered_processing_without_retry",
                message="A completed record cannot return to processing without a retry/revision reason",
                suggestion="Create an explicit retry revision and record its reason.",
            )
        )
    submitted_before = _is_submitted(before)
    changed_review_fields = sorted(
        field
        for field in REVIEW_CORE_FIELDS
        if _text(before.get(field)) != _text(after.get(field))
    )
    if submitted_before and changed_review_fields and not revision_reason:
        issues.append(
            WorkflowViolation(
                code="submitted_review_modified_without_revision",
                message=(
                    "Submitted review fields were changed without a revision reason: "
                    + ", ".join(changed_review_fields)
                ),
                suggestion="Create an explicit review revision before changing submitted evidence.",
            )
        )
    return _deduplicate_violations(issues)


def map_review_action_to_legacy_fields(
    review_action: str | ReviewSemantics,
    *,
    review_outcome: str = "",
    failure_category: str = "",
    skip_reason: str = "",
    training_selected: bool | None = None,
) -> dict[str, str]:
    """Map a typed/legacy review action back to the unchanged CSV contract."""
    semantics = review_action if isinstance(review_action, ReviewSemantics) else None
    label = semantics.legacy_review_label if semantics is not None else _text(review_action)
    mappings: dict[str, tuple[str, str, str, str]] = {
        "confirmed_ng": ("ng", "correct", "not_applicable", "yolo"),
        "confirmed_ok": ("ok", "correct", "not_applicable", "none"),
        "verified_empty": ("ok", "correct", "not_applicable", "yolo"),
        "false_positive": ("ok", "false_positive", "not_applicable", "yolo"),
        "false_negative": ("ng", "missed", "not_applicable", "yolo"),
        "wrong_box": ("ng", "wrong_box", "not_applicable", "yolo"),
        "wrong_class": ("ng", "wrong_class", "not_applicable", "yolo"),
        "image_quality_issue": (
            "unjudgeable",
            "unjudgeable",
            "unjudgeable",
            "none",
        ),
        "uncertain": ("unjudgeable", "unjudgeable", "unjudgeable", "none"),
        "color_confirmed_ng": ("ng", "correct", "confirmed_ng", "color"),
        "color_false_reject": ("ok", "correct", "actually_ok", "color"),
        "position_false_reject": (
            "ok",
            "correct",
            "not_applicable",
            "position",
        ),
    }
    if label not in mappings:
        raise ValueError(f"Unsupported review action: {label}")
    product, detection, color, route = mappings[label]
    if semantics is not None and semantics.legacy_action_route:
        route = semantics.legacy_action_route
    if review_outcome == "skip":
        route = "none"
    default_training = label not in {
        "confirmed_ok",
        "image_quality_issue",
        "uncertain",
    }
    selected = default_training if training_selected is None else training_selected
    if review_outcome == "skip":
        selected = False
    return {
        "review_selected": "1",
        "review_outcome": review_outcome,
        "review_label": label,
        "failure_category": failure_category,
        "skip_reason": skip_reason,
        "product_verdict": product,
        "detection_verdict": detection,
        "color_verdict": color,
        "action_route": route,
        "training_selected": "1" if selected else "0",
    }


def explain_invalid_record(record: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable human-readable diagnostics without changing the record."""
    issues = validate_record_consistency(record)
    if not issues:
        return ("Record is consistent.",)
    return tuple(
        f"[{issue.code}] {issue.message} Suggested action: {issue.suggestion}"
        for issue in issues
    )


def blocking_violations(
    violations: tuple[WorkflowViolation, ...],
) -> tuple[WorkflowViolation, ...]:
    return tuple(violation for violation in violations if violation.blocking)


def ensure_record_consistent(record: Mapping[str, Any]) -> None:
    """Raise a domain error when a new record contains blocking contradictions."""
    violations = blocking_violations(validate_record_consistency(record))
    if violations:
        raise ReviewWorkflowValidationError(record, violations)


def ensure_transition_valid(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    """Raise a domain error when a new transition is not auditable."""
    violations = blocking_violations(validate_transition(before, after))
    if violations:
        raise ReviewWorkflowValidationError(after, violations)


def record_identity(record: Mapping[str, Any]) -> str:
    return str(
        record.get("sample_id")
        or record.get("inspection_id")
        or record.get("config_snapshot_path")
        or record.get("original_path")
        or "unknown"
    )


def _derive_product_verdict(
    record: Mapping[str, Any], label: str
) -> ProductVerdict:
    product = _text(record.get("product_verdict"))
    if product == "ok":
        return ProductVerdict.OK
    if product == "ng":
        return ProductVerdict.NG
    if product == "unjudgeable":
        return ProductVerdict.UNKNOWN
    return {
        "confirmed_ng": ProductVerdict.NG,
        "confirmed_ok": ProductVerdict.OK,
        "verified_empty": ProductVerdict.OK,
        "false_positive": ProductVerdict.OK,
        "false_negative": ProductVerdict.NG,
        "wrong_box": ProductVerdict.NG,
        "wrong_class": ProductVerdict.NG,
        "color_confirmed_ng": ProductVerdict.NG,
        "color_false_reject": ProductVerdict.OK,
        "position_false_reject": ProductVerdict.OK,
    }.get(label, ProductVerdict.UNKNOWN)


def _derive_ai_correctness(
    record: Mapping[str, Any], label: str
) -> AICorrectness:
    detection = _text(record.get("detection_verdict"))
    if detection == "correct":
        return AICorrectness.CORRECT
    if detection == "false_positive":
        return AICorrectness.FALSE_POSITIVE
    if detection in {"missed", "false_negative"}:
        return AICorrectness.FALSE_NEGATIVE
    if detection == "wrong_class":
        return AICorrectness.WRONG_CLASS
    if detection in {"unjudgeable", "wrong_box"}:
        return AICorrectness.UNKNOWN
    return {
        "confirmed_ng": AICorrectness.CORRECT,
        "confirmed_ok": AICorrectness.CORRECT,
        "verified_empty": AICorrectness.CORRECT,
        "false_positive": AICorrectness.FALSE_POSITIVE,
        "false_negative": AICorrectness.FALSE_NEGATIVE,
        "wrong_class": AICorrectness.WRONG_CLASS,
        "color_confirmed_ng": AICorrectness.CORRECT,
        "color_false_reject": AICorrectness.FALSE_POSITIVE,
        "position_false_reject": AICorrectness.CORRECT,
    }.get(label, AICorrectness.UNKNOWN)


def _derive_annotation_validity(
    record: Mapping[str, Any], label: str
) -> AnnotationValidity:
    annotation_status = _text(record.get("annotation_status"))
    if annotation_status in {"verified_annotation", "verified_empty"}:
        return AnnotationValidity.VERIFIED
    if label in {
        "confirmed_ng",
        "confirmed_ok",
        "verified_empty",
        "position_false_reject",
    }:
        return AnnotationValidity.VERIFIED
    if label in {"false_positive", "wrong_box"}:
        return AnnotationValidity.NEEDS_BOX_FIX
    if label == "false_negative":
        return AnnotationValidity.MISSING
    if label == "wrong_class":
        return AnnotationValidity.NEEDS_CLASS_FIX
    if label == "image_quality_issue":
        return AnnotationValidity.UNUSABLE
    if label in COLOR_LABELS:
        detection = _text(record.get("detection_verdict"))
        if detection == "wrong_box":
            return AnnotationValidity.NEEDS_BOX_FIX
        if detection == "wrong_class":
            return AnnotationValidity.NEEDS_CLASS_FIX
        return AnnotationValidity.VERIFIED
    return AnnotationValidity.UNKNOWN


def _derive_required_action(
    record: Mapping[str, Any],
    *,
    label: str,
    route: str,
    outcome: str,
    annotation: AnnotationValidity,
) -> RequiredAction:
    if outcome == "skip" or label in EXCLUDED_LABELS:
        return RequiredAction.EXCLUDE
    if _text(record.get("failure_category")) == "lighting_issue":
        return RequiredAction.EXCLUDE
    if annotation in {AnnotationValidity.NEEDS_BOX_FIX, AnnotationValidity.MISSING}:
        return RequiredAction.ANNOTATION
    if annotation == AnnotationValidity.NEEDS_CLASS_FIX:
        return RequiredAction.CLASS_FIX
    if label in COLOR_LABELS or route == "color":
        return RequiredAction.COLOR_CALIBRATION
    if label in POSITION_LABELS or route == "position":
        return RequiredAction.POSITION_CALIBRATION
    if label == "uncertain":
        return RequiredAction.MANUAL_REVIEW
    return RequiredAction.NONE


def _legacy_route(label: str, record: Mapping[str, Any]) -> str:
    if label in COLOR_LABELS:
        detection = _text(record.get("detection_verdict")) or "correct"
        return "both" if detection in {"wrong_box", "wrong_class"} else "color"
    if label in POSITION_LABELS:
        return "position"
    if label in DIRECT_TRAIN_LABELS | CORRECTION_LABELS:
        return "yolo"
    return "none"


def _validate_known_value(
    issues: list[WorkflowViolation],
    field: str,
    value: str,
    allowed: frozenset[str],
) -> None:
    if value and value not in allowed:
        issues.append(
            WorkflowViolation(
                code=f"invalid_{field}",
                message=f"Unsupported {field}: {value!r}",
                suggestion=f"Choose one of: {', '.join(sorted(allowed))}.",
            )
        )


def _has_false_positive_negative_conflict(label: str, detection: str) -> bool:
    return (label == "false_positive" and detection in {"missed", "false_negative"}) or (
        label == "false_negative" and detection == "false_positive"
    )


def _has_partial_review(record: Mapping[str, Any]) -> bool:
    return any(
        _text(record.get(field))
        for field in (
            "failure_category",
            "failure_source",
            "failure_note",
            "product_verdict",
            "detection_verdict",
            "color_verdict",
            "action_route",
        )
    )


def _job_status(record: Mapping[str, Any]) -> str:
    explicit = _text(record.get("job_status")) or _text(
        record.get("submission_status")
    )
    if explicit:
        return explicit
    if record.get("job_id") or record.get("status_path"):
        return _text(record.get("state"))
    return ""


def _is_submitted(record: Mapping[str, Any]) -> bool:
    status = _job_status(record)
    return bool(
        _bool(record.get("submitted"))
        or record.get("submission_id")
        or status
        in QUEUED_JOB_STATES
        | PROCESSING_JOB_STATES
        | COMPLETED_JOB_STATES
        | ERROR_JOB_STATES
    )


def _deduplicate_violations(
    issues: list[WorkflowViolation],
) -> tuple[WorkflowViolation, ...]:
    return tuple({issue.code: issue for issue in issues}.values())


def _bool(value: Any) -> bool:
    return _text(value) in {"1", "true", "yes", "on"}


def _text(value: Any) -> str:
    return str(value or "").strip().lower()
