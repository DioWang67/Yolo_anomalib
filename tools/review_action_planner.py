"""Pure Phase 2B action planning on top of the Phase 1B review domain."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from tools.review_classification import (
    COLOR_ISSUE,
    LIGHTING_ISSUE,
    MISCLASSIFICATION,
    MISSED_DETECTION,
    NEW_DEFECT_TYPE,
    THRESHOLD_NOT_MET,
    WRONG_BOX,
    WRONG_CLASS,
    ReviewFailureClassification,
    normalize_source_key,
)
from tools.review_routing import (
    has_color_failure,
    has_non_color_failure,
    has_position_only_failure,
)
from tools.review_workflow import (
    ReviewSemantics,
    derive_review_semantics,
    map_review_action_to_legacy_fields,
)

DETECTION_PRESENT = "present"
DETECTION_ABSENT = "absent"
DETECTION_UNKNOWN = "unknown"

SKIP_UNJUDGEABLE = "unjudgeable"
SKIP_IMAGE_QUALITY = "image_quality"
SKIP_EQUIPMENT_LIGHTING = "equipment_lighting"
SKIP_CONFIRMED_FAILURE = "confirmed_failure"
SKIP_OTHER = "other"
VALID_SKIP_UI_REASONS = frozenset(
    {
        SKIP_UNJUDGEABLE,
        SKIP_IMAGE_QUALITY,
        SKIP_EQUIPMENT_LIGHTING,
        SKIP_CONFIRMED_FAILURE,
        SKIP_OTHER,
    }
)


@dataclass(frozen=True)
class ReviewActionPlan:
    """Immutable proposed update and its Phase 1B-derived meaning."""

    outcome: str
    review_label: str
    updates: Mapping[str, str]
    semantics: ReviewSemantics

    def proposed_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        return {**record, **self.updates}


def plan_pass(record: Mapping[str, Any]) -> ReviewActionPlan:
    """Plan an operator PASS without exposing an action route to the UI."""
    if has_color_failure(record):
        label = "color_false_reject"
    elif has_position_only_failure(record):
        label = "position_false_reject"
    elif str(record.get("status") or "").strip().upper() == "PASS":
        label = "confirmed_ok"
    else:
        label = "false_positive"
    plan = _build_plan(record, outcome="pass", label=label)
    if label == "color_false_reject" and has_non_color_failure(record):
        updates = {**plan.updates, "product_verdict": "ng"}
        return _finalize_plan(record, "pass", label, updates)
    return plan


def plan_fail(
    record: Mapping[str, Any],
    *,
    category: str,
    note: str = "",
) -> ReviewActionPlan:
    """Plan one FAIL category using the existing deterministic legacy mapping."""
    source = normalize_source_key(record.get("detector")) or "yolo"
    classification = ReviewFailureClassification(
        category=category,
        source=source,
        note=note,
    )
    label = _failure_label(record, category)
    updates = dict(
        map_review_action_to_legacy_fields(
            label,
            review_outcome="fail",
            failure_category=category,
            training_selected=category != LIGHTING_ISSUE,
        )
    )
    updates.update(classification.to_columns())
    if label == "color_false_reject" and has_non_color_failure(record):
        updates["product_verdict"] = "ng"
    updates["review_note"] = ""
    return _finalize_plan(record, "fail", label, updates)


def plan_skip(
    record: Mapping[str, Any],
    *,
    ui_reason: str,
    note: str = "",
) -> ReviewActionPlan:
    """Map five operator-facing skip reasons onto the unchanged legacy enum."""
    if ui_reason not in VALID_SKIP_UI_REASONS:
        raise ValueError(f"Unsupported skip reason: {ui_reason}")
    normalized_note = note.strip()
    if ui_reason == SKIP_OTHER and not normalized_note:
        raise ValueError("A custom skip note is required for Other")
    legacy_reason = (
        "confirmed_failure"
        if ui_reason == SKIP_CONFIRMED_FAILURE
        else "image_quality_issue"
    )
    label = (
        "confirmed_ng"
        if ui_reason == SKIP_CONFIRMED_FAILURE
        else "image_quality_issue"
    )
    updates = dict(
        map_review_action_to_legacy_fields(
            label,
            review_outcome="skip",
            skip_reason=legacy_reason,
            training_selected=False,
        )
    )
    updates.update(
        {
            "failure_category": "",
            "failure_source": "",
            "failure_note": "",
            "review_note": _encode_skip_note(ui_reason, normalized_note),
        }
    )
    return _finalize_plan(record, "skip", label, updates)


def skip_ui_reason_from_record(record: Mapping[str, Any]) -> str:
    """Restore the richer UI choice from legacy reason plus existing note."""
    review_note = str(record.get("review_note") or "")
    if review_note.startswith("[skip:") and "]" in review_note:
        reason = review_note[6 : review_note.index("]")]
        if reason in VALID_SKIP_UI_REASONS:
            return reason
    if str(record.get("skip_reason") or "") == "confirmed_failure":
        return SKIP_CONFIRMED_FAILURE
    return SKIP_IMAGE_QUALITY


def skip_custom_note_from_record(record: Mapping[str, Any]) -> str:
    review_note = str(record.get("review_note") or "")
    if review_note.startswith("[skip:") and "]" in review_note:
        return review_note[review_note.index("]") + 1 :].strip()
    return ""


def detection_state(record: Mapping[str, Any]) -> str:
    """Return present, absent, or unknown from persisted structured evidence."""
    try:
        detections = json.loads(str(record.get("detections_json") or "[]"))
    except (TypeError, json.JSONDecodeError):
        detections = []
    if isinstance(detections, list) and any(
        isinstance(detection, dict)
        and isinstance(detection.get("bbox"), list)
        and len(detection["bbox"]) == 4
        for detection in detections
    ):
        return DETECTION_PRESENT

    source = str(record.get("detection_evidence_source") or "").strip().lower()
    try:
        count = int(str(record.get("detected_box_count") or "").strip())
    except ValueError:
        count = None
    if count is not None and count > 0:
        return DETECTION_PRESENT
    if source == "snapshot" and count == 0:
        return DETECTION_ABSENT
    if source == "saved_crops":
        return DETECTION_PRESENT if count else DETECTION_UNKNOWN
    if source == "unknown":
        return DETECTION_UNKNOWN
    return DETECTION_ABSENT


def _build_plan(
    record: Mapping[str, Any],
    *,
    outcome: str,
    label: str,
) -> ReviewActionPlan:
    updates = dict(
        map_review_action_to_legacy_fields(label, review_outcome=outcome)
    )
    updates.update(
        {
            "failure_category": "",
            "failure_source": "",
            "failure_note": "",
            "review_note": "",
        }
    )
    return _finalize_plan(record, outcome, label, updates)


def _finalize_plan(
    record: Mapping[str, Any],
    outcome: str,
    label: str,
    updates: dict[str, str],
) -> ReviewActionPlan:
    proposed = {**record, **updates}
    return ReviewActionPlan(
        outcome=outcome,
        review_label=label,
        updates=MappingProxyType(dict(updates)),
        semantics=derive_review_semantics(proposed),
    )


def _failure_label(record: Mapping[str, Any], category: str) -> str:
    if category == MISCLASSIFICATION:
        return "false_positive"
    if category in {MISSED_DETECTION, NEW_DEFECT_TYPE}:
        return "false_negative"
    if category == WRONG_BOX:
        return "wrong_box"
    if category == WRONG_CLASS:
        return "wrong_class"
    if category in {COLOR_ISSUE, THRESHOLD_NOT_MET} and has_color_failure(record):
        return "color_false_reject"
    if category == LIGHTING_ISSUE:
        return "image_quality_issue"
    if detection_state(record) == DETECTION_ABSENT:
        return "false_negative"
    return "false_positive"


def _encode_skip_note(ui_reason: str, note: str) -> str:
    suffix = f" {note}" if note else ""
    return f"[skip:{ui_reason}]{suffix}"
