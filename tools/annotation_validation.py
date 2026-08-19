"""Pure YOLO annotation validation and revision-diff rules for Phase 3C2."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum


class AnnotationErrorCode(str, Enum):
    ANNOTATION_PACKAGE_STALE = "ANNOTATION_PACKAGE_STALE"
    IMAGE_STALE = "IMAGE_STALE"
    PARENT_LABEL_STALE = "PARENT_LABEL_STALE"
    LABEL_MISSING = "LABEL_MISSING"
    LABEL_PARSE_ERROR = "LABEL_PARSE_ERROR"
    CLASS_ID_INVALID = "CLASS_ID_INVALID"
    BBOX_INVALID = "BBOX_INVALID"
    CLASS_FIX_GEOMETRY_CHANGED = "CLASS_FIX_GEOMETRY_CHANGED"
    EMPTY_LABEL_NOT_ALLOWED = "EMPTY_LABEL_NOT_ALLOWED"
    NO_EFFECTIVE_CHANGE = "NO_EFFECTIVE_CHANGE"
    WORKING_PATH_ESCAPE = "WORKING_PATH_ESCAPE"
    REVISION_COMMIT_FAILED = "REVISION_COMMIT_FAILED"
    REVISION_VERIFY_FAILED = "REVISION_VERIFY_FAILED"
    CANONICAL_CONFLICT = "CANONICAL_CONFLICT"
    REVISION_REASON_REQUIRED = "REVISION_REASON_REQUIRED"
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    TOOL_LAUNCH_FAILED = "TOOL_LAUNCH_FAILED"


class AnnotationOperation(str, Enum):
    CREATE_ANNOTATION = "CREATE_ANNOTATION"
    FIX_BOUNDING_BOX = "FIX_BOUNDING_BOX"
    FIX_CLASS_ONLY = "FIX_CLASS_ONLY"
    REVIEW_EMPTY_LABEL = "REVIEW_EMPTY_LABEL"


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    @property
    def geometry(self) -> tuple[float, float, float, float]:
        return (self.center_x, self.center_y, self.width, self.height)


@dataclass(frozen=True)
class AnnotationDiff:
    bbox_count_before: int
    bbox_count_after: int
    class_changes: int
    bbox_changes: int
    added_boxes: int
    removed_boxes: int


@dataclass(frozen=True)
class AnnotationValidationResult:
    valid: bool
    errors: tuple[AnnotationErrorCode, ...]
    messages: tuple[str, ...]
    warnings: tuple[str, ...]
    boxes: tuple[YoloBox, ...]
    label_sha256: str
    actual_operation: AnnotationOperation
    diff: AnnotationDiff


def normalized_label_text(value: str) -> str:
    return "\n".join(
        line.strip() for line in value.replace("\r\n", "\n").splitlines() if line.strip()
    )


def normalized_label_sha256(value: str) -> str:
    return hashlib.sha256(normalized_label_text(value).encode("utf-8")).hexdigest()


def parse_yolo_label(
    label_text: str,
    *,
    class_count: int,
) -> tuple[tuple[YoloBox, ...], tuple[AnnotationErrorCode, ...], tuple[str, ...]]:
    boxes: list[YoloBox] = []
    errors: list[AnnotationErrorCode] = []
    messages: list[str] = []
    for line_number, raw in enumerate(label_text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(AnnotationErrorCode.LABEL_PARSE_ERROR)
            messages.append(f"line {line_number}: expected five YOLO values")
            continue
        try:
            class_id = int(parts[0])
            center_x, center_y, width, height = (float(value) for value in parts[1:])
        except ValueError:
            errors.append(AnnotationErrorCode.LABEL_PARSE_ERROR)
            messages.append(f"line {line_number}: invalid numeric value")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(AnnotationErrorCode.CLASS_ID_INVALID)
            messages.append(f"line {line_number}: class ID {class_id} is out of range")
        values = (center_x, center_y, width, height)
        if (
            not all(math.isfinite(value) for value in values)
            or not 0 <= center_x <= 1
            or not 0 <= center_y <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
            or center_x - width / 2 < 0
            or center_x + width / 2 > 1
            or center_y - height / 2 < 0
            or center_y + height / 2 > 1
        ):
            errors.append(AnnotationErrorCode.BBOX_INVALID)
            messages.append(f"line {line_number}: bbox is not finite and normalized")
        boxes.append(YoloBox(class_id, center_x, center_y, width, height))
    return tuple(boxes), _unique(errors), tuple(messages)


def validate_annotation_revision(
    *,
    parent_label_text: str,
    working_label_text: str,
    class_count: int,
    requested_operation: AnnotationOperation,
    allow_empty: bool,
    revision_reason: str,
    escalation_reason: str = "",
) -> AnnotationValidationResult:
    parent_boxes, parent_errors, parent_messages = parse_yolo_label(
        parent_label_text, class_count=class_count
    )
    boxes, errors, messages = parse_yolo_label(
        working_label_text, class_count=class_count
    )
    combined_errors = list(parent_errors) + list(errors)
    combined_messages = list(parent_messages) + list(messages)
    if not revision_reason.strip():
        combined_errors.append(AnnotationErrorCode.REVISION_REASON_REQUIRED)
        combined_messages.append("A revision reason is required.")
    if not boxes and not allow_empty:
        combined_errors.append(AnnotationErrorCode.EMPTY_LABEL_NOT_ALLOWED)
        combined_messages.append("An empty label is not allowed for this reviewed case.")
    diff = _diff(parent_boxes, boxes)
    changed = normalized_label_sha256(parent_label_text) != normalized_label_sha256(
        working_label_text
    )
    if not changed:
        combined_errors.append(AnnotationErrorCode.NO_EFFECTIVE_CHANGE)
        combined_messages.append("The working label has no effective normalized change.")
    actual_operation = requested_operation
    if requested_operation == AnnotationOperation.FIX_CLASS_ONLY and diff.bbox_changes:
        if not escalation_reason.strip():
            combined_errors.append(AnnotationErrorCode.CLASS_FIX_GEOMETRY_CHANGED)
            combined_messages.append(
                "A class-only correction changed bbox geometry without an escalation reason."
            )
        else:
            actual_operation = AnnotationOperation.FIX_BOUNDING_BOX
    warnings = _duplicate_warnings(boxes)
    unique_errors = _unique(combined_errors)
    return AnnotationValidationResult(
        valid=not unique_errors,
        errors=unique_errors,
        messages=tuple(combined_messages),
        warnings=warnings,
        boxes=boxes,
        label_sha256=normalized_label_sha256(working_label_text),
        actual_operation=actual_operation,
        diff=diff,
    )


def _diff(before: tuple[YoloBox, ...], after: tuple[YoloBox, ...]) -> AnnotationDiff:
    paired = min(len(before), len(after))
    class_changes = sum(before[index].class_id != after[index].class_id for index in range(paired))
    bbox_changes = sum(before[index].geometry != after[index].geometry for index in range(paired))
    return AnnotationDiff(
        bbox_count_before=len(before),
        bbox_count_after=len(after),
        class_changes=class_changes,
        bbox_changes=bbox_changes + abs(len(before) - len(after)),
        added_boxes=max(0, len(after) - len(before)),
        removed_boxes=max(0, len(before) - len(after)),
    )


def _duplicate_warnings(boxes: tuple[YoloBox, ...]) -> tuple[str, ...]:
    seen: set[tuple[int, float, float, float, float]] = set()
    warnings: list[str] = []
    for index, box in enumerate(boxes, start=1):
        signature = (box.class_id, *box.geometry)
        if signature in seen:
            warnings.append(f"duplicate_bbox: exact duplicate at item {index}")
        seen.add(signature)
    return tuple(warnings)


def _unique(values):
    return tuple(dict.fromkeys(values))
