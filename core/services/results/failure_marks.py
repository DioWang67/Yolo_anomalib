"""Operator-facing emphasis for the targets that failed inspection.

The annotated result image carries every detection, so a failure drawn in the
same weight as a pass is easy to miss on a dense board: until this module, an
NG target differed from an OK one only by hue on a 2px outline, which is a weak
cue over an orange PCB and useless to an operator glancing at the station
monitor.

Failing targets are therefore collected first (:func:`collect_detection_marks`,
:func:`missing_marks`) and then drawn as one separate top layer
(:func:`draw_failure_marks`): a translucent wash, a heavy three-tone frame with
targeting corners, and a reason label. Collection is pure so the "what failed"
rules are testable without a frame, and drawing last keeps the emphasis from
being overdrawn by the boxes of neighbouring passing detections.

All rendered text is ASCII: these labels go through ``cv2.putText`` with
``FONT_HERSHEY_SIMPLEX``, which draws non-Latin glyphs as ``?``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.services.results.customer_message import (
    COLOR_FAILURE_LOW_CONFIDENCE,
    COLOR_FAILURE_MISMATCH,
    classify_color_check_failure,
)
from core.services.results.position_summary import (
    POSITION_OK_STATES,
    POSITION_SKIP_STATES,
)
from core.visualization import contrast_text_color

Color = tuple[int, int, int]
Bbox = tuple[int, int, int, int]

# Reason codes are shared with the NG evidence crop filenames
# (``core.services.results.crops``) so an operator can match a highlighted box
# on the annotated image to the crop saved for the same defect.
REASON_MISSING = "MISSING"
REASON_WRONG_COMPONENT = "WRONG_COMPONENT"
REASON_SEQUENCE = "SEQUENCE"
REASON_COLOR = "COLOR"
REASON_POSITION = "POSITION_SHIFT"
REASON_EXTRA = "EXTRA"
REASON_INVALID = "INVALID"

# Reasons whose detail already reads as the whole story on the box: an
# ``exp=Orange`` caption says "this slot should hold Orange" without the code in
# front of it, which only pushes the part the operator acts on further right.
# The code is still named in the panel summary and in the NG crop filename, and
# it comes back on the box as soon as a target fails more than one check, so a
# detail is never printed under the wrong heading.
SELF_EVIDENT_REASONS = frozenset({REASON_SEQUENCE})

FAIL_COLOR: Color = (0, 0, 255)
EXTRA_COLOR: Color = (0, 140, 255)
INVALID_COLOR: Color = (255, 0, 255)
# Keeps the per-reason hues the detection boxes already use
# (``annotations._position_color``), so this layer changes the weight of a
# failure cue without changing its established color language.
REASON_COLORS: dict[str, Color] = {
    REASON_EXTRA: EXTRA_COLOR,
    REASON_INVALID: INVALID_COLOR,
}

WASH_ALPHA = 0.22
BORDER_THICKNESS = 3
DARK_RING_COLOR: Color = (0, 0, 0)
LIGHT_RING_COLOR: Color = (255, 255, 255)
# OpenCV grows a thickness-3 outline two pixels outside the box, so the rings
# start at 3 to sit beside the color band rather than under it.
DARK_RING_OFFSET = 3
LIGHT_RING_OFFSET = 4
BRACKET_OFFSET = 7
BRACKET_THICKNESS = 3
BRACKET_BACKING_THICKNESS = 5
BRACKET_MIN_ARM = 8
BRACKET_MAX_ARM = 28
# Clears the light ring, and leaves the pointer tail room to read as a shape.
LABEL_GAP = 10
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1
LABEL_PADDING_X = 5
LABEL_PADDING_Y = 4
SUMMARY_TOKEN_LIMIT = 4
POINTER_HALF_BASE = 5


@dataclass(frozen=True)
class FailureMark:
    """One failing target to emphasize, independent of how it is drawn.

    Attributes:
        bbox: Target box in frame pixels.
        reason: Primary reason code, the one the color and label lead with.
        detail: Short ASCII detail (e.g. ``Black->Orange``), may be empty.
        index: Source detection index, or ``None`` for a target that was never
            detected (a missing item has an expected box but no detection).
        extra_reasons: Further reasons for the same target, in priority order.
    """

    bbox: Bbox
    reason: str
    detail: str = ""
    index: int | None = None
    extra_reasons: tuple[str, ...] = ()

    @property
    def color(self) -> Color:
        return REASON_COLORS.get(self.reason, FAIL_COLOR)

    @property
    def label(self) -> str:
        """On-image caption, e.g. ``#5 NG COLOR+POSITION_SHIFT Black->Orange``.

        The reason code is dropped for a lone self-evident reason, leaving
        ``#1 NG exp=Orange``. It is kept whenever a target failed more than one
        check, so a detail never appears under a heading it does not belong to.
        """
        codes = (self.reason, *self.extra_reasons)
        head = f"#{self.index} " if self.index is not None else ""
        parts = [f"{head}NG"]
        if len(codes) > 1 or self.reason not in SELF_EVIDENT_REASONS:
            parts.append("+".join(codes))
        if self.detail:
            parts.append(self.detail)
        return " ".join(parts)

    @property
    def token(self) -> str:
        """Compact form for the info panel summary line."""
        head = f"#{self.index} " if self.index is not None else ""
        return f"{head}{self.reason}"


def collect_detection_marks(
    detections: Sequence[dict[str, Any]] | None,
    color_items_by_index: Mapping[int, dict[str, Any]] | None = None,
    sequence_expectations: Mapping[int, str] | None = None,
) -> list[FailureMark]:
    """Return one mark per failing detection, richest reason first.

    A single target can fail several checks at once (a swapped wire sits in the
    wrong sequence slot, reads as the wrong color, and may be off position too).
    Emitting one mark per target rather than one per check keeps the image from
    stacking overlapping highlights on the same box; the secondary reasons
    survive in :attr:`FailureMark.extra_reasons` and are shown in the label.

    Args:
        sequence_expectations: Source index to the class the sequence slot that
            detection occupies should have held, from
            :func:`core.services.detection_sequence.sequence_mismatches`.
    """
    marks: list[FailureMark] = []
    items = color_items_by_index or {}
    expectations = sequence_expectations or {}
    for position, detection in enumerate(detections or []):
        if not isinstance(detection, dict):
            continue
        bbox = coerce_bbox(detection.get("bbox"))
        if bbox is None:
            continue
        index = _source_index(detection, position)
        reasons = _detection_reasons(
            detection, items.get(index), expectations.get(index)
        )
        if not reasons:
            continue
        primary_reason, primary_detail = reasons[0]
        marks.append(
            FailureMark(
                bbox=bbox,
                reason=primary_reason,
                detail=primary_detail,
                index=index,
                extra_reasons=tuple(reason for reason, _ in reasons[1:]),
            )
        )
    return marks


def missing_marks(entries: Sequence[tuple[Bbox, str]]) -> list[FailureMark]:
    """Return marks for expected targets that were never detected."""
    marks: list[FailureMark] = []
    for bbox, name in entries:
        coerced = coerce_bbox(bbox)
        if coerced is None:
            continue
        marks.append(
            FailureMark(bbox=coerced, reason=REASON_MISSING, detail=ascii_text(name))
        )
    return marks


def summarize_failure_marks(
    marks: Sequence[FailureMark],
    limit: int = SUMMARY_TOKEN_LIMIT,
) -> list[tuple[str, Color]]:
    """Return info-panel lines naming what failed, for the text-only readers."""
    if not marks:
        return []
    tokens = [mark.token for mark in marks[:limit]]
    hidden = len(marks) - len(tokens)
    if hidden > 0:
        tokens.append(f"+{hidden} more")
    return [(f"NG x{len(marks)}: {', '.join(tokens)}", FAIL_COLOR)]


def draw_failure_marks(
    frame: np.ndarray,
    marks: Sequence[FailureMark],
) -> None:
    """Draw every failure mark as one top layer.

    Drawn in three passes so a mark never partially covers its neighbour: all
    washes, then all frames, then all labels.
    """
    if frame is None or frame.size == 0 or not marks:
        return
    for mark in marks:
        _wash_region(frame, mark.bbox, mark.color)
    for mark in marks:
        _draw_emphasis_frame(frame, mark.bbox, mark.color)
    placed: list[Bbox] = []
    for mark in marks:
        _draw_reason_label(frame, mark, placed)


def coerce_bbox(value: Any) -> Bbox | None:
    """Return an ordered integer bbox, or ``None`` when it is unusable."""
    if value is None:
        return None
    try:
        if len(value) < 4:
            return None
        x1, y1, x2, y2 = (int(float(item)) for item in tuple(value)[:4])
    except (TypeError, ValueError):
        return None
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if x1 == x2 or y1 == y2:
        return None
    return (x1, y1, x2, y2)


def ascii_text(value: Any) -> str:
    """Return text ``cv2.putText`` can actually render.

    Non-Latin class names would otherwise be drawn as a row of ``?`` with no
    hint of what was lost, so they are replaced by a single marker.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    return "".join(char if 32 <= ord(char) < 127 else "?" for char in text)


def _detection_reasons(
    detection: dict[str, Any],
    color_item: dict[str, Any] | None,
    sequence_expectation: str | None = None,
) -> list[tuple[str, str]]:
    """Return ``(reason, detail)`` pairs for one detection, most specific first.

    Order decides which reason the caption leads with. A wrong or out-of-order
    part names what the slot should have held, which is what the operator acts
    on; the color and position mechanics behind it follow.
    """
    reasons: list[tuple[str, str]] = []

    expected_class = detection.get("slot_mismatch_expected_class")
    if expected_class:
        reasons.append((REASON_WRONG_COMPONENT, f"exp={ascii_text(expected_class)}"))

    if sequence_expectation:
        reasons.append((REASON_SEQUENCE, f"exp={ascii_text(sequence_expectation)}"))

    if color_item is not None and not _is_ok(color_item):
        reasons.append((REASON_COLOR, _color_detail(color_item)))

    status = str(detection.get("position_status") or "").upper()
    if status not in POSITION_SKIP_STATES and status not in POSITION_OK_STATES:
        if status == "UNEXPECTED":
            reasons.append((REASON_EXTRA, ""))
        elif status in {"INVALID", "ERROR"}:
            reasons.append((REASON_INVALID, ""))
        else:
            reasons.append((REASON_POSITION, _position_detail(detection)))
    return reasons


def _is_ok(item: dict[str, Any]) -> bool:
    try:
        return bool(item.get("is_ok", True))
    except AttributeError:
        return True


def _color_detail(item: dict[str, Any]) -> str:
    description = classify_color_check_failure(item)
    class_name = ascii_text(description.class_name)
    if description.kind == COLOR_FAILURE_MISMATCH:
        return f"{class_name}->{ascii_text(description.predicted_color) or '?'}"
    if description.kind == COLOR_FAILURE_LOW_CONFIDENCE:
        return f"{class_name} low-conf" if class_name else "low-conf"
    return class_name


def _position_detail(detection: dict[str, Any]) -> str:
    error = detection.get("position_error")
    if isinstance(error, (int, float)):
        return f"d={float(error):.1f}"
    return ""


def _source_index(detection: dict[str, Any], fallback: int) -> int:
    try:
        return int(detection.get("source_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _wash_region(frame: np.ndarray, bbox: Bbox, color: Color) -> None:
    """Tint the failing area so the eye lands on it before reading any text.

    Only the box ROI is blended, not the whole frame, so cost scales with the
    defect count rather than with image size.
    """
    if frame.ndim != 3 or frame.shape[2] != len(color):
        return
    height, width = frame.shape[:2]
    x1 = max(0, min(width, bbox[0]))
    y1 = max(0, min(height, bbox[1]))
    x2 = max(0, min(width, bbox[2]))
    y2 = max(0, min(height, bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    wash = np.full(roi.shape, color, dtype=roi.dtype)
    cv2.addWeighted(roi, 1.0 - WASH_ALPHA, wash, WASH_ALPHA, 0.0, dst=roi)


def _draw_emphasis_frame(frame: np.ndarray, bbox: Bbox, color: Color) -> None:
    """Draw a light/dark/color sandwich that survives any background."""
    x1, y1, x2, y2 = bbox
    # OpenCV clips drawing to the image, so boxes at the frame edge need no
    # manual clamping here.
    cv2.rectangle(
        frame,
        (x1 - LIGHT_RING_OFFSET, y1 - LIGHT_RING_OFFSET),
        (x2 + LIGHT_RING_OFFSET, y2 + LIGHT_RING_OFFSET),
        LIGHT_RING_COLOR,
        1,
    )
    cv2.rectangle(
        frame,
        (x1 - DARK_RING_OFFSET, y1 - DARK_RING_OFFSET),
        (x2 + DARK_RING_OFFSET, y2 + DARK_RING_OFFSET),
        DARK_RING_COLOR,
        1,
    )
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BORDER_THICKNESS)
    _draw_corner_brackets(frame, bbox, color)


def _draw_corner_brackets(frame: np.ndarray, bbox: Bbox, color: Color) -> None:
    """Draw outward targeting corners, which stay legible on tiny boxes."""
    x1, y1, x2, y2 = bbox
    arm = max(
        BRACKET_MIN_ARM,
        min(BRACKET_MAX_ARM, int(0.3 * min(x2 - x1, y2 - y1))),
    )
    left = x1 - BRACKET_OFFSET
    top = y1 - BRACKET_OFFSET
    right = x2 + BRACKET_OFFSET
    bottom = y2 + BRACKET_OFFSET
    corners = (
        ((left, top), (arm, arm)),
        ((right, top), (-arm, arm)),
        ((left, bottom), (arm, -arm)),
        ((right, bottom), (-arm, -arm)),
    )
    for (corner_x, corner_y), (dx, dy) in corners:
        segments = (
            ((corner_x, corner_y), (corner_x + dx, corner_y)),
            ((corner_x, corner_y), (corner_x, corner_y + dy)),
        )
        for start, end in segments:
            cv2.line(frame, start, end, DARK_RING_COLOR, BRACKET_BACKING_THICKNESS)
            cv2.line(frame, start, end, color, BRACKET_THICKNESS)


def _draw_reason_label(
    frame: np.ndarray,
    mark: FailureMark,
    placed: list[Bbox],
) -> None:
    """Draw the reason caption near the box without covering another caption.

    The caption prefers the slot the plain ``#index`` tag would have used, so a
    dense layout does not gain a second row of text per defect. Captions are
    wider than the boxes they name, though, so neighbouring defects collide;
    each one therefore takes the first free slot above, below, or inside its own
    box. ``placed`` accumulates the rectangles already taken.
    """
    height, width = frame.shape[:2]
    text = _fit_text(mark.label, width - 2 * LABEL_PADDING_X - 4)
    if not text:
        return
    text_size, baseline = cv2.getTextSize(
        text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS
    )
    tag_width = text_size[0] + LABEL_PADDING_X * 2
    tag_height = text_size[1] + baseline + LABEL_PADDING_Y * 2

    x1, y1, x2, y2 = mark.bbox
    # Centered on the target rather than aligned to its left edge: a caption
    # wider than its box then still reads as belonging to what sits under its
    # middle, which is where the eye assigns it.
    left = max(0, min(width - tag_width - 1, (x1 + x2 - tag_width) // 2))
    right = min(width - 1, left + tag_width)
    tops = (
        y1 - LABEL_GAP - tag_height,  # above, the preferred slot
        y2 + LABEL_GAP,  # below
        y1 + LABEL_GAP,  # inside the box, for a crowded row
    )
    top = _first_free_top(tops, tag_height, height, left, right, placed)
    bottom = top + tag_height
    placed.append((left, top, right, bottom))

    _draw_label_pointer(frame, mark.bbox, (left, top, right, bottom), mark.color)
    cv2.rectangle(frame, (left, top), (right, bottom), mark.color, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), LIGHT_RING_COLOR, 1)
    cv2.putText(
        frame,
        text,
        (left + LABEL_PADDING_X, bottom - LABEL_PADDING_Y - baseline),
        LABEL_FONT,
        LABEL_FONT_SCALE,
        contrast_text_color(mark.color),
        LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_label_pointer(
    frame: np.ndarray,
    bbox: Bbox,
    label_rect: Bbox,
    color: Color,
) -> None:
    """Point a caption at the box it names.

    A caption is several times wider than the small box it describes, so on a
    crowded row it reaches well over its neighbours and the eye assigns it to
    whatever sits under its middle. Centering the caption on its box fixes the
    common case; this tail settles the rest, including the captions pushed aside
    by a neighbour or clamped by the frame edge, which no longer sit over their
    own target at all.
    """
    x1, y1, x2, y2 = bbox
    left, top, right, bottom = label_rect
    box_center_x = (x1 + x2) // 2
    base_x = max(
        left + POINTER_HALF_BASE + 1,
        min(right - POINTER_HALF_BASE - 1, box_center_x),
    )
    if bottom <= y1:
        base_y, apex = bottom, (box_center_x, y1)
    elif top >= y2:
        base_y, apex = top, (box_center_x, y2)
    else:
        # The caption is inside its own box; there is nothing to disambiguate.
        return
    tail = np.array(
        [
            (base_x - POINTER_HALF_BASE, base_y),
            (base_x + POINTER_HALF_BASE, base_y),
            apex,
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, tail, color, cv2.LINE_AA)
    cv2.polylines(frame, [tail], True, DARK_RING_COLOR, 1, cv2.LINE_AA)


def _first_free_top(
    candidates: Sequence[int],
    tag_height: int,
    frame_height: int,
    left: int,
    right: int,
    placed: Sequence[Bbox],
) -> int:
    """Return the first candidate top that fits the frame and is unoccupied."""
    fallback: int | None = None
    for top in candidates:
        if top < 0 or top + tag_height > frame_height - 1:
            continue
        if fallback is None:
            fallback = top
        if not any(
            _overlaps((left, top, right, top + tag_height), taken) for taken in placed
        ):
            return top
    if fallback is not None:
        return fallback
    return max(0, min(frame_height - 1 - tag_height, candidates[0]))


def _overlaps(first: Bbox, second: Bbox) -> bool:
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def _fit_text(text: str, max_width: int) -> str:
    """Trim ``text`` until it fits ``max_width`` pixels, marking the cut."""
    if not text or max_width <= 0:
        return text
    if _text_width(text) <= max_width:
        return text
    trimmed = text
    while trimmed and _text_width(f"{trimmed}..") > max_width:
        trimmed = trimmed[:-1]
    return f"{trimmed}.." if trimmed else text[:1]


def _text_width(text: str) -> int:
    return int(
        cv2.getTextSize(text, LABEL_FONT, LABEL_FONT_SCALE, LABEL_THICKNESS)[0][0]
    )
