"""Single source of truth for the left-to-right detection sequence.

The overlay panel and ``SequenceCheckStep`` must report the same ordering: an
operator reading ``LR:`` on a saved image has to see exactly the labels the
PASS/FAIL decision was made from.

Deriving the sequence in two places broke that. The panel rendered the color
checker's raw ``best_color`` while the decision used the fail-closed
``verified_class``, so a rejected board displayed ``LR: ... -> Red -> ...`` in
the very slot the log reported as ``missing=['Red']``. The evidence image
contradicted its own log. The color checker's opinion is not lost: it is still
reported per item as ``#5 Orange -> Red (mismatch)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "effective_class_name",
    "left_right_detections",
    "left_right_sequence",
    "sequence_mismatches",
]


def effective_class_name(detection: Mapping[str, Any]) -> str:
    """Return the label downstream checks judge ``detection`` by.

    ``ColorCheckStep`` writes ``verified_class`` only when the color result was
    accepted; a rejected match deliberately falls back to the detector class so
    an unverified color can never be promoted into a decision.
    """
    return str(detection.get("verified_class") or detection.get("class", "")).strip()


def _center_x(bbox: Any) -> float | None:
    """Return the horizontal box center, or ``None`` for an unusable bbox."""
    try:
        x1, _, x2, _ = bbox[:4]
        return (float(x1) + float(x2)) / 2.0
    except (TypeError, ValueError):
        return None


def left_right_detections(
    detections: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Return the detections that hold a sequence slot, ordered left to right.

    Detections without a usable bbox or class name are skipped rather than
    reordered: an unplaceable box has no position in the sequence.
    """
    ordered: list[tuple[float, int, Mapping[str, Any]]] = []
    for position, detection in enumerate(detections or []):
        if not isinstance(detection, Mapping):
            continue
        center = _center_x(detection.get("bbox"))
        if center is None:
            continue
        if not effective_class_name(detection):
            continue
        # The input position breaks ties so two boxes sharing a center keep a
        # stable order instead of depending on the sort's view of a Mapping.
        ordered.append((center, position, detection))
    ordered.sort(key=lambda entry: (entry[0], entry[1]))
    return [detection for _, _, detection in ordered]


def left_right_sequence(
    detections: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """Return effective class names ordered by horizontal box center."""
    return [
        effective_class_name(detection)
        for detection in left_right_detections(detections)
    ]


def sequence_mismatches(
    detections: Sequence[Mapping[str, Any]] | None,
    sequence_check: Mapping[str, Any] | None,
) -> list[tuple[int, Mapping[str, Any], str, str]]:
    """Return ``(slot, detection, observed, expected)`` for each slot out of order.

    ``SequenceCheckStep`` records what the order was and what it should have
    been, but not which box occupies a broken slot, so the slots are paired back
    to detections here by re-deriving the same ordering.

    Nothing is returned unless the recorded ``observed`` can be reproduced from
    ``detections`` exactly. A stale or unreproducible record would otherwise
    make the image blame a specific part for a defect belonging to another one,
    which is worse than the panel's text alone. A length mismatch is likewise
    unpairable: the missing or extra items are reported by their own checks.
    """
    if not isinstance(sequence_check, Mapping) or sequence_check.get("is_ok", True):
        return []
    expected = [str(item).strip() for item in (sequence_check.get("expected") or [])]
    observed = [str(item).strip() for item in (sequence_check.get("observed") or [])]
    if not expected or len(expected) != len(observed):
        return []

    ordered = left_right_detections(detections)
    direction = str(sequence_check.get("direction", "left_to_right")).lower()
    if direction in {"right_to_left", "rtl"}:
        ordered = list(reversed(ordered))
    if [effective_class_name(detection) for detection in ordered] != observed:
        return []

    return [
        (slot, detection, observed[slot], expected[slot])
        for slot, detection in enumerate(ordered)
        if observed[slot] != expected[slot]
    ]
