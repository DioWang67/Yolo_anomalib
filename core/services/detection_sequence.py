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

__all__ = ["effective_class_name", "left_right_sequence"]


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


def left_right_sequence(
    detections: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """Return effective class names ordered by horizontal box center.

    Detections without a usable bbox or class name are skipped rather than
    reordered: an unplaceable box has no position in the sequence.
    """
    ordered: list[tuple[float, str]] = []
    for detection in detections or []:
        if not isinstance(detection, Mapping):
            continue
        center = _center_x(detection.get("bbox"))
        if center is None:
            continue
        name = effective_class_name(detection)
        if not name:
            continue
        ordered.append((center, name))
    ordered.sort(key=lambda entry: entry[0])
    return [name for _, name in ordered]
