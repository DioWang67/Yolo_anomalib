from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExpectedLayoutAlignment:
    """Shared expected-layout transform derived from detected parts.

    Current scope is intentionally conservative:
    - translation only
    - source evidence comes from validated detections

    This is the first alignment layer for downstream rule/ROI consumers.
    It is not a full board homography solution.
    """

    dx: float = 0.0
    dy: float = 0.0
    source_count: int = 0

    def shift_box(self, box: dict[str, Any]) -> tuple[int, int, int, int]:
        """Return an integer box translated by the estimated layout shift."""
        return (
            int(round(float(box["x1"]) + self.dx)),
            int(round(float(box["y1"]) + self.dy)),
            int(round(float(box["x2"]) + self.dx)),
            int(round(float(box["y2"]) + self.dy)),
        )

    def shift_center(self, cx: float, cy: float) -> tuple[float, float]:
        """Return a center point translated by the estimated layout shift."""
        return float(cx) + self.dx, float(cy) + self.dy

    def to_dict(self) -> dict[str, Any]:
        """Serialize for result metadata."""
        return {"dx": self.dx, "dy": self.dy, "source_count": self.source_count}

    @classmethod
    def from_detections(
        cls, detections: list[dict[str, Any]] | None
    ) -> ExpectedLayoutAlignment:
        """Estimate translation from validated detections using a robust median."""
        shifts_x: list[float] = []
        shifts_y: list[float] = []

        for det in detections or []:
            dx, dy = _extract_detection_shift(det)
            if dx is None or dy is None:
                continue
            shifts_x.append(dx)
            shifts_y.append(dy)

        if not shifts_x or not shifts_y:
            return cls()
        return cls(
            dx=float(np.median(shifts_x)),
            dy=float(np.median(shifts_y)),
            source_count=min(len(shifts_x), len(shifts_y)),
        )


def extract_layout_alignment(
    detections: list[dict[str, Any]] | None,
) -> ExpectedLayoutAlignment:
    """Convenience wrapper for callers that prefer a function API."""
    return ExpectedLayoutAlignment.from_detections(detections)


def build_aligned_expected_boxes(
    expected_boxes: dict[str, dict[str, Any]] | None,
    alignment: ExpectedLayoutAlignment,
) -> dict[str, dict[str, float]]:
    """Return expected boxes translated into the current aligned layout."""
    aligned: dict[str, dict[str, float]] = {}
    for key, box in (expected_boxes or {}).items():
        if not isinstance(box, dict):
            continue
        try:
            shifted = alignment.shift_box(box)
            aligned[key] = {
                "x1": float(shifted[0]),
                "y1": float(shifted[1]),
                "x2": float(shifted[2]),
                "y2": float(shifted[3]),
            }
            if "cx" in box and "cy" in box:
                cx, cy = alignment.shift_center(float(box["cx"]), float(box["cy"]))
                aligned[key]["cx"] = float(cx)
                aligned[key]["cy"] = float(cy)
        except (KeyError, TypeError, ValueError):
            continue
    return aligned


def resolve_missing_expected_keys(
    missing_items: list[str],
    expected_boxes: dict[str, dict[str, Any]],
    used_expected_keys: set[str],
) -> list[str]:
    """Match missing class names to concrete expected keys.

    Supports multi-instance keys like ``LED#1`` by matching on the base class
    and consuming keys in deterministic order.
    """
    matched_keys: list[str] = []
    consumed = set(used_expected_keys)
    expected_keys = list(expected_boxes.keys())
    for missing_name in missing_items:
        base_name = str(missing_name or "").strip()
        if not base_name:
            continue
        for key in expected_keys:
            if key in consumed:
                continue
            if base_class_name(key) != base_name:
                continue
            matched_keys.append(key)
            consumed.add(key)
            break
    return matched_keys


def base_class_name(key: str) -> str:
    """Strip a trailing ``#N`` instance index from a class key."""
    idx = key.rfind("#")
    if idx > 0 and key[idx + 1 :].isdigit():
        return key[:idx]
    return key


def _extract_detection_shift(
    detection: dict[str, Any],
) -> tuple[float | None, float | None]:
    layout_alignment = detection.get("position_layout_alignment")
    if isinstance(layout_alignment, dict):
        try:
            source_count = int(layout_alignment.get("source_count", 0))
            if source_count > 0:
                return float(layout_alignment["dx"]), float(layout_alignment["dy"])
        except (KeyError, TypeError, ValueError):
            pass

    offset = detection.get("position_offset")
    if isinstance(offset, dict):
        try:
            return float(offset["dx"]), float(offset["dy"])
        except (KeyError, TypeError, ValueError):
            pass

    expected_center = detection.get("position_expected_center")
    bbox = detection.get("bbox")
    if isinstance(expected_center, dict) and bbox and len(bbox) >= 4:
        try:
            exp_x = float(expected_center["cx"])
            exp_y = float(expected_center["cy"])
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            return ((x1 + x2) / 2.0) - exp_x, ((y1 + y2) / 2.0) - exp_y
        except (KeyError, TypeError, ValueError):
            return None, None
    return None, None
