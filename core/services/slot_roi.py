from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.services.alignment import (
    ExpectedLayoutAlignment,
    base_class_name,
    build_aligned_expected_boxes,
)


@dataclass(frozen=True)
class SlotROI:
    """One aligned slot crop prepared for downstream fine inspection."""

    expected_key: str
    class_name: str
    bbox: tuple[int, int, int, int]
    image: np.ndarray


def extract_slot_rois(
    image: np.ndarray,
    expected_boxes: dict[str, dict[str, Any]] | None,
    alignment: ExpectedLayoutAlignment,
    *,
    keys: list[str] | None = None,
    margin: int = 0,
    min_size: int = 4,
) -> list[SlotROI]:
    """Crop aligned slot ROIs from an image.

    This prepares fixed, geometry-driven ROIs for anomaly/OCR/inspection steps
    without depending on YOLO detection boxes.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    aligned_boxes = build_aligned_expected_boxes(expected_boxes, alignment)
    selected_keys = keys if keys is not None else list(aligned_boxes.keys())
    height, width = image.shape[:2]
    rois: list[SlotROI] = []

    for expected_key in selected_keys:
        box = aligned_boxes.get(expected_key)
        if not isinstance(box, dict):
            continue
        try:
            x1 = max(0, int(round(float(box["x1"]))) - margin)
            y1 = max(0, int(round(float(box["y1"]))) - margin)
            x2 = min(width, int(round(float(box["x2"]))) + margin)
            y2 = min(height, int(round(float(box["y2"]))) + margin)
        except (KeyError, TypeError, ValueError):
            continue
        if x2 - x1 < min_size or y2 - y1 < min_size:
            continue
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        rois.append(
            SlotROI(
                expected_key=expected_key,
                class_name=base_class_name(expected_key),
                bbox=(x1, y1, x2, y2),
                image=roi,
            )
        )
    return rois


def clamp_bbox(
    bbox: Any,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp an ``xyxy`` box into an image, or return None if nothing is left.

    Callers crop measurement ROIs with this, so a box that survives is
    guaranteed to yield a non-empty array. Returning None rather than an empty
    slice is deliberate: an empty ROI reaches OpenCV as an assertion failure,
    which surfaces as a whole-frame ERROR (or, in the async pipeline, a line
    stop) for what is really a single unusable detection.
    """
    if not isinstance(bbox, (list, tuple, np.ndarray)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in tuple(bbox)[:4])
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def extract_bbox_roi(
    image: np.ndarray | None,
    bbox: Any,
    *,
    min_size: int = 1,
) -> np.ndarray | None:
    """Crop ``bbox`` out of ``image``, or return None when it is unusable.

    ``min_size`` rejects slivers that carry too few pixels for the statistic
    the caller intends to compute.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    height, width = image.shape[:2]
    clamped = clamp_bbox(bbox, width=width, height=height)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    if x2 - x1 < min_size or y2 - y1 < min_size:
        return None
    roi = image[y1:y2, x1:x2]
    return roi if roi.size else None
