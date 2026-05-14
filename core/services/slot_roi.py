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
