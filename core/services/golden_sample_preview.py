"""Read visual evidence without changing the reference or the verdict."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.services.golden_sample import GoldenSampleError, align_positions


def _crop(position: dict | None):
    if not position or not position.get("crop_path"):
        return None
    try:
        return cv2.imdecode(np.frombuffer(Path(position["crop_path"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def build_preview(reference: dict, result: dict | None = None) -> list[dict]:
    """Pair each position's baseline crop with the frame the verdict was drawn from.

    Cells, worst frame and the aligned sampling window are read off the
    evaluated row rather than measured a second time, so the picture on screen
    is the one that was judged.
    """
    anchors = reference.get("positions", [])
    baseline_readings = reference.get("readings", [])
    if not anchors or not baseline_readings:
        return []
    baseline = align_positions(baseline_readings[0]["positions"], anchors)
    current = []
    if result and result.get("readings"):
        current = [align_positions(r["positions"], anchors) for r in result["readings"]]
    rows = result.get("rows", []) if result else []
    previews = []
    for index, anchor in enumerate(anchors):
        row = rows[index] if index < len(rows) else None
        selected = None
        heatmap = None
        frame_index = None
        if current and row is not None and row.get("cells") is not None:
            frame_index = int(row.get("worst_frame", 0))
            selected = current[frame_index][index]
            heatmap = [row["cells"][start : start + 4] for start in range(0, 16, 4)]
        previews.append(
            {
                "position": index + 1,
                "color": anchor["color"],
                "reference_image": _crop(baseline[index]),
                "reference_bbox": baseline[index]["bbox"],
                "current_image": _crop(selected),
                "current_bbox": selected["bbox"] if selected else None,
                "roi": anchor.get("measurement_bbox"),
                "current_roi": row.get("measured_roi") if row else None,
                "heatmap": heatmap,
                "frame": frame_index + 1 if frame_index is not None else None,
                "row": row,
                "delta_e_limit": reference["delta_e_limit"],
                "repeatability_limit": reference["repeatability_limit"],
            }
        )
    return previews


def preview_or_empty(reference: dict, result: dict | None = None) -> list[dict]:
    """Unavailable pictures never alter an otherwise valid measurement result."""
    try:
        return build_preview(reference, result)
    except (GoldenSampleError, OSError, ValueError, TypeError, KeyError, IndexError, cv2.error):
        return []
