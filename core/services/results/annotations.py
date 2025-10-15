from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
from ultralytics.utils.plotting import colors  # type: ignore[import]

from core.utils import ImageUtils


def annotate_yolo_frame(
    image_utils: ImageUtils,
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    color_result: Optional[Dict[str, Any]],
    status: str,
) -> None:
    """Render detection metadata and color check cues onto the frame."""
    status_text = str(status or "").upper()
    if status_text:
        status_color = (0, 255, 0) if status_text == "PASS" else (0, 0, 255)
        image_utils.draw_label(
            frame,
            f"Result: {status_text}",
            (10, 40),
            status_color,
            font_scale=1.1,
            thickness=2,
        )

    color_items: List[Dict[str, Any]] = []
    if color_result:
        color_items = (color_result or {}).get("items", []) or []

    if detections:
        fail_indices: List[int] = []
        for idx, det in enumerate(detections):
            _draw_detection_box(image_utils, frame, det)
            if idx < len(color_items):
                item = color_items[idx]
                try:
                    if not item.get("is_ok", True):
                        fail_indices.append(idx)
                except Exception:
                    continue
        if fail_indices:
            _draw_fail_indices(image_utils, frame, fail_indices)

    if color_result:
        _draw_color_summary(image_utils, frame, color_result, origin_y=80)
        _highlight_color_failures(frame, detections, color_result)


def _draw_detection_box(
    image_utils: ImageUtils, frame: np.ndarray, detection: Dict[str, Any]
) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    label = f"{detection['class']} {detection['confidence']:.2f}"
    color = colors(detection["class_id"], True)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    image_utils.draw_label(frame, label, (x1, y1 - 10), color)


def _draw_fail_indices(
    image_utils: ImageUtils, frame: np.ndarray, indices: List[int]
) -> None:
    try:
        text = f"NG idx: {', '.join(str(i) for i in indices)}"
        image_utils.draw_label(
            frame, text, (10, 10), (0, 0, 255), font_scale=1.0, thickness=2
        )
    except Exception:
        pass


def _highlight_color_failures(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    color_result: Dict[str, Any],
) -> None:
    try:
        items = (color_result or {}).get("items", []) or []
        for idx, det in enumerate(detections or []):
            if idx >= len(items):
                continue
            item = items[idx]
            if not item.get("is_ok", True):
                color = (0, 0, 255)
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    except Exception:
        pass


def _draw_color_summary(
    image_utils: ImageUtils,
    frame: np.ndarray,
    color_result: Dict[str, Any],
    origin_y: Optional[int] = None,
) -> None:
    try:
        text_x = 10
        text_y = origin_y if origin_y is not None else max(50, frame.shape[0] - 120)
        status = "PASS" if color_result.get("is_ok", False) else "FAIL"
        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        image_utils.draw_label(
            frame,
            f"Color: {status}",
            (text_x, text_y),
            color,
            font_scale=1.0,
            thickness=2,
        )
    except Exception:
        pass
