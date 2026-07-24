"""Shared, backend-independent detection overlay rendering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from typing import Any

import cv2
import numpy as np

Detection = dict[str, Any]
Color = tuple[int, int, int]
ColorResolver = Callable[[Detection], Color]
LEGEND_TEXT_COLOR: Color = (235, 235, 235)


def draw_detection_overlay(
    frame: np.ndarray,
    detections: Sequence[Detection],
    color_resolver: ColorResolver,
    *,
    heading: tuple[str, Color] | None = None,
) -> None:
    """Draw compact box indices and a readable legend.

    Dense industrial layouts often place multiple narrow detections on the same
    horizontal line. Full labels above every box overlap in that geometry, so
    boxes carry only a stable ``#index`` tag while class and confidence details
    live in a compact legend.
    """
    legend_entries: list[tuple[str, Color]] = []
    for index, detection in enumerate(detections):
        bbox = _coerce_bbox(detection.get("bbox"))
        if bbox is None:
            continue
        color = color_resolver(detection)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_box_tag(frame, bbox, f"#{index}", color)
        legend_entries.append(
            (_format_detection_legend(index, detection), LEGEND_TEXT_COLOR)
        )

    if heading is not None:
        legend_entries.insert(0, heading)
    draw_legend_panel(frame, legend_entries)


def draw_box_tag(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    text: str,
    color: Color,
) -> None:
    """Draw a short, clamped label tag immediately above a detection box."""
    if frame.size == 0 or not text:
        return
    height, width = frame.shape[:2]
    x1, y1, _, _ = bbox
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding_x = 4
    padding_y = 3
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tag_width = text_size[0] + padding_x * 2
    tag_height = text_size[1] + baseline + padding_y * 2

    left = max(0, min(width - tag_width - 1, x1))
    bottom = max(tag_height, min(height - 1, y1))
    top = max(0, bottom - tag_height)
    right = min(width - 1, left + tag_width)

    cv2.rectangle(frame, (left, top), (right, bottom), color, -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (230, 230, 230), 1)
    text_color = (0, 0, 0) if _perceived_brightness(color) >= 135 else (255, 255, 255)
    cv2.putText(
        frame,
        text,
        (left + padding_x, bottom - padding_y - baseline),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_legend_panel(
    frame: np.ndarray,
    entries: Sequence[tuple[str, Color]],
    *,
    origin: tuple[int, int] = (8, 8),
) -> None:
    """Draw detection details in one or two columns without text collisions."""
    if frame.size == 0 or not entries:
        return

    frame_height, frame_width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    line_height = 21
    padding = 8
    column_gap = 18
    column_count = 2 if frame_width >= 480 and len(entries) >= 4 else 1
    row_count = ceil(len(entries) / column_count)
    columns = [
        entries[column * row_count : (column + 1) * row_count]
        for column in range(column_count)
    ]
    column_widths = [
        max(
            (
                cv2.getTextSize(text, font, font_scale, thickness)[0][0]
                for text, _ in column
            ),
            default=0,
        )
        for column in columns
    ]

    panel_width = sum(column_widths) + column_gap * (column_count - 1) + padding * 2
    panel_height = row_count * line_height + padding * 2
    left = max(0, min(frame_width - panel_width - 1, origin[0]))
    top = max(0, min(frame_height - panel_height - 1, origin[1]))
    right = min(frame_width - 1, left + panel_width)
    bottom = min(frame_height - 1, top + panel_height)
    cv2.rectangle(frame, (left, top), (right, bottom), (25, 25, 25), -1)
    cv2.rectangle(frame, (left, top), (right, bottom), (90, 90, 90), 1)

    column_x = left + padding
    for column, column_width in zip(columns, column_widths, strict=True):
        for row, (text, color) in enumerate(column):
            text_y = top + padding + (row + 1) * line_height - 5
            cv2.putText(
                frame,
                text,
                (column_x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
        column_x += column_width + column_gap


def _format_detection_legend(index: int, detection: Detection) -> str:
    class_name = str(detection.get("class") or "?")
    try:
        confidence = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return f"#{index} {class_name} {confidence:.2f}"


def _coerce_bbox(value: Any) -> tuple[int, int, int, int] | None:
    if not value or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(float(item)) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def _perceived_brightness(color: Color) -> float:
    blue, green, red = color
    return 0.114 * blue + 0.587 * green + 0.299 * red
