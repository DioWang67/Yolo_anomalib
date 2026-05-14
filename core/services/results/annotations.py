from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from core.services.alignment import (
    base_class_name,
    extract_layout_alignment,
    resolve_missing_expected_keys,
)
from core.services.results.position_summary import (
    POSITION_FAIL_STATES,
    POSITION_OK_STATES,
    format_fixture_shift_hint,
    summarize_position_records,
)
from core.utils import ImageUtils

COLOR_PANEL_MAX_ITEMS = 8
POSITION_PANEL_MAX_ITEMS = 4
EXPECTED_BOX_COLOR = (170, 170, 170)
EXPECTED_CENTER_COLOR = (255, 255, 0)
POSITION_LINE_COLOR = (0, 215, 255)


def colors(class_id: int | str, bgr: bool = True) -> tuple[int, int, int]:
    """Return a deterministic class color without importing Ultralytics."""
    try:
        idx = int(class_id)
    except (TypeError, ValueError):
        idx = abs(hash(str(class_id)))
    palette = (
        (255, 56, 56),
        (255, 157, 151),
        (255, 112, 31),
        (255, 178, 29),
        (207, 210, 49),
        (72, 249, 10),
        (146, 204, 23),
        (61, 219, 134),
        (26, 147, 52),
        (0, 212, 187),
    )
    rgb = palette[idx % len(palette)]
    return rgb[::-1] if bgr else rgb


def annotate_yolo_frame(
    image_utils: ImageUtils,
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    color_result: dict[str, Any] | None,
    status: str,
    *,
    missing_items: list[str] | None = None,
    expected_boxes: dict[str, dict[str, Any]] | None = None,
    missing_locations: list[dict[str, Any]] | None = None,
) -> None:
    """Render detection, color, and position cues onto the frame."""
    panel_lines: list[tuple[str, tuple[int, int, int]]] = []
    status_text = str(status or "").upper()
    if status_text:
        status_color = (0, 255, 0) if status_text == "PASS" else (0, 0, 255)
        panel_lines.append((f"Result: {status_text}", status_color))

    color_items: list[dict[str, Any]] = []
    if color_result:
        color_items = (color_result or {}).get("items", []) or []

    fail_indices: list[int] = []
    if detections:
        for idx, det in enumerate(detections):
            color_item: dict[str, Any] | None = None
            if idx < len(color_items):
                color_item = color_items[idx]
            _draw_detection_box(image_utils, frame, det, color_item)
            if color_item is not None:
                try:
                    if not color_item.get("is_ok", True):
                        fail_indices.append(idx)
                except Exception:
                    continue
    _draw_missing_expected_boxes(frame, detections, missing_items, expected_boxes)
    if fail_indices:
        panel_lines.append(
            (f"NG idx: {', '.join(str(i) for i in fail_indices)}", (0, 0, 255))
        )

    panel_lines.extend(_build_position_summary_lines(detections))

    if color_result:
        panel_lines.extend(_build_color_summary_lines(color_result, detections))
        _highlight_color_failures(frame, detections, color_result)

    if missing_locations:
        missing_names = [str(item.get("class", "")) for item in missing_locations]
        panel_lines.append(
            (
                f"Missing: {', '.join(name for name in missing_names if name)}",
                (0, 0, 255),
            )
        )

    if panel_lines:
        _draw_info_panel(frame, panel_lines, origin=(15, 35))

    if missing_locations:
        _draw_missing_locations(image_utils, frame, missing_locations)


def _draw_detection_box(
    image_utils: ImageUtils,
    frame: np.ndarray,
    detection: dict[str, Any],
    color_item: dict[str, Any] | None = None,
) -> None:
    x1, y1, x2, y2 = _coerce_bbox(detection.get("bbox"))
    label = f"{detection['class']} {detection['confidence']:.2f}"
    color = _position_color(detection)

    _draw_expected_position(frame, detection)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    _draw_position_offset(frame, detection)

    label_y = max(y1 - 10, 20)
    status_suffix = _short_position_status(detection.get("position_status"))
    image_utils.draw_label(frame, f"{label}{status_suffix}", (x1, label_y), color)

    if color_item and not color_item.get("is_ok", True):
        _draw_color_tag(image_utils, frame, x1, label_y, color_item)


def _draw_expected_position(frame: np.ndarray, detection: dict[str, Any]) -> None:
    expected_box = detection.get("position_expected_box")
    if isinstance(expected_box, dict):
        try:
            x1 = int(float(expected_box["x1"]))
            y1 = int(float(expected_box["y1"]))
            x2 = int(float(expected_box["x2"]))
            y2 = int(float(expected_box["y2"]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), EXPECTED_BOX_COLOR, 1)
        except (KeyError, TypeError, ValueError):
            pass

    expected_center = detection.get("position_expected_center")
    if isinstance(expected_center, dict):
        try:
            cx = int(float(expected_center["cx"]))
            cy = int(float(expected_center["cy"]))
            cv2.drawMarker(
                frame,
                (cx, cy),
                EXPECTED_CENTER_COLOR,
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=1,
            )
        except (KeyError, TypeError, ValueError):
            pass


def _draw_position_offset(frame: np.ndarray, detection: dict[str, Any]) -> None:
    bbox = _coerce_bbox(detection.get("bbox"))
    center = _bbox_center(bbox)
    expected_center = detection.get("position_expected_center")
    if not isinstance(expected_center, dict):
        return
    try:
        exp_x = int(float(expected_center["cx"]))
        exp_y = int(float(expected_center["cy"]))
    except (KeyError, TypeError, ValueError):
        return

    cv2.circle(frame, center, 3, _position_color(detection), -1)
    cv2.line(frame, center, (exp_x, exp_y), POSITION_LINE_COLOR, 1, cv2.LINE_AA)

    error_distance = detection.get("position_error")
    if not isinstance(error_distance, (int, float)):
        return
    cv2.putText(
        frame,
        f"d={float(error_distance):.1f}",
        (center[0] + 6, center[1] + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        POSITION_LINE_COLOR,
        1,
        cv2.LINE_AA,
        )


def _draw_missing_expected_boxes(
    frame: np.ndarray,
    detections: list[dict[str, Any]] | None,
    missing_items: list[str] | None,
    expected_boxes: dict[str, dict[str, Any]] | None,
) -> None:
    if not missing_items or not expected_boxes:
        return

    alignment = extract_layout_alignment(detections or [])
    used_expected_keys = {
        str(det.get("position_expected_key"))
        for det in (detections or [])
        if det.get("position_expected_key")
    }
    for expected_key in resolve_missing_expected_keys(
        missing_items, expected_boxes, used_expected_keys
    ):
        expected_box = expected_boxes.get(expected_key)
        if not isinstance(expected_box, dict):
            continue
        try:
            x1, y1, x2, y2 = alignment.shift_box(expected_box)
        except (KeyError, TypeError, ValueError):
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"MISSING {base_class_name(expected_key)}",
            (x1, max(y1 - 8, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_missing_locations(
    image_utils: ImageUtils,
    frame: np.ndarray,
    missing_locations: list[dict[str, Any]],
) -> None:
    for item in missing_locations:
        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        except (TypeError, ValueError):
            continue

        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w - 1, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h - 1, y2))
        if x1 >= x2 or y1 >= y2:
            continue

        color = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"MISSING {item.get('class', '')}".strip()
        label_y = max(y1 - 10, 20)
        image_utils.draw_label(
            frame,
            label,
            (x1, label_y),
            color,
            font_scale=0.65,
            thickness=2,
        )


def _highlight_color_failures(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    color_result: dict[str, Any],
) -> None:
    try:
        items = (color_result or {}).get("items", []) or []
        for idx, det in enumerate(detections or []):
            if idx >= len(items):
                continue
            item = items[idx]
            if not item.get("is_ok", True):
                color = (0, 0, 255)
                x1, y1, x2, y2 = _coerce_bbox(det.get("bbox"))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    except Exception:
        pass


def _build_position_summary_lines(
    detections: list[dict[str, Any]] | None,
) -> list[tuple[str, tuple[int, int, int]]]:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    if not detections:
        return lines

    summary = summarize_position_records(detections)
    if summary.total_with_position <= 0 and summary.skipped_count <= 0:
        return lines

    summary_color = (0, 255, 0) if summary.fail_count == 0 else (0, 0, 255)
    lines.append(
        (
            f"Pos: ok={summary.correct_count} ng={summary.fail_count} skip={summary.skipped_count}",
            summary_color,
        )
    )

    fixture_hint = format_fixture_shift_hint(summary)
    if fixture_hint:
        lines.append((fixture_hint, (0, 165, 255)))

    for issue in summary.issues[:POSITION_PANEL_MAX_ITEMS]:
        detail = issue.label
        if issue.error is not None:
            detail += f" d={issue.error:.1f}"
        if issue.dx is not None and issue.dy is not None:
            detail += f" ({issue.dx:+.1f},{issue.dy:+.1f})"
        lines.append((f"{issue.status}: {detail}", (0, 0, 255)))
    return lines


def _build_color_summary_lines(
    color_result: dict[str, Any], detections: list[dict[str, Any]] | None = None
) -> list[tuple[str, tuple[int, int, int]]]:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    try:
        status = "PASS" if color_result.get("is_ok", False) else "FAIL"
        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        lines.append((f"Color: {status}", color))
        details = _format_color_lines(color_result)
        for line, is_ok in details:
            line_color = (0, 255, 0) if is_ok else (0, 0, 255)
            lines.append((line, line_color))
        if detections:
            seq = _left_right_color_sequence(detections, color_result)
            if seq:
                seq_line = seq[0] if len(seq) == 1 else " -> ".join(seq)
                lines.append((f"LR: {seq_line}", (200, 200, 200)))
    except Exception:
        pass
    return lines


def _draw_info_panel(
    frame: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int] = (15, 35),
) -> None:
    if not lines:
        return
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_gap = 20
    try:
        max_width = 0
        for text, _ in lines:
            size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, size[0])
        panel_height = line_gap * len(lines) + 10
        top_left = (max(x - 12, 0), max(y - 30, 0))
        bottom_right = (
            min(top_left[0] + max_width + 30, frame.shape[1] - 1),
            min(top_left[1] + panel_height + 20, frame.shape[0] - 1),
        )
        cv2.rectangle(frame, top_left, bottom_right, (25, 25, 25), -1)
        cv2.rectangle(frame, top_left, bottom_right, (90, 90, 90), 1)
        for idx, (text, color) in enumerate(lines):
            text_y = y + idx * line_gap
            cv2.putText(frame, text, (x, text_y), font, font_scale, color, thickness)
    except Exception:
        pass


def _draw_color_tag(
    image_utils: ImageUtils,
    frame: np.ndarray,
    x: int,
    label_y: int,
    color_item: dict[str, Any],
) -> None:
    try:
        name = str(color_item.get("best_color") or "-")
        diff = color_item.get("diff")
        threshold = color_item.get("threshold")
        diff_text = ""
        if isinstance(diff, (int, float)) and isinstance(threshold, (int, float)):
            diff_text = f" d={float(diff):.2f}/{float(threshold):.2f}"
        status = "OK" if color_item.get("is_ok", True) else "NG"
        tag_text = f"{name}{diff_text} {status}".strip()
        tag_color = (0, 255, 0) if status == "OK" else (0, 0, 255)
        offset_y = max(label_y - 18, 20)
        image_utils.draw_label(
            frame,
            tag_text,
            (x, offset_y),
            tag_color,
            font_scale=0.55,
            thickness=1,
        )
    except Exception:
        pass


def _format_color_lines(
    color_result: dict[str, Any], max_items: int | None = None
) -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = []
    try:
        items = (color_result or {}).get("items", []) or []
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for pos, item in enumerate(items):
            is_ok = bool(item.get("is_ok", True))
            rank = 0 if not is_ok else 1
            ranked.append((rank, pos, item))
        ranked.sort(key=lambda entry: (entry[0], entry[1]))
        limit = max_items if max_items is not None else COLOR_PANEL_MAX_ITEMS
        for _, _, item in ranked[:limit]:
            idx = item.get("index", "?")
            cls_name = item.get("class_name") or "-"
            best = item.get("best_color") or "-"
            diff = item.get("diff")
            threshold = item.get("threshold")
            diff_str = "-"
            if isinstance(diff, (int, float)) and isinstance(threshold, (int, float)):
                diff_str = f"{float(diff):.2f}/{float(threshold):.2f}"
            status_ok = bool(item.get("is_ok", True))
            status_text = "OK" if status_ok else "NG"
            line = f"#{idx} {cls_name} -> {best} (d={diff_str}) {status_text}"
            lines.append((line, status_ok))
        hidden = max(0, len(ranked) - limit)
        if hidden > 0:
            lines.append((f"... +{hidden} more", True))
    except Exception:
        pass
    return lines


def _left_right_color_sequence(
    detections: list[dict[str, Any]], color_result: dict[str, Any]
) -> list[str]:
    try:
        items = (color_result or {}).get("items", []) or []
        seq: list[tuple[float, str]] = []
        for idx, det in enumerate(detections or []):
            if idx >= len(items):
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, _, x2, _ = bbox
            center = (float(x1) + float(x2)) / 2.0
            best = str(items[idx].get("best_color") or "-")
            seq.append((center, best))
        seq.sort(key=lambda item: item[0])
        return [color for _, color in seq]
    except Exception:
        return []


def _position_color(detection: dict[str, Any]) -> tuple[int, int, int]:
    status = str(detection.get("position_status") or "").upper()
    if status in POSITION_OK_STATES:
        return (0, 200, 0)
    if status == "WRONG":
        return (0, 0, 255)
    if status == "UNEXPECTED":
        return (0, 140, 255)
    if status in {"INVALID", "ERROR"}:
        return (255, 0, 255)
    return colors(detection.get("class_id", 0), True)


def _short_position_status(status: Any) -> str:
    value = str(status or "").upper()
    if not value:
        return ""
    if value == "CORRECT":
        return " [OK]"
    if value == "WRONG":
        return " [NG]"
    if value == "UNEXPECTED":
        return " [UNEXP]"
    if value in {"INVALID", "ERROR", "UNKNOWN"}:
        return f" [{value}]"
    return ""


def _coerce_bbox(bbox: Any) -> tuple[int, int, int, int]:
    if not bbox or len(bbox) < 4:
        return (0, 0, 0, 0)
    return tuple(int(float(v)) for v in bbox[:4])


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)
