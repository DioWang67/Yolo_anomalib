from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics.utils.plotting import colors  # type: ignore[import]

from core.utils import ImageUtils

COLOR_PANEL_MAX_ITEMS = 8

def annotate_yolo_frame(
    image_utils: ImageUtils,
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    color_result: Optional[Dict[str, Any]],
    status: str,
) -> None:
    """Render detection metadata and color check cues onto the frame."""
    panel_lines: List[Tuple[str, Tuple[int, int, int]]] = []
    status_text = str(status or "").upper()
    if status_text:
        status_color = (0, 255, 0) if status_text == "PASS" else (0, 0, 255)
        panel_lines.append((f"Result: {status_text}", status_color))

    color_items: List[Dict[str, Any]] = []
    if color_result:
        color_items = (color_result or {}).get("items", []) or []

    fail_indices: List[int] = []
    if detections:
        for idx, det in enumerate(detections):
            color_item: Optional[Dict[str, Any]] = None
            if idx < len(color_items):
                color_item = color_items[idx]
            _draw_detection_box(image_utils, frame, det, color_item)
            if color_item is not None:
                try:
                    if not color_item.get("is_ok", True):
                        fail_indices.append(idx)
                except Exception:
                    continue
    if fail_indices:
        panel_lines.append(
            (f"NG idx: {', '.join(str(i) for i in fail_indices)}", (0, 0, 255))
        )

    if color_result:
        panel_lines.extend(_build_color_summary_lines(color_result, detections))
        _highlight_color_failures(frame, detections, color_result)

    if panel_lines:
        _draw_info_panel(frame, panel_lines, origin=(15, 35))


def _draw_detection_box(
    image_utils: ImageUtils,
    frame: np.ndarray,
    detection: Dict[str, Any],
    color_item: Optional[Dict[str, Any]] = None,
) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    label = f"{detection['class']} {detection['confidence']:.2f}"
    color = colors(detection["class_id"], True)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_y = max(y1 - 10, 20)
    image_utils.draw_label(frame, label, (x1, label_y), color)

    if color_item and not color_item.get("is_ok", True):
        _draw_color_tag(image_utils, frame, x1, label_y, color_item)
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


def _build_color_summary_lines(
    color_result: Dict[str, Any], detections: Optional[List[Dict[str, Any]]] = None
) -> List[Tuple[str, Tuple[int, int, int]]]:
    lines: List[Tuple[str, Tuple[int, int, int]]] = []
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
    lines: List[Tuple[str, Tuple[int, int, int]]],
    origin: Tuple[int, int] = (15, 35),
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
    color_item: Dict[str, Any],
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
    color_result: Dict[str, Any], max_items: Optional[int] = None
) -> List[tuple[str, bool]]:
    lines: List[tuple[str, bool]] = []
    try:
        items = (color_result or {}).get("items", []) or []
        ranked: List[tuple[int, int, Dict[str, Any]]] = []
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
    detections: List[Dict[str, Any]], color_result: Dict[str, Any]
) -> List[str]:
    try:
        items = (color_result or {}).get("items", []) or []
        seq: List[tuple[float, str]] = []
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
