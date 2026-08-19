from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from core.services.alignment import (
    base_class_name,
    extract_layout_alignment,
    resolve_missing_expected_keys,
)
from core.services.detection_sequence import left_right_sequence
from core.services.results.customer_message import (
    COLOR_FAILURE_LOW_CONFIDENCE,
    COLOR_FAILURE_MISMATCH,
    classify_color_check_failure,
)
from core.services.results.position_summary import (
    POSITION_OK_STATES,
    format_fixture_shift_hint,
    summarize_position_records,
)
from core.utils import ImageUtils
from core.visualization import draw_box_tag

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
    duplicate_filter: dict[str, Any] | None = None,
    raw_detections: list[dict[str, Any]] | None = None,
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
    color_items_by_index = _items_by_index(color_items)

    fail_indices: list[int] = []
    if duplicate_filter:
        panel_lines.extend(_build_duplicate_summary_lines(duplicate_filter))
        _draw_duplicate_suppressions(
            frame,
            duplicate_filter,
            raw_detections or detections,
        )
    if detections:
        for idx, det in enumerate(detections):
            source_index = _source_index(det, idx)
            color_item = color_items_by_index.get(source_index)
            _draw_detection_box(frame, source_index, det, color_item)
            if color_item is not None:
                try:
                    if not color_item.get("is_ok", True):
                        fail_indices.append(source_index)
                except Exception:
                    continue
    if not missing_locations:
        _draw_missing_expected_boxes(frame, detections, missing_items, expected_boxes)
    if fail_indices:
        panel_lines.append(
            (f"NG idx: {', '.join(str(i) for i in fail_indices)}", (0, 0, 255))
        )

    panel_lines.extend(_build_position_summary_lines(detections))

    if color_result:
        panel_lines.extend(_build_color_summary_lines(color_result, detections))
        _highlight_color_failures(frame, detections, color_result)
    elif detections:
        panel_lines.extend(_build_detection_summary_lines(detections))

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
    frame: np.ndarray,
    index: int,
    detection: dict[str, Any],
    color_item: dict[str, Any] | None = None,
) -> None:
    x1, y1, x2, y2 = _coerce_bbox(detection.get("bbox"))
    color = _position_color(detection)
    position_status = str(detection.get("position_status") or "").upper()
    position_is_ok = position_status in POSITION_OK_STATES

    if position_status and not position_is_ok:
        _draw_expected_position(frame, detection)
        _draw_position_offset(frame, detection)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    tag_text = f"#{index}"
    tag_color = color
    if color_item and not color_item.get("is_ok", True):
        tag_text += " NG"
        tag_color = (0, 0, 255)
    draw_box_tag(frame, (x1, y1, x2, y2), tag_text, tag_color)


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
        items_by_index = _items_by_index(items)
        for idx, det in enumerate(detections or []):
            item = items_by_index.get(_source_index(det, idx))
            if item is None:
                continue
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
    if summary.total_with_position <= 0:
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
        effective_indices = {
            _source_index(detection, position)
            for position, detection in enumerate(detections or [])
        }
        details = _format_color_lines(
            color_result,
            effective_indices=effective_indices,
        )
        for line, is_ok in details:
            line_color = (0, 255, 0) if is_ok else (0, 0, 255)
            lines.append((line, line_color))
        if detections:
            # Must stay the shared helper: the panel reports the labels the
            # decision was made from, not the color checker's raw best match.
            seq = left_right_sequence(detections)
            if seq:
                seq_line = seq[0] if len(seq) == 1 else " -> ".join(seq)
                lines.append((f"LR: {seq_line}", (200, 200, 200)))
    except Exception:
        pass
    return lines


def _build_detection_summary_lines(
    detections: list[dict[str, Any]],
) -> list[tuple[str, tuple[int, int, int]]]:
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for index, detection in enumerate(detections[:COLOR_PANEL_MAX_ITEMS]):
        class_name = str(detection.get("class") or "?")
        try:
            confidence = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        lines.append(
            (
                f"#{_source_index(detection, index)} {class_name} confidence={confidence:.2f}",
                _position_color(detection),
            )
        )
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


def _color_item_overlay_text(item: dict[str, Any], is_ok: bool) -> str:
    """Return an ASCII-safe description of one color-check item for overlay text.

    Shares its classification with the main verdict message
    (:func:`core.services.results.customer_message.classify_color_check_failure`)
    so this image overlay never shows a bare class name for a failure the
    operator guidance card separately reports as a color mismatch or a
    confidence shortfall.
    """
    if is_ok:
        cls_name = item.get("class_name") or "-"
        best = item.get("best_color") or "-"
        return f"{cls_name} -> {best}"
    description = classify_color_check_failure(item)
    if description.kind == COLOR_FAILURE_MISMATCH:
        return f"{description.class_name} -> {description.predicted_color} (mismatch)"
    if description.kind == COLOR_FAILURE_LOW_CONFIDENCE:
        return f"{description.class_name} (low confidence)"
    return item.get("class_name") or "-"


def _format_color_lines(
    color_result: dict[str, Any],
    max_items: int | None = None,
    effective_indices: set[int] | None = None,
) -> list[tuple[str, bool]]:
    lines: list[tuple[str, bool]] = []
    try:
        items = (color_result or {}).get("items", []) or []
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        for pos, item in enumerate(items):
            try:
                item_index = int(item.get("index", pos))
            except (TypeError, ValueError):
                continue
            if effective_indices is not None and item_index not in effective_indices:
                continue
            is_ok = bool(item.get("is_ok", True))
            rank = 0 if not is_ok else 1
            ranked.append((rank, pos, item))
        ranked.sort(key=lambda entry: (entry[0], entry[1]))
        limit = max_items if max_items is not None else COLOR_PANEL_MAX_ITEMS
        for _, _, item in ranked[:limit]:
            idx = item.get("index", "-")
            diff = item.get("diff")
            threshold = item.get("threshold")
            diff_str = "-"
            if isinstance(diff, (int, float)) and isinstance(threshold, (int, float)):
                diff_str = f"{float(diff):.2f}/{float(threshold):.2f}"
            status_ok = bool(item.get("is_ok", True))
            status_text = "OK" if status_ok else "NG"
            # ASCII-only: this line is drawn with cv2.putText / FONT_HERSHEY_SIMPLEX,
            # which cannot render non-Latin glyphs, so it cannot reuse the
            # Chinese wording customer_message.py uses for the same
            # mismatch-vs-low-confidence classification. The distinction (not
            # just diff/threshold, which are debugging context kept either way)
            # still needs to survive in this ASCII form.
            description = _color_item_overlay_text(item, status_ok)
            line = f"#{idx} {description} (d={diff_str}) {status_text}"
            lines.append((line, status_ok))
        hidden = max(0, len(ranked) - limit)
        if hidden > 0:
            lines.append((f"... +{hidden} more", True))
    except Exception:
        pass
    return lines


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


def _coerce_bbox(bbox: Any) -> tuple[int, int, int, int]:
    if not bbox or len(bbox) < 4:
        return (0, 0, 0, 0)
    x1, y1, x2, y2 = (int(float(v)) for v in bbox[:4])
    return (x1, y1, x2, y2)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _source_index(detection: dict[str, Any], fallback: int) -> int:
    try:
        return int(detection.get("source_index", fallback))
    except (TypeError, ValueError):
        return fallback


def _items_by_index(
    items: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            continue
        indexed[index] = item
    return indexed


def _build_duplicate_summary_lines(
    duplicate_filter: dict[str, Any],
) -> list[tuple[str, tuple[int, int, int]]]:
    status = str(duplicate_filter.get("status") or "")
    suppressions = duplicate_filter.get("suppressions", []) or []
    proposals = duplicate_filter.get("proposed_suppressions", []) or []
    records = suppressions if status == "suppressed" else proposals
    if not records:
        return []

    action = "removed" if status == "suppressed" else "candidate"
    lines: list[tuple[str, tuple[int, int, int]]] = []
    for record in records[:2]:
        try:
            suppressed = int(record["suppressed_index"])
            kept = int(record["kept_index"])
            iou = float(record.get("iou", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        lines.append(
            (
                f"DUP {action}: #{suppressed} -> #{kept} IoU={iou:.2f}",
                (0, 215, 255),
            )
        )
    if len(records) > 2:
        lines.append((f"DUP ... +{len(records) - 2} more", (0, 215, 255)))
    return lines


def _draw_duplicate_suppressions(
    frame: np.ndarray,
    duplicate_filter: dict[str, Any],
    raw_detections: list[dict[str, Any]],
) -> None:
    if str(duplicate_filter.get("status") or "") != "suppressed":
        return
    raw_by_index = {
        _source_index(detection, position): detection
        for position, detection in enumerate(raw_detections or [])
    }
    for record in duplicate_filter.get("suppressions", []) or []:
        try:
            suppressed_index = int(record["suppressed_index"])
            kept_index = int(record["kept_index"])
        except (KeyError, TypeError, ValueError):
            continue
        detection = raw_by_index.get(suppressed_index)
        if detection is None:
            continue
        x1, y1, x2, y2 = _coerce_bbox(detection.get("bbox"))
        color = (255, 0, 255)
        _draw_dashed_rectangle(frame, (x1, y1, x2, y2), color)
        cv2.putText(
            frame,
            f"#{suppressed_index} DUP -> #{kept_index}",
            (x1, min(frame.shape[0] - 5, y2 + 17)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_dashed_rectangle(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    dash_length: int = 6,
) -> None:
    x1, y1, x2, y2 = bbox
    segments = (
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    )
    for start, end in segments:
        length = int(math.dist(start, end))
        if length <= 0:
            continue
        for offset in range(0, length, dash_length * 2):
            end_offset = min(offset + dash_length, length)
            ratio_start = offset / length
            ratio_end = end_offset / length
            line_start = (
                round(start[0] + (end[0] - start[0]) * ratio_start),
                round(start[1] + (end[1] - start[1]) * ratio_start),
            )
            line_end = (
                round(start[0] + (end[0] - start[0]) * ratio_end),
                round(start[1] + (end[1] - start[1]) * ratio_end),
            )
            cv2.line(frame, line_start, line_end, color, 1, cv2.LINE_AA)
