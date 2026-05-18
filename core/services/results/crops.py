from __future__ import annotations

import os
import re
from typing import Any

import numpy as np

from .image_queue import ImageWriteQueue
from .path_manager import SavePathBundle


def save_detection_crops(
    queue: ImageWriteQueue,
    crop_source: np.ndarray,
    detections: list[dict[str, Any]],
    bundle: SavePathBundle,
    *,
    product: str | None,
    area: str | None,
    timestamp_text: str,
    params: list[int],
    limit: int | None = None,
) -> list[str]:
    """Persist detection crops and return their paths."""
    cropped_paths: list[str] = []
    for idx, det in enumerate(detections):
        if limit is not None and idx >= limit:
            break
        x1, y1, x2, y2 = det["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(crop_source.shape[1], x2)
        y2 = min(crop_source.shape[0], y2)
        cropped_img = crop_source[y1:y2, x1:x2]
        crop_name = (
            f"{bundle.detector_prefix}_{product}_{area}_"
            f"{timestamp_text}_{det['class']}_{idx}.png"
        )
        crop_path = os.path.join(bundle.cropped_dir, crop_name)
        queue.enqueue(crop_path, cropped_img, params)
        cropped_paths.append(crop_path)
    return cropped_paths


def save_failure_crops(
    queue: ImageWriteQueue,
    crop_source: np.ndarray,
    bundle: SavePathBundle,
    *,
    product: str | None,
    area: str | None,
    timestamp_text: str,
    params: list[int],
    missing_locations: list[dict[str, Any]] | None = None,
    detections: list[dict[str, Any]] | None = None,
    slot_mismatches: list[dict[str, Any]] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Persist NG evidence crops with stable reason-based filenames.

    Args:
        queue: Image write queue.
        crop_source: Image used for crop extraction.
        bundle: Resolved result path bundle.
        product: Product name for filenames.
        area: Area name for filenames.
        timestamp_text: Timestamp segment shared with result images.
        params: cv2 write parameters.
        missing_locations: Expected boxes for missing items.
        detections: Detection records; ``position_status == WRONG`` is cropped.
        slot_mismatches: Wrong component records.
        limit: Optional maximum number of failure crops.

    Returns:
        List of paths queued for persistence.
    """
    if crop_source is None or crop_source.size == 0:
        return []

    requests = _build_failure_crop_requests(
        missing_locations=missing_locations,
        detections=detections,
        slot_mismatches=slot_mismatches,
    )
    if limit is not None:
        requests = requests[: max(0, limit)]

    paths: list[str] = []
    for index, request in enumerate(requests):
        crop = _extract_crop(crop_source, request["bbox"])
        if crop is None:
            continue
        crop_name = (
            f"{bundle.detector_prefix}_{_safe_name(product)}_{_safe_name(area)}_"
            f"{timestamp_text}_NG_{request['reason']}_{request['name']}_{index}.png"
        )
        crop_path = os.path.join(bundle.cropped_dir, crop_name)
        queue.enqueue(crop_path, crop, params)
        paths.append(crop_path)
    return paths


def _build_failure_crop_requests(
    *,
    missing_locations: list[dict[str, Any]] | None,
    detections: list[dict[str, Any]] | None,
    slot_mismatches: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    for item in missing_locations or []:
        bbox = item.get("bbox")
        if not _valid_bbox(bbox):
            continue
        name = item.get("expected_key") or item.get("class") or "unknown"
        requests.append(
            {"reason": "MISSING", "name": _safe_name(name), "bbox": list(bbox[:4])}
        )

    for item in slot_mismatches or []:
        bbox = item.get("bbox")
        if not _valid_bbox(bbox):
            continue
        name = item.get("expected_key") or item.get("expected_class") or "unknown"
        requests.append(
            {
                "reason": "WRONG_COMPONENT",
                "name": _safe_name(name),
                "bbox": list(bbox[:4]),
            }
        )

    for det in detections or []:
        if str(det.get("position_status") or "").upper() != "WRONG":
            continue
        bbox = det.get("bbox")
        if not _valid_bbox(bbox):
            continue
        name = det.get("position_expected_key") or det.get("class") or "unknown"
        requests.append(
            {
                "reason": "POSITION_SHIFT",
                "name": _safe_name(name),
                "bbox": list(bbox[:4]),
            }
        )
    return requests


def _extract_crop(image: np.ndarray, bbox: Any) -> np.ndarray | None:
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
    except (TypeError, ValueError):
        return None
    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _valid_bbox(bbox: Any) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) >= 4


def _safe_name(value: Any) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
