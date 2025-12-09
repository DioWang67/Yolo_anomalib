from __future__ import annotations

import os
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
