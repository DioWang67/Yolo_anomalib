"""Pure image sampling helpers shared by color calibration and inference."""

from __future__ import annotations

import numpy as np


def center_crop_by_ratio(image: np.ndarray, margin_ratio: float) -> np.ndarray:
    """Remove the same fraction from every edge of an image.

    The ratio is evaluated independently on width and height.  Keeping this
    operation shared is part of the baseline contract: statistics measured on
    a 70% x 70% calibration crop must be compared with a 70% x 70% runtime
    crop, including for elongated detector boxes.
    """
    if not isinstance(image, np.ndarray) or image.ndim < 2 or image.size == 0:
        return image
    ratio = float(margin_ratio)
    if not np.isfinite(ratio) or ratio <= 0.0:
        return image
    height, width = image.shape[:2]
    margin_y = int(height * ratio)
    margin_x = int(width * ratio)
    if (
        margin_y <= 0
        and margin_x <= 0
        or margin_y * 2 >= height
        or margin_x * 2 >= width
    ):
        return image
    cropped = image[
        margin_y : height - margin_y,
        margin_x : width - margin_x,
    ]
    return cropped if cropped.size else image
