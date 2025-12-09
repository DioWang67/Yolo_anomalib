from __future__ import annotations

"""Context object passed between pipeline steps during a detection run."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class DetectionContext:
    product: str
    area: str
    inference_type: str
    frame: np.ndarray
    processed_image: np.ndarray
    result: dict[str, Any]
    status: str
    color_result: dict[str, Any] | None = None
    save_result: dict[str, Any] | None = None
    config: Any | None = None
