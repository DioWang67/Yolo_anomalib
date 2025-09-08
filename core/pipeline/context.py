from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class DetectionContext:
    product: str
    area: str
    inference_type: str
    frame: np.ndarray
    processed_image: np.ndarray
    result: Dict[str, Any]
    status: str
    color_result: Optional[Dict[str, Any]] = None
    save_result: Optional[Dict[str, Any]] = None
    config: Any | None = None
