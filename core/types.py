from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

DetectionStatus = Literal["PASS", "FAIL", "ERROR"]

@dataclass
class DetectionItem:
    """Represents a single detected object or anomaly."""
    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectionResult:
    """Standardized result object for a detection cycle."""
    status: DetectionStatus
    items: list[DetectionItem]
    latency: float
    timestamp: float = field(default_factory=time.time)
    frame_id: int | None = None
    image_path: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for legacy compatibility or serialization."""
        return {
            "status": self.status,
            "items": [
                {
                    "label": item.label,
                    "confidence": item.confidence,
                    "bbox_xyxy": item.bbox_xyxy,
                    "metadata": item.metadata,
                }
                for item in self.items
            ],
            "latency": self.latency,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "error": self.error,
            "metadata": self.metadata,
        }
