"""核心資料型別：統一檢測結果的強型別定義。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

DetectionStatus = Literal["PASS", "FAIL", "ERROR", "CANCELED"]


@dataclass
class DetectionItem:
    """Represents a single detected object or anomaly."""

    label: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Standardized result object for a detection cycle.

    This is the **single source of truth** returned by
    ``DetectionSystem.detect()`` and consumed by GUI, CLI, and tests.
    All previously dict-based fields are now explicit attributes.
    """

    # --- Core fields ---
    status: DetectionStatus
    items: list[DetectionItem] = field(default_factory=list)
    latency: float = 0.0
    timestamp: float = field(default_factory=time.time)
    frame_id: int | None = None
    error: str | None = None

    # --- Domain-specific fields ---
    product: str = ""
    area: str = ""
    inference_type: str = ""
    ckpt_path: str = ""
    anomaly_score: float | str | None = None

    # --- Artifact paths ---
    original_image_path: str = ""
    preprocessed_image_path: str = ""
    annotated_path: str = ""
    heatmap_path: str = ""
    cropped_paths: list[str] = field(default_factory=list)

    # --- Pipeline outputs ---
    missing_items: list[str] = field(default_factory=list)
    unexpected_items: list[str] = field(default_factory=list)
    color_check: dict[str, Any] | None = None
    sequence_check: dict[str, Any] | None = None

    # --- Image data (not serialized) ---
    result_frame: np.ndarray | None = field(default=None, repr=False)

    # --- Extensible metadata bucket ---
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def image_path(self) -> str | None:
        """Backward-compatible alias for ``original_image_path``."""
        return self.original_image_path or None

    @property
    def detections(self) -> list[dict[str, Any]]:
        """Legacy-compatible list of detection dicts.

        Useful for code that still expects raw dict format.
        """
        return [
            {
                "class": item.label,
                "confidence": item.confidence,
                "bbox": list(item.bbox_xyxy),
                **item.metadata,
            }
            for item in self.items
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization or legacy compatibility."""
        return {
            "status": self.status,
            "product": self.product,
            "area": self.area,
            "inference_type": self.inference_type,
            "error": self.error or "",
            "ckpt_path": self.ckpt_path,
            "anomaly_score": self.anomaly_score,
            "detections": self.detections,
            "missing_items": self.missing_items,
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
            "original_image_path": self.original_image_path,
            "preprocessed_image_path": self.preprocessed_image_path,
            "annotated_path": self.annotated_path,
            "heatmap_path": self.heatmap_path,
            "cropped_paths": self.cropped_paths,
            "color_check": self.color_check,
            "sequence_check": self.sequence_check,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Pipeline DTO — carries context through the Producer-Consumer pipeline.
# ---------------------------------------------------------------------------

@dataclass
class DetectionTask:
    """Unit of work flowing through the async detection pipeline.

    Created by AcquisitionWorker at capture time.  Carries the raw frame
    *together with* all context required by downstream workers so that no
    information is lost when items sit in a queue.

    Attributes:
        task_id:        Unique identifier (UUID or hardware trigger sequence).
        timestamp:      ``time.time()`` at the moment the frame was captured.
                        StorageWorker uses this to record the true capture
                        time, **not** the (later) inference-complete time.
        product:        Product identifier for model dispatch.
        area:           Area identifier for model dispatch.
        inference_type: The requested inference mode ('yolo', 'anomalib',
                        'fusion').
        frame:          Raw BGR image from the camera.
        is_poison_pill: If ``True`` the task is a shutdown sentinel — workers
                        must propagate it downstream and then exit.
        error:          Populated by InferenceWorker when inference fails so
                        that StorageWorker can persist the error context.
        result:         Raw inference result dict, populated by
                        InferenceWorker for StorageWorker to persist.
    """

    task_id: str
    timestamp: float
    product: str
    area: str
    inference_type: str = "yolo"
    frame: np.ndarray | None = field(default=None, repr=False)
    is_poison_pill: bool = False
    error: str | None = None
    result: dict[str, Any] | None = field(default=None, repr=False)

    # ---- Convenience factories ----

    @classmethod
    def poison_pill(cls) -> DetectionTask:
        """Create a minimal shutdown sentinel task."""
        return cls(
            task_id="__POISON_PILL__",
            timestamp=0.0,
            product="",
            area="",
            is_poison_pill=True,
        )
