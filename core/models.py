from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class DetectionItem:
    class_name: str
    class_id: int
    confidence: float
    bbox: List[int]
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    position_status: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Backward compatible keys used elsewhere
        d["class"] = d.pop("class_name")
        return d


@dataclass
class ColorCheckItemResult:
    index: int
    class_name: Optional[str]
    bbox: Optional[List[int]]
    best_color: str
    diff: float
    threshold: float
    is_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d


@dataclass
class ColorCheckResult:
    is_ok: bool
    items: List[ColorCheckItemResult]

    def diff_string(self) -> str:
        return ";".join([f"{it.diff:.2f}" for it in self.items])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_ok": self.is_ok,
            "items": [it.to_dict() for it in self.items],
            "diff": self.diff_string(),
        }


@dataclass
class DetectionResultModel:
    status: str
    product: str
    area: str
    inference_type: str
    ckpt_path: str = ""
    anomaly_score: Optional[float] = None
    detections: List[DetectionItem] = None  # type: ignore
    missing_items: List[str] = None  # type: ignore
    original_image_path: str = ""
    preprocessed_image_path: str = ""
    annotated_path: str = ""
    heatmap_path: str = ""
    cropped_paths: List[str] = None  # type: ignore
    color_check: Optional[ColorCheckResult] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "product": self.product,
            "area": self.area,
            "inference_type": self.inference_type,
            "ckpt_path": self.ckpt_path,
            "anomaly_score": self.anomaly_score if self.anomaly_score is not None else "",
            "detections": [d.to_dict() for d in (self.detections or [])],
            "missing_items": list(self.missing_items or []),
            "original_image_path": self.original_image_path,
            "preprocessed_image_path": self.preprocessed_image_path,
            "annotated_path": self.annotated_path,
            "heatmap_path": self.heatmap_path,
            "cropped_paths": list(self.cropped_paths or []),
            "color_check": self.color_check.to_dict() if self.color_check else None,
            "error": self.error,
        }

