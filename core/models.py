from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DetectionItem:
    class_name: str
    class_id: int
    confidence: float
    bbox: list[int]
    image_width: int | None = None
    image_height: int | None = None
    cx: float | None = None
    cy: float | None = None
    position_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Backward compatible keys used elsewhere
        d["class"] = d.pop("class_name")
        return d


@dataclass
class ColorCheckItemResult:
    index: int
    class_name: str | None
    bbox: list[int] | None
    best_color: str
    diff: float
    threshold: float
    is_ok: bool

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["class"] = d.pop("class_name")
        return d


@dataclass
class ColorCheckResult:
    is_ok: bool
    items: list[ColorCheckItemResult]

    def diff_string(self) -> str:
        return ";".join([f"{it.diff:.2f}" for it in self.items])

    def to_dict(self) -> dict[str, Any]:
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
    anomaly_score: float | None = None
    detections: list[DetectionItem] = None  # type: ignore
    missing_items: list[str] = None  # type: ignore
    original_image_path: str = ""
    preprocessed_image_path: str = ""
    annotated_path: str = ""
    heatmap_path: str = ""
    cropped_paths: list[str] = None  # type: ignore
    color_check: ColorCheckResult | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "product": self.product,
            "area": self.area,
            "inference_type": self.inference_type,
            "ckpt_path": self.ckpt_path,
            "anomaly_score": (
                self.anomaly_score if self.anomaly_score is not None else ""
            ),
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
