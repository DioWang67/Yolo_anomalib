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
    #: Whether the item satisfies the color check overall: the measurement is
    #: trustworthy *and* it agrees with the detector class.
    is_ok: bool
    #: Whether the measurement alone cleared its own threshold, independent of
    #: whether it agrees with the detector class. ``is_ok`` collapses those two
    #: questions, but they call for opposite responses: a measurement that is
    #: untrustworthy tells us nothing, while a trustworthy one that contradicts
    #: the detector is evidence *against the detector*. Only the latter may
    #: correct ``verified_class``, so the distinction has to survive here.
    measurement_is_ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Keep the explicit field for typed/UI consumers and expose the
        # legacy alias for persisted-result compatibility.
        d["class"] = d["class_name"]
        return d


@dataclass
class ColorCheckResult:
    is_ok: bool
    items: list[ColorCheckItemResult]
    # Why the verdict looks the way it does. ``evaluated`` means every item was
    # actually measured; other values record that no measurement was possible,
    # which downstream consumers must not read as a passing result.
    status: str = "evaluated"

    def diff_string(self) -> str:
        return ";".join([f"{it.diff:.2f}" for it in self.items])

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ok": self.is_ok,
            "items": [it.to_dict() for it in self.items],
            "diff": self.diff_string(),
            "status": self.status,
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
