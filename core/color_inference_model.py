import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from core.base_model import BaseInferenceModel


@dataclass
class ColorRegion:
    """表示需要檢測的區域與其目標顏色。"""
    label: str
    bbox: Tuple[int, int, int, int]
    color: Tuple[int, int, int]


@dataclass
class EnhancedReferenceModel:
    """由 JSON 載入的顏色參考模型。"""
    regions: List[ColorRegion]
    threshold: float = 30.0

    @classmethod
    def from_json(cls, path: str) -> "EnhancedReferenceModel":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        threshold = data.get("threshold", 30.0)
        regions = []
        for r in data.get("regions", []):
            label = r.get("label") or r.get("name", "")
            bbox = tuple(r.get("bbox", (0, 0, 0, 0)))
            color = tuple(r.get("color", (0, 0, 0)))
            regions.append(ColorRegion(label, bbox, color))
        return cls(regions=regions, threshold=threshold)


def _mean_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    x1, y1, x2, y2 = bbox
    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return 0, 0, 0
    mean = region.mean(axis=(0, 1))
    return int(mean[0]), int(mean[1]), int(mean[2])


def _color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    return float(np.linalg.norm(np.array(c1, dtype=float) - np.array(c2, dtype=float)))


def enhanced_detect_one(image: np.ndarray, model: EnhancedReferenceModel) -> Dict[str, Any]:
    detections = []
    for region in model.regions:
        mean_color = _mean_color(image, region.bbox)
        diff = _color_distance(mean_color, region.color)
        detections.append({
            "label": region.label,
            "bbox": list(region.bbox),
            "mean_color": mean_color,
            "target_color": region.color,
            "color_diff": diff,
        })
    status = "PASS" if all(d["color_diff"] <= model.threshold for d in detections) else "FAIL"
    return {"detections": detections, "status": status, "result_frame": image}


class ColorInferenceModel(BaseInferenceModel):
    def __init__(self, config):
        super().__init__(config)
        self.reference_model: Optional[EnhancedReferenceModel] = None

    def initialize(self, product: str = None, area: str = None) -> bool:
        try:
            model_path = self.config.color_model_path
            self.logger.logger.info(f"正在載入顏色模型: {model_path}")
            self.reference_model = EnhancedReferenceModel.from_json(model_path)
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.logger.error(f"顏色模型初始化失敗: {e}")
            return False

    def infer(self, image: np.ndarray, product: str, area: str, output_path: str = None) -> Dict[str, Any]:
        if not self.is_initialized or self.reference_model is None:
            raise RuntimeError("顏色模型未初始化")
        try:
            result = enhanced_detect_one(image, self.reference_model)
            return {
                "inference_type": "color",
                "status": result["status"],
                "detections": result["detections"],
                "processed_image": image,
                "result_frame": result.get("result_frame", image),
                "expected_items": [],
            }
        except Exception as e:
            self.logger.logger.error(f"顏色推理失敗: {e}")
            raise
