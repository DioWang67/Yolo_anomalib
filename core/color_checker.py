# -*- coding: utf-8 -*-
"""color_checker.py

提供 LED 顏色檢測的簡易介面。此模組自原始 ``led_qc_enhanced.py``
抽取與推理相關的資料結構與函式，移除 CLI、批次處理與模型建置等不必要內容。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, List
import math

try:  # numpy 可能不存在，測試環境將改以純 Python 版本運作
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


# -------------------- 資料結構 --------------------
@dataclass
class EnhancedDetectionResult:
    """單張影像的檢測結果。"""

    diff: float
    is_ok: bool


# 為了保持與原始程式一致，提供別名
DetectionResult = EnhancedDetectionResult


class EnhancedReferenceModel:
    """儲存參考顏色特徵與允許的誤差。"""

    def __init__(self, mean_bgr: Iterable[float], threshold: float) -> None:
        self.mean_bgr = list(mean_bgr)
        self.threshold = threshold

    @classmethod
    def from_json(cls, model_path: Any) -> "EnhancedReferenceModel":
        """從 JSON 檔載入模型。"""
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mean = data["mean_bgr"]
        threshold = float(data.get("threshold", 10.0))
        return cls(mean, threshold)


# -------------------- 特徵計算與推理 --------------------
def compute_enhanced_features(image_bgr) -> List[float]:
    """計算圖像的平均 BGR 特徵。"""
    if np is not None and isinstance(image_bgr, np.ndarray):
        if image_bgr.size == 0:
            raise ValueError("輸入圖像為空")
        mean = image_bgr.mean(axis=(0, 1))
        return mean.tolist() if hasattr(mean, "tolist") else list(mean)

    # 純 Python 陣列 (list of list)
    h = len(image_bgr)
    w = len(image_bgr[0]) if h else 0
    if h == 0 or w == 0:
        raise ValueError("輸入圖像為空")
    sum_b = sum_g = sum_r = 0.0
    for row in image_bgr:
        for b, g, r in row:
            sum_b += b
            sum_g += g
            sum_r += r
    count = h * w
    return [sum_b / count, sum_g / count, sum_r / count]


def enhanced_detect_one(image_bgr, model: EnhancedReferenceModel) -> EnhancedDetectionResult:
    """比較圖像與參考模型，回傳檢測結果。"""
    features = compute_enhanced_features(image_bgr)
    if np is not None:
        diff = float(np.linalg.norm(np.array(features) - np.array(model.mean_bgr)))
    else:  # 純 Python 計算歐氏距離
        diff = math.sqrt(sum((f - m) ** 2 for f, m in zip(features, model.mean_bgr)))
    is_ok = diff <= model.threshold
    return EnhancedDetectionResult(diff=diff, is_ok=is_ok)


# -------------------- 封裝成檢測器 --------------------
class ColorChecker:
    """顏色檢測器。"""

    def __init__(self, model_path: Any) -> None:
        self.model = EnhancedReferenceModel.from_json(model_path)

    def check(self, image_bgr) -> DetectionResult:
        """對單張 BGR 影像進行檢測。"""
        return enhanced_detect_one(image_bgr, self.model)
