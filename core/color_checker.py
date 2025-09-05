# -*- coding: utf-8 -*-
"""color_checker.py

提供 LED 顏色檢測的簡易介面。此模組自原始 ``led_qc_enhanced.py``
抽取與推理相關的資料結構與函式，移除 CLI、批次處理與模型建置等不必要內容。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Dict, Iterable, List
import math
import colorsys

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
    """儲存參考顏色直方圖與統計資料。"""

    def __init__(
        self,
        mean_bgr: Iterable[float],
        hist_h: Iterable[float],
        mask_ratio: float,
        white_ratio: float,
        threshold: float,
        hist_threshold: float,
        white_threshold: float,
    ) -> None:
        self.mean_bgr = list(mean_bgr)
        self.hist_h = list(hist_h)
        self.mask_ratio = float(mask_ratio)
        self.white_ratio = float(white_ratio)
        self.threshold = float(threshold)
        self.hist_threshold = float(hist_threshold)
        self.white_threshold = float(white_threshold)

    @classmethod
    def from_json(cls, model_path: Any) -> "EnhancedReferenceModel":
        """從 JSON 檔載入模型。"""
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mean = data["mean_bgr"]
        hist = data.get("hist_h", [0.0] * 18)
        mask_ratio = float(data.get("mask_ratio", 1.0))
        white_ratio = float(data.get("white_ratio", 0.0))
        threshold = float(data.get("threshold", 10.0))
        hist_threshold = float(data.get("hist_threshold", 0.2))
        white_threshold = float(data.get("white_threshold", 0.5))
        return cls(
            mean,
            hist,
            mask_ratio,
            white_ratio,
            threshold,
            hist_threshold,
            white_threshold,
        )


def compute_enhanced_features(image_bgr) -> Dict[str, Any]:
    """計算平均 BGR、HSV 直方圖、遮罩比例與白光比例。"""
    # 統一取得像素陣列
    if np is not None and isinstance(image_bgr, np.ndarray):
        if image_bgr.size == 0:
            raise ValueError("輸入圖像為空")
        pixels = image_bgr.reshape(-1, 3).tolist()
        mean = image_bgr.mean(axis=(0, 1))
        mean_list = mean.tolist() if hasattr(mean, "tolist") else list(mean)
    else:
        h = len(image_bgr)
        w = len(image_bgr[0]) if h else 0
        if h == 0 or w == 0:
            raise ValueError("輸入圖像為空")
        pixels = []
        sum_b = sum_g = sum_r = 0.0
        for row in image_bgr:
            for b, g, r in row:
                pixels.append([b, g, r])
                sum_b += b
                sum_g += g
                sum_r += r
        count = h * w
        mean_list = [sum_b / count, sum_g / count, sum_r / count]

    # 直方圖與統計
    hist_bins = 18
    hist = [0] * hist_bins
    mask_count = 0
    white_count = 0
    for b, g, r in pixels:
        r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
        h_val, s_val, v_val = colorsys.rgb_to_hsv(r_n, g_n, b_n)
        v = v_val * 255.0
        s = s_val * 255.0
        if v > 30:  # 遮罩：排除過暗像素
            mask_count += 1
            h_deg = h_val * 360.0
            bin_idx = int(h_deg // (360 / hist_bins)) % hist_bins
            hist[bin_idx] += 1
        if s < 30 and v > 200:  # 白光判斷條件
            white_count += 1

    total_pixels = len(pixels)
    mask_ratio = mask_count / total_pixels if total_pixels else 0.0
    white_ratio = white_count / total_pixels if total_pixels else 0.0
    if mask_count > 0:
        hist = [h / mask_count for h in hist]

    return {
        "mean_bgr": mean_list,
        "hist_h": hist,
        "mask_ratio": mask_ratio,
        "white_ratio": white_ratio,
    }


def enhanced_detect_one(image_bgr, model: EnhancedReferenceModel) -> EnhancedDetectionResult:
    """比較圖像與參考模型，回傳檢測結果。"""
    features = compute_enhanced_features(image_bgr)
    feat_mean = features["mean_bgr"]
    feat_hist = features["hist_h"]
    # 顏色距離
    if np is not None:
        diff_color = float(
            np.linalg.norm(np.array(feat_mean) - np.array(model.mean_bgr))
        )
    else:
        diff_color = math.sqrt(
            sum((f - m) ** 2 for f, m in zip(feat_mean, model.mean_bgr))
        )
    # 直方圖差異 (L1)
    diff_hist = sum(abs(f - m) for f, m in zip(feat_hist, model.hist_h))
    diff = diff_color + diff_hist
    is_ok = (
        diff_color <= model.threshold
        and diff_hist <= model.hist_threshold
        and features["white_ratio"] <= model.white_threshold
    )
    return EnhancedDetectionResult(diff=diff, is_ok=is_ok)


# -------------------- 封裝成檢測器 --------------------
class ColorChecker:
    """顏色檢測器。"""

    def __init__(self, model_path: Any) -> None:
        self.model = EnhancedReferenceModel.from_json(model_path)

    def check(self, image_bgr) -> DetectionResult:
        """對單張 BGR 影像進行檢測。"""
        return enhanced_detect_one(image_bgr, self.model)
