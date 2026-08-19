"""
完整改進版 LED 顏色檢測程式
可直接替換原 color_verifier.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Default sampling thresholds
DEFAULT_SAT_THRESHOLD = 20.0
DEFAULT_EDGE_MARGIN = 0.12
DEFAULT_MIN_VALID_PIXELS = 40

# Black detection thresholds
BLACK_S_THRESHOLD = 50.0
BLACK_V_THRESHOLD = 80.0
BLACK_MIN_COVERAGE = 0.6
# Median statistics are less sensitive to highlights, so require a stricter
# fraction of the black thresholds before accepting that signal alone.
BLACK_MEDIAN_STRICTNESS_RATIO = 0.8

# Yellow detection
YELLOW_H_RANGE = (20, 35)
YELLOW_S_MIN = 80
YELLOW_V_MIN = 150

ORANGE_RED_TIE_MARGIN = 0.15
GREEN_DOMINANCE_RATIO = 0.3
MIN_HSV_MATCH_RATIO = 0.01
COLOR_CONF_THRESHOLDS = {
    "Black": 0.45,
    "Yellow": 0.20,
    "Orange": 0.25,
    "Red": 0.25,
    "Green": 0.30,
}
CANONICAL_COLOR_NAMES = {
    name.casefold(): name for name in COLOR_CONF_THRESHOLDS
}


@dataclass
class ColorRange:
    name: str
    hsv_min: np.ndarray
    hsv_max: np.ndarray
    lab_min: np.ndarray
    lab_max: np.ndarray
    hsv_mean: np.ndarray | None = None
    lab_mean: np.ndarray | None = None
    coverage_mean: float | None = None
    hsv_p10: np.ndarray | None = None
    hsv_p90: np.ndarray | None = None
    lab_p10: np.ndarray | None = None
    lab_p90: np.ndarray | None = None


@dataclass
class ColorDecision:
    image: Path
    predicted_color: str
    expected_color: str | None
    confidence: float
    status: str
    ratios: dict[str, float]
    debug_info: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        result = {
            "image": str(self.image),
            "predicted_color": self.predicted_color,
            "expected_color": self.expected_color,
            "confidence": self.confidence,
            "status": self.status,
            "match": (self.predicted_color == self.expected_color) if self.expected_color else None,
            "ratios": self.ratios,
        }
        if self.debug_info:
            result["debug"] = self.debug_info
        return result


@dataclass
class DecisionContext:
    ratios: dict[str, float]
    debug_info: dict[str, object]
    hsv_img: np.ndarray
    lab_img: np.ndarray
    edge_margin: float
    sat_threshold: float


DecisionRule = Callable[[str, float, "DecisionContext"], tuple[str, float] | None]


class VerificationReportWriteError(OSError):
    """Raised when color analysis succeeds but a requested report cannot be written."""


# ============= 新增: 核心改進函數 =============

def _validate_evaluation_options(
    *,
    edge_margin: float,
    sat_threshold: float,
    min_valid_pixels: int,
) -> None:
    if not 0.0 <= edge_margin < 0.5:
        raise ValueError("edge_margin must be in the range [0.0, 0.5).")
    if not 0.0 <= sat_threshold <= 255.0:
        raise ValueError("sat_threshold must be in the range [0.0, 255.0].")
    if min_valid_pixels < 1:
        raise ValueError("min_valid_pixels must be at least 1.")


def _center_slices(shape: Sequence[int], margin_ratio: float) -> tuple[slice, slice]:
    """Return per-axis center slices while preserving rectangular image geometry."""
    if len(shape) < 2:
        raise ValueError("An image must have at least two dimensions.")
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("margin_ratio must be in the range [0.0, 0.5).")

    height, width = int(shape[0]), int(shape[1])
    margin_y = int(height * margin_ratio)
    margin_x = int(width * margin_ratio)
    return slice(margin_y, height - margin_y), slice(margin_x, width - margin_x)


def _crop_center(img: np.ndarray, margin_ratio: float = DEFAULT_EDGE_MARGIN) -> np.ndarray:
    """Return the single center region used by every color decision path."""
    y_slice, x_slice = _center_slices(img.shape, margin_ratio)
    return img[y_slice, x_slice]


def _expand_center_mask(
    center_mask: np.ndarray,
    image_shape: Sequence[int],
    margin_ratio: float,
) -> np.ndarray:
    full_mask = np.zeros((int(image_shape[0]), int(image_shape[1])), dtype=bool)
    y_slice, x_slice = _center_slices(image_shape, margin_ratio)
    full_mask[y_slice, x_slice] = center_mask
    return full_mask


def _clamp_confidence(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _hsv_color_mask(
    hsv_vals: np.ndarray,
    color_range: ColorRange,
) -> np.ndarray:
    h_vals = hsv_vals[:, 0]
    s_vals = hsv_vals[:, 1]
    v_vals = hsv_vals[:, 2]

    hue_min = float(color_range.hsv_min[0])
    hue_max = float(color_range.hsv_max[0])
    hue_mask = (h_vals >= hue_min) & (h_vals <= hue_max)
    if hue_min > hue_max:
        hue_mask = (h_vals >= hue_min) | (h_vals <= hue_max)
    return (
        hue_mask
        & (s_vals >= color_range.hsv_min[1])
        & (s_vals <= color_range.hsv_max[1])
        & (v_vals >= color_range.hsv_min[2])
        & (v_vals <= color_range.hsv_max[2])
    )


def _lab_color_mask(lab_vals: np.ndarray, color_range: ColorRange) -> np.ndarray:
    return (
        (lab_vals[:, 0] >= color_range.lab_min[0])
        & (lab_vals[:, 0] <= color_range.lab_max[0])
        & (lab_vals[:, 1] >= color_range.lab_min[1])
        & (lab_vals[:, 1] <= color_range.lab_max[1])
        & (lab_vals[:, 2] >= color_range.lab_min[2])
        & (lab_vals[:, 2] <= color_range.lab_max[2])
    )

def circular_hue_distance(h1: float, h2: float) -> float:
    """計算色相的循環距離 (0-180 度)"""
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def improved_match_ratio(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    color_range: ColorRange,
    color_name: str
) -> tuple[float, dict[str, float]]:
    """改進的匹配比例計算"""
    if hsv_vals.size == 0 or lab_vals.size == 0:
        return 0.0, {}

    debug = {}
    h_vals = hsv_vals[:, 0]
    h_mask = _hsv_color_mask(hsv_vals, color_range)

    hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
    debug["hsv_ratio"] = hsv_ratio
    if hsv_ratio < MIN_HSV_MATCH_RATIO:
        debug["hsv_gate_rejected"] = True
        debug["final_score"] = 0.0
        return 0.0, debug

    # LAB 匹配
    lab_mask = _lab_color_mask(lab_vals, color_range)
    lab_ratio = float(np.count_nonzero(lab_mask)) / len(lab_vals)
    debug["lab_ratio"] = lab_ratio

    # 色相平均值檢查
    mean_h = float(np.mean(h_vals))
    debug["mean_hue"] = mean_h

    hue_similarity = 1.0
    if color_range.hsv_mean is not None:
        expected_h = float(color_range.hsv_mean[0])
        hue_dist = circular_hue_distance(mean_h, expected_h)
        hue_similarity = np.exp(-hue_dist / 15.0)
        debug["hue_distance"] = hue_dist
        debug["hue_similarity"] = float(hue_similarity)

    # LAB 色度分析 (對 Orange/Red 重要)
    lab_chroma_similarity = 1.0
    if color_range.lab_mean is not None and color_name in ["Orange", "Red"]:
        mean_a = float(np.mean(lab_vals[:, 1]))
        mean_b = float(np.mean(lab_vals[:, 2]))
        expected_a = float(color_range.lab_mean[1])
        expected_b = float(color_range.lab_mean[2])

        lab_chroma_dist = np.sqrt((mean_a - expected_a)**2 + (mean_b - expected_b)**2)
        lab_chroma_similarity = np.exp(-lab_chroma_dist / 20.0)

        debug["lab_chroma_dist"] = float(lab_chroma_dist)
        debug["lab_chroma_similarity"] = float(lab_chroma_similarity)

    # 動態權重
    if color_name in ["Orange", "Red"]:
        weights = {"hsv": 0.35, "lab": 0.25, "hue_sim": 0.25, "lab_chroma": 0.15}
    elif color_name == "Green":
        weights = {"hsv": 0.6, "lab": 0.2, "hue_sim": 0.2, "lab_chroma": 0.0}
    elif color_name == "Yellow":
        weights = {"hsv": 0.5, "lab": 0.2, "hue_sim": 0.3, "lab_chroma": 0.0}
    else:
        weights = {"hsv": 0.5, "lab": 0.3, "hue_sim": 0.2, "lab_chroma": 0.0}

    final_score = (
        hsv_ratio * weights["hsv"] +
        lab_ratio * weights["lab"] +
        hue_similarity * weights["hue_sim"] +
        lab_chroma_similarity * weights["lab_chroma"]
    )

    final_score = _clamp_confidence(final_score)
    debug["final_score"] = final_score
    return final_score, debug

def separate_orange_red_improved(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    orange_score: float,
    red_score: float
) -> tuple[str, float, dict]:
    """改進的 Orange vs Red 分離"""
    if len(hsv_vals) == 0:
        return (
            "Red" if red_score >= orange_score else "Orange",
            _clamp_confidence(max(red_score, orange_score)),
            {},
        )

    debug = {}
    hue_vals = hsv_vals[:, 0]

    # 色相分布
    orange_core = np.sum((hue_vals >= 8) & (hue_vals <= 16))
    red_core = np.sum((hue_vals <= 5) | (hue_vals >= 175))

    orange_hue_ratio = orange_core / len(hue_vals)
    red_hue_ratio = red_core / len(hue_vals)

    debug["orange_hue_ratio"] = float(orange_hue_ratio)
    debug["red_hue_ratio"] = float(red_hue_ratio)

    # LAB a*/b* 分析
    mean_a = float(np.mean(lab_vals[:, 1]))
    mean_b = float(np.mean(lab_vals[:, 2]))
    ab_ratio = mean_b / max(mean_a, 1.0)

    debug["mean_a"] = mean_a
    debug["mean_b"] = mean_b
    debug["ab_ratio"] = float(ab_ratio)

    # 判斷邏輯
    if ab_ratio > 1.05:
        lab_vote = "Orange"
    elif ab_ratio < 0.90:
        lab_vote = "Red"
    else:
        lab_vote = "Unclear"

    hue_vote = "Orange" if orange_hue_ratio > red_hue_ratio * 1.2 else \
               "Red" if red_hue_ratio > orange_hue_ratio * 1.2 else "Unclear"

    debug["lab_vote"] = lab_vote
    debug["hue_vote"] = hue_vote

    # 最終決策
    if hue_vote == lab_vote and hue_vote != "Unclear":
        predicted = hue_vote
        confidence = max(orange_score, red_score) * 1.3
    elif hue_vote != "Unclear":
        predicted = hue_vote
        confidence = (orange_score if hue_vote == "Orange" else red_score) * 1.1
    elif lab_vote != "Unclear":
        predicted = lab_vote
        confidence = (orange_score if lab_vote == "Orange" else red_score) * 1.1
    else:
        predicted = "Orange" if orange_score > red_score else "Red"
        confidence = max(orange_score, red_score) * 0.9

    debug["decision"] = predicted
    return predicted, _clamp_confidence(confidence), debug


# ============= 主要評估函數 (整合改進邏輯) =============

def _evaluate_image_improved(
    hsv_img: np.ndarray,
    lab_img: np.ndarray,
    color_ranges: dict[str, ColorRange],
    *,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
    sat_threshold: float = DEFAULT_SAT_THRESHOLD,
    min_valid_pixels: int = DEFAULT_MIN_VALID_PIXELS,
) -> tuple[dict[str, float], dict[str, np.ndarray], dict[str, object]]:
    """整合改進邏輯的圖片評估函數"""
    _validate_evaluation_options(
        edge_margin=edge_margin,
        sat_threshold=sat_threshold,
        min_valid_pixels=min_valid_pixels,
    )
    debug_info: dict[str, object] = {
        "edge_margin": edge_margin,
        "sat_threshold": sat_threshold,
        "min_valid_pixels": min_valid_pixels,
    }

    # 快速檢查黑色
    is_black, black_conf, black_mask = _is_black_image(
        hsv_img,
        BLACK_S_THRESHOLD,
        BLACK_V_THRESHOLD,
        edge_margin=edge_margin,
    )
    debug_info["is_black_detected"] = is_black
    debug_info["black_confidence"] = float(black_conf)

    if is_black and "Black" in color_ranges:
        ratios = dict.fromkeys(color_ranges.keys(), 0.0)
        ratios["Black"] = black_conf
        masks = {color: np.zeros(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
        masks["Black"] = black_mask
        debug_info["shortcut"] = "Black"
        return ratios, masks, debug_info

    # 快速檢查黃色
    is_yellow, yellow_conf, yellow_mask = _detect_yellow_special(
        hsv_img,
        edge_margin=edge_margin,
    )
    debug_info["is_yellow_detected"] = is_yellow
    debug_info["yellow_confidence"] = float(yellow_conf)

    if is_yellow and "Yellow" in color_ranges:
        ratios = dict.fromkeys(color_ranges.keys(), 0.0)
        ratios["Yellow"] = yellow_conf
        masks = {color: np.zeros(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
        masks["Yellow"] = yellow_mask
        debug_info["shortcut"] = "Yellow"
        return ratios, masks, debug_info

    center_hsv = _crop_center(hsv_img, edge_margin)
    center_lab = _crop_center(lab_img, edge_margin)

    # 過濾低飽和度
    sat_mask = center_hsv[:, :, 1] >= sat_threshold
    valid_hsv = center_hsv[sat_mask].reshape(-1, 3)
    valid_lab = center_lab[sat_mask].reshape(-1, 3)
    debug_info["valid_pixel_count"] = len(valid_hsv)

    if len(valid_hsv) < min_valid_pixels:
        ratios = dict.fromkeys(color_ranges.keys(), 0.0)
        masks = {color: np.zeros(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
        debug_info["low_saturation_fallback"] = True
        return ratios, masks, debug_info

    ratios: dict[str, float] = {}
    all_debug: dict[str, dict[str, float]] = {}
    masks: dict[str, np.ndarray] = {}
    flat_center_hsv = center_hsv.reshape(-1, 3)
    flat_center_lab = center_lab.reshape(-1, 3)
    flat_sat_mask = sat_mask.reshape(-1)

    for color_name, color_range in color_ranges.items():
        score, color_debug = improved_match_ratio(
            valid_hsv, valid_lab, color_range, color_name
        )
        ratios[color_name] = score
        all_debug[color_name] = color_debug

        center_color_mask = _hsv_color_mask(flat_center_hsv, color_range)
        if color_name != "Black":
            center_color_mask &= flat_sat_mask
        center_color_mask &= _lab_color_mask(flat_center_lab, color_range)
        masks[color_name] = _expand_center_mask(
            center_color_mask.reshape(center_hsv.shape[:2]),
            hsv_img.shape,
            edge_margin,
        )

    debug_info["color_details"] = all_debug
    return ratios, masks, debug_info


def _is_black_image(
    hsv_img: np.ndarray,
    s_thresh: float,
    v_thresh: float,
    *,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> tuple[bool, float, np.ndarray]:
    """檢測是否為黑色圖片"""
    center_region = _crop_center(hsv_img, edge_margin)

    mean_s = float(np.mean(center_region[:, :, 1]))
    mean_v = float(np.mean(center_region[:, :, 2]))
    median_s = float(np.median(center_region[:, :, 1]))
    median_v = float(np.median(center_region[:, :, 2]))

    black_mask = (center_region[:, :, 1] < s_thresh) & (center_region[:, :, 2] < v_thresh)
    black_coverage = float(np.count_nonzero(black_mask)) / black_mask.size

    is_black = (
        (mean_s < s_thresh and mean_v < v_thresh) or
        (
            median_s < s_thresh * BLACK_MEDIAN_STRICTNESS_RATIO
            and median_v < v_thresh * BLACK_MEDIAN_STRICTNESS_RATIO
        ) or
        (black_coverage > BLACK_MIN_COVERAGE)
    )

    confidence = black_coverage if is_black else 0.0
    full_mask = _expand_center_mask(black_mask, hsv_img.shape, edge_margin)
    return is_black, confidence, full_mask


def _detect_yellow_special(
    hsv_img: np.ndarray,
    *,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> tuple[bool, float, np.ndarray]:
    """快速檢測黃色"""
    center = _crop_center(hsv_img, edge_margin)

    h_vals = center[:, :, 0]
    s_vals = center[:, :, 1]
    v_vals = center[:, :, 2]

    yellow_mask_primary = (
        (h_vals >= YELLOW_H_RANGE[0]) &
        (h_vals <= YELLOW_H_RANGE[1]) &
        (s_vals >= YELLOW_S_MIN) &
        (v_vals >= YELLOW_V_MIN)
    )

    orange_like_mask = (h_vals < YELLOW_H_RANGE[0]) & (h_vals > 5) & (s_vals > 100)
    yellow_mask_secondary = (
        (h_vals >= 18) & (h_vals <= 38) &
        (s_vals >= 60) & (v_vals >= 180) &
        ~orange_like_mask
    )

    yellow_mask = yellow_mask_primary | yellow_mask_secondary
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / yellow_mask.size

    orange_ratio = float(np.count_nonzero(orange_like_mask)) / orange_like_mask.size

    is_yellow = (yellow_ratio > 0.25) and (yellow_ratio > orange_ratio * 1.3)
    full_mask = _expand_center_mask(yellow_mask, hsv_img.shape, edge_margin)
    return is_yellow, yellow_ratio, full_mask

def _initial_prediction(ratios: dict[str, float]) -> tuple[str, float]:
    if not ratios:
        raise ValueError("No ratios provided for prediction.")
    predicted_color, confidence = max(ratios.items(), key=lambda item: item[1])
    if confidence <= 0.0:
        return "Unknown", 0.0
    return predicted_color, confidence


def _apply_color_rules(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> tuple[str, float]:
    for rule in _COLOR_RULES:
        result = rule(predicted_color, confidence, context)
        if result is not None:
            predicted_color, confidence = result
    return predicted_color, _clamp_confidence(confidence)


def _rule_orange_red_tiebreak(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> tuple[str, float] | None:
    ratios = context.ratios
    if (
        "Orange" not in ratios
        or "Red" not in ratios
        or predicted_color not in {"Orange", "Red"}
        or abs(ratios["Orange"] - ratios["Red"]) >= ORANGE_RED_TIE_MARGIN
    ):
        return None

    center_hsv = _crop_center(context.hsv_img, context.edge_margin)
    center_lab = _crop_center(context.lab_img, context.edge_margin)

    if center_hsv.size == 0 or center_lab.size == 0:
        return None

    flat_hsv = center_hsv.reshape(-1, 3)
    flat_lab = center_lab.reshape(-1, 3)

    if flat_hsv.size == 0 or flat_lab.size == 0:
        return None

    sat_mask = flat_hsv[:, 1] >= context.sat_threshold
    valid_hsv = flat_hsv[sat_mask]
    valid_lab = flat_lab[sat_mask]

    if len(valid_hsv) == 0 or len(valid_lab) == 0:
        return None

    new_color, new_conf, sep_debug = separate_orange_red_improved(
        valid_hsv, valid_lab, ratios["Orange"], ratios["Red"]
    )
    context.debug_info["orange_red_separated"] = True
    context.debug_info["separation_details"] = sep_debug
    return new_color, new_conf


def _rule_green_correction(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> tuple[str, float] | None:
    if predicted_color != "Red" or "Green" not in context.ratios:
        return None

    center_hsv = _crop_center(context.hsv_img, context.edge_margin)
    if center_hsv.size == 0:
        return None

    h_vals = center_hsv[:, :, 0]
    s_vals = center_hsv[:, :, 1]
    total_pixels = h_vals.size
    if total_pixels == 0:
        return None

    green_pixels = np.sum(
        (h_vals >= 70)
        & (h_vals <= 100)
        & (s_vals >= context.sat_threshold)
    )
    green_ratio = green_pixels / total_pixels

    if green_ratio > GREEN_DOMINANCE_RATIO:
        context.debug_info["green_correction"] = True
        return "Green", green_ratio
    return None


_COLOR_RULES: list[DecisionRule] = [
    _rule_orange_red_tiebreak,
    _rule_green_correction,
]


def _validate_confidence_threshold(default_threshold: float) -> None:
    if not np.isfinite(default_threshold) or not 0.0 <= default_threshold <= 1.0:
        raise ValueError("ratio_threshold must be in the range [0.0, 1.0].")


def _confidence_threshold_for(color: str, default_threshold: float) -> float:
    _validate_confidence_threshold(default_threshold)
    return max(default_threshold, COLOR_CONF_THRESHOLDS.get(color, default_threshold))

# ============= 載入與驗證函數 =============

def _margin_vector(margin: Sequence[float] | float) -> np.ndarray:
    if isinstance(margin, Sequence) and not isinstance(margin, (str, bytes)):
        values = [float(value) for value in margin]
    else:
        values = [float(margin)]
    if len(values) == 1:
        values *= 3
    if len(values) != 3:
        raise ValueError("margin must contain 1 or 3 values.")
    return np.asarray(values, dtype=np.float32)


def _canonical_color_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("color_stats contains an empty or non-string color name.")
    name = value.strip()
    return CANONICAL_COLOR_NAMES.get(name.casefold(), name)


def _required_stat_array(stats: Mapping[str, object], key: str) -> np.ndarray:
    if key not in stats:
        raise ValueError(f"color_stats entry is missing {key}.")
    value = np.asarray(stats[key], dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"color_stats {key} must contain three finite values.")
    return value


def load_color_ranges(
    stats_path: Path,
    hsv_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
    lab_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
) -> dict[str, ColorRange]:
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("color_stats summary missing.")

    hsv_margin_vec = _margin_vector(hsv_margin)
    lab_margin_vec = _margin_vector(lab_margin)
    ranges: dict[str, ColorRange] = {}
    seen_names: set[str] = set()
    for color, stats in summary.items():
        canonical_color = _canonical_color_name(color)
        collision_key = canonical_color.casefold()
        if collision_key in seen_names:
            raise ValueError(f"color_stats contains duplicate color name: {canonical_color}")
        if not isinstance(stats, Mapping):
            raise ValueError(f"color_stats entry for {canonical_color} must be an object.")
        seen_names.add(collision_key)

        hsv_min = _required_stat_array(stats, "hsv_min") - hsv_margin_vec
        hsv_max = _required_stat_array(stats, "hsv_max") + hsv_margin_vec
        lab_min = _required_stat_array(stats, "lab_min") - lab_margin_vec
        lab_max = _required_stat_array(stats, "lab_max") + lab_margin_vec

        ranges[canonical_color] = ColorRange(
            canonical_color,
            hsv_min,
            hsv_max,
            lab_min,
            lab_max,
            hsv_mean=_optional_stat_array(stats, "hsv_mean"),
            lab_mean=_optional_stat_array(stats, "lab_mean"),
            coverage_mean=float(stats["coverage_mean"]) if "coverage_mean" in stats else None,
            hsv_p10=_optional_stat_array(stats, "hsv_p10"),
            hsv_p90=_optional_stat_array(stats, "hsv_p90"),
            lab_p10=_optional_stat_array(stats, "lab_p10"),
            lab_p90=_optional_stat_array(stats, "lab_p90"),
        )
    return ranges


def _optional_stat_array(stats: Mapping[str, object], key: str) -> np.ndarray | None:
    if key not in stats:
        return None
    value = np.asarray(stats[key], dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"color_stats {key} must contain three finite values.")
    return value


def _load_expected_map(path: Path | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not path:
        return lookup
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("items", [])
        for row in rows:
            if isinstance(row, dict) and row.get("image") and row.get("color"):
                lookup[Path(row["image"]).name.lower()] = str(row["color"])
    else:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                img = row.get("image") or row.get("file") or row.get("path")
                color = row.get("color") or row.get("label")
                if img and color:
                    lookup[Path(img).name.lower()] = str(color)
    return lookup


def _resolve_expected_color(
    image_path: Path,
    lookup: MutableMapping[str, str],
    known_colors: Iterable[str],
    infer_from_name: bool,
) -> str | None:
    name = image_path.name.lower()
    known_by_name = {color.casefold(): color for color in known_colors}
    if name in lookup:
        expected = lookup[name].strip()
        return known_by_name.get(expected.casefold(), expected)
    if infer_from_name:
        for color in known_by_name.values():
            if color.lower() in name:
                return color
    return None


# ============= 主驗證函數 =============

def verify_directory(
    input_dir: Path,
    color_stats: Path,
    *,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    recursive: bool = False,
    expected_map: Path | None = None,
    infer_expected_from_name: bool = True,
    hsv_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
    lab_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
    ratio_threshold: float = 0.35,
    sat_threshold: float = DEFAULT_SAT_THRESHOLD,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
    min_valid_pixels: int = DEFAULT_MIN_VALID_PIXELS,
    debug_plot: bool = False,
    debug_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> tuple[dict[str, object], list[ColorDecision]]:
    logger = logger or logging.getLogger(__name__)
    _validate_evaluation_options(
        edge_margin=edge_margin,
        sat_threshold=sat_threshold,
        min_valid_pixels=min_valid_pixels,
    )
    _validate_confidence_threshold(ratio_threshold)
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    ranges = load_color_ranges(color_stats.resolve(), hsv_margin, lab_margin)
    expected_lookup = _load_expected_map(expected_map)

    debug_root: Path | None = None
    if debug_plot:
        base = debug_dir or (output_json.parent if output_json else input_dir)
        debug_root = (Path(base) / "color_debug").resolve()
        debug_root.mkdir(parents=True, exist_ok=True)

    def iter_images() -> Iterable[Path]:
        if recursive:
            yield from (p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS)
        else:
            yield from (p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_FORMATS)

    images = sorted(iter_images())
    if not images:
        raise FileNotFoundError(f"No images with suffix {SUPPORTED_FORMATS} in {input_dir}")

    results: list[ColorDecision] = []
    counters = {"total": 0, "matched": 0, "mismatched": 0, "predicted_only": 0, "low_confidence": 0}

    for image_path in images:
        counters["total"] += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Unable to read %s, skipping.", image_path)
            continue

        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        ratios, masks, debug_info = _evaluate_image_improved(
            hsv_img,
            lab_img,
            ranges,
            edge_margin=edge_margin,
            sat_threshold=sat_threshold,
            min_valid_pixels=min_valid_pixels,
        )

        predicted_color, confidence = _initial_prediction(ratios)
        context = DecisionContext(
            ratios=ratios,
            debug_info=debug_info,
            hsv_img=hsv_img,
            lab_img=lab_img,
            edge_margin=edge_margin,
            sat_threshold=sat_threshold,
        )
        predicted_color, confidence = _apply_color_rules(predicted_color, confidence, context)

        expected = _resolve_expected_color(image_path, expected_lookup, ranges.keys(), infer_expected_from_name)
        base_threshold = _confidence_threshold_for(predicted_color, ratio_threshold)
        if confidence < base_threshold:
            status = "low_confidence"
            counters["low_confidence"] += 1
        elif expected is None:
            status = "predicted_only"
            counters["predicted_only"] += 1
        elif expected != predicted_color:
            status = "mismatch"
            counters["mismatched"] += 1
        else:
            status = "match"
            counters["matched"] += 1

        decision = ColorDecision(
            image=image_path.relative_to(input_dir),
            predicted_color=predicted_color,
            expected_color=expected,
            confidence=confidence,
            status=status,
            ratios=ratios,
            debug_info=debug_info,
        )
        results.append(decision)

        if debug_root:
            mask = masks.get(predicted_color, np.zeros(image.shape[:2], dtype=bool))
            visualize_debug(
                image_bgr=image,
                predicted_color=predicted_color,
                confidence=confidence,
                ratios=ratios,
                mask=mask,
                output_path=debug_root / f"{image_path.stem}_analysis.png",
                debug_info=debug_info,
                edge_margin=edge_margin,
            )

    summary = {
        "input_dir": str(input_dir),
        "color_stats": str(color_stats),
        "total_images": counters["total"],
        "matched": counters["matched"],
        "mismatched": counters["mismatched"],
        "predicted_only": counters["predicted_only"],
        "low_confidence": counters["low_confidence"],
        "accuracy": f"{counters['matched'] / max(counters['matched'] + counters['mismatched'], 1) * 100:.2f}%",
        "debug_output_dir": str(debug_root) if debug_root else None,
    }

    report = {"summary": summary, "items": [item.to_dict() for item in results]}
    if output_json:
        try:
            output_json.parent.mkdir(parents=True, exist_ok=True)

            def _safe_convert(obj):
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _safe_convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_safe_convert(v) for v in obj]
                return obj

            report = _safe_convert(report)
            output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            raise VerificationReportWriteError(f"Failed to write JSON report: {output_json}") from exc
    if output_csv:
        try:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["image", "predicted_color", "expected_color", "status", "confidence", "match", "ratios_json"]
                )
                for item in results:
                    data = item.to_dict()
                    writer.writerow(
                        [
                            data["image"],
                            data["predicted_color"],
                            data["expected_color"],
                            data["status"],
                            f"{data['confidence']:.4f}",
                            data["match"],
                            json.dumps(data["ratios"], ensure_ascii=False),
                        ]
                    )
        except OSError as exc:
            raise VerificationReportWriteError(f"Failed to write CSV report: {output_csv}") from exc

    return summary, results


def visualize_debug(
    image_bgr: np.ndarray,
    predicted_color: str,
    confidence: float,
    ratios: dict[str, float],
    mask: np.ndarray,
    output_path: Path,
    debug_info: Mapping[str, object] | None = None,
    *,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except (ImportError, OSError) as exc:
        logging.getLogger(__name__).warning("Debug visualization unavailable: %s", exc)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_image = _crop_center(image_bgr, edge_margin)
    analysis_mask = _crop_center(mask, edge_margin)
    image_rgb = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2RGB)
    mask_uint8 = analysis_mask.astype(np.uint8) * 255 if analysis_mask.dtype != np.uint8 else analysis_mask
    mask_uint8 = mask_uint8 if mask_uint8.max() > 1 else mask_uint8 * 255
    overlay = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_rgb, 0.7, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 0.3, 0)

    hsv = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].flatten()
    sat = hsv[:, :, 1].flatten()

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    ax0 = fig.add_subplot(gs[:2, 0])
    ax0.imshow(image_rgb)
    ax0.set_title("Input ROI", fontsize=10)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[:2, 1])
    ax1.imshow(overlay)
    ax1.set_title("Mask Overlay", fontsize=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(hue, bins=30, range=(0, 180), color="orange", alpha=0.7)
    ax2.set_title("Hue Histogram", fontsize=9)
    ax2.set_xlim(0, 180)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(sat, bins=30, range=(0, 255), color="teal", alpha=0.7)
    ax3.set_title("Saturation Histogram", fontsize=9)
    ax3.set_xlim(0, 255)

    ax4 = fig.add_subplot(gs[2, :])
    colors_list = list(ratios.keys())
    values_list = list(ratios.values())
    bars = ax4.bar(range(len(ratios)), values_list, tick_label=colors_list)

    max_idx = values_list.index(max(values_list))
    bars[max_idx].set_color("red")
    bars[max_idx].set_alpha(0.8)

    ax4.set_ylim(0, 1)
    ax4.set_title("Color Confidence", fontsize=10)
    ax4.axhline(y=0.35, color="green", linestyle="--", linewidth=1, alpha=0.5, label="threshold")
    ax4.legend(fontsize=8)

    debug_text = f"Prediction: {predicted_color} (confidence={confidence:.3f})\n"
    if debug_info:
        if debug_info.get("is_black_detected"):
            debug_text += "Black shortcut: True\n"
        if debug_info.get("is_yellow_detected"):
            debug_text += "Yellow shortcut: True\n"
        if debug_info.get("orange_red_separated"):
            debug_text += "Orange/Red disambiguation triggered\n"
        if debug_info.get("green_correction"):
            debug_text += "Green correction applied\n"
        if debug_info.get("low_saturation_fallback"):
            debug_text += "Low-saturation fallback: no color accepted\n"

    fig.text(0.02, 0.02, debug_text, fontsize=8, verticalalignment="bottom",
             bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

    fig.suptitle(f"Color decision: {predicted_color}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved LED color verification")
    parser.add_argument("--input-dir", required=True, help="Directory containing inference images")
    parser.add_argument("--color-stats", required=True, help="JSON file produced by color_inspection.collect")
    parser.add_argument("--output-json", default="./reports/led_qc/color_verification_improved.json")
    parser.add_argument("--output-csv", default="./reports/led_qc/color_verification_improved.csv")
    parser.add_argument("--expected-map", help="CSV/JSON mapping from filename to expected color")
    parser.add_argument("--recursive", action="store_true", help="Scan directories recursively")
    parser.add_argument("--no-filename-expectation", action="store_true", help="Disable automatic expectation from filenames")
    parser.add_argument("--hsv-margin", type=float, nargs="*", default=[8.0, 35.0, 40.0])
    parser.add_argument("--lab-margin", type=float, nargs="*", default=[12.0, 8.0, 12.0])
    parser.add_argument("--ratio-threshold", type=float, default=0.35, help="Base confidence threshold")
    parser.add_argument(
        "--sat-threshold",
        type=float,
        default=DEFAULT_SAT_THRESHOLD,
        help="Minimum HSV saturation accepted for color matching",
    )
    parser.add_argument(
        "--edge-margin",
        type=float,
        default=DEFAULT_EDGE_MARGIN,
        help="Fraction removed from each edge before color analysis (0.0 <= value < 0.5)",
    )
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=DEFAULT_MIN_VALID_PIXELS,
        help="Minimum saturated pixels required for a color decision",
    )
    parser.add_argument("--debug-plot", action="store_true", help="Save per-image debug visualizations")
    parser.add_argument("--debug-dir", help="Directory to store debug visualizations")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting IMPROVED color verification pipeline...")
    logger.info("Input directory: %s", args.input_dir)
    logger.info("Color stats: %s", args.color_stats)

    try:
        summary, _ = verify_directory(
            input_dir=Path(args.input_dir),
            color_stats=Path(args.color_stats),
            output_json=Path(args.output_json) if args.output_json else None,
            output_csv=Path(args.output_csv) if args.output_csv else None,
            recursive=args.recursive,
            expected_map=Path(args.expected_map) if args.expected_map else None,
            infer_expected_from_name=not args.no_filename_expectation,
            hsv_margin=args.hsv_margin,
            lab_margin=args.lab_margin,
            ratio_threshold=args.ratio_threshold,
            sat_threshold=args.sat_threshold,
            edge_margin=args.edge_margin,
            min_valid_pixels=args.min_valid_pixels,
            debug_plot=args.debug_plot,
            debug_dir=Path(args.debug_dir) if args.debug_dir else None,
            logger=logger,
        )

    except FileNotFoundError as exc:
        logger.error("Color verification input was not found: %s", exc)
        return 1
    except VerificationReportWriteError as exc:
        logger.error("Color verification report could not be written: %s", exc)
        return 1
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.error("Color verification input or configuration is invalid: %s", exc)
        return 1
    except cv2.error as exc:
        logger.error("OpenCV failed while analyzing an image: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Color verification I/O failed: %s", exc)
        return 1

    logger.info("=" * 60)
    logger.info("Verification complete")
    logger.info("Total images: %s", summary["total_images"])
    logger.info("Matched: %s", summary["matched"])
    logger.info("Mismatched: %s", summary["mismatched"])
    logger.info("Accuracy: %s", summary["accuracy"])
    logger.info("Low-confidence: %s", summary["low_confidence"])
    if args.output_json:
        logger.info("JSON report: %s", args.output_json)
    if args.output_csv:
        logger.info("CSV report: %s", args.output_csv)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
