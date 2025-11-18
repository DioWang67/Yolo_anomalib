"""
完整改進版 LED 顏色檢測程式
可直接替換原 color_verifier.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import cv2
import numpy as np

SUPPORTED_FORMATS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

# Default sampling thresholds
DEFAULT_SAT_THRESHOLD = 20.0
DEFAULT_VAL_THRESHOLD = 250.0
DEFAULT_EDGE_MARGIN = 0.12
DEFAULT_CENTER_SIGMA = 0.25
DEFAULT_MIN_VALID_PIXELS = 40
DEFAULT_TOPK = 5
DEFAULT_MIN_SAT_RATIO = 0.15
DEFAULT_MAX_EDGE_RATIO = 0.25

# Black detection thresholds
BLACK_S_THRESHOLD = 50.0
BLACK_V_THRESHOLD = 80.0
BLACK_MIN_COVERAGE = 0.6

# Yellow detection
YELLOW_H_RANGE = (20, 35)
YELLOW_S_MIN = 80
YELLOW_V_MIN = 150

ORANGE_RED_TIE_MARGIN = 0.15
GREEN_DOMINANCE_RATIO = 0.3
CENTER_MARGIN_RATIO = 0.15
COLOR_CONF_THRESHOLDS = {
    "Black": 0.45,
    "Yellow": 0.20,
    "Orange": 0.25,
    "Red": 0.25,
    "Green": 0.30,
}


@dataclass
class ColorRange:
    name: str
    hsv_min: np.ndarray
    hsv_max: np.ndarray
    lab_min: np.ndarray
    lab_max: np.ndarray
    hsv_mean: Optional[np.ndarray] = None
    lab_mean: Optional[np.ndarray] = None
    coverage_mean: Optional[float] = None
    hsv_p10: Optional[np.ndarray] = None
    hsv_p90: Optional[np.ndarray] = None
    lab_p10: Optional[np.ndarray] = None
    lab_p90: Optional[np.ndarray] = None


@dataclass
class StripOptions:
    enabled: bool = False
    segments: int = 8
    orientation: str = "vertical"
    min_strip_ratio: float = 0.05
    ratio_threshold: float = 0.35
    edge_margin: float = DEFAULT_EDGE_MARGIN
    sat_threshold: float = DEFAULT_SAT_THRESHOLD
    val_threshold: float = DEFAULT_VAL_THRESHOLD
    center_bias: bool = True
    center_sigma: float = DEFAULT_CENTER_SIGMA
    min_valid_pixels: int = DEFAULT_MIN_VALID_PIXELS
    top_k: int = DEFAULT_TOPK
    min_sat_ratio: float = DEFAULT_MIN_SAT_RATIO
    max_edge_ratio: float = DEFAULT_MAX_EDGE_RATIO
    black_s_threshold: float = BLACK_S_THRESHOLD
    black_v_threshold: float = BLACK_V_THRESHOLD


@dataclass
class ColorDecision:
    image: Path
    predicted_color: str
    expected_color: Optional[str]
    confidence: float
    status: str
    ratios: Dict[str, float]
    debug_info: Dict[str, object] = None

    def to_dict(self) -> Dict[str, object]:
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
    ratios: Dict[str, float]
    debug_info: Dict[str, object]
    hsv_img: np.ndarray
    lab_img: np.ndarray


DecisionRule = Callable[[str, float, "DecisionContext"], Optional[Tuple[str, float]]]


# ============= 新增: 核心改進函數 =============

def circular_hue_distance(h1: float, h2: float) -> float:
    """計算色相的循環距離 (0-180 度)"""
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def improved_match_ratio(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    color_range: ColorRange,
    color_name: str
) -> Tuple[float, Dict[str, float]]:
    """改進的匹配比例計算"""
    if hsv_vals.size == 0 or lab_vals.size == 0:
        return 0.0, {}

    debug = {}
    h_vals = hsv_vals[:, 0]
    s_vals = hsv_vals[:, 1]
    v_vals = hsv_vals[:, 2]

    # 針對不同顏色的特殊邏輯
    if color_name == "Red":
        h_mask = ((h_vals <= 10) | (h_vals >= 170)) & \
                 (s_vals >= max(color_range.hsv_min[1], 130)) & \
                 (v_vals >= max(color_range.hsv_min[2], 80))
    elif color_name == "Orange":
        h_mask = (h_vals >= 5) & (h_vals <= 20) & \
                 (s_vals >= max(color_range.hsv_min[1], 130)) & \
                 (v_vals >= max(color_range.hsv_min[2], 100))
    elif color_name == "Yellow":
        h_mask = (h_vals >= 20) & (h_vals <= 35) & \
                 (s_vals >= 80) & (v_vals >= 150)
    elif color_name == "Green":
        h_mask = (h_vals >= 70) & (h_vals <= 100) & \
                 (s_vals >= 75) & \
                 (v_vals >= 30) & (v_vals <= 100)
    elif color_name == "Black":
        h_mask = (s_vals < BLACK_S_THRESHOLD) & (v_vals < BLACK_V_THRESHOLD)
    else:
        h_mask = (h_vals >= color_range.hsv_min[0]) & \
                 (h_vals <= color_range.hsv_max[0]) & \
                 (s_vals >= color_range.hsv_min[1]) & \
                 (s_vals <= color_range.hsv_max[1]) & \
                 (v_vals >= color_range.hsv_min[2]) & \
                 (v_vals <= color_range.hsv_max[2])

    hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)
    debug["hsv_ratio"] = hsv_ratio

    # LAB 匹配
    lab_mask = (
        (lab_vals[:, 0] >= color_range.lab_min[0]) &
        (lab_vals[:, 0] <= color_range.lab_max[0]) &
        (lab_vals[:, 1] >= color_range.lab_min[1]) &
        (lab_vals[:, 1] <= color_range.lab_max[1]) &
        (lab_vals[:, 2] >= color_range.lab_min[2]) &
        (lab_vals[:, 2] <= color_range.lab_max[2])
    )
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

    debug["final_score"] = float(final_score)
    return final_score, debug

def separate_orange_red_improved(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    orange_score: float,
    red_score: float
) -> Tuple[str, float, Dict]:
    """改進的 Orange vs Red 分離"""
    if len(hsv_vals) == 0:
        return ("Red" if red_score >= orange_score else "Orange", max(red_score, orange_score), {})

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
    return predicted, confidence, debug


# ============= 主要評估函數 (整合改進邏輯) =============

def _evaluate_image_improved(
    image: np.ndarray,
    hsv_img: np.ndarray,
    lab_img: np.ndarray,
    color_ranges: Dict[str, ColorRange],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, object]]:
    """整合改進邏輯的圖片評估函數"""
    debug_info = {}
    h, w = hsv_img.shape[:2]

    # 快速檢查黑色
    is_black, black_conf = _is_black_image(hsv_img, BLACK_S_THRESHOLD, BLACK_V_THRESHOLD)
    debug_info["is_black_detected"] = is_black
    debug_info["black_confidence"] = float(black_conf)

    if is_black and "Black" in color_ranges:
        ratios = {color: 0.0 for color in color_ranges.keys()}
        ratios["Black"] = black_conf
        masks = {color: np.zeros(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
        masks["Black"] = np.ones(hsv_img.shape[:2], dtype=bool)
        return ratios, masks, debug_info

    # 快速檢查黃色
    is_yellow, yellow_conf = _detect_yellow_special(hsv_img)
    debug_info["is_yellow_detected"] = is_yellow
    debug_info["yellow_confidence"] = float(yellow_conf)

    if is_yellow and "Yellow" in color_ranges:
        ratios = {color: 0.0 for color in color_ranges.keys()}
        ratios["Yellow"] = yellow_conf
        masks = {color: np.zeros(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
        masks["Yellow"] = np.ones(hsv_img.shape[:2], dtype=bool)
        return ratios, masks, debug_info

    # 取中心區域
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)
    center_hsv = hsv_img[margin_y:h - margin_y, margin_x:w - margin_x]
    center_lab = lab_img[margin_y:h - margin_y, margin_x:w - margin_x]

    # 過濾低飽和度
    sat_mask = center_hsv[:, :, 1] >= DEFAULT_SAT_THRESHOLD
    valid_hsv = center_hsv[sat_mask].reshape(-1, 3)
    valid_lab = center_lab[sat_mask].reshape(-1, 3)

    if len(valid_hsv) < 50:
        ratios = {color: 0.0 for color in color_ranges.keys()}
        if "Black" in ratios:
            ratios["Black"] = 0.7
        masks = {color: np.zeros(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
        return ratios, masks, debug_info

    ratios = {}
    all_debug = {}

    for color_name, color_range in color_ranges.items():
        score, color_debug = improved_match_ratio(
            valid_hsv, valid_lab, color_range, color_name
        )
        ratios[color_name] = score
        all_debug[color_name] = color_debug

    debug_info["color_details"] = all_debug

    masks = {color: np.ones(hsv_img.shape[:2], dtype=bool) for color in color_ranges.keys()}
    return ratios, masks, debug_info


def _is_black_image(hsv_img: np.ndarray, s_thresh: float, v_thresh: float) -> Tuple[bool, float]:
    """檢測是否為黑色圖片"""
    h, w = hsv_img.shape[:2]
    margin_y = int(h * 0.15)
    margin_x = int(w * 0.15)
    center_region = hsv_img[margin_y:h - margin_y, margin_x:w - margin_x]

    if center_region.size == 0:
        center_region = hsv_img

    mean_s = float(np.mean(center_region[:, :, 1]))
    mean_v = float(np.mean(center_region[:, :, 2]))
    median_s = float(np.median(center_region[:, :, 1]))
    median_v = float(np.median(center_region[:, :, 2]))

    black_mask = (center_region[:, :, 1] < s_thresh) & (center_region[:, :, 2] < v_thresh)
    black_coverage = float(np.count_nonzero(black_mask)) / black_mask.size

    is_black = (
        (mean_s < s_thresh and mean_v < v_thresh) or
        (median_s < s_thresh * 0.8 and median_v < v_thresh * 0.8) or
        (black_coverage > BLACK_MIN_COVERAGE)
    )

    confidence = black_coverage if is_black else 0.0
    return is_black, confidence


def _detect_yellow_special(hsv_img: np.ndarray) -> Tuple[bool, float]:
    """快速檢測黃色"""
    h, w = hsv_img.shape[:2]
    margin = int(min(h, w) * 0.15)
    center = hsv_img[margin:h - margin, margin:w - margin]

    if center.size == 0:
        center = hsv_img

    h_vals = center[:, :, 0]
    s_vals = center[:, :, 1]
    v_vals = center[:, :, 2]

    yellow_mask_primary = (
        (h_vals >= YELLOW_H_RANGE[0]) &
        (h_vals <= YELLOW_H_RANGE[1]) &
        (s_vals >= YELLOW_S_MIN) &
        (v_vals >= YELLOW_V_MIN)
    )

    yellow_mask_secondary = (
        (h_vals >= 18) & (h_vals <= 38) &
        (s_vals >= 60) & (v_vals >= 180)
    )

    yellow_mask = yellow_mask_primary | yellow_mask_secondary
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / yellow_mask.size

    orange_like_mask = (h_vals < 20) & (h_vals > 5) & (s_vals > 100)
    orange_ratio = float(np.count_nonzero(orange_like_mask)) / orange_like_mask.size

    is_yellow = (yellow_ratio > 0.25) and (yellow_ratio > orange_ratio * 1.3)
    return is_yellow, yellow_ratio

def _extract_center_pixels(img: np.ndarray, margin_ratio: float = CENTER_MARGIN_RATIO) -> np.ndarray:
    """Return the center crop used for rule-based decisions."""
    h, w = img.shape[:2]
    margin = int(min(h, w) * margin_ratio)
    if margin == 0:
        return img
    center = img[margin:h - margin, margin:w - margin]
    return center if center.size else img


def _initial_prediction(ratios: Dict[str, float]) -> Tuple[str, float]:
    if not ratios:
        raise ValueError("No ratios provided for prediction.")
    return max(ratios.items(), key=lambda item: item[1])


def _apply_color_rules(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> Tuple[str, float]:
    for rule in _COLOR_RULES:
        result = rule(predicted_color, confidence, context)
        if result is not None:
            predicted_color, confidence = result
    return predicted_color, confidence


def _rule_black(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> Optional[Tuple[str, float]]:
    if context.debug_info.get("is_black_detected") and "Black" in context.ratios:
        new_conf = context.debug_info.get("black_confidence", confidence)
        return "Black", new_conf
    return None


def _rule_yellow(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> Optional[Tuple[str, float]]:
    if context.debug_info.get("is_yellow_detected") and "Yellow" in context.ratios:
        new_conf = context.debug_info.get("yellow_confidence", confidence)
        return "Yellow", new_conf
    return None


def _rule_orange_red_tiebreak(
    predicted_color: str,
    confidence: float,
    context: DecisionContext,
) -> Optional[Tuple[str, float]]:
    ratios = context.ratios
    if (
        "Orange" not in ratios
        or "Red" not in ratios
        or predicted_color not in {"Orange", "Red"}
        or abs(ratios["Orange"] - ratios["Red"]) >= ORANGE_RED_TIE_MARGIN
    ):
        return None

    center_hsv = _extract_center_pixels(context.hsv_img)
    center_lab = _extract_center_pixels(context.lab_img)

    if center_hsv.size == 0 or center_lab.size == 0:
        return None

    flat_hsv = center_hsv.reshape(-1, 3)
    flat_lab = center_lab.reshape(-1, 3)

    if flat_hsv.size == 0 or flat_lab.size == 0:
        return None

    sat_mask = flat_hsv[:, 1] >= DEFAULT_SAT_THRESHOLD
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
) -> Optional[Tuple[str, float]]:
    if predicted_color != "Red" or "Green" not in context.ratios:
        return None

    center_hsv = _extract_center_pixels(context.hsv_img)
    if center_hsv.size == 0:
        return None

    h_vals = center_hsv[:, :, 0] if center_hsv.ndim == 3 else center_hsv
    total_pixels = h_vals.size
    if total_pixels == 0:
        return None

    green_pixels = np.sum((h_vals >= 70) & (h_vals <= 100))
    green_ratio = green_pixels / total_pixels

    if green_ratio > GREEN_DOMINANCE_RATIO:
        context.debug_info["green_correction"] = True
        return "Green", green_ratio
    return None


_COLOR_RULES: List[DecisionRule] = [
    _rule_black,
    _rule_yellow,
    _rule_orange_red_tiebreak,
    _rule_green_correction,
]


def _confidence_threshold_for(color: str, default_threshold: float) -> float:
    return COLOR_CONF_THRESHOLDS.get(color, default_threshold)

# ============= 載入與驗證函數 =============

def _margin_vector(margin: Sequence[float] | float, default: float) -> np.ndarray:
    if isinstance(margin, Sequence) and not isinstance(margin, (str, bytes)):
        values = list(margin)
        if len(values) == 1:
            values *= 3
    else:
        values = [float(margin if not isinstance(margin, Sequence) else margin[0])] * 3
    if len(values) != 3:
        raise ValueError("margin must contain 1 or 3 values.")
    return np.asarray(values, dtype=np.float32)


def load_color_ranges(
    stats_path: Path,
    hsv_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
    lab_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
) -> Dict[str, ColorRange]:
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("color_stats summary missing.")

    hsv_margin_vec = _margin_vector(hsv_margin, 0.0)
    lab_margin_vec = _margin_vector(lab_margin, 0.0)
    ranges: Dict[str, ColorRange] = {}
    for color, stats in summary.items():
        hsv_min = np.asarray(stats["hsv_min"], dtype=np.float32) - hsv_margin_vec
        hsv_max = np.asarray(stats["hsv_max"], dtype=np.float32) + hsv_margin_vec
        lab_min = np.asarray(stats["lab_min"], dtype=np.float32) - lab_margin_vec
        lab_max = np.asarray(stats["lab_max"], dtype=np.float32) + lab_margin_vec

        def _optional_array(key: str) -> Optional[np.ndarray]:
            if key not in stats:
                return None
            return np.asarray(stats[key], dtype=np.float32)

        ranges[color] = ColorRange(
            color,
            hsv_min,
            hsv_max,
            lab_min,
            lab_max,
            hsv_mean=_optional_array("hsv_mean"),
            lab_mean=_optional_array("lab_mean"),
            coverage_mean=float(stats["coverage_mean"]) if "coverage_mean" in stats else None,
            hsv_p10=_optional_array("hsv_p10"),
            hsv_p90=_optional_array("hsv_p90"),
            lab_p10=_optional_array("lab_p10"),
            lab_p90=_optional_array("lab_p90"),
        )
    return ranges


def _load_expected_map(path: Optional[Path]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
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
) -> Optional[str]:
    name = image_path.name.lower()
    if name in lookup:
        return lookup[name]
    if infer_from_name:
        for color in known_colors:
            if color.lower() in name:
                return color
    return None


# ============= 主驗證函數 =============

def verify_directory(
    input_dir: Path,
    color_stats: Path,
    *,
    output_json: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    recursive: bool = False,
    expected_map: Optional[Path] = None,
    infer_expected_from_name: bool = True,
    hsv_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
    lab_margin: Sequence[float] | float = (0.0, 0.0, 0.0),
    ratio_threshold: float = 0.35,
    debug_plot: bool = False,
    debug_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Tuple[Dict[str, object], List[ColorDecision]]:
    logger = logger or logging.getLogger(__name__)
    input_dir = input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    ranges = load_color_ranges(color_stats.resolve(), hsv_margin, lab_margin)
    expected_lookup = _load_expected_map(expected_map)

    debug_root: Optional[Path] = None
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

    results: List[ColorDecision] = []
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
            image, hsv_img, lab_img, ranges
        )
        if debug_info is None:
            debug_info = {}

        predicted_color, confidence = _initial_prediction(ratios)
        context = DecisionContext(
            ratios=ratios,
            debug_info=debug_info,
            hsv_img=hsv_img,
            lab_img=lab_img,
        )
        predicted_color, confidence = _apply_color_rules(predicted_color, confidence, context)

        expected = _resolve_expected_color(image_path, expected_lookup, ranges.keys(), infer_expected_from_name)
        status = "match"
        if expected is None:
            status = "predicted_only"
            counters["predicted_only"] += 1
        elif expected != predicted_color:
            status = "mismatch"
            counters["mismatched"] += 1
        else:
            counters["matched"] += 1

        base_threshold = _confidence_threshold_for(predicted_color, ratio_threshold)
        if confidence < base_threshold:
            status = "low_confidence" if status == "match" else f"{status}_low_conf"
            counters["low_confidence"] += 1

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
            mask = masks.get(predicted_color, np.ones(image.shape[:2], dtype=bool))
            visualize_debug(
                image_bgr=image,
                predicted_color=predicted_color,
                confidence=confidence,
                ratios=ratios,
                mask=mask,
                output_path=debug_root / f"{image_path.stem}_analysis.png",
                debug_info=debug_info,
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
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "predicted_color", "expected_color", "status", "confidence", "match", "ratios_json"])
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

    return summary, results


def visualize_debug(
    image_bgr: np.ndarray,
    predicted_color: str,
    confidence: float,
    ratios: Dict[str, float],
    mask: np.ndarray,
    output_path: Path,
    debug_info: Dict[str, object] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except Exception:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_uint8 = mask.astype(np.uint8) * 255 if mask.dtype != np.uint8 else mask
    mask_uint8 = mask_uint8 if mask_uint8.max() > 1 else mask_uint8 * 255
    overlay = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_rgb, 0.7, cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 0.3, 0)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
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

    fig.text(0.02, 0.02, debug_text, fontsize=8, verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

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
    parser.add_argument("--debug-plot", action="store_true", help="Save per-image debug visualizations")
    parser.add_argument("--debug-dir", help="Directory to store debug visualizations")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting IMPROVED color verification pipeline...")
        logger.info(f"Input directory: {args.input_dir}")
        logger.info(f"Color stats: {args.color_stats}")

        summary, results = verify_directory(
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
            debug_plot=args.debug_plot,
            debug_dir=Path(args.debug_dir) if args.debug_dir else None,
            logger=logger,
        )

        logger.info("=" * 60)
        logger.info("Verification complete")
        logger.info(f"Total images: {summary['total_images']}")
        logger.info(f"Matched: {summary['matched']}")
        logger.info(f"Mismatched: {summary['mismatched']}")
        logger.info(f"Accuracy: {summary['accuracy']}")
        logger.info(f"Low-confidence: {summary['low_confidence']}")
        if args.output_json:
            logger.info(f"JSON report: {args.output_json}")
        if args.output_csv:
            logger.info(f"CSV report: {args.output_csv}")
        logger.info("=" * 60)

    except Exception as exc:
        logger.error(f"Verification failed: {exc}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
