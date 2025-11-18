from __future__ import annotations

"""Stats-based color checker derived from the improved color_verifier script."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json

import cv2
import numpy as np

from core.led_qc_enhanced import LEDQCAdvancedResult

# Default sampling thresholds
DEFAULT_SAT_THRESHOLD = 20.0
BLACK_S_THRESHOLD = 50.0
BLACK_V_THRESHOLD = 80.0
BLACK_MIN_COVERAGE = 0.6
YELLOW_H_RANGE = (20, 35)
YELLOW_S_MIN = 80
YELLOW_V_MIN = 150
ORANGE_RED_TIE_MARGIN = 0.15
GREEN_DOMINANCE_RATIO = 0.3
CENTER_MARGIN_RATIO = 0.15
DEFAULT_RATIO_THRESHOLD = 0.35

COLOR_CONF_THRESHOLDS = {
    "black": 0.45,
    "yellow": 0.20,
    "orange": 0.25,
    "red": 0.25,
    "green": 0.30,
}


@dataclass
class _ColorRange:
    name: str
    hsv_min: np.ndarray
    hsv_max: np.ndarray
    lab_min: np.ndarray
    lab_max: np.ndarray
    hsv_mean: Optional[np.ndarray] = None
    lab_mean: Optional[np.ndarray] = None


def _margin_vector(margin: Sequence[float] | float | None) -> np.ndarray:
    if margin is None:
        return np.zeros(3, dtype=np.float32)
    if isinstance(margin, Sequence) and not isinstance(margin, (str, bytes)):
        values = list(margin)
        if len(values) == 1:
            values *= 3
    else:
        values = [float(margin if not isinstance(margin, Sequence) else margin[0])] * 3
    if len(values) != 3:
        raise ValueError("margin must contain 1 or 3 values.")
    return np.asarray(values, dtype=np.float32)


def _load_color_ranges(
    stats_path: Path,
    hsv_margin: Sequence[float] | float | None = None,
    lab_margin: Sequence[float] | float | None = None,
) -> Dict[str, _ColorRange]:
    payload = stats_path.read_text(encoding="utf-8")
    data = json.loads(payload)
    summary = data.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("color_stats summary missing.")

    hsv_margin_vec = _margin_vector(hsv_margin)
    lab_margin_vec = _margin_vector(lab_margin)
    ranges: Dict[str, _ColorRange] = {}
    for color, stats in summary.items():
        color_name = str(color)
        hsv_min = np.asarray(stats["hsv_min"], dtype=np.float32) - hsv_margin_vec
        hsv_max = np.asarray(stats["hsv_max"], dtype=np.float32) + hsv_margin_vec
        lab_min = np.asarray(stats["lab_min"], dtype=np.float32) - lab_margin_vec
        lab_max = np.asarray(stats["lab_max"], dtype=np.float32) + lab_margin_vec

        def _opt_array(key: str) -> Optional[np.ndarray]:
            if key not in stats:
                return None
            return np.asarray(stats[key], dtype=np.float32)

        ranges[color_name.lower()] = _ColorRange(
            name=color_name,
            hsv_min=hsv_min,
            hsv_max=hsv_max,
            lab_min=lab_min,
            lab_max=lab_max,
            hsv_mean=_opt_array("hsv_mean"),
            lab_mean=_opt_array("lab_mean"),
        )
    return ranges


def _circular_hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def _improved_match_ratio(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    color_range: _ColorRange,
    color_name: str,
) -> float:
    if hsv_vals.size == 0 or lab_vals.size == 0:
        return 0.0

    h_vals = hsv_vals[:, 0]
    s_vals = hsv_vals[:, 1]
    v_vals = hsv_vals[:, 2]

    if color_name == "red":
        h_mask = ((h_vals <= 10) | (h_vals >= 170)) & (
            (s_vals >= max(color_range.hsv_min[1], 130))
            & (v_vals >= max(color_range.hsv_min[2], 80))
        )
    elif color_name == "orange":
        h_mask = ((h_vals >= 5) & (h_vals <= 20)) & (
            (s_vals >= max(color_range.hsv_min[1], 130))
            & (v_vals >= max(color_range.hsv_min[2], 100))
        )
    elif color_name == "yellow":
        h_mask = (h_vals >= 20) & (h_vals <= 35) & (s_vals >= 80) & (v_vals >= 150)
    elif color_name == "green":
        h_mask = (
            (h_vals >= 70)
            & (h_vals <= 100)
            & (s_vals >= 75)
            & (v_vals >= 30)
            & (v_vals <= 100)
        )
    elif color_name == "black":
        h_mask = (s_vals < BLACK_S_THRESHOLD) & (v_vals < BLACK_V_THRESHOLD)
    else:
        h_mask = (
            (h_vals >= color_range.hsv_min[0])
            & (h_vals <= color_range.hsv_max[0])
            & (s_vals >= color_range.hsv_min[1])
            & (s_vals <= color_range.hsv_max[1])
            & (v_vals >= color_range.hsv_min[2])
            & (v_vals <= color_range.hsv_max[2])
        )

    hsv_ratio = float(np.count_nonzero(h_mask)) / len(hsv_vals)

    lab_mask = (
        (lab_vals[:, 0] >= color_range.lab_min[0])
        & (lab_vals[:, 0] <= color_range.lab_max[0])
        & (lab_vals[:, 1] >= color_range.lab_min[1])
        & (lab_vals[:, 1] <= color_range.lab_max[1])
        & (lab_vals[:, 2] >= color_range.lab_min[2])
        & (lab_vals[:, 2] <= color_range.lab_max[2])
    )
    lab_ratio = float(np.count_nonzero(lab_mask)) / len(lab_vals)

    mean_h = float(np.mean(h_vals))
    hue_similarity = 1.0
    if color_range.hsv_mean is not None:
        expected_h = float(color_range.hsv_mean[0])
        hue_dist = _circular_hue_distance(mean_h, expected_h)
        hue_similarity = np.exp(-hue_dist / 15.0)

    lab_chroma_similarity = 1.0
    if color_range.lab_mean is not None and color_name in {"orange", "red"}:
        mean_a = float(np.mean(lab_vals[:, 1]))
        mean_b = float(np.mean(lab_vals[:, 2]))
        expected_a = float(color_range.lab_mean[1])
        expected_b = float(color_range.lab_mean[2])
        lab_chroma_dist = np.sqrt((mean_a - expected_a) ** 2 + (mean_b - expected_b) ** 2)
        lab_chroma_similarity = np.exp(-lab_chroma_dist / 20.0)

    if color_name in {"orange", "red"}:
        weights = (0.35, 0.25, 0.25, 0.15)
    elif color_name == "green":
        weights = (0.6, 0.2, 0.2, 0.0)
    elif color_name == "yellow":
        weights = (0.5, 0.2, 0.3, 0.0)
    else:
        weights = (0.5, 0.3, 0.2, 0.0)

    return (
        hsv_ratio * weights[0]
        + lab_ratio * weights[1]
        + hue_similarity * weights[2]
        + lab_chroma_similarity * weights[3]
    )


def _is_black_image(hsv_img: np.ndarray) -> Tuple[bool, float]:
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

    black_mask = (center_region[:, :, 1] < BLACK_S_THRESHOLD) & (
        center_region[:, :, 2] < BLACK_V_THRESHOLD
    )
    black_coverage = float(np.count_nonzero(black_mask)) / max(black_mask.size, 1)
    is_black = (
        (mean_s < BLACK_S_THRESHOLD and mean_v < BLACK_V_THRESHOLD)
        or (median_s < BLACK_S_THRESHOLD * 0.8 and median_v < BLACK_V_THRESHOLD * 0.8)
        or (black_coverage > BLACK_MIN_COVERAGE)
    )
    return is_black, (black_coverage if is_black else 0.0)


def _detect_yellow_special(hsv_img: np.ndarray) -> Tuple[bool, float]:
    h, w = hsv_img.shape[:2]
    margin = int(min(h, w) * 0.15)
    center = hsv_img[margin:h - margin, margin:w - margin]
    if center.size == 0:
        center = hsv_img

    h_vals = center[:, :, 0]
    s_vals = center[:, :, 1]
    v_vals = center[:, :, 2]

    yellow_mask = (
        (h_vals >= YELLOW_H_RANGE[0])
        & (h_vals <= YELLOW_H_RANGE[1])
        & (s_vals >= YELLOW_S_MIN)
        & (v_vals >= YELLOW_V_MIN)
    )
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / max(yellow_mask.size, 1)
    orange_like_mask = (h_vals < 20) & (h_vals > 5) & (s_vals > 100)
    orange_ratio = float(np.count_nonzero(orange_like_mask)) / max(orange_like_mask.size, 1)
    return (yellow_ratio > 0.25 and yellow_ratio > orange_ratio * 1.3), yellow_ratio


class StatsColorChecker:
    """Single-ROI color checker driven by color_stats summary JSON."""

    def __init__(
        self,
        color_ranges: Dict[str, _ColorRange],
        *,
        default_threshold: float = DEFAULT_RATIO_THRESHOLD,
        color_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        if not color_ranges:
            raise ValueError("color_ranges must not be empty")
        self._ranges = color_ranges
        self._default_threshold = default_threshold or DEFAULT_RATIO_THRESHOLD
        base_thresholds = {**COLOR_CONF_THRESHOLDS}
        if color_thresholds:
            for name, val in color_thresholds.items():
                base_thresholds[name.lower()] = float(val)
        self._color_thresholds = base_thresholds

    @classmethod
    def from_json(
        cls,
        stats_path: str | Path,
        *,
        default_threshold: float = DEFAULT_RATIO_THRESHOLD,
        color_thresholds: Optional[Dict[str, float]] = None,
        hsv_margin: Sequence[float] | float | None = None,
        lab_margin: Sequence[float] | float | None = None,
    ) -> "StatsColorChecker":
        stats_path = Path(stats_path)
        ranges = _load_color_ranges(stats_path, hsv_margin=hsv_margin, lab_margin=lab_margin)
        return cls(
            ranges,
            default_threshold=default_threshold,
            color_thresholds=color_thresholds,
        )

    def check(
        self,
        image_bgr: np.ndarray,
        *,
        allowed_colors: Optional[Iterable[str]] = None,
    ) -> LEDQCAdvancedResult:
        ranges = self._filter_ranges(allowed_colors)
        if not ranges:
            ranges = self._ranges

        hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        is_black, black_conf = _is_black_image(hsv_img)
        if is_black and "black" in ranges:
            score_map = {name: 0.0 for name in ranges}
            score_map["black"] = black_conf
            return self._result_from_scores(score_map, debug={"shortcut": "black"})

        is_yellow, yellow_conf = _detect_yellow_special(hsv_img)
        if is_yellow and "yellow" in ranges:
            score_map = {name: 0.0 for name in ranges}
            score_map["yellow"] = yellow_conf
            return self._result_from_scores(score_map, debug={"shortcut": "yellow"})

        center_hsv = self._center_crop(hsv_img)
        center_lab = self._center_crop(lab_img)
        sat_mask = center_hsv[:, :, 1] >= DEFAULT_SAT_THRESHOLD
        valid_hsv = center_hsv[sat_mask].reshape(-1, 3)
        valid_lab = center_lab[sat_mask].reshape(-1, 3)
        if len(valid_hsv) == 0 or len(valid_lab) == 0:
            score_map = {name: 0.0 for name in ranges}
            score_map.setdefault("black", 0.7)
            return self._result_from_scores(score_map, debug={"no_pixels": True})

        scores = {
            name: _improved_match_ratio(valid_hsv, valid_lab, color_range, name)
            for name, color_range in ranges.items()
        }
        return self._result_from_scores(scores, debug={})

    def _result_from_scores(
        self,
        score_map: Dict[str, float],
        debug: Dict[str, object],
    ) -> LEDQCAdvancedResult:
        best_name, best_score = ("", 0.0)
        for name, score in score_map.items():
            if best_name == "" or score > best_score:
                best_name = name
                best_score = score

        if not best_name:
            best_name = next(iter(self._ranges))
            best_score = 0.0

        threshold = self._color_thresholds.get(best_name.lower(), self._default_threshold)
        is_ok = best_score >= threshold
        metrics = {
            "score": float(best_score),
            "threshold": float(threshold),
            "ratios": score_map,
            "debug": debug,
        }
        ordered_scores = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        return LEDQCAdvancedResult(
            best_color=self._ranges.get(best_name, self._ranges[next(iter(self._ranges))]).name,
            diff=float(max(0.0, 1.0 - best_score)),
            threshold=float(max(0.0, 1.0 - threshold)),
            is_ok=is_ok,
            scores=[(name, float(score)) for name, score in ordered_scores],
            metrics=metrics,
        )

    def apply_threshold_overrides(self, overrides: Optional[Dict[str, float]]) -> None:
        if not overrides:
            return
        for name, value in overrides.items():
            try:
                self._color_thresholds[str(name).lower()] = float(value)
            except Exception:
                continue

    def set_default_threshold(self, threshold: Optional[float]) -> None:
        if threshold is None:
            return
        try:
            self._default_threshold = float(threshold)
        except Exception:
            pass

    def _filter_ranges(
        self, allowed_colors: Optional[Iterable[str]]
    ) -> Dict[str, _ColorRange]:
        if not allowed_colors:
            return self._ranges
        selected: Dict[str, _ColorRange] = {}
        for name in allowed_colors:
            if not name:
                continue
            key = str(name).lower()
            if key in self._ranges:
                selected[key] = self._ranges[key]
        return selected

    @staticmethod
    def _center_crop(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        margin = int(min(h, w) * CENTER_MARGIN_RATIO)
        if margin <= 0 or margin * 2 >= h or margin * 2 >= w:
            return img
        cropped = img[margin:h - margin, margin:w - margin]
        return cropped if cropped.size else img
