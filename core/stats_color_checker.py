"""Stats-based color checker derived from the improved color_verifier script."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.color_qc_enhanced import ColorQCAdvancedResult

# Default sampling thresholds. These are only fallbacks: per-product values
# belong in the model config.yaml under ``color_decision_tuning`` (loaded via
# ColorOverrideLoader) so threshold changes never require a code release.
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

# Hue/S/V gates for the colors that cannot be expressed as a plain box in the
# stats summary: red wraps the 0/179 seam, and orange/green need tighter gates
# than their recorded percentile range. These were hard-coded in the matcher
# and therefore carried one product's calibration in module code; they are
# constants only so that an absent ``color_decision_tuning`` section keeps the
# historical behavior exactly.
RED_H_LOW_MAX = 10
RED_H_HIGH_MIN = 170
RED_S_MIN = 130
RED_V_MIN = 80
ORANGE_H_RANGE = (5, 20)
ORANGE_S_MIN = 130
ORANGE_V_MIN = 100
GREEN_H_RANGE = (70, 100)
GREEN_S_MIN = 75
GREEN_V_RANGE = (30, 100)

COLOR_CONF_THRESHOLDS = {
    "black": 0.45,
    "yellow": 0.20,
    "orange": 0.25,
    "red": 0.25,
    "green": 0.30,
}


@dataclass(frozen=True)
class ColorDecisionTuning:
    """Per-product decision knobs for StatsColorChecker.

    Every field defaults to the historical module constant, so an absent or
    partial ``color_decision_tuning`` config section keeps behavior identical.
    """

    sat_threshold: float = DEFAULT_SAT_THRESHOLD
    black_s_threshold: float = BLACK_S_THRESHOLD
    black_v_threshold: float = BLACK_V_THRESHOLD
    black_min_coverage: float = BLACK_MIN_COVERAGE
    yellow_h_min: float = YELLOW_H_RANGE[0]
    yellow_h_max: float = YELLOW_H_RANGE[1]
    yellow_s_min: float = YELLOW_S_MIN
    yellow_v_min: float = YELLOW_V_MIN
    orange_red_tie_margin: float = ORANGE_RED_TIE_MARGIN
    center_margin_ratio: float = CENTER_MARGIN_RATIO
    red_h_low_max: float = RED_H_LOW_MAX
    red_h_high_min: float = RED_H_HIGH_MIN
    red_s_min: float = RED_S_MIN
    red_v_min: float = RED_V_MIN
    orange_h_min: float = ORANGE_H_RANGE[0]
    orange_h_max: float = ORANGE_H_RANGE[1]
    orange_s_min: float = ORANGE_S_MIN
    orange_v_min: float = ORANGE_V_MIN
    green_h_min: float = GREEN_H_RANGE[0]
    green_h_max: float = GREEN_H_RANGE[1]
    green_s_min: float = GREEN_S_MIN
    green_v_min: float = GREEN_V_RANGE[0]
    green_v_max: float = GREEN_V_RANGE[1]

    @classmethod
    def from_dict(cls, data: dict | None) -> ColorDecisionTuning:
        """Build tuning from a config mapping, ignoring unknown keys.

        Args:
            data: ``color_decision_tuning`` mapping from config, or None.

        Returns:
            Tuning with provided values coerced to float; defaults elsewhere.

        Raises:
            ValueError: If a provided value cannot be coerced to float.
        """
        if not data:
            return cls()
        known = {field.name for field in fields(cls)}
        kwargs = {
            key: float(value)
            for key, value in data.items()
            if key in known and value is not None
        }
        return cls(**kwargs)


_DEFAULT_TUNING = ColorDecisionTuning()


@dataclass
class _ColorRange:
    name: str
    hsv_min: np.ndarray
    hsv_max: np.ndarray
    lab_min: np.ndarray
    lab_max: np.ndarray
    hsv_mean: np.ndarray | None = None
    lab_mean: np.ndarray | None = None


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
) -> dict[str, _ColorRange]:
    payload = stats_path.read_text(encoding="utf-8")
    data = json.loads(payload)
    summary = data.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise ValueError("color_stats summary missing.")

    hsv_margin_vec = _margin_vector(hsv_margin)
    lab_margin_vec = _margin_vector(lab_margin)
    ranges: dict[str, _ColorRange] = {}
    for color, stats in summary.items():
        color_name = str(color)
        hsv_min = _required_stat_array(stats, "hsv_min", color_name) - hsv_margin_vec
        hsv_max = _required_stat_array(stats, "hsv_max", color_name) + hsv_margin_vec
        lab_min = _required_stat_array(stats, "lab_min", color_name) - lab_margin_vec
        lab_max = _required_stat_array(stats, "lab_max", color_name) + lab_margin_vec

        ranges[color_name.lower()] = _ColorRange(
            name=color_name,
            hsv_min=hsv_min,
            hsv_max=hsv_max,
            lab_min=lab_min,
            lab_max=lab_max,
            hsv_mean=_optional_stat_array(stats, "hsv_mean"),
            lab_mean=_optional_stat_array(stats, "lab_mean"),
        )
    return ranges


def stats_color_model_load_failure(stats_path: str | Path) -> str:
    """Return why this file cannot back a stats color checker, or ``""`` if it can.

    Deliberately implemented by running the real loader rather than by
    re-listing the keys it needs: a separate schema check is free to drift from
    the loader, and then a model would pass validation and still fail at
    inference. Callers use this to decide whether to *offer* a color model at
    all, which is the difference between a greyed-out row explaining itself and
    a whole acceptance combination of ERROR results.
    """

    try:
        _load_color_ranges(Path(stats_path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"顏色模型無法讀取：{exc}"
    except (KeyError, TypeError, ValueError) as exc:
        missing = exc.args[0] if isinstance(exc, KeyError) and exc.args else ""
        return (
            f"顏色模型缺少必要統計量 {missing}"
            if missing
            else f"顏色模型格式無效：{exc}"
        )
    return ""


def _required_stat_array(
    stats: Mapping[str, object], key: str, color_name: str
) -> np.ndarray:
    """Read a mandatory 3-channel statistic, rejecting a malformed one here.

    Without the shape check a two-element array loaded fine and only failed
    later as an IndexError from inside the matcher, where it reads as an
    inference crash rather than a bad model file.
    """
    if key not in stats:
        raise KeyError(key)
    array = np.asarray(stats[key], dtype=np.float32).ravel()
    if array.size != 3:
        raise ValueError(
            f"{color_name}.{key} must contain 3 channel values, got {array.size}"
        )
    return array


def _optional_stat_array(stats: Mapping[str, object], key: str) -> np.ndarray | None:
    if key not in stats:
        return None
    array = np.asarray(stats[key], dtype=np.float32).ravel()
    return array if array.size == 3 else None


def _circular_hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180 - diff)


def circular_hue_mean(hue_values: np.ndarray) -> float:
    """Mean hue on the OpenCV 0..179 circle.

    A plain arithmetic mean breaks at the 0/179 seam: red pixels at 3 and 178
    average to ~90, which is green, so the resulting score handed genuine red
    parts to whichever color sits near hue 90. Every consumer compares this
    value through :func:`_circular_hue_distance`, so the mean has to be
    circular too. Hue is expanded to a full turn, averaged as unit vectors,
    then mapped back.
    """
    values = np.asarray(hue_values, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    angles = values * (np.pi / 90.0)
    mean_angle = np.arctan2(
        float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
    )
    return float(np.mod(mean_angle * (90.0 / np.pi), 180.0))


def _hue_in_range(h_vals: np.ndarray, hue_min: float, hue_max: float) -> np.ndarray:
    """Hue membership test that survives the 0/179 seam.

    A margin can push a recorded range past either end of the circle, and a
    color calibrated around hue 0 (pink, magenta) records a range that wraps by
    construction. Both read as ``hue_min > hue_max`` once normalized, which the
    plain ``>= min and <= max`` test answered with "never matches".
    """
    lo = float(hue_min)
    hi = float(hue_max)
    if hi - lo >= 180.0:
        return np.ones(h_vals.shape, dtype=bool)
    lo %= 180.0
    hi %= 180.0
    if lo <= hi:
        return (h_vals >= lo) & (h_vals <= hi)
    return (h_vals >= lo) | (h_vals <= hi)


def _improved_match_ratio(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    color_range: _ColorRange,
    color_name: str,
    tuning: ColorDecisionTuning = _DEFAULT_TUNING,
) -> float:
    if hsv_vals.size == 0 or lab_vals.size == 0:
        return 0.0

    h_vals = hsv_vals[:, 0]
    s_vals = hsv_vals[:, 1]
    v_vals = hsv_vals[:, 2]

    if color_name == "red":
        h_mask = (
            (h_vals <= tuning.red_h_low_max) | (h_vals >= tuning.red_h_high_min)
        ) & (
            (s_vals >= max(color_range.hsv_min[1], tuning.red_s_min))
            & (v_vals >= max(color_range.hsv_min[2], tuning.red_v_min))
        )
    elif color_name == "orange":
        h_mask = ((h_vals >= tuning.orange_h_min) & (h_vals <= tuning.orange_h_max)) & (
            (s_vals >= max(color_range.hsv_min[1], tuning.orange_s_min))
            & (v_vals >= max(color_range.hsv_min[2], tuning.orange_v_min))
        )
    elif color_name == "yellow":
        h_mask = (
            (h_vals >= tuning.yellow_h_min)
            & (h_vals <= tuning.yellow_h_max)
            & (s_vals >= tuning.yellow_s_min)
            & (v_vals >= tuning.yellow_v_min)
        )
    elif color_name == "green":
        h_mask = (
            (h_vals >= tuning.green_h_min)
            & (h_vals <= tuning.green_h_max)
            & (s_vals >= tuning.green_s_min)
            & (v_vals >= tuning.green_v_min)
            & (v_vals <= tuning.green_v_max)
        )
    elif color_name == "black":
        h_mask = (s_vals < tuning.black_s_threshold) & (
            v_vals < tuning.black_v_threshold
        )
    else:
        h_mask = (
            _hue_in_range(h_vals, color_range.hsv_min[0], color_range.hsv_max[0])
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

    mean_h = circular_hue_mean(h_vals)
    hue_similarity: float | None = None
    if color_range.hsv_mean is not None:
        expected_h = float(color_range.hsv_mean[0])
        hue_dist = _circular_hue_distance(mean_h, expected_h)
        hue_similarity = float(np.exp(-hue_dist / 15.0))

    lab_chroma_similarity: float | None = None
    if color_range.lab_mean is not None and color_name in {"orange", "red"}:
        mean_a = float(np.mean(lab_vals[:, 1]))
        mean_b = float(np.mean(lab_vals[:, 2]))
        expected_a = float(color_range.lab_mean[1])
        expected_b = float(color_range.lab_mean[2])
        lab_chroma_dist = np.sqrt(
            (mean_a - expected_a) ** 2 + (mean_b - expected_b) ** 2
        )
        lab_chroma_similarity = float(np.exp(-lab_chroma_dist / 20.0))

    if color_name in {"orange", "red"}:
        weights = (0.35, 0.25, 0.25, 0.15)
    elif color_name == "green":
        weights = (0.6, 0.2, 0.2, 0.0)
    elif color_name == "yellow":
        weights = (0.5, 0.2, 0.3, 0.0)
    else:
        weights = (0.5, 0.3, 0.2, 0.0)

    # A similarity term with no baseline behind it used to default to 1.0 and
    # still collect its full weight, so a color with incomplete stats scored up
    # to 0.3 higher than one with complete stats -- exactly the wrong way
    # round. Drop absent terms and renormalize instead. This is a no-op when
    # every statistic is present, because each weight tuple already sums to 1.
    terms: list[tuple[float, float]] = [
        (hsv_ratio, weights[0]),
        (lab_ratio, weights[1]),
    ]
    if hue_similarity is not None:
        terms.append((hue_similarity, weights[2]))
    if lab_chroma_similarity is not None:
        terms.append((lab_chroma_similarity, weights[3]))

    total_weight = sum(weight for _, weight in terms)
    if total_weight <= 0.0:
        return 0.0
    return sum(value * weight for value, weight in terms) / total_weight


def _separate_orange_red(
    hsv_vals: np.ndarray,
    lab_vals: np.ndarray,
    orange_score: float,
    red_score: float,
) -> tuple[str, float, dict]:
    """Resolve Orange versus Red without inflating their absolute score.

    The tie-breaker may transfer the pair's existing best score to its chosen
    winner, but it must never create confidence that can overtake an unrelated
    color such as Black or Yellow.
    """
    pair_score = max(orange_score, red_score)
    if len(hsv_vals) == 0:
        return (
            "red" if red_score >= orange_score else "orange",
            pair_score,
            {},
        )

    debug = {}
    hue_vals = hsv_vals[:, 0]

    # Core hue counts
    orange_core = np.sum((hue_vals >= 8) & (hue_vals <= 16))
    red_core = np.sum((hue_vals <= 5) | (hue_vals >= 175))

    orange_hue_ratio = orange_core / len(hue_vals)
    red_hue_ratio = red_core / len(hue_vals)

    # LAB a*/b* analysis
    mean_a = float(np.mean(lab_vals[:, 1]))
    mean_b = float(np.mean(lab_vals[:, 2]))
    ab_ratio = mean_b / max(mean_a, 1.0)

    debug.update(
        {
            "orange_hue_ratio": float(orange_hue_ratio),
            "red_hue_ratio": float(red_hue_ratio),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "ab_ratio": float(ab_ratio),
        }
    )

    # Decisions
    if ab_ratio > 1.05:
        lab_vote = "orange"
    elif ab_ratio < 0.90:
        lab_vote = "red"
    else:
        lab_vote = "unclear"

    hue_vote = (
        "orange"
        if orange_hue_ratio > red_hue_ratio * 1.2
        else "red"
        if red_hue_ratio > orange_hue_ratio * 1.2
        else "unclear"
    )

    debug["lab_vote"] = lab_vote
    debug["hue_vote"] = hue_vote

    if hue_vote == lab_vote and hue_vote != "unclear":
        predicted = hue_vote
    elif hue_vote != "unclear":
        predicted = hue_vote
    elif lab_vote != "unclear":
        predicted = lab_vote
    else:
        predicted = "orange" if orange_score > red_score else "red"

    return predicted, float(pair_score), debug


def _center_crop_array(img: np.ndarray, margin_ratio: float) -> np.ndarray:
    """Crop a centered region, falling back to the full image when it cannot.

    Shared so the black shortcut, the yellow shortcut and the main scoring path
    all judge the *same* pixels. They previously used three different crops --
    per-axis 15%, ``min(h, w)`` at a hard-coded 15%, and the tuned ratio -- so
    the three decisions could legitimately disagree about an elongated ROI.
    """
    h, w = img.shape[:2]
    margin = int(min(h, w) * margin_ratio)
    if margin <= 0 or margin * 2 >= h or margin * 2 >= w:
        return img
    cropped = img[margin : h - margin, margin : w - margin]
    return cropped if cropped.size else img


def _is_black_image(
    hsv_img: np.ndarray,
    tuning: ColorDecisionTuning = _DEFAULT_TUNING,
) -> tuple[bool, float, tuple[str, ...]]:
    """Decide whether a region is black, and say which rule decided it.

    Returns ``(is_black, coverage, rules)``. ``coverage`` is the fraction of
    the center that is both unsaturated and dark, and it is what the caller
    scores against black's threshold.

    An earlier version reported the *margin* of whichever rule fired instead,
    on the grounds that answering with coverage when the mean rule decided is
    incoherent. It is -- but coverage is also what the threshold was calibrated
    against, and swapping it collapsed real black scores from ~0.5 to ~0.02 and
    rejected every good board on the line. The incoherence is real and is now
    answered by naming the rules that fired, which costs nothing, rather than
    by moving a number the verdict depends on.
    """
    # Deliberately a per-axis margin, not the shared ``_center_crop_array``.
    # The other paths use ``min(h, w)`` and unifying them looks like tidying,
    # but black's threshold has ~1% of headroom on real crops and was
    # calibrated against *this* crop: switching shaved a genuine black from
    # 0.456 to 0.446 against a 0.45 threshold and rejected good boards. Making
    # the three crops agree is a recalibration, not a cleanup, and has to be
    # done with the threshold in the same change.
    h, w = hsv_img.shape[:2]
    margin_y = int(h * tuning.center_margin_ratio)
    margin_x = int(w * tuning.center_margin_ratio)
    center_region = hsv_img[margin_y : h - margin_y, margin_x : w - margin_x]
    if center_region.size == 0:
        center_region = hsv_img

    mean_s = float(np.mean(center_region[:, :, 1]))
    mean_v = float(np.mean(center_region[:, :, 2]))
    median_s = float(np.median(center_region[:, :, 1]))
    median_v = float(np.median(center_region[:, :, 2]))

    black_s = tuning.black_s_threshold
    black_v = tuning.black_v_threshold
    black_mask = (center_region[:, :, 1] < black_s) & (
        center_region[:, :, 2] < black_v
    )
    black_coverage = float(np.count_nonzero(black_mask)) / max(black_mask.size, 1)

    rules: list[str] = []
    if mean_s < black_s and mean_v < black_v:
        rules.append("mean")
    if median_s < black_s * 0.8 and median_v < black_v * 0.8:
        rules.append("median")
    if black_coverage > tuning.black_min_coverage:
        rules.append("coverage")

    is_black = bool(rules)
    return is_black, (black_coverage if is_black else 0.0), tuple(rules)


def _detect_yellow_special(
    hsv_img: np.ndarray,
    tuning: ColorDecisionTuning = _DEFAULT_TUNING,
) -> tuple[bool, float]:
    center = _center_crop_array(hsv_img, tuning.center_margin_ratio)

    h_vals = center[:, :, 0]
    s_vals = center[:, :, 1]
    v_vals = center[:, :, 2]

    yellow_mask = (
        (h_vals >= tuning.yellow_h_min)
        & (h_vals <= tuning.yellow_h_max)
        & (s_vals >= tuning.yellow_s_min)
        & (v_vals >= tuning.yellow_v_min)
    )
    yellow_ratio = float(np.count_nonzero(yellow_mask)) / max(yellow_mask.size, 1)
    orange_like_mask = (h_vals < 20) & (h_vals > 5) & (s_vals > 100)
    orange_ratio = float(np.count_nonzero(orange_like_mask)) / max(
        orange_like_mask.size, 1
    )
    return (yellow_ratio > 0.25 and yellow_ratio > orange_ratio * 1.3), yellow_ratio


class StatsColorChecker:
    """Single-ROI color checker driven by color_stats summary JSON."""

    def __init__(
        self,
        color_ranges: dict[str, _ColorRange],
        *,
        default_threshold: float = DEFAULT_RATIO_THRESHOLD,
        color_thresholds: dict[str, float] | None = None,
        tuning: ColorDecisionTuning | None = None,
    ) -> None:
        if not color_ranges:
            raise ValueError("color_ranges must not be empty")
        self._ranges = color_ranges
        self._tuning = tuning or _DEFAULT_TUNING

        # Immutable baseline: the effective configuration this checker returns to
        # whenever a caller applies a runtime configuration that omits a key. It
        # is a private copy, so instance-level tuning can never write back into
        # the module constant shared by every other checker.
        self._baseline_default_threshold = DEFAULT_RATIO_THRESHOLD
        self._baseline_color_thresholds: dict[str, float] = dict(COLOR_CONF_THRESHOLDS)

        self._default_threshold = self._baseline_default_threshold
        self._color_thresholds = dict(self._baseline_color_thresholds)
        # Constructor arguments are runtime configuration, not baseline, so they
        # go through the same validate-then-commit path as any later update.
        self.apply_runtime_configuration(
            # ``or None`` preserves the historical constructor contract where a
            # falsy threshold means "unset" and falls back to the baseline.
            default_threshold=default_threshold or None,
            color_thresholds=color_thresholds,
        )

    @property
    def supported_colors(self) -> tuple[str, ...]:
        """Return the immutable color vocabulary exposed by this checker."""
        return tuple(color_range.name for color_range in self._ranges.values())

    @classmethod
    def from_json(
        cls,
        stats_path: str | Path,
        *,
        default_threshold: float = DEFAULT_RATIO_THRESHOLD,
        color_thresholds: dict[str, float] | None = None,
        hsv_margin: Sequence[float] | float | None = None,
        lab_margin: Sequence[float] | float | None = None,
        tuning: ColorDecisionTuning | None = None,
    ) -> StatsColorChecker:
        stats_path = Path(stats_path)
        ranges = _load_color_ranges(
            stats_path, hsv_margin=hsv_margin, lab_margin=lab_margin
        )
        return cls(
            ranges,
            default_threshold=default_threshold,
            color_thresholds=color_thresholds,
            tuning=tuning,
        )

    def check(
        self,
        image_bgr: np.ndarray,
        *,
        allowed_colors: Iterable[str] | None = None,
    ) -> ColorQCAdvancedResult:
        debug: dict[str, object] = {}
        ranges = self._filter_ranges(allowed_colors)
        if not ranges:
            ranges = self._ranges

        hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        is_black, black_conf, black_rules = _is_black_image(hsv_img, self._tuning)
        if is_black and "black" in ranges:
            score_map = dict.fromkeys(ranges, 0.0)
            score_map["black"] = black_conf
            # Naming the rules that fired keeps the decision auditable without
            # changing the number the threshold compares against.
            return self._result_from_scores(
                score_map,
                debug={"shortcut": "black", "black_rules": list(black_rules)},
            )

        is_yellow, yellow_conf = _detect_yellow_special(hsv_img, self._tuning)
        if is_yellow and "yellow" in ranges:
            score_map = dict.fromkeys(ranges, 0.0)
            score_map["yellow"] = yellow_conf
            return self._result_from_scores(score_map, debug={"shortcut": "yellow"})

        center_hsv = self._center_crop(hsv_img)
        center_lab = self._center_crop(lab_img)
        sat_mask = center_hsv[:, :, 1] >= self._tuning.sat_threshold
        valid_hsv = center_hsv[sat_mask].reshape(-1, 3)
        valid_lab = center_lab[sat_mask].reshape(-1, 3)
        if len(valid_hsv) == 0 or len(valid_lab) == 0:
            # Nothing cleared the saturation gate, so no color was measured.
            # This branch used to answer with a hard-coded ``black: 0.7`` --
            # a score chosen to clear black's threshold, invented for a ROI
            # carrying no evidence, and written past the allowed vocabulary
            # (a caller restricted to Red got a passing Black back). Worse,
            # the pipeline reads a passing measurement as grounds to overwrite
            # the detector's class. Report the absence instead: a verdict with
            # no evidence behind it fails closed.
            return self._result_from_scores(
                dict.fromkeys(ranges, 0.0),
                debug={"no_pixels": True},
                has_evidence=False,
            )

        scores = {
            name: _improved_match_ratio(
                valid_hsv, valid_lab, color_range, name, self._tuning
            )
            for name, color_range in ranges.items()
        }
        # Tie-break for Orange vs Red
        if "orange" in scores and "red" in scores:
            o_score = scores["orange"]
            r_score = scores["red"]
            if abs(o_score - r_score) < self._tuning.orange_red_tie_margin:
                winner, new_conf, tie_debug = _separate_orange_red(
                    valid_hsv, valid_lab, o_score, r_score
                )
                scores[winner] = max(scores[winner], new_conf)
                # Slightly suppress the loser to ensure winner is picked
                loser = "red" if winner == "orange" else "orange"
                scores[loser] = min(scores[loser], scores[winner] * 0.8)
                debug["orange_red_tiebreak"] = tie_debug

        return self._result_from_scores(scores, debug=debug)

    def _result_from_scores(
        self,
        score_map: dict[str, float],
        debug: dict[str, object],
        *,
        has_evidence: bool = True,
    ) -> ColorQCAdvancedResult:
        """Assemble a result, refusing to pass when nothing was measured.

        Args:
            score_map: Per-color confidence, keyed by the lower-cased name.
            debug: Opaque diagnostic payload carried into ``metrics``.
            has_evidence: False when the ROI yielded no measurable pixels. The
                verdict is then forced closed rather than left to the
                threshold comparison, because a product may legitimately
                configure a threshold of 0 and ``0.0 >= 0.0`` would otherwise
                pass every unmeasurable ROI.
        """
        best_name, best_score = ("", 0.0)
        for name, score in score_map.items():
            if best_name == "" or score > best_score:
                best_name = name
                best_score = score

        if not best_name:
            best_name = next(iter(self._ranges))
            best_score = 0.0

        threshold = self._color_thresholds.get(
            best_name.lower(), self._default_threshold
        )
        is_ok = has_evidence and best_score >= threshold
        metrics = {
            "score": float(best_score),
            "threshold": float(threshold),
            "has_evidence": bool(has_evidence),
            "ratios": score_map,
            "debug": debug,
        }
        ordered_scores = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
        # Name the color that was actually scored. Substituting an arbitrary
        # first entry when the key is unknown reported one color's name against
        # another color's score.
        scored_range = self._ranges.get(best_name)
        return ColorQCAdvancedResult(
            best_color=scored_range.name if scored_range is not None else best_name,
            diff=float(max(0.0, 1.0 - best_score)),
            threshold=float(max(0.0, 1.0 - threshold)),
            is_ok=is_ok,
            scores=[(name, float(score)) for name, score in ordered_scores],
            metrics=metrics,
        )

    def apply_runtime_configuration(
        self,
        *,
        default_threshold: float | None = None,
        # Values arrive straight from product config and are validated here, so
        # they are deliberately untyped rather than assumed to be floats.
        color_thresholds: Mapping[str, Any] | None = None,
    ) -> None:
        """Replace all runtime-tunable state with this invocation's configuration.

        Any key not supplied returns to the immutable baseline, so configuration
        belonging to a previously inspected product cannot survive into the next
        one when the same checker instance is reused.

        Every value is validated before any state changes, so a successful call
        commits the whole configuration and a rejected one commits none of it.
        A rejected call additionally resets to baseline, so the checker is never
        left holding a half-applied or previous-product configuration.

        Args:
            default_threshold: Fallback threshold for colors without an explicit
                one; ``None`` restores the baseline.
            color_thresholds: Per-color thresholds, matched case-insensitively;
                colors omitted here return to the baseline.

        Raises:
            TypeError: If ``color_thresholds`` is not a mapping.
            ValueError: If any value cannot be coerced to float.
        """
        try:
            resolved_default = self._baseline_default_threshold
            if default_threshold is not None:
                try:
                    resolved_default = float(default_threshold)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid default color threshold: {default_threshold!r}"
                    ) from exc

            resolved = dict(self._baseline_color_thresholds)
            if color_thresholds:
                if not isinstance(color_thresholds, Mapping):
                    raise TypeError(
                        "color threshold overrides must be a mapping, got "
                        f"{type(color_thresholds).__name__}"
                    )
                for name, value in color_thresholds.items():
                    try:
                        resolved[str(name).lower()] = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid color threshold overrides: {name!r} -> {value!r}"
                        ) from exc
        except (TypeError, ValueError):
            self.reset_runtime_configuration()
            raise

        self._default_threshold = resolved_default
        self._color_thresholds = resolved

    def reset_runtime_configuration(self) -> None:
        """Discard every runtime override and return to the immutable baseline."""
        self._default_threshold = self._baseline_default_threshold
        self._color_thresholds = dict(self._baseline_color_thresholds)

    def apply_threshold_overrides(self, overrides: dict[str, float] | None) -> None:
        """Merge per-color threshold overrides onto the *current* configuration.

        Prefer :meth:`apply_runtime_configuration` when switching products: this
        method deliberately keeps values already applied, so on its own it cannot
        clear a previous product's configuration.

        Raises:
            ValueError: If any value cannot be coerced to float. Nothing is
                applied in that case, so a rejected batch cannot leave the
                checker half-configured.
        """
        if not overrides:
            return
        coerced: dict[str, float] = {}
        for name, value in overrides.items():
            try:
                coerced[str(name).lower()] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid color threshold overrides: {name!r} -> {value!r}"
                ) from exc
        self._color_thresholds.update(coerced)

    def set_default_threshold(self, threshold: float | None) -> None:
        """Set the fallback threshold used by colors without an explicit one.

        Raises:
            ValueError: If the threshold cannot be coerced to float.
        """
        if threshold is None:
            return
        try:
            self._default_threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid default color threshold: {threshold!r}"
            ) from exc

    def _filter_ranges(
        self, allowed_colors: Iterable[str] | None
    ) -> dict[str, _ColorRange]:
        if not allowed_colors:
            return self._ranges
        selected: dict[str, _ColorRange] = {}
        for name in allowed_colors:
            if not name:
                continue
            key = str(name).lower()
            if key in self._ranges:
                selected[key] = self._ranges[key]
        return selected

    def _center_crop(self, img: np.ndarray) -> np.ndarray:
        return _center_crop_array(img, self._tuning.center_margin_ratio)
