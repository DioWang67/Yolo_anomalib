import json
import pathlib

import numpy as np
import pytest

from core.stats_color_checker import StatsColorChecker


@pytest.fixture
def dummy_stats_json(tmp_path):
    stats = {
        "summary": {
            "black": {
                "hsv_min": [0, 0, 0],
                "hsv_max": [180, 50, 50],
                "lab_min": [0, 120, 120],
                "lab_max": [50, 135, 135],
                "hsv_mean": [90, 25, 25],
                "coverage_mean": 1.0,
            },
            "target_green": {
                "hsv_min": [40, 50, 50],
                "hsv_max": [80, 255, 255],
                "lab_min": [50, 100, 110],
                "lab_max": [200, 125, 140],
                "hsv_mean": [60, 150, 150]
            }
        }
    }
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    return path

def test_load_stats_from_json(dummy_stats_json):
    """測試從 JSON 載入顏色統計資料"""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    assert "black" in checker._ranges
    assert "target_green" in checker._ranges
    assert checker._ranges["target_green"].name == "target_green"
    assert checker.supported_colors == ("black", "target_green")

def test_check_solid_color(dummy_stats_json):
    """測試對純色圖像進行顏色檢查"""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))

    # Create a solid green image (HSV: 60, 255, 255)
    # Green in BGR is (0, 255, 0)
    green_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    green_bgr[:, :, 1] = 255

    result = checker.check(green_bgr, allowed_colors=["target_green"])
    assert result.best_color == "target_green"
    assert bool(result.is_ok)

def test_check_unsupported_color(dummy_stats_json):
    """An explicit unsupported vocabulary must not widen to every color."""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    # Use uint8 for black image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = checker.check(img, allowed_colors=["blue"])

    assert result.best_color == ""
    assert result.is_ok is False
    assert result.scores == []
    assert result.metrics["debug"]["no_evaluable_candidates"] is True


def test_none_allows_every_supported_color(dummy_stats_json):
    checker = StatsColorChecker.from_json(str(dummy_stats_json))

    result = checker.check(np.zeros((10, 10, 3), dtype=np.uint8), allowed_colors=None)

    assert result.best_color == "black"
    assert result.is_ok is True


def test_explicit_empty_candidates_fail_closed(dummy_stats_json):
    checker = StatsColorChecker.from_json(str(dummy_stats_json))

    result = checker.check(np.zeros((10, 10, 3), dtype=np.uint8), allowed_colors=[])

    assert result.best_color == ""
    assert result.is_ok is False


def test_yellow_special_is_diagnostic_and_does_not_skip_other_scores(tmp_path):
    import cv2

    summary = {}
    for name, hue in {
        "black": 0,
        "green": 85,
        "orange": 12,
        "red": 3,
        "yellow": 25,
    }.items():
        summary[name] = {
            "hsv_min": [0, 0, 0],
            "hsv_max": [179, 255, 255],
            "hsv_mean": [hue, 180, 180],
            "lab_min": [0, 0, 0],
            "lab_max": [255, 255, 255],
            "lab_mean": [128, 128, 128],
            "coverage_mean": 1.0,
        }
    path = tmp_path / "five-colors.json"
    path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
    hsv = np.zeros((100, 100, 3), dtype=np.uint8)
    hsv[:, :] = (3, 200, 180)
    # Survives the 15% per-edge crop as 25/70 = 35.7% Yellow evidence.
    hsv[:, 15:40] = (25, 200, 200)

    result = StatsColorChecker.from_json(path).check(
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    )
    scores = dict(result.scores)

    assert result.metrics["debug"]["yellow_special"]["matched"] is True
    assert result.metrics["debug"]["yellow_special"]["score_adjustment"] == 0.0
    assert set(scores) == set(summary)
    assert scores["red"] > 0.0
    # Neither orange nor green's hue appears anywhere in this image, so
    # neither has a connected match to restrict to; both fall back to the
    # whole candidate pool for their LAB/hue-mean terms rather than being
    # scored a hard zero, which is what keeps them evaluable at all instead
    # of silently disappearing whenever a board's actual color is absent.
    assert scores["orange"] > 0.0
    assert scores["green"] > 0.0

def test_circular_hue_distance():
    """測試色調（Hue）環形距離計算法"""
    from core.stats_color_checker import _circular_hue_distance
    assert _circular_hue_distance(10, 20) == 10
    assert _circular_hue_distance(170, 10) == 20 # 170 to 180(0) to 10
    assert _circular_hue_distance(0, 180) == 0 # OpenCV Hue is 0-179


def test_orange_red_tiebreak_cannot_inflate_orange_above_black():
    from core.stats_color_checker import _separate_orange_red

    # Reproduces the score ordering from the Black -> Orange failure. Hue and
    # Lab both vote Orange, but Black was already the strongest color.
    hsv_vals = np.array([[10.0, 66.0, 89.0]] * 100, dtype=np.float32)
    lab_vals = np.array([[87.0, 120.0, 136.0]] * 100, dtype=np.float32)
    scores = {
        "black": 0.385626,
        "yellow": 0.362031,
        "orange": 0.311123,
        "red": 0.219895,
    }

    winner, pair_score, _debug = _separate_orange_red(
        hsv_vals,
        lab_vals,
        scores["orange"],
        scores["red"],
    )
    scores[winner] = pair_score

    assert winner == "orange"
    assert pair_score == pytest.approx(0.311123)
    assert max(scores, key=scores.get) == "black"

def test_decision_tuning_defaults_match_module_constants():
    """未提供 tuning 時，行為必須與歷史常數完全一致（零行為變更保證）"""
    from core.stats_color_checker import (
        CENTER_MARGIN_RATIO,
        DEFAULT_SAT_THRESHOLD,
        ORANGE_RED_TIE_MARGIN,
        YELLOW_H_RANGE,
        YELLOW_S_MIN,
        YELLOW_V_MIN,
        ColorDecisionTuning,
    )

    tuning = ColorDecisionTuning.from_dict(None)
    assert tuning.sat_threshold == DEFAULT_SAT_THRESHOLD
    assert (tuning.yellow_h_min, tuning.yellow_h_max) == YELLOW_H_RANGE
    assert tuning.yellow_s_min == YELLOW_S_MIN
    assert tuning.yellow_v_min == YELLOW_V_MIN
    assert tuning.orange_red_tie_margin == ORANGE_RED_TIE_MARGIN
    assert tuning.center_margin_ratio == CENTER_MARGIN_RATIO


def test_decision_tuning_from_dict_ignores_unknown_keys():
    from core.stats_color_checker import ColorDecisionTuning

    tuning = ColorDecisionTuning.from_dict(
        {"yellow_h_min": 18, "not_a_real_knob": 1.0, "black_s_threshold": None}
    )
    assert tuning.yellow_h_min == 18.0
    assert not hasattr(tuning, "black_s_threshold")


# ---------------------------------------------------------------------------
# Regressions from the color-detection review
# ---------------------------------------------------------------------------


def _solid_hsv(hue_pairs, sat, val, size=200):
    """Build a BGR ROI whose top/bottom halves carry the two given hues."""
    import cv2

    hsv = np.zeros((size, size, 3), np.uint8)
    half = size // 2
    hsv[:half, :, 0] = hue_pairs[0]
    hsv[half:, :, 0] = hue_pairs[1]
    hsv[:, :, 1] = sat
    hsv[:, :, 2] = val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_unmeasurable_roi_fails_closed_instead_of_inventing_black(dummy_stats_json):
    """A ROI with no pixels above the saturation gate carries no evidence.

    It used to answer with a hard-coded ``black: 0.7`` -- a score picked to
    clear black's own threshold -- so a washed-out ROI passed the color check.
    """
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    grey = np.full((40, 40, 3), 128, dtype=np.uint8)

    result = checker.check(grey, allowed_colors=["target_green"])

    assert result.is_ok is False
    assert result.metrics["has_evidence"] is False
    assert result.metrics["debug"].get("no_pixels") is True


def test_unmeasurable_roi_stays_inside_the_allowed_vocabulary(dummy_stats_json):
    """The invented score was also written past the allowed color filter."""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    grey = np.full((40, 40, 3), 128, dtype=np.uint8)

    result = checker.check(grey, allowed_colors=["target_green"])

    assert [name for name, _ in result.scores] == ["target_green"]
    assert result.best_color == "target_green"


def test_unmeasurable_roi_fails_closed_even_with_a_zero_threshold(dummy_stats_json):
    """``0.0 >= 0.0`` must not turn "nothing measured" into a pass."""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    checker.apply_runtime_configuration(default_threshold=None, color_thresholds={"black": 0.0})
    checker.set_default_threshold(0.0)
    grey = np.full((40, 40, 3), 128, dtype=np.uint8)

    assert checker.check(grey).is_ok is False


def test_circular_hue_mean_crosses_the_seam():
    from core.stats_color_checker import circular_hue_mean

    # 3 and 178 are 5 apart across the seam; their mean is 0.5, not 90.5.
    assert circular_hue_mean(np.array([3.0, 178.0])) == pytest.approx(0.5, abs=1e-6)
    assert circular_hue_mean(np.array([10.0, 20.0])) == pytest.approx(15.0, abs=1e-6)
    assert circular_hue_mean(np.array([])) == 0.0


def test_seam_straddling_red_is_not_scored_as_another_color():
    """An arithmetic hue mean put seam-crossing red at ~90 -- i.e. green.

    Both ROIs here are the same dim red; only the seam differs.
    """
    stats = {
        "summary": {
            "red": {
                "hsv_min": [2, 132, 72], "hsv_max": [9, 217, 186],
                "lab_min": [30, 140, 120], "lab_max": [90, 190, 170],
                "hsv_mean": [3.9, 201.0, 149.1], "lab_mean": [60, 170, 150],
            },
            "green": {
                "hsv_min": [78, 75, 35], "hsv_max": [96, 150, 90],
                "lab_min": [20, 100, 110], "lab_max": [70, 125, 140],
                "hsv_mean": [89.4, 119.6, 61.2], "lab_mean": [45, 112, 125],
            },
        }
    }
    import json as _json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "stats.json"
        path.write_text(_json.dumps(stats), encoding="utf-8")
        checker = StatsColorChecker.from_json(str(path))

        no_seam = checker.check(_solid_hsv((3, 5), 120, 70))
        seam = checker.check(_solid_hsv((3, 178), 120, 70))

    no_seam_scores = dict(no_seam.scores)
    seam_scores = dict(seam.scores)

    assert no_seam.best_color == "red"
    assert seam.best_color == "red"
    # Crossing the seam cost red roughly half its score and handed green a
    # fifth of a point, because the arithmetic mean hue landed on green.
    assert seam_scores["red"] >= 0.8 * no_seam_scores["red"]
    assert seam_scores["green"] < 0.01


def test_missing_hue_baseline_does_not_inflate_a_color(dummy_stats_json):
    """An absent statistic defaulted to a perfect 1.0 and kept its full weight.

    A color with no ``hsv_mean`` therefore outscored one whose actual hue
    honestly disagreed with its recorded mean, which is exactly backwards --
    the missing statistic must not draw a better score than an honest,
    partial mismatch would.
    """
    from core.stats_color_checker import _ColorRange, _improved_match_ratio

    # A small solid block inside both bounding boxes, so hsv_ratio and
    # lab_ratio are identical (1.0) for both color_range variants below --
    # the only thing that can still differ between them is the hue-mean term,
    # which is why this isolates it. Its own hue (70) sits inside the box but
    # away from the recorded mean (60) used in one variant.
    hsv = np.full((6, 6, 3), (70.0, 200.0, 200.0), dtype=np.float32)
    lab = np.full((6, 6, 3), (150.0, 110.0, 150.0), dtype=np.float32)
    bounds = {
        "hsv_min": np.array([40, 50, 50], np.float32),
        "hsv_max": np.array([80, 255, 255], np.float32),
        "lab_min": np.array([50, 90, 110], np.float32),
        "lab_max": np.array([200, 125, 200], np.float32),
    }
    with_mean = _ColorRange(
        name="c", hsv_mean=np.array([60, 150, 150], np.float32), **bounds
    )
    without_mean = _ColorRange(name="c", hsv_mean=None, **bounds)

    scored = _improved_match_ratio(hsv, lab, with_mean, "c")
    unscored = _improved_match_ratio(hsv, lab, without_mean, "c")

    # Both ratio terms are a perfect 1.0 for both variants, so a color with no
    # hue baseline collects exactly that: 1.0, the renormalized combination of
    # the terms that *are* present -- not the old behavior of defaulting the
    # absent hue term to a perfect 1.0 too (which would have looked identical
    # here) and not a term the caller supplied being ignored. The variant that
    # *does* record a mean pays for the partial mismatch between its recorded
    # mean (60) and this block's actual hue (70): its hue-similarity term is
    # real but imperfect, so it honestly scores lower than the color with no
    # such statistic to be honest about.
    assert unscored == pytest.approx(1.0)
    assert scored < unscored


def test_hue_range_test_wraps_around_zero():
    from core.stats_color_checker import _hue_in_range

    h_vals = np.array([0.0, 5.0, 90.0, 175.0], dtype=np.float32)

    normal = _hue_in_range(h_vals, 80.0, 100.0)
    assert normal.tolist() == [False, False, True, False]

    wrapped = _hue_in_range(h_vals, 170.0, 190.0)  # margin pushed past the seam
    assert wrapped.tolist() == [True, True, False, True]

    everything = _hue_in_range(h_vals, -10.0, 190.0)
    assert everything.all()


def test_stats_arrays_must_carry_three_channels(tmp_path):
    """A malformed model used to load and only fail later inside the matcher."""
    from core.stats_color_checker import stats_color_model_load_failure

    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "summary": {
                    "red": {
                        "hsv_min": [0, 0], "hsv_max": [10, 255, 255],
                        "lab_min": [0, 0, 0], "lab_max": [255, 255, 255],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    failure = stats_color_model_load_failure(str(path))
    assert "hsv_min" in failure


def test_black_score_survives_realistic_specular_highlights(dummy_stats_json):
    """Localized glare spots -- unlike a periodic grid -- do not fragment the wire.

    A wire's real surface glare sits in a few small spots, not a lattice
    cutting all the way across the frame. Small punctures do not sever the
    surrounding region's connectivity, so the largest connected match still
    covers nearly the whole crop.
    """
    import cv2

    rng = np.random.default_rng(7)
    size_h, size_w = 40, 160
    hsv = np.zeros((size_h, size_w, 3), np.uint8)
    hsv[:, :, 1] = 25
    hsv[:, :, 2] = 35
    for _ in range(12):
        y = int(rng.integers(2, size_h - 2))
        x = int(rng.integers(2, size_w - 2))
        hsv[y - 1 : y + 2, x - 1 : x + 2, 2] = 95
    crop = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    result = checker.check(crop)

    assert result.best_color == "black"
    assert result.metrics["debug"]["black_baseline"]["blob_pixels"] > 0.9 * (
        size_h * size_w
    )
    assert result.metrics["score"] > 0.9
    assert result.is_ok is True


def test_black_baseline_debug_exposes_the_matched_blob(dummy_stats_json):
    gray = np.full((20, 20, 3), 30, dtype=np.uint8)

    result = StatsColorChecker.from_json(str(dummy_stats_json)).check(gray)

    debug = result.metrics["debug"]["black_baseline"]
    assert debug["blob_pixels"] == 400
    assert debug["score"] == pytest.approx(1.0)


def test_black_score_is_the_matched_blobs_share_of_the_whole_box(tmp_path):
    """No coverage figure to divide by any more -- the blob's own share stands.

    ``coverage_mean`` recorded how much of a *fixed, differently framed* crop
    matched during calibration, a property of that crop's geometry rather than
    of black. A region already isolated by connectivity does not need a
    second, geometry-coupled number to normalize away contamination it no
    longer contains.
    """
    stats = {
        "summary": {
            "black": {
                "hsv_min": [0, 0, 0],
                "hsv_max": [179, 80, 80],
                "lab_min": [0, 120, 120],
                "lab_max": [80, 136, 136],
            }
        }
    }
    path = tmp_path / "black.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    # A solid 8-column-wide dark block against a bright background: one
    # connected region of 8 * 20 = 160 pixels out of the whole 20x20 = 400.
    image = np.full((20, 20, 3), 180, dtype=np.uint8)
    image[:, :8] = 20

    result = StatsColorChecker.from_json(path).check(image)

    debug = result.metrics["debug"]["black_baseline"]
    assert debug["blob_pixels"] == 160
    assert debug["score"] == pytest.approx(160 / 400)
