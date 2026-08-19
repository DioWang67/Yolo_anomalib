import json

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
                "hsv_mean": [90, 25, 25]
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
    """測試請求不支援的顏色（應回退到所有可用顏色）處理"""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    # Use uint8 for black image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = checker.check(img, allowed_colors=["blue"])

    # Fallback to all. All zeros image will match 'black'.
    assert result.best_color == "black"
    assert bool(result.is_ok)

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
        BLACK_MIN_COVERAGE,
        BLACK_S_THRESHOLD,
        BLACK_V_THRESHOLD,
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
    assert tuning.black_s_threshold == BLACK_S_THRESHOLD
    assert tuning.black_v_threshold == BLACK_V_THRESHOLD
    assert tuning.black_min_coverage == BLACK_MIN_COVERAGE
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
    assert tuning.black_s_threshold == 50.0  # None -> default kept


def test_decision_tuning_changes_black_shortcut(dummy_stats_json):
    """tuning 確實生效：放寬黑色門檻後，灰圖被黑色捷徑捕捉"""
    from core.stats_color_checker import ColorDecisionTuning

    # V=100 的深灰圖：預設 black_v_threshold=80 不會判黑
    gray_bgr = np.full((20, 20, 3), 100, dtype=np.uint8)

    default_checker = StatsColorChecker.from_json(str(dummy_stats_json))
    default_result = default_checker.check(gray_bgr)

    relaxed = ColorDecisionTuning.from_dict({"black_v_threshold": 120})
    relaxed_checker = StatsColorChecker.from_json(
        str(dummy_stats_json), tuning=relaxed
    )
    relaxed_result = relaxed_checker.check(gray_bgr)

    assert relaxed_result.best_color == "black"
    assert relaxed_result.metrics["debug"].get("shortcut") == "black"
    assert default_result.metrics["debug"].get("shortcut") != "black"
