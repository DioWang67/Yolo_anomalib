import json
import pytest
import numpy as np
from pathlib import Path
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

def test_check_solid_color(dummy_stats_json):
    """測試對純色圖像進行顏色檢查"""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    
    # Create a solid green image (HSV: 60, 255, 255)
    # Green in BGR is (0, 255, 0)
    green_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    green_bgr[:, :, 1] = 255 
    
    result = checker.check(green_bgr, allowed_colors=["target_green"])
    assert result.best_color == "target_green"
    assert result.is_ok == True

def test_check_unsupported_color(dummy_stats_json):
    """測試請求不支援的顏色（應回退到所有可用顏色）處理"""
    checker = StatsColorChecker.from_json(str(dummy_stats_json))
    # Use uint8 for black image
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    result = checker.check(img, allowed_colors=["blue"])
    
    # Fallback to all. All zeros image will match 'black'.
    assert result.best_color == "black"
    assert result.is_ok == True

def test_circular_hue_distance():
    """測試色調（Hue）環形距離計算法"""
    from core.stats_color_checker import _circular_hue_distance
    assert _circular_hue_distance(10, 20) == 10
    assert _circular_hue_distance(170, 10) == 20 # 170 to 180(0) to 10
    assert _circular_hue_distance(0, 180) == 0 # OpenCV Hue is 0-179
