"""Colour evidence must not amplify a small neighbouring wire into a winner."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.services.color_checker import ColorCheckerService
from core.stats_color_checker import StatsColorChecker, _ColorRange

FIXTURES = Path(__file__).parent / "fixtures" / "green_red_contamination"


@pytest.mark.parametrize("width", [1, 2, 5, 10, 20])
def test_red_similarity_cannot_exceed_its_relative_connected_support(width):
    hsv = np.full((20, 20, 3), (85, 180, 80), dtype=np.uint8)
    hsv[:, :width] = (3, 200, 180)
    red = _ColorRange(
        name="Red",
        hsv_min=np.array([0, 130, 80], dtype=np.float32),
        hsv_max=np.array([10, 255, 255], dtype=np.float32),
        hsv_mean=np.array([3, 200, 180], dtype=np.float32),
        lab_min=np.array([110, 170, 160], dtype=np.float32),
        lab_max=np.array([130, 190, 180], dtype=np.float32),
        lab_mean=np.array([120, 180, 170], dtype=np.float32),
    )

    green = _ColorRange(
        name="Green",
        hsv_min=np.array([70, 75, 30], dtype=np.float32),
        hsv_max=np.array([100, 255, 100], dtype=np.float32),
        hsv_mean=np.array([85, 180, 80], dtype=np.float32),
        lab_min=np.array([0, 0, 0], dtype=np.float32),
        lab_max=np.array([255, 255, 255], dtype=np.float32),
    )
    result = StatsColorChecker({"red": red, "green": green}).check(
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    )
    score = dict(result.scores)["red"]

    assert score <= width / max(width, 20 - width) + 1e-7
    if width == 20:
        assert result.best_color == "Red"
        assert result.is_ok
    elif width < 10:
        assert result.best_color == "Green"
        assert result.is_ok


@pytest.mark.parametrize(
    ("filename", "expected"),
    [("Red_0.png", "Red"), ("Green_1.png", "Green"),
     ("Yellow_2.png", "Yellow"), ("Black_3.png", "Black"),
     ("Orange_4.png", "Orange"), ("Black_5.png", "Black"),
     ("Black_pcb_175017.png", "Black"), ("Black_pcb_175010.png", "Black"),
     ("Orange_pcb_174708.png", "Orange")],
)
def test_saved_station_crops_keep_their_actual_colour(filename, expected):
    checker = StatsColorChecker.from_json(FIXTURES / "color_model.json")
    crop = cv2.imread(str(FIXTURES / filename))

    result = checker.check(crop)

    assert result.best_color == expected
    assert result.is_ok


def test_green_with_neighbouring_red_passes_service_but_real_red_is_rejected():
    service = ColorCheckerService()
    service.ensure_loaded(str(FIXTURES / "color_model.json"), checker_type="stats")
    for filename, expected_ok in [("Green_1.png", True), ("Red_0.png", False)]:
        crop = cv2.imread(str(FIXTURES / filename))
        height, width = crop.shape[:2]
        result = service.check_items(
            frame=crop,
            processed_image=crop,
            detections=[{"class": "Green", "bbox": [0, 0, width, height]}],
        )
        assert result.is_ok is expected_ok
        assert result.items[0].is_ok is expected_ok


def test_black_and_chromatic_ranking_preserves_calibrated_thresholds():
    checker = StatsColorChecker.from_json(FIXTURES / "color_model.json")
    crop = cv2.imread(str(FIXTURES / "Black_pcb_175017.png"))
    result = checker.check(crop)

    assert result.best_color == "Black"
    assert result.is_ok
    assert result.diff == pytest.approx(1 - 636 / 1508)
    assert result.metrics["ranking_scores"]["black"] > result.metrics["ranking_scores"]["green"]
    assert result.scores[0][0] == "black"

    checker.apply_threshold_overrides({"Black": 0.5})
    rejected = checker.check(crop)
    assert rejected.best_color == "Black"
    assert not rejected.is_ok
    assert rejected.diff == pytest.approx(result.diff)


def test_true_green_cannot_pass_a_black_detector_expectation():
    service = ColorCheckerService()
    service.ensure_loaded(str(FIXTURES / "color_model.json"), checker_type="stats")
    crop = cv2.imread(str(FIXTURES / "Green_1.png"))
    height, width = crop.shape[:2]
    result = service.check_items(
        crop, crop, [{"class": "Black", "bbox": [0, 0, width, height]}]
    )
    assert not result.is_ok
    assert result.items[0].best_color == "Green"
