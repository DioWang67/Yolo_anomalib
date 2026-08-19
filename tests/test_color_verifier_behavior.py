from __future__ import annotations

import builtins
import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import color_verifier


def _color_range(name: str) -> color_verifier.ColorRange:
    return color_verifier.ColorRange(
        name=name,
        hsv_min=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        hsv_max=np.asarray([179.0, 255.0, 255.0], dtype=np.float32),
        lab_min=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        lab_max=np.asarray([255.0, 255.0, 255.0], dtype=np.float32),
    )


def test_low_saturation_fallback_never_fabricates_black_prediction():
    hsv_img = np.full((20, 20, 3), (0.0, 0.0, 255.0), dtype=np.float32)
    lab_img = np.full((20, 20, 3), (255.0, 128.0, 128.0), dtype=np.float32)
    ranges = {name: _color_range(name) for name in ("Black", "White")}

    ratios, masks, debug = color_verifier._evaluate_image_improved(
        hsv_img,
        lab_img,
        ranges,
        edge_margin=0.1,
        sat_threshold=20.0,
        min_valid_pixels=40,
    )

    assert ratios == {"Black": 0.0, "White": 0.0}
    assert color_verifier._initial_prediction(ratios) == ("Unknown", 0.0)
    assert debug["low_saturation_fallback"] is True
    assert debug["valid_pixel_count"] == 0
    assert all(not np.any(mask) for mask in masks.values())


def test_saturation_threshold_is_applied_to_evaluation():
    hsv_img = np.full((10, 10, 3), (110.0, 30.0, 120.0), dtype=np.float32)
    lab_img = np.full((10, 10, 3), (120.0, 128.0, 128.0), dtype=np.float32)
    ranges = {"Blue": _color_range("Blue")}

    accepted, _, accepted_debug = color_verifier._evaluate_image_improved(
        hsv_img,
        lab_img,
        ranges,
        edge_margin=0.0,
        sat_threshold=20.0,
        min_valid_pixels=1,
    )
    rejected, rejected_masks, rejected_debug = color_verifier._evaluate_image_improved(
        hsv_img,
        lab_img,
        ranges,
        edge_margin=0.0,
        sat_threshold=40.0,
        min_valid_pixels=1,
    )

    assert accepted["Blue"] > 0.0
    assert accepted_debug.get("low_saturation_fallback") is None
    assert rejected == {"Blue": 0.0}
    assert rejected_debug["low_saturation_fallback"] is True
    assert not np.any(rejected_masks["Blue"])


def test_orange_red_separator_clamps_boosted_confidence():
    hsv_vals = np.full((20, 3), (10.0, 200.0, 200.0), dtype=np.float32)
    lab_vals = np.full((20, 3), (150.0, 100.0, 120.0), dtype=np.float32)

    predicted, confidence, _ = color_verifier.separate_orange_red_improved(
        hsv_vals,
        lab_vals,
        orange_score=0.95,
        red_score=0.90,
    )

    assert predicted == "Orange"
    assert confidence == 1.0


def test_global_confidence_threshold_can_only_tighten_color_threshold():
    assert color_verifier._confidence_threshold_for("Yellow", 0.50) == 0.50
    assert color_verifier._confidence_threshold_for("Black", 0.20) == 0.45


def test_hsv_mask_uses_trained_range_instead_of_hardcoded_red_saturation():
    red_range = _color_range("Red")
    red_range.hsv_min = np.asarray([0.0, 80.0, 80.0], dtype=np.float32)
    red_range.hsv_max = np.asarray([10.0, 120.0, 180.0], dtype=np.float32)
    hsv_vals = np.asarray([[5.0, 100.0, 120.0]], dtype=np.float32)

    mask = color_verifier._hsv_color_mask(hsv_vals, red_range)

    assert mask.tolist() == [True]


def test_hsv_mask_supports_trained_hue_range_crossing_zero():
    red_range = _color_range("Red")
    red_range.hsv_min = np.asarray([170.0, 80.0, 80.0], dtype=np.float32)
    red_range.hsv_max = np.asarray([10.0, 255.0, 255.0], dtype=np.float32)
    hsv_vals = np.asarray(
        [[175.0, 100.0, 120.0], [5.0, 100.0, 120.0], [90.0, 100.0, 120.0]],
        dtype=np.float32,
    )

    mask = color_verifier._hsv_color_mask(hsv_vals, red_range)

    assert mask.tolist() == [True, True, False]


def test_match_ratio_is_zero_when_no_hsv_pixels_match():
    red_range = _color_range("Red")
    red_range.hsv_min = np.asarray([0.0, 80.0, 80.0], dtype=np.float32)
    red_range.hsv_max = np.asarray([10.0, 255.0, 255.0], dtype=np.float32)
    hsv_vals = np.full((100, 3), (90.0, 200.0, 200.0), dtype=np.float32)
    lab_vals = np.full((100, 3), (150.0, 128.0, 128.0), dtype=np.float32)

    score, debug = color_verifier.improved_match_ratio(hsv_vals, lab_vals, red_range, "Red")

    assert score == 0.0
    assert debug["hsv_gate_rejected"] is True
    assert debug["final_score"] == 0.0


def test_center_crop_uses_each_axis_for_rectangular_images():
    image = np.zeros((10, 100, 3), dtype=np.uint8)

    cropped = color_verifier._crop_center(image, 0.1)

    assert cropped.shape == (8, 80, 3)


def test_yellow_shortcut_returns_real_center_mask():
    hsv_img = np.full((10, 20, 3), (25.0, 200.0, 200.0), dtype=np.float32)
    lab_img = np.full((10, 20, 3), (150.0, 128.0, 128.0), dtype=np.float32)

    ratios, masks, debug = color_verifier._evaluate_image_improved(
        hsv_img,
        lab_img,
        {"Yellow": _color_range("Yellow")},
        edge_margin=0.1,
    )

    assert ratios["Yellow"] == pytest.approx(1.0)
    assert debug["shortcut"] == "Yellow"
    assert 0 < np.count_nonzero(masks["Yellow"]) < masks["Yellow"].size
    assert not np.any(masks["Yellow"][[0, -1], :])


def test_yellow_secondary_mask_excludes_high_saturation_orange_pixels():
    hsv_img = np.full((10, 10, 3), (19.0, 200.0, 220.0), dtype=np.float32)

    is_yellow, confidence, mask = color_verifier._detect_yellow_special(
        hsv_img,
        edge_margin=0.0,
    )

    assert is_yellow is False
    assert confidence == 0.0
    assert not np.any(mask)


def test_general_path_mask_contains_only_pixels_matching_hsv_and_lab():
    hsv_img = np.full((10, 10, 3), (110.0, 100.0, 120.0), dtype=np.float32)
    hsv_img[5, 5] = (50.0, 100.0, 120.0)
    lab_img = np.full((10, 10, 3), (120.0, 128.0, 128.0), dtype=np.float32)
    blue_range = _color_range("Blue")
    blue_range.hsv_min = np.asarray([100.0, 0.0, 0.0], dtype=np.float32)
    blue_range.hsv_max = np.asarray([120.0, 255.0, 255.0], dtype=np.float32)

    _, masks, debug = color_verifier._evaluate_image_improved(
        hsv_img,
        lab_img,
        {"Blue": blue_range},
        edge_margin=0.1,
        sat_threshold=20.0,
        min_valid_pixels=1,
    )

    assert debug.get("shortcut") is None
    assert np.count_nonzero(masks["Blue"]) == 63
    assert not masks["Blue"][5, 5]
    assert not np.any(masks["Blue"][[0, -1], :])


def test_green_correction_ignores_low_saturation_hue_noise():
    hsv_img = np.full((10, 10, 3), (80.0, 0.0, 120.0), dtype=np.float32)
    context = color_verifier.DecisionContext(
        ratios={"Red": 0.4, "Green": 0.3},
        debug_info={},
        hsv_img=hsv_img,
        lab_img=np.zeros_like(hsv_img),
        edge_margin=0.0,
        sat_threshold=20.0,
    )

    result = color_verifier._rule_green_correction("Red", 0.4, context)

    assert result is None
    assert "green_correction" not in context.debug_info


def test_shortcut_rules_are_not_reapplied_after_evaluation():
    assert [rule.__name__ for rule in color_verifier._COLOR_RULES] == [
        "_rule_orange_red_tiebreak",
        "_rule_green_correction",
    ]


def test_verify_directory_rejects_unknown_keyword_instead_of_ignoring_it(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword"):
        color_verifier.verify_directory(
            tmp_path,
            tmp_path / "stats.json",
            sat_thresold=99.0,
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("edge_margin", 0.5, "edge_margin"),
        ("sat_threshold", 256.0, "sat_threshold"),
        ("min_valid_pixels", 0, "min_valid_pixels"),
    ],
)
def test_evaluation_options_fail_fast(option, value, message):
    options = {
        "edge_margin": 0.1,
        "sat_threshold": 20.0,
        "min_valid_pixels": 1,
    }
    options[option] = value

    with pytest.raises(ValueError, match=message):
        color_verifier._validate_evaluation_options(**options)


def test_verify_directory_reports_low_saturation_as_unknown(tmp_path):
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    assert cv2.imwrite(str(input_dir / "white.png"), np.full((20, 20, 3), 255, dtype=np.uint8))
    stats_path = tmp_path / "color_stats.json"
    stats_path.write_text(
        """
        {
          "summary": {
            "Black": {"hsv_min": [0, 0, 0], "hsv_max": [179, 255, 255], "lab_min": [0, 0, 0], "lab_max": [255, 255, 255]},
            "White": {"hsv_min": [0, 0, 0], "hsv_max": [179, 255, 255], "lab_min": [0, 0, 0], "lab_max": [255, 255, 255]}
          }
        }
        """,
        encoding="utf-8",
    )

    summary, decisions = color_verifier.verify_directory(input_dir, stats_path)

    assert summary["low_confidence"] == 1
    assert summary["matched"] == 0
    assert summary["mismatched"] == 0
    assert decisions[0].predicted_color == "Unknown"
    assert decisions[0].confidence == 0.0
    assert decisions[0].status == "low_confidence"


def test_load_color_ranges_normalizes_known_color_names(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        '{"summary":{"red":{"hsv_min":[0,80,80],"hsv_max":[10,255,255],'
        '"lab_min":[0,0,0],"lab_max":[255,255,255]}}}',
        encoding="utf-8",
    )

    ranges = color_verifier.load_color_ranges(stats_path)

    assert list(ranges) == ["Red"]
    assert ranges["Red"].name == "Red"


def test_load_color_ranges_rejects_case_insensitive_name_collision(tmp_path):
    stats_path = tmp_path / "stats.json"
    color_data = (
        '{"hsv_min":[0,80,80],"hsv_max":[10,255,255],'
        '"lab_min":[0,0,0],"lab_max":[255,255,255]}'
    )
    stats_path.write_text(
        f'{{"summary":{{"Red":{color_data},"red":{color_data}}}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate color name: Red"):
        color_verifier.load_color_ranges(stats_path)


def test_margin_vector_accepts_only_scalar_or_three_values():
    assert color_verifier._margin_vector(2.0).tolist() == [2.0, 2.0, 2.0]
    assert color_verifier._margin_vector([1.0, 2.0, 3.0]).tolist() == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="1 or 3"):
        color_verifier._margin_vector([])


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (FileNotFoundError("missing"), "input was not found"),
        (ValueError("bad margin"), "input or configuration is invalid"),
        (OSError("disk error"), "I/O failed"),
        (color_verifier.VerificationReportWriteError("report error"), "report could not be written"),
    ],
)
def test_main_reports_the_actual_failure_category(monkeypatch, caplog, error, message):
    def fail_verification(**_kwargs):
        raise error

    monkeypatch.setattr(color_verifier, "verify_directory", fail_verification)

    with caplog.at_level(logging.ERROR):
        exit_code = color_verifier.main(
            [
                "--input-dir",
                "images",
                "--color-stats",
                "stats.json",
                "--output-json",
                "",
                "--output-csv",
                "",
            ]
        )

    assert exit_code == 1
    assert message in caplog.text


def test_visualize_debug_warns_when_optional_backend_is_unavailable(monkeypatch, caplog, tmp_path):
    real_import = builtins.__import__

    def reject_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib.pyplot":
            raise ImportError("matplotlib disabled for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_matplotlib)

    with caplog.at_level(logging.WARNING):
        color_verifier.visualize_debug(
            image_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
            predicted_color="Unknown",
            confidence=0.0,
            ratios={"Black": 0.0},
            mask=np.zeros((2, 2), dtype=bool),
            output_path=Path(tmp_path) / "debug.png",
        )

    assert "Debug visualization unavailable" in caplog.text
    assert not (tmp_path / "debug.png").exists()
