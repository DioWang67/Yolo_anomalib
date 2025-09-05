# -*- coding: utf-8 -*-
import json
import pytest

from core.color_checker import ColorChecker, compute_enhanced_features


def _make_model(
    tmp_path,
    image,
    threshold=5.0,
    hist_threshold=0.2,
    white_threshold=0.5,
):
    """建立包含直方圖與統計資料的模型檔。"""
    feats = compute_enhanced_features(image)
    model_file = tmp_path / "model.json"
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_bgr": feats["mean_bgr"],
                "hist_h": feats["hist_h"],
                "mask_ratio": feats["mask_ratio"],
                "white_ratio": feats["white_ratio"],
                "threshold": threshold,
                "hist_threshold": hist_threshold,
                "white_threshold": white_threshold,
            },
            f,
        )
    return model_file


def _make_image(color, size=(4, 4)):
    h, w = size
    return [[list(color) for _ in range(w)] for _ in range(h)]


def test_check_pass(tmp_path):
    ref = _make_image([10, 20, 30])
    model_path = _make_model(tmp_path, ref, threshold=5.0)
    checker = ColorChecker(model_path)
    img = _make_image([10, 20, 30])
    result = checker.check(img)
    assert result.is_ok
    assert result.diff == pytest.approx(0.0)


def test_check_fail_color(tmp_path):
    ref = _make_image([10, 20, 30])
    model_path = _make_model(tmp_path, ref, threshold=5.0)
    checker = ColorChecker(model_path)
    img = _make_image([100, 100, 100])
    result = checker.check(img)
    assert not result.is_ok
    assert result.diff > 5.0


def test_check_fail_white(tmp_path):
    ref = _make_image([10, 20, 30])
    model_path = _make_model(tmp_path, ref, white_threshold=0.01)
    checker = ColorChecker(model_path)
    img = _make_image([255, 255, 255])
    result = checker.check(img)
    assert not result.is_ok
