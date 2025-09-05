# -*- coding: utf-8 -*-
import json
import pytest

from core.color_checker import ColorChecker


def _make_model(tmp_path, mean, threshold=5.0):
    model_file = tmp_path / "model.json"
    with open(model_file, "w", encoding="utf-8") as f:
        json.dump({"mean_bgr": mean, "threshold": threshold}, f)
    return model_file


def _make_image(color, size=(4, 4)):
    h, w = size
    return [[list(color) for _ in range(w)] for _ in range(h)]


def test_check_pass(tmp_path):
    model_path = _make_model(tmp_path, [10, 20, 30], threshold=5.0)
    checker = ColorChecker(model_path)
    img = _make_image([10, 20, 30])
    result = checker.check(img)
    assert result.is_ok
    assert result.diff == pytest.approx(0.0)


def test_check_fail(tmp_path):
    model_path = _make_model(tmp_path, [10, 20, 30], threshold=5.0)
    checker = ColorChecker(model_path)
    img = _make_image([100, 100, 100])
    result = checker.check(img)
    assert not result.is_ok
    assert result.diff > 5.0
