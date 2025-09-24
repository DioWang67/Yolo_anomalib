import textwrap

import pytest

from core.config import ConfigValidationError, DetectionConfig


def write_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


def test_from_yaml_requires_weights(tmp_path):
    cfg_path = write_config(tmp_path, """
        device: cuda:0
    """)
    with pytest.raises(ConfigValidationError):
        DetectionConfig.from_yaml(str(cfg_path))


def test_from_yaml_normalizes_pipeline_and_numeric(tmp_path):
    cfg_path = write_config(tmp_path, """
        weights: "models/model.pt"
        pipeline: [save_results]
        jpeg_quality: 88
        png_compression: 7
        max_crops_per_frame: 5
    """)
    cfg = DetectionConfig.from_yaml(str(cfg_path))
    assert cfg.pipeline == ["save_results"]
    assert cfg.jpeg_quality == 88
    assert cfg.png_compression == 7
    assert cfg.max_crops_per_frame == 5


def test_normalize_model_dict_allows_missing_imgsz(tmp_path):
    cfg = DetectionConfig.normalize_model_dict({}, "test")
    assert cfg.get("imgsz") is None


def test_normalize_model_dict_validates_imgsz(tmp_path):
    with pytest.raises(ConfigValidationError):
        DetectionConfig.normalize_model_dict({"imgsz": [640]}, "test")
