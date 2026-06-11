import textwrap

import pytest

from core.config import ConfigValidationError, DetectionConfig


def write_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")
    return path


def test_from_yaml_requires_weights(tmp_path):
    cfg_path = write_config(
        tmp_path,
        """
        device: cuda:0
    """,
    )
    with pytest.raises(ConfigValidationError):
        DetectionConfig.from_yaml(str(cfg_path))


def test_from_yaml_normalizes_pipeline_and_numeric(tmp_path):
    cfg_path = write_config(
        tmp_path,
        """
        weights: "models/model.pt"
        pipeline: [save_results]
        jpeg_quality: 88
        png_compression: 7
        max_crops_per_frame: 5
    """,
    )
    cfg = DetectionConfig.from_yaml(str(cfg_path))
    assert cfg.pipeline == ["save_results"]
    assert cfg.jpeg_quality == 88
    assert cfg.png_compression == 7
    assert cfg.max_crops_per_frame == 5


def test_from_yaml_parses_camera_resilience_fields(tmp_path):
    cfg_path = write_config(
        tmp_path,
        """
        weights: "models/model.pt"
        camera_lost_threshold: 3
        camera_reconnect_attempts: 4
        camera_reconnect_backoff: 1.5
    """,
    )
    cfg = DetectionConfig.from_yaml(str(cfg_path))
    assert cfg.camera_lost_threshold == 3
    assert cfg.camera_reconnect_attempts == 4
    assert cfg.camera_reconnect_backoff == 1.5


def test_camera_resilience_defaults_keep_legacy_behavior(tmp_path):
    cfg_path = write_config(tmp_path, 'weights: "models/model.pt"')
    cfg = DetectionConfig.from_yaml(str(cfg_path))
    assert cfg.camera_lost_threshold == 5
    assert cfg.camera_reconnect_attempts == 0  # auto-reconnect disabled
    assert cfg.camera_reconnect_backoff == 2.0


def test_local_overlay_overrides_global_values(tmp_path):
    write_config(
        tmp_path,
        """
        weights: "models/model.pt"
        exposure_time: "1000"
        gain: "1.0"
    """,
    )
    (tmp_path / "config.local.yaml").write_text(
        'exposure_time: "51170.0"\noutput_dir: "Result_station2"\n',
        encoding="utf-8",
    )
    cfg = DetectionConfig.from_yaml(str(tmp_path / "config.yaml"))
    assert cfg.exposure_time == "51170.0"  # overridden
    assert cfg.output_dir == "Result_station2"  # added
    assert cfg.gain == "1.0"  # untouched


def test_local_overlay_must_be_mapping(tmp_path):
    write_config(tmp_path, 'weights: "models/model.pt"')
    (tmp_path / "config.local.yaml").write_text(
        "- not\n- a\n- mapping\n", encoding="utf-8"
    )
    with pytest.raises(ConfigValidationError, match="Local overlay"):
        DetectionConfig.from_yaml(str(tmp_path / "config.yaml"))


def test_normalize_model_dict_allows_missing_imgsz(tmp_path):
    cfg = DetectionConfig.normalize_model_dict({}, "test")
    assert cfg.get("imgsz") is None


def test_normalize_model_dict_validates_imgsz(tmp_path):
    with pytest.raises(ConfigValidationError):
        DetectionConfig.normalize_model_dict({"imgsz": [640]}, "test")


@pytest.mark.parametrize(
    "payload",
    [
        {"conf_thres": 1.5},
        {"iou_thres": -0.1},
        {"jpeg_quality": 101},
        {"png_compression": 10},
        {"buffer_limit": 0},
        {"imgsz": [640, 0]},
    ],
)
def test_normalize_model_dict_rejects_out_of_range_values(payload):
    with pytest.raises(ConfigValidationError):
        DetectionConfig.normalize_model_dict(payload, "test")
