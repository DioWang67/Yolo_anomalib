from __future__ import annotations

import yaml
import pytest

from core.services.model_config_editor import (
    ModelConfigEditError,
    load_model_config,
    update_model_config,
)


def test_update_model_config_writes_common_fields_and_expected_items(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "old.pt",
                "conf_thres": 0.25,
                "expected_items": {"PCBA1": {"A": ["J1"]}},
                "custom": {"keep": True},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = update_model_config(
        config_path,
        {
            "weights": "models/PCBA1/A/yolo/best.onnx",
            "conf_thres": 0.85,
            "iou_thres": 0.45,
            "imgsz": [640, 640],
            "enable_yolo": True,
            "expected_items": "J5-1\nJ5-2\nJ5-1\n",
        },
        product="PCBA1",
        area="A",
    )

    saved = load_model_config(config_path)
    assert saved["weights"] == "models/PCBA1/A/yolo/best.onnx"
    assert saved["conf_thres"] == 0.85
    assert saved["imgsz"] == [640, 640]
    assert saved["expected_items"]["PCBA1"]["A"] == ["J5-1", "J5-2"]
    assert saved["custom"] == {"keep": True}
    assert result.backup_path.exists()


def test_update_model_config_rejects_invalid_threshold(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: old.pt\n", encoding="utf-8")

    with pytest.raises(ModelConfigEditError, match="conf_thres"):
        update_model_config(
            config_path,
            {"conf_thres": 1.5},
            product="PCBA1",
            area="A",
        )


def test_update_model_config_rejects_unknown_field(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: old.pt\n", encoding="utf-8")

    with pytest.raises(ModelConfigEditError, match="不支援"):
        update_model_config(
            config_path,
            {"dangerous": "value"},
            product="PCBA1",
            area="A",
        )
