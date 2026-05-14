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


def test_update_model_config_writes_position_and_count_options(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "old.pt",
                "position_config": {
                    "PCBA1": {
                        "A": {
                            "enabled": True,
                            "expected_boxes": {"J1": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                        }
                    }
                },
                "steps": {"count_check": {"strict": False}},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    update_model_config(
        config_path,
        {
            "position_check_enabled": False,
            "position_mode": "center",
            "position_tolerance": 2.5,
            "position_tolerance_unit": "pixel",
            "position_alignment_enabled": False,
            "missing_slot_check_enabled": True,
            "count_check_strict": True,
        },
        product="PCBA1",
        area="A",
    )

    saved = load_model_config(config_path)
    area_cfg = saved["position_config"]["PCBA1"]["A"]
    assert area_cfg["enabled"] is False
    assert area_cfg["mode"] == "center"
    assert area_cfg["tolerance"] == 2.5
    assert area_cfg["tolerance_unit"] == "pixel"
    assert area_cfg["alignment"]["enabled"] is False
    assert area_cfg["missing_slot_check"]["enabled"] is True
    assert area_cfg["expected_boxes"] == {"J1": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}
    assert saved["steps"]["count_check"]["strict"] is True


def test_update_model_config_rejects_invalid_position_unit(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: old.pt\n", encoding="utf-8")

    with pytest.raises(ModelConfigEditError, match="position_tolerance_unit"):
        update_model_config(
            config_path,
            {"position_tolerance_unit": "mm"},
            product="PCBA1",
            area="A",
        )
