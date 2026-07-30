from __future__ import annotations

import yaml

from app.gui.model_config_dialog import ModelConfigDialog


def test_model_config_dialog_loads_duplicate_filter_controls(qtbot, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "best.onnx",
                "enable_color_check": True,
                "pipeline": [
                    "color_check",
                    "cross_class_duplicate_filter",
                    "save_results",
                ],
                "steps": {
                    "cross_class_duplicate_filter": {
                        "enabled": True,
                        "mode": "suppress",
                        "iou_threshold": 0.91,
                        "center_distance_ratio_max": 0.08,
                        "area_similarity_min": 0.85,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    dialog = ModelConfigDialog(
        product="Cable1",
        area="A",
        inference_type="yolo",
        config_path=config_path,
        language="zh",
    )
    qtbot.addWidget(dialog)

    changes = dialog.changes()
    assert changes["duplicate_filter_enabled"] is True
    assert changes["duplicate_filter_mode"] == "suppress"
    assert changes["duplicate_filter_iou_threshold"] == 0.91
    assert changes["duplicate_filter_center_distance_ratio_max"] == 0.08
    assert changes["duplicate_filter_area_similarity_min"] == 0.85
