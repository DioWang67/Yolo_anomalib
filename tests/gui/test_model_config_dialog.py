from __future__ import annotations

import yaml
from PyQt5.QtWidgets import QDialog, QGroupBox, QMessageBox

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
                "position_config": {
                    "Cable1": {
                        "A": {
                            "enabled": True,
                            "mode": "region",
                            "tolerance": 8.5,
                            "tolerance_unit": "pixel",
                        }
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
    assert dialog.windowTitle() == "檢測參數設定｜Cable1 / A / yolo"
    assert {group.title() for group in dialog.findChildren(QGroupBox)} >= {
        "推論模型",
        "檢測模組與判定規則",
        "跨類別重複框處理",
        "位置與缺件判定",
        "結果與證據保存",
        "應檢元件清單",
    }
    assert changes["duplicate_filter_enabled"] is True
    assert changes["duplicate_filter_mode"] == "suppress"
    assert changes["duplicate_filter_iou_threshold"] == 0.91
    assert changes["duplicate_filter_center_distance_ratio_max"] == 0.08
    assert changes["duplicate_filter_area_similarity_min"] == 0.85
    assert dialog.position_mode_combo.currentText() == "允許區域"
    assert changes["position_mode"] == "region"
    assert dialog.position_unit_combo.currentText() == "像素"
    assert changes["position_tolerance_unit"] == "pixel"


def test_model_config_dialog_requires_confirmation_before_active_update(
    qtbot, tmp_path, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: best.onnx\n", encoding="utf-8")
    dialog = ModelConfigDialog(
        product="PCBA1",
        area="A",
        inference_type="yolo",
        config_path=config_path,
        language="zh",
    )
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.Save,
    )

    dialog._confirm_active_update()

    assert dialog.result() == QDialog.Accepted
