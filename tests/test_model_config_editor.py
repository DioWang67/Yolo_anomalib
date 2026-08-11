from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
import yaml

import core.services.model_config_editor as model_config_editor
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


def test_update_model_config_fails_closed_when_deploy_lock_exists(tmp_path):
    config_path = tmp_path / "config.yaml"
    original = "weights: old.pt\n"
    config_path.write_text(original, encoding="utf-8")
    (tmp_path / ".deploy.lock").mkdir()

    with pytest.raises(ModelConfigEditError, match="deployment is in progress"):
        update_model_config(
            config_path,
            {"weights": "new.pt"},
            product="PCBA1",
            area="A",
        )

    assert config_path.read_text(encoding="utf-8") == original
    assert not config_path.with_suffix(".yaml.bak").exists()
    assert (tmp_path / ".deploy.lock").is_dir()


def test_update_model_config_holds_deploy_lock_through_atomic_publish(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "config.yaml"
    backup_path = config_path.with_suffix(".yaml.bak")
    original = "weights: old.pt\n"
    config_path.write_text(original, encoding="utf-8")
    original_replace = Path.replace
    publish_started = Event()
    allow_publish = Event()

    def replace_while_locked(path, target):
        assert path.parent == config_path.parent
        assert target == config_path
        publish_started.set()
        if not allow_publish.wait(timeout=5):
            raise TimeoutError("test did not release the atomic config publish")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace_while_locked)

    with ThreadPoolExecutor(max_workers=1) as executor:
        edit = executor.submit(
            update_model_config,
            config_path,
            {"weights": "new.pt"},
            product="PCBA1",
            area="A",
        )
        try:
            assert publish_started.wait(timeout=5)
            lock_path = tmp_path / ".deploy.lock"
            assert lock_path.is_dir()
            with pytest.raises(FileExistsError):
                lock_path.mkdir()
        finally:
            allow_publish.set()
        edit.result(timeout=5)

    assert load_model_config(config_path)["weights"] == "new.pt"
    assert backup_path.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".deploy.lock").exists()
    assert not list(tmp_path.glob(".config.yaml.*.tmp"))


def test_update_model_config_holds_deploy_lock_before_read(
    tmp_path,
    monkeypatch,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: old.pt\n", encoding="utf-8")
    original_load = model_config_editor.load_model_config
    read_started = Event()
    allow_read = Event()

    def load_while_locked(path):
        assert (tmp_path / ".deploy.lock").is_dir()
        read_started.set()
        if not allow_read.wait(timeout=5):
            raise TimeoutError("test did not release the locked config read")
        return original_load(path)

    monkeypatch.setattr(model_config_editor, "load_model_config", load_while_locked)

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_edit = executor.submit(
            update_model_config,
            config_path,
            {"weights": "first.pt"},
            product="PCBA1",
            area="A",
        )
        try:
            assert read_started.wait(timeout=5)
            with pytest.raises(ModelConfigEditError, match="deployment is in progress"):
                update_model_config(
                    config_path,
                    {"weights": "stale-second.pt"},
                    product="PCBA1",
                    area="A",
                )
        finally:
            allow_read.set()
        first_edit.result(timeout=5)

    assert load_model_config(config_path)["weights"] == "first.pt"


def test_update_model_config_lock_cleanup_failure_keeps_successful_publication(
    tmp_path,
    monkeypatch,
    caplog,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: old.pt\n", encoding="utf-8")
    real_rmdir = Path.rmdir

    def deny_lock_cleanup(path):
        if path.name == ".deploy.lock":
            raise PermissionError("simulated lock cleanup denial")
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", deny_lock_cleanup)
    caplog.set_level(logging.WARNING, logger=model_config_editor.__name__)

    update_model_config(
        config_path,
        {"weights": "new.pt"},
        product="PCBA1",
        area="A",
    )

    assert load_model_config(config_path)["weights"] == "new.pt"
    assert (tmp_path / ".deploy.lock").is_dir()
    assert "deployment-lock cleanup was deferred" in caplog.text
    assert "simulated lock cleanup denial" in caplog.text


def test_update_model_config_lock_cleanup_failure_does_not_mask_primary_error(
    tmp_path,
    monkeypatch,
    caplog,
):
    config_path = tmp_path / "config.yaml"
    original = "weights: old.pt\nenable_color_check: false\n"
    config_path.write_text(original, encoding="utf-8")
    real_rmdir = Path.rmdir

    def deny_lock_cleanup(path):
        if path.name == ".deploy.lock":
            raise PermissionError("simulated lock cleanup denial")
        return real_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", deny_lock_cleanup)
    caplog.set_level(logging.WARNING, logger=model_config_editor.__name__)

    with pytest.raises(ModelConfigEditError, match="顏色檢查"):
        update_model_config(
            config_path,
            {"duplicate_filter_enabled": True},
            product="PCBA1",
            area="A",
        )

    assert config_path.read_text(encoding="utf-8") == original
    assert (tmp_path / ".deploy.lock").is_dir()
    assert "deployment-lock cleanup was deferred" in caplog.text
    assert "simulated lock cleanup denial" in caplog.text


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


def test_update_model_config_writes_duplicate_filter_and_pipeline_order(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "old.pt",
                "enable_color_check": True,
                "pipeline": [
                    "color_check",
                    "count_check",
                    "sequence_check",
                    "save_results",
                ],
                "steps": {"count_check": {"strict": True}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    update_model_config(
        config_path,
        {
            "duplicate_filter_enabled": True,
            "duplicate_filter_mode": "suppress",
            "duplicate_filter_iou_threshold": 0.90,
            "duplicate_filter_center_distance_ratio_max": 0.10,
            "duplicate_filter_area_similarity_min": 0.80,
        },
        product="PCBA1",
        area="A",
    )

    saved = load_model_config(config_path)
    assert saved["pipeline"] == [
        "color_check",
        "cross_class_duplicate_filter",
        "count_check",
        "sequence_check",
        "save_results",
    ]
    duplicate = saved["steps"]["cross_class_duplicate_filter"]
    assert duplicate["enabled"] is True
    assert duplicate["mode"] == "suppress"
    assert duplicate["iou_threshold"] == 0.90
    assert duplicate["require_same_verified_class"] is True
    assert duplicate["require_color_check_pass"] is True
    assert duplicate["require_position_disabled"] is True
    assert saved["steps"]["count_check"]["strict"] is True


def test_update_model_config_disabling_duplicate_filter_removes_pipeline_step(
    tmp_path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "old.pt",
                "pipeline": [
                    "color_check",
                    "cross_class_duplicate_filter",
                    "save_results",
                ],
                "steps": {
                    "cross_class_duplicate_filter": {
                        "enabled": True,
                        "mode": "suppress",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    update_model_config(
        config_path,
        {"duplicate_filter_enabled": False},
        product="PCBA1",
        area="A",
    )

    saved = load_model_config(config_path)
    assert "cross_class_duplicate_filter" not in saved["pipeline"]
    assert saved["steps"]["cross_class_duplicate_filter"]["enabled"] is False


def test_update_model_config_disabled_duplicate_filter_preserves_implicit_pipeline(
    tmp_path,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weights: old.pt\nenable_color_check: false\n",
        encoding="utf-8",
    )

    update_model_config(
        config_path,
        {
            "duplicate_filter_enabled": False,
            "duplicate_filter_mode": "report_only",
            "duplicate_filter_iou_threshold": 0.90,
            "duplicate_filter_center_distance_ratio_max": 0.10,
            "duplicate_filter_area_similarity_min": 0.80,
        },
        product="PCBA1",
        area="A",
    )

    saved = load_model_config(config_path)
    assert "pipeline" not in saved
    assert "steps" not in saved


def test_update_model_config_requires_color_check_for_duplicate_filter(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weights: old.pt\nenable_color_check: false\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelConfigEditError, match="顏色檢查"):
        update_model_config(
            config_path,
            {"duplicate_filter_enabled": True},
            product="PCBA1",
            area="A",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duplicate_filter_mode", "automatic"),
        ("duplicate_filter_iou_threshold", 0.0),
        ("duplicate_filter_area_similarity_min", 1.1),
    ],
)
def test_update_model_config_rejects_invalid_duplicate_filter_values(
    tmp_path,
    field,
    value,
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: old.pt\n", encoding="utf-8")

    with pytest.raises(ModelConfigEditError, match=field):
        update_model_config(
            config_path,
            {field: value},
            product="PCBA1",
            area="A",
        )


def test_save_calibration_settings_writes_camera_and_target(tmp_path):
    from core.services.model_config_editor import save_calibration_settings

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"weights": "best.onnx"}, sort_keys=False),
        encoding="utf-8",
    )

    result = save_calibration_settings(
        config_path,
        exposure_time=51170.0,
        gain=23.0,
        light_brightness=80,
        target_luma=138.5,
        tolerance=3.0,
        roi=(10, 20, 300, 400),
    )

    saved = load_model_config(config_path)
    assert saved["exposure_time"] == "51170.0000"
    assert saved["gain"] == "23.0"
    assert saved["light_brightness"] == 80
    assert saved["calibration"] == {
        "target_luma": 138.5,
        "tolerance": 3.0,
        "roi": [10, 20, 300, 400],
    }
    assert saved["weights"] == "best.onnx"  # untouched
    assert result.backup_path.exists()


def test_save_calibration_settings_partial_update_preserves_existing(tmp_path):
    from core.services.model_config_editor import save_calibration_settings

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "exposure_time": "1000.0000",
                "gain": "5.0",
                "calibration": {"target_luma": 100.0, "tolerance": 4.0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    # Only update exposure and the target luma; gain + tolerance must remain.
    save_calibration_settings(config_path, exposure_time=2000.0, target_luma=150.0)

    saved = load_model_config(config_path)
    assert saved["exposure_time"] == "2000.0000"
    assert saved["gain"] == "5.0"
    assert saved["calibration"]["target_luma"] == 150.0
    assert saved["calibration"]["tolerance"] == 4.0


def test_save_calibration_settings_rejects_out_of_range(tmp_path):
    from core.services.model_config_editor import (
        ModelConfigEditError,
        save_calibration_settings,
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text("weights: best.onnx\n", encoding="utf-8")

    with pytest.raises(ModelConfigEditError):
        save_calibration_settings(config_path, light_brightness=150)
    with pytest.raises(ModelConfigEditError):
        save_calibration_settings(config_path, target_luma=300.0)
    with pytest.raises(ModelConfigEditError):
        save_calibration_settings(config_path, exposure_time=-1.0)
