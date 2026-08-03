import copy
import sys
import types
from pathlib import Path

import pytest
import yaml

from core.config import DetectionConfig
from core.exceptions import ModelConfigError
from core.logging_config import DetectionLogger
from core.security import SecurityError
from core.services.model_manager import ModelManager


class _FakeInferenceEngine:
    """Config-only test double that avoids importing Torch native DLLs."""

    def __init__(self, config):
        self.config = config

    def initialize(self):
        return True

    def shutdown(self):
        return None


def test_detection_config_defers_device_resolution():
    config = DetectionConfig(weights="model.onnx")

    assert config.device == "auto"


def _write_global_config(tmp_path: Path, weights_path: Path) -> Path:
    cfg = {
        "weights": str(weights_path),
        "enable_yolo": True,
        "enable_anomalib": False,
        "output_dir": "Result",
        "exposure_time": "51170.0000",
        "gain": "1.0",
        "light_brightness": 80,
        "calibration": {"target_luma": 100.0, "tolerance": 4.0},
        "expected_items": {"Cable1": {"A": ["Item1"]}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _write_model_config(model_dir: Path, weights_path: Path) -> Path:
    config = {
        "weights": str(weights_path),
        "enable_yolo": True,
        "color_model_path": "color.json",
        "output_dir": "outputs",
        "exposure_time": "22380.0000",
        "gain": "23.0",
        "light_brightness": 0,
        "calibration": {"target_luma": 60.3, "tolerance": 2.0},
    }
    path = model_dir / "config.yaml"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "color.json").write_text("{}", encoding="utf-8")
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _write_pcba_model_config(model_dir: Path, weights_path: Path) -> Path:
    config = {
        "weights": str(weights_path),
        "enable_yolo": True,
        "output_dir": "outputs",
        "expected_items": {"PCBA1": {"A": ["J5-1", "J5-2"]}},
        "position_config": {
            "PCBA1": {
                "A": {
                    "enabled": True,
                    "expected_boxes": {
                        "J5-1": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                        "J5-2": {"x1": 5, "y1": 6, "x2": 7, "y2": 8},
                    },
                }
            }
        },
    }
    path = model_dir / "config.yaml"
    model_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_model_overrides_resolve_relative_paths_and_keep_globals(tmp_path, monkeypatch):
    # Arrange: global config + model config under temp models path
    weights_path = tmp_path / "dummy.pt"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)

    models_root = tmp_path / "models" / "Cable1" / "A" / "yolo"
    _write_model_config(models_root, weights_path)

    # Ensure cwd points to temp repo root so models/<...>/config.yaml is found
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    logger = DetectionLogger()
    station_root = tmp_path / "station"
    manager = ModelManager(
        logger,
        engine_factory=_FakeInferenceEngine,
        output_root=station_root,
    )

    # Act
    engine, cfg_snapshot = manager.switch(
        base_config, product="Cable1", area="A", inference_type="yolo"
    )

    # Result output paths stay station-relative, not under the model bundle.
    expected_output_dir = str((station_root / "outputs").resolve())
    assert cfg_snapshot.output_dir == expected_output_dir

    # Model resources still resolve relative to the model config folder.
    assert cfg_snapshot.color_model_path == str((models_root / "color.json").resolve())

    # Global-only fields should stay when not overridden
    assert cfg_snapshot.expected_items == {"Cable1": {"A": ["Item1"]}}

    # Per-model camera/calibration values must override global startup values.
    assert cfg_snapshot.exposure_time == "22380.0000"
    assert cfg_snapshot.gain == "23.0"
    assert cfg_snapshot.light_brightness == 0
    assert cfg_snapshot.calibration == {"target_luma": 60.3, "tolerance": 2.0}

    # Engine initialized lazily
    assert engine is not None


def test_exact_config_override_reads_history_without_changing_active_config(
    tmp_path,
):
    weights_path = tmp_path / "historical.onnx"
    weights_path.write_bytes(b"historical")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    active_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    active_path = _write_model_config(active_dir, weights_path)
    active_before = active_path.read_bytes()
    historical_path = tmp_path / "history" / "v1.0.5.config.yaml"
    historical_path.parent.mkdir()
    historical_path.write_text(
        yaml.safe_dump(
            {
                "weights": str(weights_path),
                "enable_yolo": True,
                "exposure_time": "10500.0000",
            }
        ),
        encoding="utf-8",
    )
    manager = ModelManager(
        DetectionLogger(),
        engine_factory=_FakeInferenceEngine,
        models_root=tmp_path / "models",
        model_config_overrides={
            ("Cable1", "A", "yolo"): historical_path
        },
    )

    _, config = manager.switch(
        DetectionConfig.from_yaml(str(global_cfg_path)),
        product="Cable1",
        area="A",
        inference_type="yolo",
    )

    assert config.exposure_time == "10500.0000"
    assert active_path.read_bytes() == active_before


def test_model_overrides_apply_expected_items_from_model_config(tmp_path, monkeypatch):
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    models_root = tmp_path / "models" / "PCBA1" / "A" / "yolo"
    _write_pcba_model_config(models_root, weights_path)
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    _, cfg_snapshot = manager.switch(
        base_config, product="PCBA1", area="A", inference_type="yolo"
    )

    assert cfg_snapshot.get_items_by_area("PCBA1", "A") == ["J5-1", "J5-2"]


def test_model_camera_and_calibration_settings_override_global_values():
    """Saved per-model hardware settings must survive restart-time merging."""
    base_config = DetectionConfig(
        weights="global.onnx",
        exposure_time="51170.0000",
        gain="1.0",
        light_brightness=80,
        calibration={"target_luma": 100.0, "tolerance": 4.0},
    )
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    manager._apply_model_config(
        base_config,
        {
            "exposure_time": "22380.0000",
            "gain": "23.0",
            "light_brightness": 0,
            "calibration": {"target_luma": 60.3, "tolerance": 2.0},
        },
    )

    assert base_config.exposure_time == "22380.0000"
    assert base_config.gain == "23.0"
    assert base_config.light_brightness == 0
    assert base_config.calibration == {"target_luma": 60.3, "tolerance": 2.0}


def test_missing_model_camera_settings_preserve_global_values():
    """Schema-produced None values must not erase global hardware defaults."""
    base_config = DetectionConfig(
        weights="global.onnx",
        exposure_time="51170.0000",
        gain="1.0",
        light_brightness=80,
        calibration={"target_luma": 100.0, "tolerance": 4.0},
    )
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    manager._apply_model_config(
        base_config,
        {
            "exposure_time": None,
            "gain": None,
            "light_brightness": None,
            "calibration": None,
        },
    )

    assert base_config.exposure_time == "51170.0000"
    assert base_config.gain == "1.0"
    assert base_config.light_brightness == 80
    assert base_config.calibration == {"target_luma": 100.0, "tolerance": 4.0}


def test_schema_none_values_preserve_global_scalar_settings():
    """Schema defaults must not silently disable or corrupt global settings."""
    base_config = DetectionConfig(
        weights="global.onnx",
        device="cpu",
        conf_thres=0.61,
        iou_thres=0.37,
        enable_yolo=True,
        enable_color_check=True,
    )
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    manager._apply_model_config(
        base_config,
        {
            "device": None,
            "conf_thres": None,
            "iou_thres": None,
            "enable_yolo": None,
            "enable_color_check": None,
        },
    )

    assert base_config.device == "cpu"
    assert base_config.conf_thres == pytest.approx(0.61)
    assert base_config.iou_thres == pytest.approx(0.37)
    assert base_config.enable_yolo is True
    assert base_config.enable_color_check is True


def test_switch_never_mutates_base_config(tmp_path, monkeypatch):
    """switch() must treat the shared global config as read-only.

    Workers and the GUI may read the global config concurrently; overrides
    go onto a returned copy (adopted by atomic reference swap), both on
    cache miss and cache hit.
    """
    weights_path = tmp_path / "dummy.pt"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    models_root = tmp_path / "models" / "Cable1" / "A" / "yolo"
    _write_model_config(models_root, weights_path)
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    snapshot_before = copy.deepcopy(base_config.__dict__)
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    _, merged = manager.switch(
        base_config, product="Cable1", area="A", inference_type="yolo"
    )

    assert merged is not base_config
    assert base_config.__dict__ == snapshot_before, "cache-miss switch mutated base"

    _, merged_again = manager.switch(
        base_config, product="Cable1", area="A", inference_type="yolo"
    )

    assert merged_again is not merged, "cache hit must return a fresh copy"
    assert base_config.__dict__ == snapshot_before, "cache-hit switch mutated base"
    assert merged_again.output_dir == merged.output_dir


def test_switch_reloads_engine_after_deployed_config_changes(tmp_path, monkeypatch):
    """Atomic deployment updates config last; the next switch must reload it."""
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"model")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_config = _write_model_config(model_dir, weights_path)
    monkeypatch.chdir(tmp_path)

    class FakeEngine:
        instances = []

        def __init__(self, config):
            self.config = config
            self.shutdown_count = 0
            self.instances.append(self)

        def initialize(self):
            return True

        def shutdown(self):
            self.shutdown_count += 1

    engine_module = types.ModuleType("core.inference_engine")
    engine_module.InferenceEngine = FakeEngine
    monkeypatch.setitem(sys.modules, "core.inference_engine", engine_module)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
    )
    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(DetectionLogger())

    first, _ = manager.switch(base_config, "Cable1", "A", "yolo")
    unchanged, _ = manager.switch(base_config, "Cable1", "A", "yolo")
    config = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    config["conf_thres"] = 0.412345
    model_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    reloaded, merged = manager.switch(base_config, "Cable1", "A", "yolo")

    assert unchanged is first
    assert reloaded is not first
    assert first.shutdown_count == 1
    assert merged.conf_thres == pytest.approx(0.412345)


def test_failed_replacement_keeps_previous_engine_active(tmp_path, monkeypatch):
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"model")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_config = _write_model_config(model_dir, weights_path)
    monkeypatch.chdir(tmp_path)

    class FakeEngine:
        instances = []

        def __init__(self, config):
            self.config = config
            self.shutdown_count = 0
            self.instances.append(self)

        def initialize(self):
            if len(self.instances) == 2:
                assert self.instances[0].shutdown_count == 0
                return False
            return True

        def shutdown(self):
            self.shutdown_count += 1

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(DetectionLogger(), engine_factory=FakeEngine)
    active, _ = manager.switch(base_config, "Cable1", "A", "yolo")

    payload = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    payload["conf_thres"] = 0.54321
    model_config.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="init failed"):
        manager.switch(base_config, "Cable1", "A", "yolo")

    assert manager.get_cached_engine("Cable1", "A", "yolo") is active
    assert active.shutdown_count == 0
    assert FakeEngine.instances[1].shutdown_count == 1


def test_switch_keeps_engine_for_station_calibration_change(tmp_path, monkeypatch):
    """Exposure/light edits return fresh config without reloading model bytes."""
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"model")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_config = _write_model_config(model_dir, weights_path)
    monkeypatch.chdir(tmp_path)

    class FakeEngine:
        def __init__(self, config):
            self.config = config
            self.shutdown_count = 0

        def initialize(self):
            return True

        def shutdown(self):
            self.shutdown_count += 1

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(DetectionLogger(), engine_factory=FakeEngine)
    first, _ = manager.switch(base_config, "Cable1", "A", "yolo")

    payload = yaml.safe_load(model_config.read_text(encoding="utf-8"))
    payload["exposure_time"] = "79979.0000"
    payload["light_brightness"] = 55
    model_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    second, merged = manager.switch(base_config, "Cable1", "A", "yolo")

    assert second is first
    assert first.shutdown_count == 0
    assert merged.exposure_time == "79979.0000"
    assert merged.light_brightness == 55


def test_switch_has_no_cross_model_contamination(tmp_path, monkeypatch):
    """Each switch merges from the pristine base — values set by a previous
    model (e.g. color_model_path) must not leak into the next one."""
    weights_path = tmp_path / "dummy.pt"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)

    area_a = tmp_path / "models" / "Cable1" / "A" / "yolo"
    _write_model_config(area_a, weights_path)  # sets color_model_path

    area_b = tmp_path / "models" / "Cable1" / "B" / "yolo"
    area_b.mkdir(parents=True)
    (area_b / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": str(weights_path),
                "enable_yolo": True,
                "expected_items": {"Cable1": {"B": ["Item9"]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    _, merged_a = manager.switch(
        base_config, product="Cable1", area="A", inference_type="yolo"
    )
    assert merged_a.color_model_path  # A defines a color model

    _, merged_b = manager.switch(
        base_config, product="Cable1", area="B", inference_type="yolo"
    )
    assert merged_b.color_model_path is None, (
        "B inherited A's color_model_path — switch order leaked state"
    )


def test_model_manager_fails_fast_when_expected_items_missing(
    tmp_path, monkeypatch
):
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    model_dir = tmp_path / "models" / "PCBA1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump({"weights": str(weights_path), "enable_yolo": True}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(DetectionLogger())

    with pytest.raises(ModelConfigError, match="missing expected_items"):
        manager.switch(base_config, product="PCBA1", area="A", inference_type="yolo")


def test_model_config_found_via_project_root_when_cwd_differs(
    tmp_path, monkeypatch
):
    """Frozen exe / shortcut launches run with an arbitrary cwd; the model
    bundle must still resolve against the project root."""
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    models_root = tmp_path / "models" / "PCBA1" / "A" / "yolo"
    _write_pcba_model_config(models_root, weights_path)

    # cwd has no models/ directory; PROJECT_ROOT points at the bundle root.
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    import core.services.model_manager as mm
    monkeypatch.setattr(mm, "PROJECT_ROOT", tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(
        DetectionLogger(), engine_factory=_FakeInferenceEngine
    )

    _, cfg_snapshot = manager.switch(
        base_config, product="PCBA1", area="A", inference_type="yolo"
    )

    assert cfg_snapshot.get_items_by_area("PCBA1", "A") == ["J5-1", "J5-2"]


def test_model_manager_rejects_output_dir_outside_injected_root(
    tmp_path,
    monkeypatch,
):
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    model_dir = tmp_path / "models" / "PCBA1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    outside_dir = tmp_path / "outside_results"
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": str(weights_path),
                "enable_yolo": True,
                "output_dir": str(outside_dir),
                "expected_items": {"PCBA1": {"A": ["J5-1"]}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(
        DetectionLogger(),
        output_root=tmp_path / "station",
    )

    with pytest.raises(SecurityError):
        manager.switch(base_config, product="PCBA1", area="A", inference_type="yolo")
