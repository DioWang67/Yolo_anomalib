import copy
from pathlib import Path

import pytest
import yaml

from core.config import DetectionConfig
from core.exceptions import ModelConfigError
from core.logging_config import DetectionLogger
from core.path_utils import project_root
from core.security import SecurityError
from core.services.model_manager import ModelManager


def _write_global_config(tmp_path: Path, weights_path: Path) -> Path:
    cfg = {
        "weights": str(weights_path),
        "enable_yolo": True,
        "enable_anomalib": False,
        "output_dir": "Result",
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
    manager = ModelManager(logger)

    # Act
    engine, cfg_snapshot = manager.switch(
        base_config, product="Cable1", area="A", inference_type="yolo"
    )

    # Result output paths stay project-root-relative, not under the model bundle.
    expected_output_dir = str((project_root() / "outputs").resolve())
    assert cfg_snapshot.output_dir == expected_output_dir

    # Model resources still resolve relative to the model config folder.
    assert cfg_snapshot.color_model_path == str((models_root / "color.json").resolve())

    # Global-only fields should stay when not overridden
    assert cfg_snapshot.expected_items == {"Cable1": {"A": ["Item1"]}}

    # Engine initialized lazily
    assert engine is not None


def test_model_overrides_apply_expected_items_from_model_config(tmp_path, monkeypatch):
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"")
    global_cfg_path = _write_global_config(tmp_path, weights_path)
    models_root = tmp_path / "models" / "PCBA1" / "A" / "yolo"
    _write_pcba_model_config(models_root, weights_path)
    monkeypatch.chdir(tmp_path)

    base_config = DetectionConfig.from_yaml(str(global_cfg_path))
    manager = ModelManager(DetectionLogger())

    _, cfg_snapshot = manager.switch(
        base_config, product="PCBA1", area="A", inference_type="yolo"
    )

    assert cfg_snapshot.get_items_by_area("PCBA1", "A") == ["J5-1", "J5-2"]


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
    manager = ModelManager(DetectionLogger())

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
    manager = ModelManager(DetectionLogger())

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
    manager = ModelManager(DetectionLogger())

    _, cfg_snapshot = manager.switch(
        base_config, product="PCBA1", area="A", inference_type="yolo"
    )

    assert cfg_snapshot.get_items_by_area("PCBA1", "A") == ["J5-1", "J5-2"]


def test_model_manager_rejects_output_dir_outside_project(tmp_path, monkeypatch):
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
    manager = ModelManager(DetectionLogger())

    with pytest.raises(SecurityError):
        manager.switch(base_config, product="PCBA1", area="A", inference_type="yolo")
