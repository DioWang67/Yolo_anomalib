from pathlib import Path

import yaml

from core.config import DetectionConfig
from core.logging_config import DetectionLogger
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

    # Assert: relative paths resolved against model config folder
    expected_output_dir = str((models_root / "outputs").resolve())
    assert cfg_snapshot.output_dir == expected_output_dir
    assert cfg_snapshot.color_model_path == str((models_root / "color.json").resolve())

    # Global-only fields should stay when not overridden
    assert cfg_snapshot.expected_items == {"Cable1": {"A": ["Item1"]}}

    # Engine initialized lazily
    assert engine is not None
