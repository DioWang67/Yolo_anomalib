from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from core.services.model_version_registry import (
    ModelVersionRegistry,
    ModelVersionRegistryError,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _create_target(tmp_path: Path) -> tuple[Path, Path]:
    models_root = tmp_path / "models"
    target = models_root / "PCBA1" / "A" / "yolo"
    weights = target / "weights"
    weights.mkdir(parents=True)
    return models_root, target


def _add_version(
    target: Path,
    filename: str,
    content: bytes,
    *,
    version: str,
    trained_at: str,
    config: dict | None = None,
) -> Path:
    weight_path = target / "weights" / filename
    weight_path.write_bytes(content)
    snapshot_path = target / "versions" / f"{filename}.config.yaml"
    _write_yaml(
        snapshot_path,
        config
        or {
            "weights": f"models/PCBA1/A/yolo/weights/{filename}",
            "conf_thres": 0.25,
        },
    )
    digest = hashlib.sha256(content).hexdigest()
    _write_yaml(
        weight_path.with_name(f"{filename}.manifest.yaml"),
        {
            "schema_version": 1,
            "product": "PCBA1",
            "area": "A",
            "model_type": "yolo",
            "deployed_version": version,
            "deployed_file": filename,
            "trained_at": trained_at,
            "deployed_at": trained_at,
            "weight_sha256": digest,
            "dataset_hash": f"dataset-{version}",
            "training_config_hash": f"config-{version}",
            "evaluation_metrics": {"map50": 0.8},
            "config_snapshot": f"versions/{filename}.config.yaml",
        },
    )
    return weight_path


def test_registry_lists_current_and_historical_versions(tmp_path: Path) -> None:
    models_root, target = _create_target(tmp_path)
    old_name = "PCBA1_A_v1.0.0_20260701.onnx"
    current_name = "PCBA1_A_v1.0.1_20260715.onnx"
    _add_version(
        target,
        old_name,
        b"old",
        version="1.0.0",
        trained_at="2026-07-01T08:00:00+08:00",
    )
    _add_version(
        target,
        current_name,
        b"new",
        version="1.0.1",
        trained_at="2026-07-15T09:30:00+08:00",
    )
    (target / "weights" / "last.pt").write_bytes(b"not deployable")
    _write_yaml(
        target / "config.yaml",
        {"weights": f"models/PCBA1/A/yolo/weights/{current_name}"},
    )

    records = ModelVersionRegistry(models_root).list_versions()

    assert [item.weight_path.name for item in records] == [current_name, old_name]
    assert records[0].is_current is True
    assert records[0].version == "1.0.1"
    assert records[0].trained_at == datetime.fromisoformat(
        "2026-07-15T09:30:00+08:00"
    )
    assert records[0].training_time_inferred is False
    assert records[0].evaluation_metrics == {"map50": 0.8}
    assert records[0].has_config_snapshot is True


def test_activate_restores_version_config_but_preserves_station_fields(
    tmp_path: Path,
) -> None:
    models_root, target = _create_target(tmp_path)
    old_name = "PCBA1_A_v1.0.0_20260701.onnx"
    current_name = "PCBA1_A_v1.0.1_20260715.onnx"
    _add_version(
        target,
        old_name,
        b"old",
        version="1.0.0",
        trained_at="2026-07-01T08:00:00+08:00",
        config={
            "weights": f"models/PCBA1/A/yolo/weights/{old_name}",
            "conf_thres": 0.2,
            "class_names": ["old-class"],
            "exposure_time": "100",
        },
    )
    _add_version(
        target,
        current_name,
        b"new",
        version="1.0.1",
        trained_at="2026-07-15T09:30:00+08:00",
    )
    _write_yaml(
        target / "config.yaml",
        {
            "weights": f"models/PCBA1/A/yolo/weights/{current_name}",
            "conf_thres": 0.6,
            "class_names": ["new-class"],
            "exposure_time": "45678",
            "gain": "12.5",
        },
    )
    registry = ModelVersionRegistry(models_root)
    old_record = next(
        item for item in registry.list_versions() if item.weight_path.name == old_name
    )

    activated = registry.activate(old_record, operator="operator-a")

    config = yaml.safe_load((target / "config.yaml").read_text(encoding="utf-8"))
    assert config["weights"].endswith(old_name)
    assert config["model_version"] == "1.0.0"
    assert config["conf_thres"] == 0.2
    assert config["class_names"] == ["old-class"]
    assert config["exposure_time"] == "45678"
    assert config["gain"] == "12.5"
    assert activated.is_current is True
    history = yaml.safe_load(
        (target / "activation_history.yaml").read_text(encoding="utf-8")
    )
    assert history["events"][-1]["from_file"] == current_name
    assert history["events"][-1]["to_file"] == old_name
    assert history["events"][-1]["operator"] == "operator-a"
    assert list((target / "versions" / "config_backups").glob("*.yaml"))
    assert registry.previous_version("PCBA1", "A", "yolo").weight_path.name == current_name


def test_registry_hides_noncurrent_best_alias(
    tmp_path: Path,
) -> None:
    models_root, target = _create_target(tmp_path)
    current_name = "PCBA1_A_v1.0.1_20260715.onnx"
    _add_version(
        target,
        current_name,
        b"new",
        version="1.0.1",
        trained_at="2026-07-15T09:30:00+08:00",
    )
    legacy_path = target / "weights" / "best.onnx"
    legacy_path.write_bytes(b"legacy")
    _write_yaml(
        target / "config.yaml",
        {"weights": f"models/PCBA1/A/yolo/weights/{current_name}"},
    )
    registry = ModelVersionRegistry(models_root)

    records = registry.list_versions()

    assert legacy_path not in {record.weight_path for record in records}
    assert [record.weight_path.name for record in records] == [current_name]


def test_registry_hides_paired_training_weight_but_keeps_current_legacy_alias(
    tmp_path: Path,
) -> None:
    models_root, target = _create_target(tmp_path)
    current_alias = target / "weights" / "best.onnx"
    paired_training = target / "weights" / "PCBA1_A_v1.0.1_20260715.training.pt"
    current_alias.write_bytes(b"runtime")
    paired_training.write_bytes(b"training")
    _write_yaml(
        target / "config.yaml",
        {"weights": "models/PCBA1/A/yolo/weights/best.onnx"},
    )

    records = ModelVersionRegistry(models_root).list_versions()

    assert [record.weight_path for record in records] == [current_alias.resolve()]
    assert records[0].is_current is True
    assert records[0].version == "legacy"


def test_activate_rejects_artifact_with_wrong_hash(tmp_path: Path) -> None:
    models_root, target = _create_target(tmp_path)
    old_name = "PCBA1_A_v1.0.0_20260701.onnx"
    current_name = "PCBA1_A_v1.0.1_20260715.onnx"
    old_path = _add_version(
        target,
        old_name,
        b"old",
        version="1.0.0",
        trained_at="2026-07-01T08:00:00+08:00",
    )
    _add_version(
        target,
        current_name,
        b"new",
        version="1.0.1",
        trained_at="2026-07-15T09:30:00+08:00",
    )
    _write_yaml(
        target / "config.yaml",
        {"weights": f"models/PCBA1/A/yolo/weights/{current_name}"},
    )
    old_path.write_bytes(b"tampered")
    registry = ModelVersionRegistry(models_root)
    old_record = next(
        item for item in registry.list_versions() if item.weight_path.name == old_name
    )

    with pytest.raises(ModelVersionRegistryError, match="雜湊不符"):
        registry.activate(old_record)

    config = yaml.safe_load((target / "config.yaml").read_text(encoding="utf-8"))
    assert config["weights"].endswith(current_name)


def test_registry_marks_missing_current_artifact(tmp_path: Path) -> None:
    models_root, target = _create_target(tmp_path)
    missing_name = "PCBA1_A_v9.9.9_20260715.onnx"
    _write_yaml(
        target / "config.yaml",
        {"weights": f"models/PCBA1/A/yolo/weights/{missing_name}"},
    )

    records = ModelVersionRegistry(models_root).list_versions()

    assert len(records) == 1
    assert records[0].is_current is True
    assert records[0].exists is False
    assert "不存在" in records[0].warning


def test_registry_recognizes_nested_anomalib_checkpoint_as_current(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "models"
    target = models_root / "PCBA1" / "B" / "anomalib"
    checkpoint = target / "weights" / "model.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_yaml(
        target / "config.yaml",
        {
            "anomalib_config": {
                "models": {
                    "PCBA1": {
                        "B": {
                            "ckpt_path": "models/PCBA1/B/anomalib/weights/model.ckpt"
                        }
                    }
                }
            }
        },
    )

    records = ModelVersionRegistry(models_root).list_versions()

    assert len(records) == 1
    assert records[0].model_type == "anomalib"
    assert records[0].is_current is True
