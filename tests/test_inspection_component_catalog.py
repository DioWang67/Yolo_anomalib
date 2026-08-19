from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import yaml

from core.services.inspection_component_catalog import (
    InspectionComponentCatalog,
)
from core.services.inspection_release_builder import build_draft_release
from core.services.inspection_release_models import ActivationMode
from core.services.inspection_release_store import InspectionReleaseStore
from core.services.model_version_registry import ModelVersionRegistry
from tools.color_calibration_service import (
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationScope,
)
from tools.color_configuration_revisions import (
    ColorConfigurationRevision,
    ColorConfigurationRevisionStore,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _model_registry(tmp_path: Path) -> ModelVersionRegistry:
    models_root = tmp_path / "models"
    target = models_root / "Cable1" / "A" / "yolo"
    weight = target / "weights" / "Cable1_A_v1.0.6_20260727.onnx"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"model-v1.0.6")
    snapshot = target / "versions" / f"{weight.name}.config.yaml"
    _write_yaml(snapshot, {"weights": str(weight), "enable_yolo": True})
    _write_yaml(
        weight.with_name(f"{weight.name}.manifest.yaml"),
        {
            "schema_version": 1,
            "product": "Cable1",
            "area": "A",
            "model_type": "yolo",
            "deployed_version": "1.0.6",
            "deployed_file": weight.name,
            "trained_at": "2026-07-27T12:00:00+08:00",
            "deployed_at": "2026-07-27T12:00:00+08:00",
            "weight_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "config_snapshot": f"versions/{snapshot.name}",
        },
    )
    _write_yaml(target / "config.yaml", {"weights": str(weight)})
    return ModelVersionRegistry(models_root)


def _color_versions(
    tmp_path: Path,
) -> tuple[
    ColorConfigurationRevisionStore,
    ColorConfigurationRevision,
    ColorConfigurationRevision,
]:
    scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", "black")
    sequence = iter(
        ("color-revision-1", "color-activation-1", "color-revision-2")
    )
    store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: datetime(2026, 7, 30, tzinfo=timezone.utc),
        id_generator=lambda: next(sequence),
    )

    def commit(source: str, value: float, **parent):
        return store.commit_configuration(
            source,
            scope,
            operator="engineer",
            reason="catalog test",
            proposal_sha256=f"proposal-{source}",
            preview_sha256=f"preview-{source}",
            proposed_config={
                "schema_version": COLOR_CONFIG_SCHEMA_VERSION,
                "scope": {
                    "product": scope.product,
                    "area": scope.area,
                    "model_type": scope.model_type,
                    "checker_type": scope.checker_type,
                    "threshold_key": scope.threshold_key,
                    "scope_hash": scope.scope_hash,
                },
                "threshold": value,
            },
            metrics={"overkill_rate": 0.01},
            **parent,
        )

    default = commit("source-1", 0.5)
    store.activate(
        default,
        operator="engineer",
        reason="default",
        expected_current_sha256="",
    )
    deployed = commit(
        "source-2",
        0.6,
        parent_revision_id=default.revision_id,
        parent_config_sha256=default.new_config_sha256,
    )
    return store, default, deployed


def test_catalog_distinguishes_production_combination_from_default_pointers(
    tmp_path: Path,
) -> None:
    registry = _model_registry(tmp_path)
    model = registry.list_versions()[0]
    _color_store, default_color, deployed_color = _color_versions(tmp_path)
    release_store = InspectionReleaseStore(tmp_path / ".inspection_releases")
    release = build_draft_release(
        model,
        display_version="inspection-v1.0.2",
        operator="engineer",
        reason="catalog test",
        color_revision=deployed_color,
    )
    release_store.commit(release)
    release_store.activate(
        release,
        mode=ActivationMode.RISK_ACCEPTED,
        operator="engineer",
        reason="catalog test",
        expected_release_id=None,
    )

    catalog = InspectionComponentCatalog(
        models_root=tmp_path / "models",
        color_revisions_root=tmp_path / ".color_revisions",
        inspection_releases_root=tmp_path / ".inspection_releases",
    )
    records = catalog.list_components()
    by_version = {record.version: record for record in records}

    assert by_version["1.0.6"].status == "DEPLOYED"
    assert by_version[deployed_color.display_version].status == "DEPLOYED"
    assert by_version[default_color.display_version].status == "DEFAULT"
    assert all(record.can_compose for record in records)


def test_catalog_ignores_color_revision_infrastructure_directories(
    tmp_path: Path,
) -> None:
    models_root = tmp_path / "models"
    models_root.mkdir()
    revisions_root = tmp_path / ".color_revisions"
    for directory_name in (
        "active",
        "activation_events",
        "future-infrastructure",
        "locks",
        "revocations",
    ):
        infrastructure_root = revisions_root / directory_name
        infrastructure_root.mkdir(parents=True)
        (infrastructure_root / "revision.json").write_text(
            "{}",
            encoding="utf-8",
        )

    catalog = InspectionComponentCatalog(
        models_root=models_root,
        color_revisions_root=revisions_root,
    )

    assert catalog.list_components() == ()
