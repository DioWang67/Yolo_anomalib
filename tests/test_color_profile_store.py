from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from core.services.color_profile_store import ColorProfileStore
from core.services.inspection_release_builder import build_draft_release
from core.services.inspection_release_models import (
    ActivationMode,
    InspectionReleaseError,
)
from core.services.inspection_release_store import (
    InspectionReleaseResolver,
    InspectionReleaseStore,
)
from core.services.model_version_registry import ModelVersionRecord
from tools.color_calibration_service import (
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationScope,
)
from tools.color_configuration_revisions import (
    ColorConfigurationRevisionStore,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path):
    color_model = tmp_path / "models" / "Cable1" / "A" / "yolo" / "color_stats.json"
    color_model.parent.mkdir(parents=True)
    color_model.write_text(
        json.dumps(
            {
                "summary": {
                    color: {"count": 10}
                    for color in (
                        "Black",
                        "Green",
                        "Orange",
                        "Red",
                        "Yellow",
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    config = color_model.with_name("config.yaml")
    _write_yaml(
        config,
        {
            "weights": "weights/model.onnx",
            "enable_color_check": True,
            "color_checker_type": "stats",
            "color_model_path": str(color_model),
        },
    )
    revision_store = ColorConfigurationRevisionStore(
        root=tmp_path / ".color_revisions",
        clock=lambda: datetime(2026, 7, 31, tzinfo=timezone.utc),
        id_generator=iter(("black-revision", "green-revision")).__next__,
    )
    revisions = []
    for index, (color, value) in enumerate(
        (("black", 0.4), ("green", 0.5)),
        start=1,
    ):
        scope = ColorCalibrationScope("Cable1", "A", "yolo", "stats", color)
        revisions.append(
            revision_store.commit_configuration(
                f"{color}-source",
                scope,
                operator="engineer",
                reason="profile test",
                proposal_sha256=f"{color}-proposal",
                preview_sha256=f"{color}-preview",
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
                    "config_value": value,
                },
                metrics={"sample_count": 10},
                display_version=f"color-v1.0.{index}",
            )
        )
    return color_model, config, tuple(revisions)


def test_profile_snapshots_five_color_baseline_and_multiple_revisions(
    tmp_path: Path,
) -> None:
    source_model, config, revisions = _fixture(tmp_path)
    store = ColorProfileStore(tmp_path / ".color_profiles")

    profile = store.create(
        product="Cable1",
        area="A",
        model_type="yolo",
        model_config_path=config,
        project_root=tmp_path,
        revisions=revisions,
    )

    assert profile is not None
    assert profile.colors == ("Black", "Green", "Orange", "Red", "Yellow")
    assert profile.revision_overrides == tuple(
        sorted(
            (revision.scope.scope_hash, revision.revision_id) for revision in revisions
        )
    )
    assert "black: color-v1.0.1" in profile.summary
    assert profile.color_model_path != source_model
    assert profile.color_model_path.read_bytes() == source_model.read_bytes()
    assert (
        store.create(
            product="Cable1",
            area="A",
            model_type="yolo",
            model_config_path=config,
            project_root=tmp_path,
            revisions=revisions,
        )
        == profile
    )


def test_profile_can_snapshot_an_explicit_baseline_candidate(
    tmp_path: Path,
) -> None:
    source_model, config, _revisions = _fixture(tmp_path)
    candidate_model = tmp_path / ".color_baselines" / "candidate" / "color_stats.json"
    candidate_model.parent.mkdir(parents=True)
    payload = json.loads(source_model.read_text(encoding="utf-8"))
    payload["summary"]["Black"]["count"] = 120
    candidate_model.write_text(json.dumps(payload), encoding="utf-8")

    profile = ColorProfileStore(tmp_path / ".color_profiles").create(
        product="Cable1",
        area="A",
        model_type="yolo",
        model_config_path=config,
        project_root=tmp_path,
        color_model_override=candidate_model,
    )

    assert profile is not None
    assert (
        json.loads(profile.color_model_path.read_text(encoding="utf-8"))["summary"][
            "Black"
        ]["count"]
        == 120
    )
    assert (
        profile.color_model_sha256
        != hashlib.sha256(source_model.read_bytes()).hexdigest()
    )


def test_profile_relocates_legacy_revision_path_without_rewriting_manifest(
    tmp_path: Path,
) -> None:
    inference_root = tmp_path / "inference"
    inference_root.mkdir()
    (tmp_path / "workspace.yaml").write_text(
        """\
schema_version: 1
projects:
  training: training
  inference: inference
paths:
  training_data: training/data
  inference_models: inference/models
  station_data: station/inference
  inference_artifacts: artifacts/inference
""",
        encoding="utf-8",
    )
    station_root = tmp_path / "station" / "inference"
    _source_model, config, revisions = _fixture(station_root)
    store = ColorProfileStore(station_root / ".color_profiles")
    profile = store.create(
        product="Cable1",
        area="A",
        model_type="yolo",
        model_config_path=config,
        project_root=station_root,
        revisions=revisions,
    )
    assert profile is not None
    manifest_payload = json.loads(profile.manifest_path.read_text(encoding="utf-8"))
    expected_paths: dict[str, Path] = {}
    for revision_payload, revision in zip(
        manifest_payload["revisions"],
        revisions,
        strict=True,
    ):
        legacy_path = inference_root / revision.config_path.relative_to(station_root)
        revision_payload["config_path"] = str(legacy_path)
        expected_paths[revision.revision_id] = revision.config_path.resolve()
    profile.manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = store.load(profile.manifest_path)

    assert {
        binding.revision_id: Path(binding.config_path) for binding in loaded.revisions
    } == expected_paths
    persisted_payload = json.loads(profile.manifest_path.read_text(encoding="utf-8"))
    assert all(
        Path(item["config_path"]).is_relative_to(inference_root)
        for item in persisted_payload["revisions"]
    )

    weight = config.parent / "weights" / "model.onnx"
    weight.parent.mkdir()
    weight.write_bytes(b"model")
    model = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.6",
        weight_path=weight,
        is_current=True,
        trained_at=None,
        deployed_at=None,
        activated_at=None,
        training_time_inferred=False,
        config_snapshot_path=config,
        file_size=weight.stat().st_size,
        weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
    )
    release = build_draft_release(
        model,
        display_version="inspection-v1.0.1",
        operator="engineer",
        reason="relocation cache test",
        color_profile=loaded,
    )
    release_store = InspectionReleaseStore(station_root / ".inspection_releases")
    committed = release_store.commit(release)
    release_store.activate(
        committed,
        mode=ActivationMode.LIMITED_TRIAL,
        operator="engineer",
        reason="verify relocated immutable evidence",
        expected_release_id=None,
    )
    resolver = InspectionReleaseResolver(release_store)
    assert resolver.resolve("Cable1", "A", "yolo") == committed
    revisions[0].config_path.write_text("{}", encoding="utf-8")
    with pytest.raises(InspectionReleaseError, match="checksum|SHA|完整性"):
        resolver.resolve("Cable1", "A", "yolo")


def test_release_binds_profile_baseline_and_detects_tampering(
    tmp_path: Path,
) -> None:
    _source_model, config, revisions = _fixture(tmp_path)
    weight = config.parent / "weights" / "model.onnx"
    weight.parent.mkdir()
    weight.write_bytes(b"model")
    model = ModelVersionRecord(
        product="Cable1",
        area="A",
        model_type="yolo",
        version="1.0.6",
        weight_path=weight,
        is_current=True,
        trained_at=None,
        deployed_at=None,
        activated_at=None,
        training_time_inferred=False,
        config_snapshot_path=config,
        file_size=weight.stat().st_size,
        weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),
    )
    profile = ColorProfileStore(tmp_path / ".color_profiles").create(
        product="Cable1",
        area="A",
        model_type="yolo",
        model_config_path=config,
        project_root=tmp_path,
        revisions=revisions,
    )
    assert profile is not None
    release = build_draft_release(
        model,
        display_version="inspection-v1.0.3",
        operator="engineer",
        reason="complete profile",
        color_profile=profile,
    )
    release_store = InspectionReleaseStore(tmp_path / ".inspection_releases")
    release_store.commit(release)

    assert release.color_model_override() == profile.color_model_path
    assert release.color_revision_overrides() == dict(profile.revision_overrides)

    profile.color_model_path.write_text("{}", encoding="utf-8")
    with pytest.raises(InspectionReleaseError, match="checksum|完整性"):
        release_store.load(release.scope, release.release_id)
