from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from core.services.acceptance_artifacts import (
    AcceptanceArtifactError,
    build_acceptance_artifact_bundle,
    color_scope_model_type,
    verify_acceptance_artifact_bundle,
)


@pytest.mark.parametrize(
    ("inference_type", "expected"),
    (
        ("fusion", "yolo"),
        ("Fusion", "yolo"),
        (" FUSION ", "yolo"),
        ("yolo", "yolo"),
        ("YOLO", "yolo"),
        (" anomaly ", "anomaly"),
    ),
)
def test_color_scope_model_type_normalizes_every_caller_the_same_way(
    inference_type: str,
    expected: str,
) -> None:
    """Fusion shares the YOLO color scope, and case never changes the scope.

    Six copies of this rule had drifted: one skipped the lowercasing, so a
    scope lookup for ``YOLO`` searched a differently named scope than the same
    lookup made from the matrix dialog and found no stored color models.
    """
    assert color_scope_model_type(inference_type) == expected


def _artifact_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    models_root = tmp_path / "models"
    station_root = models_root / "Cable1" / "A" / "yolo"
    station_root.mkdir(parents=True)
    weight = station_root / "candidate.onnx"
    color = station_root / "color.json"
    config = station_root / "config.yaml"
    global_config = tmp_path / "config.yaml"
    weight.write_bytes(b"candidate-weight")
    color.write_text('{"Black": {"threshold": 0.5}}', encoding="utf-8")
    config.write_text(
        "weights: candidate.onnx\n"
        "enable_color_check: true\n"
        "color_checker_type: stats\n"
        "color_model_path: color.json\n",
        encoding="utf-8",
    )
    global_config.write_text("device: cpu\n", encoding="utf-8")
    return models_root, global_config, config, weight


def test_bundle_binds_config_weight_embedded_color_and_contract(
    tmp_path: Path,
) -> None:
    models_root, global_config, config, weight = _artifact_files(tmp_path)
    contract = {"schema_version": 1, "identity_sha256": "c" * 64}

    bundle = build_acceptance_artifact_bundle(
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="1.2.3",
        global_config_path=global_config,
        model_config_path=config,
        models_root=models_root,
        model_weight_path=weight,
        color_revision_overrides={"scope": "revision"},
        color_revision_contract=contract,
    )

    assert bundle.model_weight.path == weight.resolve()
    assert bundle.model_weight.sha256 == hashlib.sha256(weight.read_bytes()).hexdigest()
    assert bundle.color_model is not None
    assert bundle.color_model.path == config.parent / "color.json"
    assert bundle.color_model_mode == "embedded"
    assert len(bundle.bundle_sha256) == 64
    verify_acceptance_artifact_bundle(bundle, models_root=models_root)


def test_bundle_rejects_config_and_selected_weight_mismatch(tmp_path: Path) -> None:
    models_root, global_config, config, _configured_weight = _artifact_files(tmp_path)
    selected_weight = tmp_path / "other.onnx"
    selected_weight.write_bytes(b"other-weight")

    with pytest.raises(AcceptanceArtifactError, match="do not resolve"):
        build_acceptance_artifact_bundle(
            product="Cable1",
            area="A",
            inference_type="yolo",
            version="candidate",
            global_config_path=global_config,
            model_config_path=config,
            models_root=models_root,
            model_weight_path=selected_weight,
        )


def test_bundle_detects_artifact_mutation_after_preparation(tmp_path: Path) -> None:
    models_root, global_config, config, weight = _artifact_files(tmp_path)
    bundle = build_acceptance_artifact_bundle(
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="candidate",
        global_config_path=global_config,
        model_config_path=config,
        models_root=models_root,
        model_weight_path=weight,
    )
    weight.write_bytes(b"mutated-weight")

    with pytest.raises(AcceptanceArtifactError, match="changed during acceptance"):
        verify_acceptance_artifact_bundle(bundle, models_root=models_root)


def test_disabled_color_check_does_not_claim_dormant_color_artifact(
    tmp_path: Path,
) -> None:
    models_root, global_config, config, weight = _artifact_files(tmp_path)
    config.write_text(
        "weights: candidate.onnx\n"
        "enable_color_check: false\n"
        "color_model_path: color.json\n",
        encoding="utf-8",
    )

    bundle = build_acceptance_artifact_bundle(
        product="Cable1",
        area="A",
        inference_type="yolo",
        version="candidate",
        global_config_path=global_config,
        model_config_path=config,
        models_root=models_root,
        model_weight_path=weight,
    )

    assert bundle.color_model is None
    assert bundle.color_model_mode == "disabled"
