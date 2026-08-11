from __future__ import annotations

from pathlib import Path

import pytest

from core.station_data import (
    load_station_data_paths,
    resolve_result_root,
    resolve_review_manifest,
)
from core.workspace import WorkspaceConfigurationError, load_workspace_paths

MANIFEST = """\
schema_version: 1
projects:
  training: Yolo11_auto_train
  inference: yolo11_inference
paths:
  training_data: Yolo11_auto_train/data
  inference_models: yolo11_inference/models
  station_data: station_data/yolo11_inference
  inference_results: Result
  inference_artifacts: release_artifacts/yolo11_inference
"""


def test_workspace_manifest_is_discovered_from_ancestor(tmp_path: Path) -> None:
    workspace_root = tmp_path / "vision-workspace"
    inference_child = workspace_root / "yolo11_inference" / "app" / "gui"
    inference_child.mkdir(parents=True)
    (workspace_root / "workspace.yaml").write_text(MANIFEST, encoding="utf-8")

    paths = load_workspace_paths(inference_child)

    assert paths.root == workspace_root.resolve()
    assert paths.training_project == (workspace_root / "Yolo11_auto_train").resolve()
    assert paths.training_data == (workspace_root / "Yolo11_auto_train" / "data").resolve()
    assert paths.inference_models == (workspace_root / "yolo11_inference" / "models").resolve()
    assert paths.station_data == (workspace_root / "station_data" / "yolo11_inference").resolve()
    assert paths.inference_results == (workspace_root / "Result").resolve()
    assert paths.inference_artifacts == (
        workspace_root / "release_artifacts" / "yolo11_inference"
    ).resolve()
    assert paths.manifest_path == (workspace_root / "workspace.yaml").resolve()


def test_environment_workspace_takes_precedence(tmp_path: Path, monkeypatch) -> None:
    environment_root = tmp_path / "configured"
    environment_root.mkdir()
    (environment_root / "workspace.yaml").write_text(MANIFEST, encoding="utf-8")
    nearer_root = tmp_path / "nearer"
    child = nearer_root / "yolo11_inference"
    child.mkdir(parents=True)
    (nearer_root / "workspace.yaml").write_text(MANIFEST, encoding="utf-8")
    monkeypatch.setenv("YOLO11_WORKSPACE_ROOT", str(environment_root))

    paths = load_workspace_paths(child)

    assert paths.root == environment_root.resolve()


def test_environment_workspace_requires_manifest(tmp_path: Path, monkeypatch) -> None:
    environment_root = tmp_path / "configured"
    environment_root.mkdir()
    monkeypatch.setenv("YOLO11_WORKSPACE_ROOT", str(environment_root))

    with pytest.raises(WorkspaceConfigurationError, match="does not contain"):
        load_workspace_paths(tmp_path)


@pytest.mark.parametrize(
    "unsafe_value",
    ("../outside/data", "../../outside"),
)
def test_workspace_rejects_paths_outside_manifest_root(
    tmp_path: Path, unsafe_value: str
) -> None:
    manifest = MANIFEST.replace("Yolo11_auto_train/data", unsafe_value)
    (tmp_path / "workspace.yaml").write_text(manifest, encoding="utf-8")

    with pytest.raises(WorkspaceConfigurationError, match="training_data"):
        load_workspace_paths(tmp_path)


def test_workspace_rejects_absolute_path(tmp_path: Path) -> None:
    unsafe_value = tmp_path.resolve().as_posix()
    manifest = MANIFEST.replace("Yolo11_auto_train/data", unsafe_value)
    (tmp_path / "workspace.yaml").write_text(manifest, encoding="utf-8")

    with pytest.raises(WorkspaceConfigurationError, match="training_data"):
        load_workspace_paths(tmp_path)


def test_legacy_sibling_layout_remains_supported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("YOLO11_WORKSPACE_ROOT", raising=False)
    inference_root = tmp_path / "yolo11_inference"
    nested = inference_root / "app" / "gui"
    nested.mkdir(parents=True)

    paths = load_workspace_paths(nested)

    assert paths.manifest_path is None
    assert paths.inference_project == inference_root.resolve()
    assert paths.training_data == (tmp_path / "Yolo11_auto_train" / "data").resolve()
    assert paths.station_data == inference_root.resolve()
    assert paths.inference_results == (inference_root / "Result").resolve()
    assert paths.inference_artifacts == inference_root.resolve()


def test_explicit_legacy_root_does_not_fall_back_to_source_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("YOLO11_WORKSPACE_ROOT", raising=False)
    deployment_root = tmp_path / "extracted-station"
    deployment_root.mkdir()

    paths = load_workspace_paths(deployment_root)

    assert paths.inference_project == deployment_root.resolve()
    assert paths.station_data == deployment_root.resolve()
    assert paths.inference_results == (deployment_root / "Result").resolve()


def test_optional_station_paths_default_to_inference_project(tmp_path: Path) -> None:
    manifest = MANIFEST.replace(
        "  station_data: station_data/yolo11_inference\n"
        "  inference_results: Result\n"
        "  inference_artifacts: release_artifacts/yolo11_inference\n",
        "",
    )
    (tmp_path / "workspace.yaml").write_text(manifest, encoding="utf-8")

    paths = load_workspace_paths(tmp_path)

    assert paths.station_data == (tmp_path / "yolo11_inference").resolve()
    assert paths.inference_results == (
        tmp_path / "yolo11_inference" / "Result"
    ).resolve()
    assert paths.inference_artifacts == (tmp_path / "yolo11_inference").resolve()


def test_station_cli_defaults_use_workspace_results_and_review_manifest(
    tmp_path: Path,
) -> None:
    (tmp_path / "workspace.yaml").write_text(MANIFEST, encoding="utf-8")

    assert resolve_result_root(None, start=tmp_path) == (tmp_path / "Result").resolve()
    assert resolve_review_manifest(None, start=tmp_path) == (
        tmp_path / "station_data" / "yolo11_inference" / "review_manifest.csv"
    ).resolve()
    assert resolve_review_manifest(
        None,
        start=tmp_path,
        default_name="review_manifest.json",
    ) == (
        tmp_path / "station_data" / "yolo11_inference" / "review_manifest.json"
    ).resolve()


def test_explicit_station_cli_paths_do_not_discover_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_workspace = tmp_path / "invalid-workspace"
    invalid_workspace.mkdir()
    monkeypatch.setenv("YOLO11_WORKSPACE_ROOT", str(invalid_workspace))
    explicit_result = tmp_path / "custom-results"
    explicit_manifest = tmp_path / "custom-review.csv"

    assert resolve_result_root(explicit_result) == explicit_result.resolve()
    assert resolve_review_manifest(explicit_manifest) == explicit_manifest.resolve()


def test_station_defaults_do_not_depend_on_process_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    expected = load_station_data_paths(project_root)
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.delenv("YOLO11_WORKSPACE_ROOT", raising=False)
    monkeypatch.chdir(unrelated_cwd)

    assert load_station_data_paths() == expected
    assert resolve_result_root(None) == expected.results
    assert resolve_review_manifest(None) == expected.default_review_manifest
