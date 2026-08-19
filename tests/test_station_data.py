from pathlib import Path

from core.station_data import load_station_data_paths


def test_station_data_paths_are_derived_from_workspace_manifest(
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
  inference_models: station/inference/models
  station_data: station/inference
  inference_results: Result
  inference_artifacts: artifacts/inference
""",
        encoding="utf-8",
    )

    paths = load_station_data_paths(inference_root)

    assert paths.root == (tmp_path / "station" / "inference").resolve()
    assert paths.models == (tmp_path / "station" / "inference" / "models").resolve()
    assert paths.results == (tmp_path / "Result").resolve()
    assert paths.default_review_manifest == paths.root / "review_manifest.csv"
    assert paths.artifacts_root == (tmp_path / "artifacts" / "inference").resolve()


def test_legacy_mutable_path_is_relocated_without_rebasing_models(
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
  inference_results: Result
  inference_artifacts: artifacts/inference
""",
        encoding="utf-8",
    )
    relocated_config = (
        tmp_path
        / "station"
        / "inference"
        / ".color_revisions"
        / "scope"
        / "config.json"
    )
    relocated_config.parent.mkdir(parents=True)
    relocated_config.write_text("{}", encoding="utf-8")
    paths = load_station_data_paths(inference_root)

    assert (
        paths.relocate_legacy_path(
            inference_root / ".color_revisions" / "scope" / "config.json"
        )
        == relocated_config.resolve()
    )
    assert (
        paths.relocate_legacy_path(inference_root / "models" / "model.onnx")
        == inference_root / "models" / "model.onnx"
    )


def test_legacy_result_paths_relocate_from_source_and_station_roots(
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
  inference_results: Result
  inference_artifacts: artifacts/inference
""",
        encoding="utf-8",
    )
    relocated = tmp_path / "Result" / "20260803" / "record.json"
    relocated.parent.mkdir(parents=True)
    relocated.write_text("{}", encoding="utf-8")
    paths = load_station_data_paths(inference_root)

    assert paths.relocate_legacy_result_path(
        inference_root / "Result" / "20260803" / "record.json"
    ) == relocated.resolve()
    assert paths.relocate_legacy_result_path(
        tmp_path
        / "station"
        / "inference"
        / "Result"
        / "20260803"
        / "record.json"
    ) == relocated.resolve()
    assert paths.relocate_legacy_path(
        inference_root / "Result" / "20260803" / "record.json"
    ) == relocated.resolve()
