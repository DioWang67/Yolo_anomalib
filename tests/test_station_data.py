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
  inference_artifacts: artifacts/inference
""",
        encoding="utf-8",
    )

    paths = load_station_data_paths(inference_root)

    assert paths.root == (tmp_path / "station" / "inference").resolve()
    assert paths.models == (tmp_path / "station" / "inference" / "models").resolve()
    assert paths.results == paths.root / "Result"
    assert paths.default_review_manifest == paths.root / "review_manifest.csv"
    assert paths.artifacts_root == (tmp_path / "artifacts" / "inference").resolve()
