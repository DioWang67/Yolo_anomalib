from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from tools.relocate_result_data import (
    ResultRelocationError,
    apply_result_relocation,
    rollback_result_relocation,
)


def _create_database(path: Path, legacy_file: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            connection.execute(
                "CREATE TABLE inspections (snapshot_path TEXT, crop_paths_json TEXT)"
            )
            connection.execute("CREATE TABLE inspection_artifacts (path TEXT)")
            connection.execute(
                "INSERT INTO inspections VALUES (?, ?)",
                (str(legacy_file), json.dumps([str(legacy_file)])),
            )
            connection.execute(
                "INSERT INTO inspection_artifacts VALUES (?)",
                (str(legacy_file),),
            )


def test_result_relocation_rewrites_metadata_and_supports_rollback(
    tmp_path: Path,
) -> None:
    inference_root = tmp_path / "inference"
    station_root = tmp_path / "station"
    source = station_root / "Result"
    destination = tmp_path / "Result"
    result_file = source / "20260803" / "record.json"
    result_file.parent.mkdir(parents=True)
    result_file.write_text("{}", encoding="utf-8")
    database_path = source / "inspection_records.sqlite3"
    legacy_file = inference_root / "Result" / "20260803" / "record.json"
    _create_database(database_path, legacy_file)
    review_manifest = station_root / "review_manifest.csv"
    review_manifest.write_text(
        f"config_snapshot_path\n{legacy_file}\n",
        encoding="utf-8",
    )
    (tmp_path / "workspace.yaml").write_text(
        """\
schema_version: 1
projects:
  training: training
  inference: inference
paths:
  training_data: training/data
  inference_models: inference/models
  station_data: station
  inference_results: Result
  inference_artifacts: artifacts
""",
        encoding="utf-8",
    )
    manifest = station_root / "result_relocation_manifest.json"

    apply_result_relocation(
        source,
        destination,
        station_root=station_root,
        manifest_path=manifest,
        metadata_paths=(review_manifest,),
    )

    assert not source.exists()
    assert (destination / "20260803" / "record.json").is_file()
    assert manifest.is_file()
    with closing(
        sqlite3.connect(destination / "inspection_records.sqlite3")
    ) as connection:
        snapshot_path, crop_paths_json = connection.execute(
            "SELECT snapshot_path, crop_paths_json FROM inspections"
        ).fetchone()
        artifact_path = connection.execute(
            "SELECT path FROM inspection_artifacts"
        ).fetchone()[0]
    expected = destination / "20260803" / "record.json"
    assert snapshot_path == str(expected)
    assert json.loads(crop_paths_json) == [str(expected)]
    assert artifact_path == str(expected)
    assert str(expected) in review_manifest.read_text(encoding="utf-8")

    original_manifest = manifest.read_text(encoding="utf-8")
    tampered_payload = json.loads(original_manifest)
    tampered_payload["status"] = "TAMPERED"
    manifest.write_text(json.dumps(tampered_payload), encoding="utf-8")
    with pytest.raises(ResultRelocationError, match="checksum mismatch"):
        rollback_result_relocation(manifest)
    manifest.write_text(original_manifest, encoding="utf-8")

    assert rollback_result_relocation(manifest) == 2

    assert result_file.is_file()
    assert database_path.is_file()
    assert not destination.exists()
    with closing(sqlite3.connect(database_path)) as connection:
        restored = connection.execute(
            "SELECT snapshot_path FROM inspections"
        ).fetchone()[0]
    assert restored == str(legacy_file)
    assert str(legacy_file) in review_manifest.read_text(encoding="utf-8")
