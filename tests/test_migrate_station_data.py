from pathlib import Path

from tools.migrate_station_data import (
    MIGRATION_MANIFEST_NAME,
    build_migration_plan,
    execute_migration,
    rollback_migration,
)


def test_plan_moves_only_allowlisted_runtime_data(tmp_path: Path) -> None:
    source = tmp_path / "source"
    station = tmp_path / "station"
    artifacts = tmp_path / "artifacts"
    source.mkdir()
    for name in ("Result", "acceptance", "models", "app"):
        (source / name).mkdir()
    (source / "review_manifest.csv").write_text("sample_id\n", encoding="utf-8")
    (source / "README.md").write_text("source", encoding="utf-8")

    plan = build_migration_plan(source, station, artifacts)

    moved_names = {entry.source.name for entry in plan}
    assert moved_names == {"Result", "acceptance", "review_manifest.csv"}
    assert all(entry.destination.parent == station for entry in plan)


def test_migration_manifest_supports_full_rollback(tmp_path: Path) -> None:
    source = tmp_path / "source"
    station = tmp_path / "station"
    artifacts = tmp_path / "artifacts"
    source.mkdir()
    (source / "Result").mkdir()
    (source / "Result" / "record.json").write_text("{}", encoding="utf-8")
    (source / "dist").mkdir()
    (source / "dist" / "app.exe").write_text("binary", encoding="utf-8")
    plan = build_migration_plan(source, station, artifacts)
    manifest = station / MIGRATION_MANIFEST_NAME

    execute_migration(plan, manifest_path=manifest)

    assert (station / "Result" / "record.json").is_file()
    assert (artifacts / "dist" / "app.exe").is_file()
    assert manifest.is_file()

    rollback_migration(manifest)

    assert (source / "Result" / "record.json").is_file()
    assert (source / "dist" / "app.exe").is_file()
    assert not manifest.exists()
