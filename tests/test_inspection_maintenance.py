from datetime import datetime, timezone
from pathlib import Path

from core.services.inspection_maintenance import (
    InspectionMaintenanceService,
    InspectionRetentionPolicy,
)
from core.services.inspection_repository import InspectionRepository

NOW = datetime(2026, 7, 28, 8, 0, tzinfo=timezone.utc)


def _add_inspection(
    repository: InspectionRepository,
    tmp_path: Path,
    *,
    name: str,
    status: str,
    timestamp: str,
    artifacts: dict,
) -> str:
    return repository.upsert_snapshot(
        {
            "timestamp": timestamp,
            "status": status,
            "detector": "yolo",
            "product": "Cable1",
            "area": "A",
            "artifacts": artifacts,
        },
        snapshot_path=tmp_path / f"{name}_snapshot.json",
    )


def test_plan_follows_pass_and_fail_retention_windows(tmp_path):
    result_root = tmp_path / "Result"
    result_root.mkdir()
    repository = InspectionRepository(result_root / "inspection_records.sqlite3")
    old_pass = result_root / "old_pass.jpg"
    young_pass = result_root / "young_pass.jpg"
    fail_original = result_root / "fail_original.jpg"
    fail_processed = result_root / "fail_processed.jpg"
    for path in (old_pass, young_pass, fail_original, fail_processed):
        path.write_bytes(b"image")
    _add_inspection(
        repository,
        tmp_path,
        name="old-pass",
        status="PASS",
        timestamp="2026-06-01T00:00:00+00:00",
        artifacts={"original_path": str(old_pass)},
    )
    _add_inspection(
        repository,
        tmp_path,
        name="young-pass",
        status="PASS",
        timestamp="2026-07-20T00:00:00+00:00",
        artifacts={"original_path": str(young_pass)},
    )
    _add_inspection(
        repository,
        tmp_path,
        name="old-fail",
        status="FAIL",
        timestamp="2026-04-01T00:00:00+00:00",
        artifacts={
            "original_path": str(fail_original),
            "preprocessed_path": str(fail_processed),
        },
    )

    candidates = InspectionMaintenanceService(
        result_root,
        now_provider=lambda: NOW,
    ).plan()

    assert {candidate.path for candidate in candidates} == {
        old_pass,
        fail_processed,
    }


def test_shared_artifact_survives_while_any_reference_is_young(tmp_path):
    result_root = tmp_path / "Result"
    result_root.mkdir()
    repository = InspectionRepository(result_root / "inspection_records.sqlite3")
    shared = result_root / "shared.jpg"
    shared.write_bytes(b"shared")
    _add_inspection(
        repository,
        tmp_path,
        name="old",
        status="PASS",
        timestamp="2026-01-01T00:00:00+00:00",
        artifacts={"original_path": str(shared)},
    )
    _add_inspection(
        repository,
        tmp_path,
        name="young",
        status="PASS",
        timestamp="2026-07-27T00:00:00+00:00",
        artifacts={"original_path": str(shared)},
    )

    candidates = InspectionMaintenanceService(
        result_root,
        now_provider=lambda: NOW,
    ).plan()

    assert candidates == ()
    assert shared.is_file()


def test_plan_rejects_paths_outside_result_root_and_non_images(tmp_path):
    result_root = tmp_path / "Result"
    result_root.mkdir()
    repository = InspectionRepository(result_root / "inspection_records.sqlite3")
    outside = tmp_path / "outside.jpg"
    metadata = result_root / "case.json"
    outside.write_bytes(b"outside")
    metadata.write_text("{}", encoding="utf-8")
    _add_inspection(
        repository,
        tmp_path,
        name="unsafe",
        status="PASS",
        timestamp="2026-01-01T00:00:00+00:00",
        artifacts={
            "original_path": str(outside),
            "annotated_path": str(metadata),
        },
    )

    candidates = InspectionMaintenanceService(
        result_root,
        now_provider=lambda: NOW,
    ).plan()

    assert candidates == ()


def test_apply_backs_up_database_deletes_image_and_keeps_metadata(tmp_path):
    result_root = tmp_path / "Result"
    result_root.mkdir()
    repository = InspectionRepository(result_root / "inspection_records.sqlite3")
    expired = result_root / "expired.jpg"
    expired.write_bytes(b"123456")
    inspection_id = _add_inspection(
        repository,
        tmp_path,
        name="expired",
        status="PASS",
        timestamp="2026-01-01T00:00:00+00:00",
        artifacts={"original_path": str(expired)},
    )

    report = InspectionMaintenanceService(
        result_root,
        policy=InspectionRetentionPolicy(),
        now_provider=lambda: NOW,
    ).run(dry_run=False)

    assert report.deleted_files == 1
    assert report.reclaimed_bytes == 6
    assert report.backup_path is not None and report.backup_path.is_file()
    assert not expired.exists()
    inspection = repository.query(
        "SELECT original_path FROM inspections WHERE inspection_id=?",
        (inspection_id,),
    )
    assert inspection == [{"original_path": ""}]
    assert repository.query(
        "SELECT path FROM inspection_artifacts WHERE inspection_id=?",
        (inspection_id,),
    ) == []
    events = repository.query(
        "SELECT event_type, affected_files FROM maintenance_events"
    )
    assert events == [{"event_type": "retention_cleanup", "affected_files": 1}]


def test_dry_run_never_changes_files_or_database(tmp_path):
    result_root = tmp_path / "Result"
    result_root.mkdir()
    repository = InspectionRepository(result_root / "inspection_records.sqlite3")
    expired = result_root / "expired.jpg"
    expired.write_bytes(b"image")
    _add_inspection(
        repository,
        tmp_path,
        name="expired",
        status="PASS",
        timestamp="2026-01-01T00:00:00+00:00",
        artifacts={"original_path": str(expired)},
    )

    report = InspectionMaintenanceService(
        result_root,
        now_provider=lambda: NOW,
    ).run(dry_run=True)

    assert len(report.candidates) == 1
    assert expired.is_file()
    assert repository.query("SELECT * FROM maintenance_events") == []
