from __future__ import annotations

from pathlib import Path

from core.services.inspection_database import InspectionDatabaseManager
from tools.restore_inspection_database import restore_database


def test_restore_defaults_to_verified_read_only_plan(tmp_path: Path) -> None:
    result_root = tmp_path / "Result"
    manager = InspectionDatabaseManager(result_root / "inspection_records.sqlite3")
    manager.initialize()
    backup = manager.backup()

    payload = restore_database(
        result_root=result_root,
        backup_path=backup.path,
        confirm_application_closed=False,
    )

    assert payload["mode"] == "verify_only"
    assert payload["backup_integrity"] == "ok"


def test_restore_preserves_pre_restore_safety_backup(tmp_path: Path) -> None:
    result_root = tmp_path / "Result"
    database_path = result_root / "inspection_records.sqlite3"
    manager = InspectionDatabaseManager(database_path)
    manager.initialize()
    original = manager.backup()

    with manager.connect() as connection:
        connection.execute(
            """
            INSERT INTO maintenance_events(
                event_type, started_at, completed_at, dry_run, detail_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("later_change", "2026-01-01", "2026-01-01", 0, "{}"),
        )
        connection.commit()

    payload = restore_database(
        result_root=result_root,
        backup_path=original.path,
        confirm_application_closed=True,
    )

    assert payload["mode"] == "restored"
    assert Path(str(payload["pre_restore_backup"])).is_file()
    with manager.connect(readonly=True) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM maintenance_events WHERE event_type='later_change'"
        ).fetchone()[0]
    assert count == 0
