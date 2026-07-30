import sqlite3
from pathlib import Path

import pytest

from core.services.inspection_database import (
    SCHEMA_VERSION,
    InspectionDatabaseIntegrityError,
    InspectionDatabaseManager,
    InspectionDatabaseMigrationError,
)
from core.services.inspection_repository import InspectionRepository


def _schema_version(path: Path) -> int:
    with sqlite3.connect(path) as connection:
        return int(connection.execute("SELECT version FROM schema_info").fetchone()[0])


def test_fresh_database_is_created_at_current_schema(tmp_path):
    path = tmp_path / "Result" / "inspection_records.sqlite3"

    repository = InspectionRepository(path)

    assert _schema_version(path) == SCHEMA_VERSION
    assert repository.query(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='maintenance_events'"
    ) == [{"name": "maintenance_events"}]
    assert repository.query(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='inspection_sync_outbox'"
    ) == [{"name": "inspection_sync_outbox"}]
    assert not (path.parent / "database_backups").exists()


def test_v1_database_is_backed_up_and_migrated_atomically(tmp_path):
    path = tmp_path / "inspection_records.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info(version INTEGER NOT NULL);
            INSERT INTO schema_info(version) VALUES (1);
            CREATE TABLE inspections (
                inspection_id TEXT PRIMARY KEY,
                snapshot_path TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                detector TEXT NOT NULL,
                product TEXT NOT NULL,
                station TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

    InspectionRepository(path)

    assert _schema_version(path) == SCHEMA_VERSION
    columns = {
        str(row[1])
        for row in sqlite3.connect(path).execute("PRAGMA table_info(inspections)")
    }
    assert {"machine_id", "model_version", "review_outcome"} <= columns
    backups = list((tmp_path / "database_backups").glob("*.sqlite3.bak"))
    assert len(backups) == 1
    assert _schema_version(backups[0]) == 1


def test_migration_backfills_historical_inspections_once(tmp_path):
    path = tmp_path / "inspection_records.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info(version INTEGER NOT NULL);
            INSERT INTO schema_info(version) VALUES (1);
            CREATE TABLE inspections (
                inspection_id TEXT PRIMARY KEY,
                snapshot_path TEXT NOT NULL UNIQUE,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                detector TEXT NOT NULL,
                product TEXT NOT NULL,
                station TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO inspections VALUES (
                'historical-1', 'old.json', '2026-07-01T00:00:00Z',
                'PASS', 'yolo', 'Cable1', 'A',
                '2026-07-01T00:00:00Z', '2026-07-01T00:00:00Z'
            );
            """
        )

    InspectionRepository(path)
    InspectionRepository(path)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            """
            SELECT inspection_id, revision, state
            FROM inspection_sync_outbox
            """
        ).fetchall()
    assert rows == [("historical-1", 1, "pending")]


def test_newer_database_schema_fails_closed(tmp_path):
    path = tmp_path / "inspection_records.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE schema_info(version INTEGER NOT NULL);
            INSERT INTO schema_info(version) VALUES (999);
            """
        )

    with pytest.raises(InspectionDatabaseMigrationError, match="newer"):
        InspectionRepository(path)


def test_corrupt_database_fails_closed(tmp_path):
    path = tmp_path / "inspection_records.sqlite3"
    path.write_bytes(b"not a sqlite database")

    with pytest.raises(InspectionDatabaseIntegrityError):
        InspectionRepository(path)


def test_backup_and_restore_preserve_a_verified_snapshot(tmp_path):
    path = tmp_path / "inspection_records.sqlite3"
    repository = InspectionRepository(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO maintenance_events (
                event_type, started_at, completed_at, dry_run,
                affected_files, reclaimed_bytes, detail_json
            ) VALUES ('before', 'a', 'b', 0, 0, 0, '{}')
            """
        )
    backup = repository.backup(reason="test")
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO maintenance_events (
                event_type, started_at, completed_at, dry_run,
                affected_files, reclaimed_bytes, detail_json
            ) VALUES ('after', 'a', 'b', 0, 0, 0, '{}')
            """
        )

    safety_backup = InspectionDatabaseManager(path).restore(backup.path)

    assert safety_backup is not None
    assert repository.query(
        "SELECT event_type FROM maintenance_events ORDER BY maintenance_event_id"
    ) == [{"event_type": "before"}]
    InspectionDatabaseManager(path).check_integrity()
