"""Schema migration, integrity, backup, and restore for inspection SQLite data."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3
_BUSY_TIMEOUT_MS = 5_000


class InspectionDatabaseError(RuntimeError):
    """Base error for an unusable inspection database."""


class InspectionDatabaseIntegrityError(InspectionDatabaseError):
    """Raised when SQLite reports structural corruption."""


class InspectionDatabaseMigrationError(InspectionDatabaseError):
    """Raised when a schema cannot be migrated safely."""


@dataclass(frozen=True)
class InspectionDatabaseBackup:
    """One verified SQLite backup created from a consistent snapshot."""

    path: Path
    created_at: datetime
    source_schema_version: int
    reason: str


class InspectionDatabaseManager:
    """Own database lifecycle operations independently from inspection queries."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> InspectionDatabaseBackup | None:
        """Validate and migrate the database, backing up existing data first."""
        existed_with_data = self.path.is_file() and self.path.stat().st_size > 0
        try:
            with closing(self.connect()) as connection:
                if existed_with_data:
                    _assert_integrity(connection, self.path)
                current_version = _read_schema_version(connection)
                if current_version > SCHEMA_VERSION:
                    raise InspectionDatabaseMigrationError(
                        "Inspection database schema is newer than this application: "
                        f"database={current_version}, application={SCHEMA_VERSION}"
                    )
        except InspectionDatabaseError:
            raise
        except sqlite3.DatabaseError as exc:
            raise InspectionDatabaseIntegrityError(
                f"Inspection database cannot be opened safely: {self.path}: {exc}"
            ) from exc

        backup: InspectionDatabaseBackup | None = None
        if existed_with_data and current_version < SCHEMA_VERSION:
            backup = self.backup(
                reason=f"pre_migration_v{current_version}_to_v{SCHEMA_VERSION}"
            )

        try:
            with closing(self.connect()) as connection:
                connection.execute("BEGIN EXCLUSIVE")
                locked_version = _read_schema_version(connection)
                if locked_version > SCHEMA_VERSION:
                    raise InspectionDatabaseMigrationError(
                        "Inspection database was upgraded by a newer application "
                        f"while opening it: {locked_version}"
                    )
                while locked_version < SCHEMA_VERSION:
                    migration = _MIGRATIONS.get(locked_version)
                    if migration is None:
                        raise InspectionDatabaseMigrationError(
                            f"No migration is registered for schema v{locked_version}."
                        )
                    migration(connection)
                    locked_version += 1
                    _write_schema_version(connection, locked_version)
                _ensure_schema_objects(connection)
                _backfill_sync_outbox(connection)
                connection.commit()
                _assert_integrity(connection, self.path)
        except sqlite3.Error as exc:
            raise InspectionDatabaseMigrationError(
                f"Inspection database migration failed: {exc}"
            ) from exc
        return backup

    def connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        """Open one configured connection; callers own its transaction scope."""
        if readonly:
            uri = f"{self.path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        else:
            connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        if not readonly:
            connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def check_integrity(self) -> None:
        """Fail closed when SQLite cannot verify the complete database."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Inspection database is missing: {self.path}")
        with closing(self.connect(readonly=True)) as connection:
            _assert_integrity(connection, self.path, full=True)

    def backup(
        self,
        destination: str | Path | None = None,
        *,
        reason: str = "manual",
    ) -> InspectionDatabaseBackup:
        """Create and verify an atomic online backup, including WAL contents."""
        if not self.path.is_file():
            raise FileNotFoundError(f"Inspection database is missing: {self.path}")
        created_at = datetime.now(timezone.utc)
        target = (
            Path(destination)
            if destination is not None
            else self._next_backup_path(created_at)
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.resolve() == self.path.resolve():
            raise ValueError("Inspection database backup must use another path.")

        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        temporary.unlink(missing_ok=True)
        try:
            with (
                closing(self.connect(readonly=True)) as source,
                closing(sqlite3.connect(temporary)) as output,
            ):
                source.backup(output)
                output.commit()
                _assert_integrity(output, temporary, full=True)
                source_version = _read_schema_version(source)
            os.replace(temporary, target)
        except (OSError, sqlite3.Error) as exc:
            temporary.unlink(missing_ok=True)
            raise InspectionDatabaseError(
                f"Inspection database backup failed: {exc}"
            ) from exc
        return InspectionDatabaseBackup(
            path=target,
            created_at=created_at,
            source_schema_version=source_version,
            reason=reason,
        )

    def restore(self, backup_path: str | Path) -> InspectionDatabaseBackup | None:
        """Restore a verified backup after preserving the current database."""
        source_path = Path(backup_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Inspection backup is missing: {source_path}")
        if source_path.resolve() == self.path.resolve():
            raise ValueError("Inspection backup and active database must be different files.")
        with closing(
            sqlite3.connect(
                f"{source_path.resolve().as_uri()}?mode=ro",
                uri=True,
                timeout=5.0,
            )
        ) as source:
            _assert_integrity(source, source_path, full=True)
            source_version = _read_schema_version(source)
            if source_version > SCHEMA_VERSION:
                raise InspectionDatabaseMigrationError(
                    "Cannot restore a database created by a newer application: "
                    f"{source_version}"
                )

        safety_backup = (
            self.backup(reason="pre_restore")
            if self.path.is_file() and self.path.stat().st_size > 0
            else None
        )
        try:
            with (
                closing(
                    sqlite3.connect(
                        f"{source_path.resolve().as_uri()}?mode=ro",
                        uri=True,
                        timeout=5.0,
                    )
                ) as source,
                closing(self.connect()) as destination,
            ):
                source.backup(destination)
                destination.commit()
                _assert_integrity(destination, self.path, full=True)
        except sqlite3.Error as exc:
            raise InspectionDatabaseError(
                f"Inspection database restore failed: {exc}"
            ) from exc
        self.initialize()
        return safety_backup

    def _next_backup_path(self, created_at: datetime) -> Path:
        backup_dir = self.path.parent / "database_backups"
        stamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        return backup_dir / f"{self.path.stem}.{stamp}.sqlite3.bak"


def _migrate_0_to_1(connection: sqlite3.Connection) -> None:
    _ensure_schema_objects(connection)


def _migrate_1_to_2(connection: sqlite3.Connection) -> None:
    table_exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='inspections'"
    ).fetchone()
    if table_exists is not None:
        _ensure_inspection_columns(connection)
    _ensure_schema_objects(connection)


def _migrate_2_to_3(connection: sqlite3.Connection) -> None:
    _ensure_schema_objects(connection)


_MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {
    0: _migrate_0_to_1,
    1: _migrate_1_to_2,
    2: _migrate_2_to_3,
}


def _read_schema_version(connection: sqlite3.Connection) -> int:
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_info'"
    ).fetchone()
    if table is None:
        return 0
    rows = connection.execute("SELECT version FROM schema_info").fetchall()
    if not rows:
        return 0
    versions: list[int] = []
    for row in rows:
        value = row[0]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InspectionDatabaseMigrationError(
                f"Inspection database contains an invalid schema version: {value!r}"
            )
        versions.append(value)
    if len(set(versions)) != 1:
        raise InspectionDatabaseMigrationError(
            f"Inspection database contains conflicting schema versions: {versions}"
        )
    return versions[0]


def _write_schema_version(connection: sqlite3.Connection, version: int) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS schema_info (version INTEGER NOT NULL)"
    )
    connection.execute("DELETE FROM schema_info")
    connection.execute("INSERT INTO schema_info(version) VALUES (?)", (version,))


def _assert_integrity(
    connection: sqlite3.Connection,
    path: Path,
    *,
    full: bool = False,
) -> None:
    pragma = "integrity_check" if full else "quick_check"
    rows = connection.execute(f"PRAGMA {pragma}").fetchall()
    messages = [str(row[0]) for row in rows]
    if messages != ["ok"]:
        raise InspectionDatabaseIntegrityError(
            f"Inspection database integrity check failed for {path}: "
            + "; ".join(messages[:10])
        )


def _ensure_schema_objects(connection: sqlite3.Connection) -> None:
    schema_sql = """
        CREATE TABLE IF NOT EXISTS schema_info (
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS inspections (
            inspection_id TEXT PRIMARY KEY,
            snapshot_path TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            detector TEXT NOT NULL,
            product TEXT NOT NULL,
            station TEXT NOT NULL,
            machine_id TEXT NOT NULL DEFAULT '',
            work_order TEXT NOT NULL DEFAULT '',
            camera_id TEXT NOT NULL DEFAULT '',
            model_version TEXT NOT NULL DEFAULT '',
            model_weights TEXT NOT NULL DEFAULT '',
            inference_time REAL,
            decision_reasons_json TEXT NOT NULL DEFAULT '[]',
            predictions_json TEXT NOT NULL DEFAULT '[]',
            original_path TEXT NOT NULL DEFAULT '',
            preprocessed_path TEXT NOT NULL DEFAULT '',
            annotated_path TEXT NOT NULL DEFAULT '',
            heatmap_path TEXT NOT NULL DEFAULT '',
            crop_paths_json TEXT NOT NULL DEFAULT '[]',
            mask_paths_json TEXT NOT NULL DEFAULT '[]',
            review_outcome TEXT NOT NULL DEFAULT '',
            review_label TEXT NOT NULL DEFAULT '',
            failure_category TEXT NOT NULL DEFAULT '',
            failure_source TEXT NOT NULL DEFAULT '',
            failure_note TEXT NOT NULL DEFAULT '',
            skip_reason TEXT NOT NULL DEFAULT '',
            action_route TEXT NOT NULL DEFAULT '',
            training_selected INTEGER NOT NULL DEFAULT 0
                CHECK(training_selected IN (0, 1)),
            training_set_state TEXT NOT NULL DEFAULT 'not_selected',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ai_predictions (
            inspection_id TEXT NOT NULL
                REFERENCES inspections(inspection_id) ON DELETE CASCADE,
            prediction_index INTEGER NOT NULL,
            class_id INTEGER,
            class_name TEXT NOT NULL DEFAULT '',
            confidence REAL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            mask_json TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (inspection_id, prediction_index)
        );

        CREATE TABLE IF NOT EXISTS inspection_artifacts (
            inspection_id TEXT NOT NULL
                REFERENCES inspections(inspection_id) ON DELETE CASCADE,
            artifact_type TEXT NOT NULL,
            artifact_index INTEGER NOT NULL,
            path TEXT NOT NULL,
            PRIMARY KEY (inspection_id, artifact_type, artifact_index)
        );

        CREATE TABLE IF NOT EXISTS review_events (
            review_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id TEXT NOT NULL
                REFERENCES inspections(inspection_id) ON DELETE CASCADE,
            review_outcome TEXT NOT NULL,
            review_label TEXT NOT NULL,
            failure_category TEXT NOT NULL,
            failure_source TEXT NOT NULL,
            failure_note TEXT NOT NULL,
            skip_reason TEXT NOT NULL,
            action_route TEXT NOT NULL,
            training_selected INTEGER NOT NULL
                CHECK(training_selected IN (0, 1)),
            reviewed_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS maintenance_events (
            maintenance_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            dry_run INTEGER NOT NULL CHECK(dry_run IN (0, 1)),
            affected_files INTEGER NOT NULL DEFAULT 0,
            reclaimed_bytes INTEGER NOT NULL DEFAULT 0,
            detail_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS inspection_sync_outbox (
            inspection_id TEXT PRIMARY KEY
                REFERENCES inspections(inspection_id) ON DELETE CASCADE,
            revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
            state TEXT NOT NULL DEFAULT 'pending'
                CHECK(state IN ('pending', 'inflight', 'synced', 'dead')),
            attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
            next_attempt_at TEXT NOT NULL,
            lease_token TEXT NOT NULL DEFAULT '',
            lease_expires_at TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            synced_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_inspections_product
            ON inspections(product);
        CREATE INDEX IF NOT EXISTS idx_inspections_equipment
            ON inspections(machine_id, station, camera_id);
        CREATE INDEX IF NOT EXISTS idx_inspections_model
            ON inspections(model_version);
        CREATE INDEX IF NOT EXISTS idx_inspections_review
            ON inspections(review_outcome, failure_category);
        CREATE INDEX IF NOT EXISTS idx_inspections_timestamp
            ON inspections(timestamp);
        CREATE INDEX IF NOT EXISTS idx_artifacts_path
            ON inspection_artifacts(path);
        CREATE INDEX IF NOT EXISTS idx_sync_outbox_due
            ON inspection_sync_outbox(state, next_attempt_at);
        """
    # ``executescript`` issues an implicit COMMIT. Execute statements one by
    # one so migrations remain inside the caller's EXCLUSIVE transaction.
    for statement in schema_sql.split(";"):
        if statement.strip():
            connection.execute(statement)


def _ensure_inspection_columns(connection: sqlite3.Connection) -> None:
    existing = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(inspections)").fetchall()
    }
    additions = {
        "machine_id": "TEXT NOT NULL DEFAULT ''",
        "work_order": "TEXT NOT NULL DEFAULT ''",
        "camera_id": "TEXT NOT NULL DEFAULT ''",
        "model_version": "TEXT NOT NULL DEFAULT ''",
        "model_weights": "TEXT NOT NULL DEFAULT ''",
        "inference_time": "REAL",
        "decision_reasons_json": "TEXT NOT NULL DEFAULT '[]'",
        "predictions_json": "TEXT NOT NULL DEFAULT '[]'",
        "original_path": "TEXT NOT NULL DEFAULT ''",
        "preprocessed_path": "TEXT NOT NULL DEFAULT ''",
        "annotated_path": "TEXT NOT NULL DEFAULT ''",
        "heatmap_path": "TEXT NOT NULL DEFAULT ''",
        "crop_paths_json": "TEXT NOT NULL DEFAULT '[]'",
        "mask_paths_json": "TEXT NOT NULL DEFAULT '[]'",
        "review_outcome": "TEXT NOT NULL DEFAULT ''",
        "review_label": "TEXT NOT NULL DEFAULT ''",
        "failure_category": "TEXT NOT NULL DEFAULT ''",
        "failure_source": "TEXT NOT NULL DEFAULT ''",
        "failure_note": "TEXT NOT NULL DEFAULT ''",
        "skip_reason": "TEXT NOT NULL DEFAULT ''",
        "action_route": "TEXT NOT NULL DEFAULT ''",
        "training_selected": "INTEGER NOT NULL DEFAULT 0",
        "training_set_state": "TEXT NOT NULL DEFAULT 'not_selected'",
        "created_at": "TEXT NOT NULL DEFAULT ''",
        "updated_at": "TEXT NOT NULL DEFAULT ''",
    }
    for column, declaration in additions.items():
        if column not in existing:
            connection.execute(
                f'ALTER TABLE inspections ADD COLUMN "{column}" {declaration}'
            )


def _backfill_sync_outbox(connection: sqlite3.Connection) -> None:
    """Queue historical inspections exactly once after outbox introduction."""
    now = datetime.now(timezone.utc).isoformat()
    connection.execute(
        """
        INSERT INTO inspection_sync_outbox (
            inspection_id, revision, state, attempt_count, next_attempt_at,
            lease_token, lease_expires_at, last_error, synced_at, updated_at
        )
        SELECT inspection_id, 1, 'pending', 0, ?, '', '', '', '', ?
        FROM inspections AS inspection
        WHERE NOT EXISTS (
            SELECT 1
            FROM inspection_sync_outbox AS outbox
            WHERE outbox.inspection_id = inspection.inspection_id
        )
        """,
        (now, now),
    )
