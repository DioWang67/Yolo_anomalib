"""Safe backup scheduling and evidence-retention maintenance."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.services.inspection_database import InspectionDatabaseManager
from tools.cross_process_lock import cross_process_file_lock

_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_SINGULAR_COLUMNS = {
    "original": "original_path",
    "preprocessed": "preprocessed_path",
    "annotated": "annotated_path",
    "heatmap": "heatmap_path",
}
_QUARANTINE_DIRECTORY = ".inspection_retention_quarantine"
_QUARANTINE_MANIFEST = "manifest.json"
_MAINTENANCE_LOCK = ".inspection_maintenance.lock"
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectionRetentionPolicy:
    """Retention windows from the production evidence SOP."""

    pass_image_days: int = 30
    fail_preprocessed_days: int = 90
    fail_all_image_days: int = 180

    def __post_init__(self) -> None:
        values = (
            self.pass_image_days,
            self.fail_preprocessed_days,
            self.fail_all_image_days,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Inspection retention days must be positive integers.")
        if self.fail_preprocessed_days > self.fail_all_image_days:
            raise ValueError(
                "FAIL preprocessed retention cannot exceed FAIL all-image retention."
            )


@dataclass(frozen=True)
class InspectionMaintenanceCandidate:
    path: Path
    artifact_types: tuple[str, ...]
    inspection_ids: tuple[str, ...]
    stored_paths: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True)
class InspectionMaintenanceReport:
    dry_run: bool
    candidates: tuple[InspectionMaintenanceCandidate, ...]
    deleted_files: int
    reclaimed_bytes: int
    missing_files: int
    backup_path: Path | None


@dataclass(frozen=True)
class _QuarantinedFile:
    original_path: Path
    quarantine_path: Path


@dataclass(frozen=True)
class _QuarantineOperation:
    cleanup_id: str
    directory: Path
    manifest_path: Path
    files: tuple[_QuarantinedFile, ...]


class InspectionMaintenanceService:
    """Plan and apply retention without deleting metadata or audit evidence."""

    def __init__(
        self,
        result_root: str | Path,
        *,
        database_path: str | Path | None = None,
        policy: InspectionRetentionPolicy | None = None,
        now_provider: Any = None,
    ) -> None:
        self.result_root = Path(result_root).resolve()
        self.database_path = Path(
            database_path
            if database_path is not None
            else self.result_root / "inspection_records.sqlite3"
        )
        self.policy = policy or InspectionRetentionPolicy()
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._database = InspectionDatabaseManager(self.database_path)

    def plan(self) -> tuple[InspectionMaintenanceCandidate, ...]:
        """Return safe candidates; shared paths survive until every reference expires."""
        if not self.database_path.is_file():
            return ()
        with closing(self._database.connect(readonly=True)) as connection:
            return self._plan_with_connection(connection)

    def _plan_with_connection(
        self,
        connection: sqlite3.Connection,
    ) -> tuple[InspectionMaintenanceCandidate, ...]:
        now = self._normalized_now()
        rows = connection.execute(
            """
            SELECT a.inspection_id, a.artifact_type, a.path,
                   i.timestamp, i.status
            FROM inspection_artifacts AS a
            JOIN inspections AS i ON i.inspection_id = a.inspection_id
            WHERE TRIM(a.path) <> ''
            """
        ).fetchall()

        references: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            path = self._safe_image_path(str(row["path"] or ""))
            if path is None:
                continue
            references[path].append(dict(row))

        candidates: list[InspectionMaintenanceCandidate] = []
        for path, path_rows in references.items():
            if not all(self._reference_is_expired(row, now) for row in path_rows):
                continue
            try:
                size_bytes = path.stat().st_size if path.is_file() else 0
            except OSError:
                size_bytes = 0
            candidates.append(
                InspectionMaintenanceCandidate(
                    path=path,
                    artifact_types=tuple(
                        sorted({str(row["artifact_type"]) for row in path_rows})
                    ),
                    inspection_ids=tuple(
                        sorted({str(row["inspection_id"]) for row in path_rows})
                    ),
                    stored_paths=tuple(
                        sorted({str(row["path"]) for row in path_rows})
                    ),
                    size_bytes=size_bytes,
                )
            )
        return tuple(sorted(candidates, key=lambda item: str(item.path).casefold()))

    def run(
        self,
        *,
        dry_run: bool = True,
        backup_before_apply: bool = True,
    ) -> InspectionMaintenanceReport:
        """Apply one recoverable retention operation after a verified backup."""
        if dry_run:
            candidates = self.plan()
            return InspectionMaintenanceReport(
                dry_run=True,
                candidates=candidates,
                deleted_files=0,
                reclaimed_bytes=0,
                missing_files=0,
                backup_path=None,
            )

        lock_path = self.result_root / _MAINTENANCE_LOCK
        with cross_process_file_lock(lock_path):
            self._recover_pending_quarantines()
            candidates = self.plan()
            if not candidates:
                return InspectionMaintenanceReport(
                    dry_run=False,
                    candidates=(),
                    deleted_files=0,
                    reclaimed_bytes=0,
                    missing_files=0,
                    backup_path=None,
                )

            backup_path = (
                self._database.backup(reason="pre_retention_cleanup").path
                if backup_before_apply
                else None
            )
            return self._apply_candidates(candidates, backup_path=backup_path)

    def _apply_candidates(
        self,
        candidates: tuple[InspectionMaintenanceCandidate, ...],
        *,
        backup_path: Path | None,
    ) -> InspectionMaintenanceReport:
        started_at = datetime.now(timezone.utc)
        cleanup_id = uuid.uuid4().hex
        operation: _QuarantineOperation | None = None
        with closing(self._database.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                # Re-plan under SQLite's writer lock so a newly inserted young
                # reference cannot be removed from beneath an active inspection.
                candidates = self._plan_with_connection(connection)
                if not candidates:
                    connection.rollback()
                    return InspectionMaintenanceReport(
                        dry_run=False,
                        candidates=(),
                        deleted_files=0,
                        reclaimed_bytes=0,
                        missing_files=0,
                        backup_path=backup_path,
                    )
                operation = self._prepare_quarantine(cleanup_id, candidates)
                quarantined_paths = {item.original_path for item in operation.files}
                deleted_files = len(operation.files)
                reclaimed_bytes = sum(
                    candidate.size_bytes for candidate in candidates if candidate.path in quarantined_paths
                )
                missing_files = len(candidates) - deleted_files
                for candidate in candidates:
                    self._remove_database_artifact_references(
                        connection,
                        candidate,
                    )
                completed_at = datetime.now(timezone.utc)
                self._insert_maintenance_event(
                    connection,
                    started_at=started_at,
                    completed_at=completed_at,
                    affected_files=deleted_files,
                    reclaimed_bytes=reclaimed_bytes,
                    detail={
                        "cleanup_id": cleanup_id,
                        "missing_files": missing_files,
                        "candidate_count": len(candidates),
                        "backup_path": str(backup_path or ""),
                        "policy": {
                            "pass_image_days": self.policy.pass_image_days,
                            "fail_preprocessed_days": (self.policy.fail_preprocessed_days),
                            "fail_all_image_days": self.policy.fail_all_image_days,
                        },
                    },
                )
                connection.commit()
            except (OSError, RuntimeError, sqlite3.Error, ValueError):
                connection.rollback()
                if operation is not None:
                    self._restore_quarantine(operation)
                raise

        if operation is None:  # Defensive invariant: candidates always prepare one.
            raise RuntimeError("Retention transaction committed without quarantine state.")
        try:
            self._finalize_quarantine(operation)
        except OSError as exc:
            _LOGGER.warning(
                "Retention cleanup committed but quarantine finalization failed; "
                "the next maintenance run will retry cleanup_id=%s: %s",
                cleanup_id,
                exc,
            )
        return InspectionMaintenanceReport(
            dry_run=False,
            candidates=candidates,
            deleted_files=deleted_files,
            reclaimed_bytes=reclaimed_bytes,
            missing_files=missing_files,
            backup_path=backup_path,
        )

    def _normalized_now(self) -> datetime:
        value = self._now_provider()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("Inspection maintenance clock must be timezone-aware.")
        return value.astimezone(timezone.utc)

    def _reference_is_expired(
        self,
        row: dict[str, Any],
        now: datetime,
    ) -> bool:
        timestamp = _parse_timestamp(str(row.get("timestamp") or ""))
        if timestamp is None or timestamp > now:
            return False
        age = now - timestamp
        status = str(row.get("status") or "").strip().upper()
        artifact_type = str(row.get("artifact_type") or "").strip().lower()
        if status == "PASS":
            return age >= timedelta(days=self.policy.pass_image_days)
        if age >= timedelta(days=self.policy.fail_all_image_days):
            return True
        return (
            artifact_type == "preprocessed"
            and age >= timedelta(days=self.policy.fail_preprocessed_days)
        )

    def _safe_image_path(self, raw_path: str) -> Path | None:
        if not raw_path.strip():
            return None
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.result_root / path
        try:
            resolved = path.resolve()
            resolved.relative_to(self.result_root)
        except (OSError, ValueError):
            return None
        if resolved.suffix.lower() not in _IMAGE_SUFFIXES:
            return None
        if "database_backups" in resolved.parts:
            return None
        if _QUARANTINE_DIRECTORY in resolved.parts:
            return None
        return resolved

    def _prepare_quarantine(
        self,
        cleanup_id: str,
        candidates: tuple[InspectionMaintenanceCandidate, ...],
    ) -> _QuarantineOperation:
        quarantine_root = self.result_root / _QUARANTINE_DIRECTORY
        operation_directory = quarantine_root / cleanup_id
        files_directory = operation_directory / "files"
        files_directory.mkdir(parents=True, exist_ok=False)
        files = tuple(
            _QuarantinedFile(
                original_path=candidate.path,
                quarantine_path=(files_directory / f"{index:06d}{candidate.path.suffix.lower()}"),
            )
            for index, candidate in enumerate(candidates)
            if candidate.path.is_file()
        )
        operation = _QuarantineOperation(
            cleanup_id=cleanup_id,
            directory=operation_directory,
            manifest_path=operation_directory / _QUARANTINE_MANIFEST,
            files=files,
        )
        self._write_quarantine_manifest(operation)
        try:
            for item in files:
                item.original_path.replace(item.quarantine_path)
        except OSError:
            self._restore_quarantine(operation)
            raise
        return operation

    def _write_quarantine_manifest(
        self,
        operation: _QuarantineOperation,
    ) -> None:
        payload = {
            "version": 1,
            "cleanup_id": operation.cleanup_id,
            "files": [
                {
                    "original_path": str(item.original_path),
                    "quarantine_name": item.quarantine_path.name,
                }
                for item in operation.files
            ],
        }
        temporary = operation.manifest_path.with_suffix(".json.tmp")
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, operation.manifest_path)

    def _recover_pending_quarantines(self) -> None:
        quarantine_root = self.result_root / _QUARANTINE_DIRECTORY
        if not quarantine_root.is_dir():
            return
        for operation_directory in sorted(quarantine_root.iterdir()):
            if not operation_directory.is_dir():
                raise RuntimeError(f"Retention quarantine contains an unexpected entry: {operation_directory}")
            manifest_path = operation_directory / _QUARANTINE_MANIFEST
            if not manifest_path.is_file():
                self._remove_incomplete_empty_operation(operation_directory)
                continue
            operation = self._load_quarantine_manifest(operation_directory)
            if self._maintenance_event_exists(operation.cleanup_id):
                try:
                    self._finalize_quarantine(operation)
                except OSError as exc:
                    _LOGGER.warning(
                        "Retrying committed retention quarantine failed for cleanup_id=%s: %s",
                        operation.cleanup_id,
                        exc,
                    )
            else:
                self._restore_quarantine(operation)
        self._remove_empty_quarantine_root()

    def _load_quarantine_manifest(
        self,
        operation_directory: Path,
    ) -> _QuarantineOperation:
        manifest_path = operation_directory / _QUARANTINE_MANIFEST
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Retention quarantine manifest is unreadable: {manifest_path}") from exc
        cleanup_id = str(payload.get("cleanup_id") or "")
        raw_files = payload.get("files")
        if payload.get("version") != 1 or cleanup_id != operation_directory.name or not isinstance(raw_files, list):
            raise RuntimeError(f"Retention quarantine manifest is invalid: {manifest_path}")
        files: list[_QuarantinedFile] = []
        files_directory = operation_directory / "files"
        for raw_item in raw_files:
            if not isinstance(raw_item, dict):
                raise RuntimeError(f"Retention quarantine manifest is invalid: {manifest_path}")
            original = self._safe_image_path(str(raw_item.get("original_path") or ""))
            quarantine_name = str(raw_item.get("quarantine_name") or "")
            if original is None or not quarantine_name or Path(quarantine_name).name != quarantine_name:
                raise RuntimeError(f"Retention quarantine manifest contains an unsafe path: {manifest_path}")
            quarantine_path = (files_directory / quarantine_name).resolve()
            try:
                quarantine_path.relative_to(files_directory.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Retention quarantine manifest contains an unsafe path: {manifest_path}") from exc
            files.append(
                _QuarantinedFile(
                    original_path=original,
                    quarantine_path=quarantine_path,
                )
            )
        return _QuarantineOperation(
            cleanup_id=cleanup_id,
            directory=operation_directory,
            manifest_path=manifest_path,
            files=tuple(files),
        )

    def _maintenance_event_exists(self, cleanup_id: str) -> bool:
        with closing(self._database.connect(readonly=True)) as connection:
            rows = connection.execute(
                """
                SELECT detail_json
                FROM maintenance_events
                WHERE event_type='retention_cleanup'
                """
            ).fetchall()
        for row in rows:
            try:
                detail = json.loads(str(row["detail_json"] or "{}"))
            except json.JSONDecodeError:
                continue
            if isinstance(detail, dict) and detail.get("cleanup_id") == cleanup_id:
                return True
        return False

    def _restore_quarantine(self, operation: _QuarantineOperation) -> None:
        for item in operation.files:
            if not item.quarantine_path.exists():
                continue
            if item.original_path.exists():
                raise RuntimeError(f"Cannot restore quarantined evidence over an existing file: {item.original_path}")
            item.original_path.parent.mkdir(parents=True, exist_ok=True)
            item.quarantine_path.replace(item.original_path)
        self._remove_operation_metadata(operation)

    def _finalize_quarantine(self, operation: _QuarantineOperation) -> None:
        for item in operation.files:
            self._delete_quarantined_file(item.quarantine_path)
        self._remove_operation_metadata(operation)

    def _delete_quarantined_file(self, path: Path) -> None:
        path.unlink(missing_ok=True)

    def _remove_operation_metadata(
        self,
        operation: _QuarantineOperation,
    ) -> None:
        operation.manifest_path.unlink(missing_ok=True)
        files_directory = operation.directory / "files"
        if files_directory.exists():
            files_directory.rmdir()
        if operation.directory.exists():
            operation.directory.rmdir()
        self._remove_empty_quarantine_root()

    def _remove_incomplete_empty_operation(
        self,
        operation_directory: Path,
    ) -> None:
        temporary_manifest = operation_directory / "manifest.json.tmp"
        temporary_manifest.unlink(missing_ok=True)
        files_directory = operation_directory / "files"
        if files_directory.exists():
            try:
                files_directory.rmdir()
            except OSError as exc:
                raise RuntimeError(
                    f"Manifestless retention quarantine contains evidence: {operation_directory}"
                ) from exc
        operation_directory.rmdir()
        self._remove_empty_quarantine_root()

    def _remove_empty_quarantine_root(self) -> None:
        quarantine_root = self.result_root / _QUARANTINE_DIRECTORY
        try:
            quarantine_root.rmdir()
        except FileNotFoundError:
            return
        except OSError:
            # Other operations may still be awaiting finalization.
            return

    def _remove_database_artifact_references(
        self,
        connection: sqlite3.Connection,
        candidate: InspectionMaintenanceCandidate,
    ) -> None:
        stored_paths = set(candidate.stored_paths)
        stored_paths.add(str(candidate.path))
        for stored_path in stored_paths:
            connection.execute(
                "DELETE FROM inspection_artifacts WHERE path=?",
                (stored_path,),
            )
            for column in _SINGULAR_COLUMNS.values():
                connection.execute(
                    f'UPDATE inspections SET "{column}"=\'\' WHERE "{column}"=?',
                    (stored_path,),
                )
            for column in ("crop_paths_json", "mask_paths_json"):
                rows = connection.execute(
                    f'SELECT inspection_id, "{column}" FROM inspections WHERE "{column}" LIKE ?',
                    (f"%{stored_path}%",),
                ).fetchall()
                for row in rows:
                    values = _json_string_list(row[column])
                    filtered = [value for value in values if value != stored_path]
                    if filtered == values:
                        continue
                    connection.execute(
                        f'UPDATE inspections SET "{column}"=? WHERE inspection_id=?',
                        (
                            json.dumps(
                                filtered,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                            str(row["inspection_id"]),
                        ),
                    )

    def _insert_maintenance_event(
        self,
        connection: sqlite3.Connection,
        *,
        started_at: datetime,
        completed_at: datetime,
        affected_files: int,
        reclaimed_bytes: int,
        detail: dict[str, Any],
    ) -> None:
        connection.execute(
            """
            INSERT INTO maintenance_events (
                event_type, started_at, completed_at, dry_run,
                affected_files, reclaimed_bytes, detail_json
            ) VALUES (?, ?, ?, 0, ?, ?, ?)
            """,
            (
                "retention_cleanup",
                started_at.isoformat(),
                completed_at.isoformat(),
                affected_files,
                reclaimed_bytes,
                json.dumps(
                    detail,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
        )


class InspectionMaintenanceScheduler:
    """Coalesce daily maintenance requests into one background worker."""

    def __init__(
        self,
        result_root: str | Path,
        *,
        database_path: str | Path,
        cleanup_enabled: bool = False,
        policy: InspectionRetentionPolicy | None = None,
        interval: timedelta = timedelta(hours=24),
        logger: logging.Logger | None = None,
    ) -> None:
        if interval.total_seconds() <= 0:
            raise ValueError("Inspection maintenance interval must be positive.")
        self.result_root = Path(result_root)
        self.database_path = Path(database_path)
        self.cleanup_enabled = bool(cleanup_enabled)
        self.policy = policy or InspectionRetentionPolicy()
        self.interval = interval
        self.logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._last_started_at: datetime | None = None

    def maybe_schedule(self) -> bool:
        """Start at most one due maintenance run without blocking detection."""
        now = datetime.now(timezone.utc)
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return False
            if (
                self._last_started_at is not None
                and now - self._last_started_at < self.interval
            ):
                return False
            if not _backup_is_due(self.database_path, now, self.interval):
                self._last_started_at = now
                return False
            self._last_started_at = now
            worker = threading.Thread(
                target=self._run,
                name="inspection-maintenance",
                daemon=True,
            )
            self._worker = worker
            worker.start()
            return True

    def close(self, timeout: float = 10.0) -> None:
        """Wait briefly for an in-flight verified backup during shutdown."""
        with self._lock:
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=max(0.0, timeout))

    def _run(self) -> None:
        try:
            database = InspectionDatabaseManager(self.database_path)
            database.backup(reason="scheduled")
            if self.cleanup_enabled:
                InspectionMaintenanceService(
                    self.result_root,
                    database_path=self.database_path,
                    policy=self.policy,
                ).run(dry_run=False, backup_before_apply=False)
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            self.logger.error(
                "Inspection maintenance failed; evidence was left unchanged: %s",
                exc,
            )


def _backup_is_due(
    database_path: Path,
    now: datetime,
    interval: timedelta,
) -> bool:
    backup_dir = database_path.parent / "database_backups"
    try:
        latest = max(
            (
                datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
                for path in backup_dir.glob("*.sqlite3.bak")
                if path.is_file()
            ),
            default=None,
        )
    except OSError:
        return True
    return latest is None or now - latest >= interval


def _parse_timestamp(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(hours=8)))
    return parsed.astimezone(timezone.utc)


def _json_string_list(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]
