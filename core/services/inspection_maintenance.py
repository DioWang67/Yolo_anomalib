"""Safe backup scheduling and evidence-retention maintenance."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.services.inspection_database import InspectionDatabaseManager

_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
_SINGULAR_COLUMNS = {
    "original": "original_path",
    "preprocessed": "preprocessed_path",
    "annotated": "annotated_path",
    "heatmap": "heatmap_path",
}


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
        now = self._normalized_now()
        with closing(self._database.connect(readonly=True)) as connection:
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
        """Apply one retention plan after a verified database backup."""
        candidates = self.plan()
        if dry_run or not candidates:
            return InspectionMaintenanceReport(
                dry_run=dry_run,
                candidates=candidates,
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
        started_at = datetime.now(timezone.utc)
        deleted_files = 0
        reclaimed_bytes = 0
        missing_files = 0
        for candidate in candidates:
            existed = candidate.path.is_file()
            if existed:
                candidate.path.unlink()
                deleted_files += 1
                reclaimed_bytes += candidate.size_bytes
            else:
                missing_files += 1
            self._remove_database_artifact_references(candidate)
        completed_at = datetime.now(timezone.utc)
        self._record_event(
            started_at=started_at,
            completed_at=completed_at,
            affected_files=deleted_files,
            reclaimed_bytes=reclaimed_bytes,
            detail={
                "missing_files": missing_files,
                "candidate_count": len(candidates),
                "backup_path": str(backup_path or ""),
                "policy": {
                    "pass_image_days": self.policy.pass_image_days,
                    "fail_preprocessed_days": self.policy.fail_preprocessed_days,
                    "fail_all_image_days": self.policy.fail_all_image_days,
                },
            },
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
        return resolved

    def _remove_database_artifact_references(
        self,
        candidate: InspectionMaintenanceCandidate,
    ) -> None:
        stored_paths = set(candidate.stored_paths)
        stored_paths.add(str(candidate.path))
        with closing(self._database.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            for stored_path in stored_paths:
                connection.execute(
                    "DELETE FROM inspection_artifacts WHERE path=?",
                    (stored_path,),
                )
                for column in _SINGULAR_COLUMNS.values():
                    connection.execute(
                        f'UPDATE inspections SET "{column}"=\'\' '
                        f'WHERE "{column}"=?',
                        (stored_path,),
                    )
                for column in ("crop_paths_json", "mask_paths_json"):
                    rows = connection.execute(
                        f'SELECT inspection_id, "{column}" FROM inspections '
                        f'WHERE "{column}" LIKE ?',
                        (f"%{stored_path}%",),
                    ).fetchall()
                    for row in rows:
                        values = _json_string_list(row[column])
                        filtered = [
                            value for value in values if value != stored_path
                        ]
                        if filtered == values:
                            continue
                        connection.execute(
                            f'UPDATE inspections SET "{column}"=? '
                            "WHERE inspection_id=?",
                            (
                                json.dumps(
                                    filtered,
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                ),
                                str(row["inspection_id"]),
                            ),
                        )

    def _record_event(
        self,
        *,
        started_at: datetime,
        completed_at: datetime,
        affected_files: int,
        reclaimed_bytes: int,
        detail: dict[str, Any],
    ) -> None:
        with closing(self._database.connect()) as connection, connection:
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
