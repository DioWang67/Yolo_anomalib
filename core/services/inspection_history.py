"""Read-only reporting service for persisted inspection history."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

_STATUS_FILTERS = frozenset({"", "PASS", "FAIL", "ERROR"})
_MAX_RECORDS = 500
HISTORY_TIMEZONE_NAME = "Asia/Taipei"
HISTORY_TIMEZONE = ZoneInfo(HISTORY_TIMEZONE_NAME)


class InspectionHistoryError(RuntimeError):
    """The inspection history database could not be read safely."""


@dataclass(frozen=True)
class InspectionHistoryFilters:
    """Validated filters for one bounded history snapshot."""

    product: str = ""
    station: str = ""
    status: str = ""
    limit: int = 200
    offset: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None

    def normalized(self) -> InspectionHistoryFilters:
        product = str(self.product).strip()
        station = str(self.station).strip()
        status = str(self.status).strip().upper()
        if len(product) > 128 or len(station) > 128:
            raise ValueError("Inspection history target filter is too long.")
        if status not in _STATUS_FILTERS:
            raise ValueError(f"Unsupported inspection status filter: {status}")
        if type(self.limit) is not int or not 1 <= self.limit <= _MAX_RECORDS:
            raise ValueError(
                f"Inspection history limit must be between 1 and {_MAX_RECORDS}."
            )
        if type(self.offset) is not int or not 0 <= self.offset <= 10_000_000:
            raise ValueError(
                "Inspection history offset must be between 0 and 10000000."
            )
        started_at = _normalize_boundary(self.started_at, "start")
        ended_at = _normalize_boundary(self.ended_at, "end")
        if (started_at is None) != (ended_at is None):
            raise ValueError(
                "Inspection history requires both start and end times."
            )
        if (
            started_at is not None
            and ended_at is not None
            and started_at >= ended_at
        ):
            raise ValueError(
                "Inspection history start time must be before end time."
            )
        return InspectionHistoryFilters(
            product=product,
            station=station,
            status=status,
            limit=self.limit,
            offset=self.offset,
            started_at=started_at,
            ended_at=ended_at,
        )


@dataclass(frozen=True)
class InspectionTimeRange:
    """Half-open local reporting interval: start <= timestamp < end."""

    started_at: datetime
    ended_at: datetime
    def __post_init__(self) -> None:
        if (
            self.started_at.tzinfo is None
            or self.ended_at.tzinfo is None
            or self.started_at >= self.ended_at
        ):
            raise ValueError(
                "Inspection time range must be aware and increasing."
            )


@dataclass(frozen=True)
class InspectionHistoryRecord:
    inspection_id: str
    timestamp: str
    status: str
    detector: str
    product: str
    station: str
    machine_id: str
    work_order: str
    camera_id: str
    model_version: str
    inference_time: float | None
    reason_codes: tuple[str, ...]
    review_outcome: str
    failure_category: str
    original_path: str
    annotated_path: str
    heatmap_path: str
    snapshot_path: str

    @property
    def preview_path(self) -> str:
        return self.annotated_path or self.heatmap_path or self.original_path

    @property
    def reason(self) -> str:
        """Backward-compatible machine-readable reason text."""
        return "、".join(self.reason_codes)


@dataclass(frozen=True)
class InspectionHistorySummary:
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0

    @property
    def yield_rate(self) -> float | None:
        judged = self.passed + self.failed
        return None if judged == 0 else (self.passed / judged) * 100.0


@dataclass(frozen=True)
class InspectionHistorySyncSummary:
    pending: int = 0
    inflight: int = 0
    synced: int = 0
    dead: int = 0


@dataclass(frozen=True)
class InspectionHistorySnapshot:
    records: tuple[InspectionHistoryRecord, ...] = ()
    summary: InspectionHistorySummary = InspectionHistorySummary()
    products: tuple[str, ...] = ()
    stations: tuple[str, ...] = ()
    filtered_total: int = 0
    sync: InspectionHistorySyncSummary = InspectionHistorySyncSummary()


class InspectionHistoryService:
    """Load one consistent, bounded snapshot without mutating the database."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)

    def load(
        self,
        filters: InspectionHistoryFilters = InspectionHistoryFilters(),
    ) -> InspectionHistorySnapshot:
        query_filters = filters.normalized()
        if not self.path.is_file():
            return InspectionHistorySnapshot()

        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN")
                products = self._load_products(connection)
                stations = self._load_stations(
                    connection,
                    product=query_filters.product,
                )
                summary = self._load_summary(connection, query_filters)
                filtered_total = self._load_record_count(
                    connection,
                    query_filters,
                )
                records = self._load_records(connection, query_filters)
                sync = self._load_sync_summary(connection)
                connection.execute("COMMIT")
        except sqlite3.Error as exc:
            raise InspectionHistoryError(
                f"Unable to read inspection history: {exc}"
            ) from exc

        return InspectionHistorySnapshot(
            records=records,
            summary=summary,
            products=products,
            stations=stations,
            filtered_total=filtered_total,
            sync=sync,
        )

    def _connect(self) -> sqlite3.Connection:
        uri = f"{self.path.resolve().as_uri()}?mode=ro"
        connection = sqlite3.connect(
            uri,
            uri=True,
            timeout=2.0,
            check_same_thread=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=2000")
        return connection

    @staticmethod
    def _load_products(connection: sqlite3.Connection) -> tuple[str, ...]:
        rows = connection.execute(
            """
            SELECT DISTINCT product
            FROM inspections
            WHERE TRIM(product) <> ''
            ORDER BY product COLLATE NOCASE
            """
        ).fetchall()
        return tuple(str(row["product"]) for row in rows)

    @staticmethod
    def _load_stations(
        connection: sqlite3.Connection,
        *,
        product: str,
    ) -> tuple[str, ...]:
        if product:
            rows = connection.execute(
                """
                SELECT DISTINCT station
                FROM inspections
                WHERE product = ? AND TRIM(station) <> ''
                ORDER BY station COLLATE NOCASE
                """,
                (product,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT DISTINCT station
                FROM inspections
                WHERE TRIM(station) <> ''
                ORDER BY station COLLATE NOCASE
                """
            ).fetchall()
        return tuple(str(row["station"]) for row in rows)

    @staticmethod
    def _load_summary(
        connection: sqlite3.Connection,
        filters: InspectionHistoryFilters,
    ) -> InspectionHistorySummary:
        where_sql, parameters = _target_where(filters)
        row = connection.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN UPPER(status) = 'PASS' THEN 1 ELSE 0 END)
                    AS passed,
                SUM(
                    CASE WHEN UPPER(status)
                        IN ('FAIL', 'DETECTION_FAIL', 'NG')
                    THEN 1 ELSE 0 END
                ) AS failed,
                SUM(
                    CASE WHEN UPPER(status)
                        IN ('ERROR', 'INFERENCE_ERROR')
                    THEN 1 ELSE 0 END
                ) AS errors
            FROM inspections
            {where_sql}
            """,
            parameters,
        ).fetchone()
        if row is None:
            return InspectionHistorySummary()
        return InspectionHistorySummary(
            total=int(row["total"] or 0),
            passed=int(row["passed"] or 0),
            failed=int(row["failed"] or 0),
            errors=int(row["errors"] or 0),
        )

    @staticmethod
    def _load_record_count(
        connection: sqlite3.Connection,
        filters: InspectionHistoryFilters,
    ) -> int:
        where_sql, parameters = build_record_where_clause(filters)
        row = connection.execute(
            f"SELECT COUNT(*) AS total FROM inspections {where_sql}",
            parameters,
        ).fetchone()
        return int(row["total"] or 0) if row is not None else 0

    @staticmethod
    def _load_records(
        connection: sqlite3.Connection,
        filters: InspectionHistoryFilters,
    ) -> tuple[InspectionHistoryRecord, ...]:
        where_sql, parameters = build_record_where_clause(filters)
        rows = connection.execute(
            f"""
            SELECT
                inspection_id, timestamp, status, detector, product, station,
                machine_id, work_order, camera_id, model_version,
                inference_time, decision_reasons_json, review_outcome,
                failure_category, original_path, annotated_path, heatmap_path,
                snapshot_path
            FROM inspections
            {where_sql}
            ORDER BY timestamp DESC, inspection_id DESC
            LIMIT ? OFFSET ?
            """,
            (*parameters, filters.limit, filters.offset),
        ).fetchall()
        return tuple(_record_from_row(row) for row in rows)

    @staticmethod
    def _load_sync_summary(
        connection: sqlite3.Connection,
    ) -> InspectionHistorySyncSummary:
        exists = connection.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type='table' AND name='inspection_sync_outbox'
            """
        ).fetchone()
        if exists is None:
            return InspectionHistorySyncSummary()
        rows = connection.execute(
            """
            SELECT state, COUNT(*) AS count
            FROM inspection_sync_outbox
            GROUP BY state
            """
        ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        return InspectionHistorySyncSummary(
            pending=counts.get("pending", 0),
            inflight=counts.get("inflight", 0),
            synced=counts.get("synced", 0),
            dead=counts.get("dead", 0),
        )


def _target_where(
    filters: InspectionHistoryFilters,
) -> tuple[str, tuple[Any, ...]]:
    clauses: list[str] = []
    parameters: list[Any] = []
    if filters.product:
        clauses.append("product = ?")
        parameters.append(filters.product)
    if filters.station:
        clauses.append("station = ?")
        parameters.append(filters.station)
    if filters.started_at is not None and filters.ended_at is not None:
        aware_sql = (
            "(UPPER(SUBSTR(timestamp, -1, 1)) = 'Z' "
            "OR (LENGTH(timestamp) >= 6 "
            "AND SUBSTR(timestamp, -6, 1) IN ('+', '-')))"
        )
        clauses.append(
            "("
            f"((NOT {aware_sql}) "
            "AND JULIANDAY(timestamp) >= JULIANDAY(?) "
            "AND JULIANDAY(timestamp) < JULIANDAY(?)) "
            "OR "
            f"({aware_sql} "
            "AND JULIANDAY(timestamp) >= JULIANDAY(?) "
            "AND JULIANDAY(timestamp) < JULIANDAY(?))"
            ")"
        )
        local_start = filters.started_at.astimezone(HISTORY_TIMEZONE).replace(
            tzinfo=None
        )
        local_end = filters.ended_at.astimezone(HISTORY_TIMEZONE).replace(
            tzinfo=None
        )
        utc_start = filters.started_at.astimezone(timezone.utc)
        utc_end = filters.ended_at.astimezone(timezone.utc)
        parameters.extend(
            (
                local_start.isoformat(timespec="microseconds"),
                local_end.isoformat(timespec="microseconds"),
                utc_start.isoformat(timespec="microseconds"),
                utc_end.isoformat(timespec="microseconds"),
            )
        )
    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, tuple(parameters)


def build_record_where_clause(
    filters: InspectionHistoryFilters,
) -> tuple[str, tuple[Any, ...]]:
    """Build the validated status-aware clause shared by history reports."""
    where_sql, parameters = _target_where(filters)
    clauses = [where_sql.removeprefix("WHERE ")] if where_sql else []
    values = list(parameters)
    if filters.status == "PASS":
        clauses.append("UPPER(status) = 'PASS'")
    elif filters.status == "FAIL":
        clauses.append("UPPER(status) IN ('FAIL', 'DETECTION_FAIL', 'NG')")
    elif filters.status == "ERROR":
        clauses.append("UPPER(status) IN ('ERROR', 'INFERENCE_ERROR')")
    return (
        f"WHERE {' AND '.join(clauses)}" if clauses else "",
        tuple(values),
    )


def _record_from_row(row: Mapping[str, Any]) -> InspectionHistoryRecord:
    inference_time = row["inference_time"]
    return InspectionHistoryRecord(
        inspection_id=str(row["inspection_id"] or ""),
        timestamp=str(row["timestamp"] or ""),
        status=str(row["status"] or ""),
        detector=str(row["detector"] or ""),
        product=str(row["product"] or ""),
        station=str(row["station"] or ""),
        machine_id=str(row["machine_id"] or ""),
        work_order=str(row["work_order"] or ""),
        camera_id=str(row["camera_id"] or ""),
        model_version=str(row["model_version"] or ""),
        inference_time=(
            float(inference_time) if inference_time is not None else None
        ),
        reason_codes=_parse_reasons(row["decision_reasons_json"]),
        review_outcome=str(row["review_outcome"] or ""),
        failure_category=str(row["failure_category"] or ""),
        original_path=str(row["original_path"] or ""),
        annotated_path=str(row["annotated_path"] or ""),
        heatmap_path=str(row["heatmap_path"] or ""),
        snapshot_path=str(row["snapshot_path"] or ""),
    )


def _parse_reasons(raw: Any) -> tuple[str, ...]:
    try:
        payload = json.loads(str(raw or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        fallback = str(raw or "").strip()
        return (fallback,) if fallback else ()
    if not isinstance(payload, list):
        return (str(payload),)
    reasons: list[str] = []
    for item in payload:
        if isinstance(item, Mapping):
            value = (
                item.get("message")
                or item.get("reason")
                or item.get("code")
            )
            if value:
                reasons.append(str(value))
        elif str(item).strip():
            reasons.append(str(item).strip())
    return tuple(dict.fromkeys(reasons))


def parse_inspection_timestamp(
    value: str,
    *,
    local_timezone: ZoneInfo = HISTORY_TIMEZONE,
) -> datetime | None:
    """Interpret legacy naive timestamps as local and normalize aware values."""
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized.endswith(("Z", "z")):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=local_timezone)
    return parsed.astimezone(local_timezone)


def local_day_range(
    day: date,
    *,
    timezone_name: str = HISTORY_TIMEZONE_NAME,
) -> InspectionTimeRange:
    local_timezone = ZoneInfo(timezone_name)
    started_at = datetime.combine(day, time.min, tzinfo=local_timezone)
    return InspectionTimeRange(
        started_at=started_at,
        ended_at=started_at + timedelta(days=1),
    )


def rolling_days_range(
    now: datetime,
    *,
    days: int,
    timezone_name: str = HISTORY_TIMEZONE_NAME,
) -> InspectionTimeRange:
    if type(days) is not int or days <= 0:
        raise ValueError("Rolling history days must be a positive integer.")
    local_now = _as_local_datetime(now, timezone_name)
    return InspectionTimeRange(
        started_at=local_now - timedelta(days=days),
        ended_at=local_now,
    )


def _normalize_boundary(
    value: datetime | None,
    name: str,
) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(
            f"Inspection history {name} time must be timezone-aware."
        )
    return value.astimezone(HISTORY_TIMEZONE)


def _as_local_datetime(value: datetime, timezone_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("Current history time must be timezone-aware.")
    return value.astimezone(ZoneInfo(timezone_name))
