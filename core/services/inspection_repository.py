"""Queryable SQLite index for immutable inspection evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.services.inspection_database import (
    SCHEMA_VERSION as _SCHEMA_VERSION,
)
from core.services.inspection_database import (
    InspectionDatabaseBackup,
    InspectionDatabaseManager,
)

SCHEMA_VERSION = _SCHEMA_VERSION


class InspectionRepository:
    """Store inspection metadata while keeping image files outside the database."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now_provider = now_provider or _system_utc_now
        self._database = InspectionDatabaseManager(self.path)
        self._initialize()

    def backup(
        self,
        destination: str | Path | None = None,
        *,
        reason: str = "manual",
    ) -> InspectionDatabaseBackup:
        """Create one verified online backup without stopping inspection reads."""
        return self._database.backup(destination, reason=reason)

    def check_integrity(self) -> None:
        """Run a complete SQLite integrity check."""
        self._database.check_integrity()

    def upsert_snapshot_file(self, snapshot_path: str | Path) -> str:
        """Index one schema-v2 result snapshot and return its stable ID."""
        path = Path(snapshot_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Inspection snapshot must contain an object: {path}")
        return self.upsert_snapshot(payload, snapshot_path=path)

    def upsert_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        snapshot_path: str | Path,
    ) -> str:
        """Atomically upsert one inspection, its artifacts, and AI predictions."""
        path = Path(snapshot_path).resolve()
        inspection_id = _inspection_id(path)
        artifacts = _mapping(payload.get("artifacts"))
        equipment = _mapping(payload.get("equipment"))
        model_info = _mapping(payload.get("model_info"))
        detections = [
            dict(item)
            for item in payload.get("detections", [])
            if isinstance(item, Mapping)
        ]
        now = _utc_now(self._now_provider)
        artifact_rows = _artifact_rows(inspection_id, artifacts)

        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO inspections (
                    inspection_id, snapshot_path, timestamp, status, detector,
                    product, station, machine_id, work_order, camera_id,
                    model_version, model_weights, inference_time,
                    decision_reasons_json, predictions_json, original_path,
                    preprocessed_path, annotated_path, heatmap_path,
                    crop_paths_json, mask_paths_json, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(inspection_id) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    status=excluded.status,
                    detector=excluded.detector,
                    product=excluded.product,
                    station=excluded.station,
                    machine_id=excluded.machine_id,
                    work_order=excluded.work_order,
                    camera_id=excluded.camera_id,
                    model_version=excluded.model_version,
                    model_weights=excluded.model_weights,
                    inference_time=excluded.inference_time,
                    decision_reasons_json=excluded.decision_reasons_json,
                    predictions_json=excluded.predictions_json,
                    original_path=excluded.original_path,
                    preprocessed_path=excluded.preprocessed_path,
                    annotated_path=excluded.annotated_path,
                    heatmap_path=excluded.heatmap_path,
                    crop_paths_json=excluded.crop_paths_json,
                    mask_paths_json=excluded.mask_paths_json,
                    updated_at=excluded.updated_at
                """,
                (
                    inspection_id,
                    str(path),
                    str(payload.get("timestamp") or ""),
                    str(payload.get("status") or ""),
                    str(payload.get("detector") or ""),
                    str(payload.get("product") or ""),
                    str(equipment.get("station") or payload.get("area") or ""),
                    str(equipment.get("machine_id") or ""),
                    str(equipment.get("work_order") or ""),
                    str(equipment.get("camera_id") or ""),
                    str(model_info.get("model_version") or ""),
                    str(model_info.get("weights") or ""),
                    _optional_float(payload.get("inference_time")),
                    _json(payload.get("fail_reasons") or []),
                    _json(detections),
                    str(artifacts.get("original_path") or ""),
                    str(artifacts.get("preprocessed_path") or ""),
                    str(artifacts.get("annotated_path") or ""),
                    str(artifacts.get("heatmap_path") or ""),
                    _json(artifacts.get("cropped_paths") or []),
                    _json(artifacts.get("mask_paths") or []),
                    now,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM inspection_artifacts WHERE inspection_id = ?",
                (inspection_id,),
            )
            connection.executemany(
                """
                INSERT INTO inspection_artifacts (
                    inspection_id, artifact_type, artifact_index, path
                ) VALUES (?, ?, ?, ?)
                """,
                artifact_rows,
            )
            connection.execute(
                "DELETE FROM ai_predictions WHERE inspection_id = ?",
                (inspection_id,),
            )
            connection.executemany(
                """
                INSERT INTO ai_predictions (
                    inspection_id, prediction_index, class_id, class_name,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    _prediction_row(inspection_id, index, prediction)
                    for index, prediction in enumerate(detections)
                ],
            )
            _enqueue_sync(connection, inspection_id, now)
        return inspection_id

    def sync_review_row(
        self, row: Mapping[str, Any], *, append_event: bool = True
    ) -> str:
        """Persist the latest operator decision and append an audit event."""
        snapshot_path = str(row.get("config_snapshot_path") or "").strip()
        if not snapshot_path:
            raise ValueError("config_snapshot_path is required for review indexing")
        path = Path(snapshot_path).resolve()
        inspection_id = _inspection_id(path)
        now = _utc_now(self._now_provider)
        with closing(self._connect()) as connection:
            exists = connection.execute(
                "SELECT 1 FROM inspections WHERE inspection_id=?",
                (inspection_id,),
            ).fetchone()
        if exists is None:
            if not path.is_file():
                raise FileNotFoundError(f"Inspection snapshot is missing: {path}")
            self.upsert_snapshot_file(path)
        review_values = (
            str(row.get("review_outcome") or ""),
            str(row.get("review_label") or ""),
            str(row.get("failure_category") or ""),
            str(row.get("failure_source") or ""),
            str(row.get("failure_note") or row.get("review_note") or ""),
            str(row.get("skip_reason") or ""),
            str(row.get("action_route") or ""),
            1 if str(row.get("training_selected") or "0") == "1" else 0,
            (
                "selected"
                if str(row.get("training_selected") or "0") == "1"
                else "not_selected"
            ),
            now,
            inspection_id,
        )
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE inspections SET
                    review_outcome=?, review_label=?, failure_category=?,
                    failure_source=?, failure_note=?, skip_reason=?, action_route=?,
                    training_selected=?, training_set_state=?, updated_at=?
                WHERE inspection_id=?
                """,
                review_values,
            )
            if append_event:
                connection.execute(
                    """
                    INSERT INTO review_events (
                        inspection_id, review_outcome, review_label, failure_category,
                        failure_source, failure_note, skip_reason, action_route,
                        training_selected, reviewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        inspection_id,
                        *review_values[:8],
                        now,
                    ),
                )
            _enqueue_sync(connection, inspection_id, now)
        return inspection_id

    def load_manifest_sync_state(self) -> dict[str, dict[str, str]]:
        """Load the indexed snapshot/review state in one read transaction.

        Inspection snapshots are immutable.  Callers can therefore skip an
        expensive snapshot UPSERT when its resolved path is already indexed,
        and only synchronize review columns that differ from the manifest.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT snapshot_path, review_outcome, review_label,
                       failure_category, failure_source, failure_note,
                       skip_reason, action_route, training_selected
                FROM inspections
                """
            ).fetchall()
        return {
            str(row["snapshot_path"]): {
                "review_outcome": str(row["review_outcome"] or ""),
                "review_label": str(row["review_label"] or ""),
                "failure_category": str(row["failure_category"] or ""),
                "failure_source": str(row["failure_source"] or ""),
                "failure_note": str(row["failure_note"] or ""),
                "skip_reason": str(row["skip_reason"] or ""),
                "action_route": str(row["action_route"] or ""),
                "training_selected": "1" if row["training_selected"] else "0",
            }
            for row in rows
        }

    def query(self, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Execute a read-only SELECT for reporting and return named rows."""
        if not sql.lstrip().lower().startswith(("select", "with")):
            raise ValueError("Only SELECT/CTE queries are allowed")
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute(sql, parameters).fetchall()
        return [dict(row) for row in rows]

    def _initialize(self) -> None:
        self._database.initialize()

    def _connect(self) -> sqlite3.Connection:
        return self._database.connect()


def _inspection_id(snapshot_path: Path) -> str:
    return hashlib.sha256(str(snapshot_path).encode("utf-8")).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _optional_float(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _artifact_rows(
    inspection_id: str, artifacts: Mapping[str, Any]
) -> list[tuple[str, str, int, str]]:
    rows: list[tuple[str, str, int, str]] = []
    singular = {
        "original": artifacts.get("original_path"),
        "preprocessed": artifacts.get("preprocessed_path"),
        "annotated": artifacts.get("annotated_path"),
        "heatmap": artifacts.get("heatmap_path"),
    }
    for artifact_type, path in singular.items():
        if str(path or "").strip():
            rows.append((inspection_id, artifact_type, 0, str(path)))
    for artifact_type, key in (("crop", "cropped_paths"), ("mask", "mask_paths")):
        paths = artifacts.get(key)
        if isinstance(paths, list):
            rows.extend(
                (inspection_id, artifact_type, index, str(path))
                for index, path in enumerate(paths)
                if str(path or "").strip()
            )
    return rows


def _prediction_row(
    inspection_id: str,
    index: int,
    prediction: Mapping[str, Any],
) -> tuple[Any, ...]:
    bbox = prediction.get("bbox")
    coordinates: list[float | None] = [None, None, None, None]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        coordinates = [_optional_float(value) for value in bbox[:4]]
    mask = prediction.get("mask") or prediction.get("segmentation") or ""
    try:
        class_id = int(prediction["class_id"])
    except (KeyError, TypeError, ValueError):
        class_id = None
    return (
        inspection_id,
        index,
        class_id,
        str(prediction.get("class") or prediction.get("class_name") or ""),
        _optional_float(prediction.get("confidence")),
        *coordinates,
        _json(mask) if mask != "" else "",
    )


def _system_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now(now_provider: Callable[[], datetime]) -> str:
    current = now_provider()
    if not isinstance(current, datetime):
        raise TypeError("Inspection repository clock must return datetime.")
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError(
            "Inspection repository clock must return a timezone-aware datetime."
        )
    return current.astimezone(timezone.utc).isoformat()


def _enqueue_sync(
    connection: sqlite3.Connection,
    inspection_id: str,
    now: str,
) -> None:
    """Publish the latest committed inspection revision to the local outbox."""
    connection.execute(
        """
        INSERT INTO inspection_sync_outbox (
            inspection_id, revision, state, attempt_count, next_attempt_at,
            lease_token, lease_expires_at, last_error, synced_at, updated_at
        ) VALUES (?, 1, 'pending', 0, ?, '', '', '', '', ?)
        ON CONFLICT(inspection_id) DO UPDATE SET
            revision=inspection_sync_outbox.revision + 1,
            state='pending',
            attempt_count=0,
            next_attempt_at=excluded.next_attempt_at,
            lease_token='',
            lease_expires_at='',
            last_error='',
            synced_at='',
            updated_at=excluded.updated_at
        """,
        (inspection_id, now, now),
    )
