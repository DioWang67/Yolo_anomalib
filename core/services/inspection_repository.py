"""Queryable SQLite index for immutable inspection evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class InspectionRepository:
    """Store inspection metadata while keeping image files outside the database."""

    def __init__(self, database_path: str | Path) -> None:
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

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
        now = _utc_now()
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
        now = _utc_now()
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
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER NOT NULL
                );
                INSERT INTO schema_info(version)
                SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_info);

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
                    training_selected INTEGER NOT NULL DEFAULT 0 CHECK(training_selected IN (0, 1)),
                    training_set_state TEXT NOT NULL DEFAULT 'not_selected',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ai_predictions (
                    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
                    prediction_index INTEGER NOT NULL,
                    class_id INTEGER,
                    class_name TEXT NOT NULL DEFAULT '',
                    confidence REAL,
                    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
                    mask_json TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (inspection_id, prediction_index)
                );

                CREATE TABLE IF NOT EXISTS inspection_artifacts (
                    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
                    artifact_type TEXT NOT NULL,
                    artifact_index INTEGER NOT NULL,
                    path TEXT NOT NULL,
                    PRIMARY KEY (inspection_id, artifact_type, artifact_index)
                );

                CREATE TABLE IF NOT EXISTS review_events (
                    review_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id) ON DELETE CASCADE,
                    review_outcome TEXT NOT NULL,
                    review_label TEXT NOT NULL,
                    failure_category TEXT NOT NULL,
                    failure_source TEXT NOT NULL,
                    failure_note TEXT NOT NULL,
                    skip_reason TEXT NOT NULL,
                    action_route TEXT NOT NULL,
                    training_selected INTEGER NOT NULL CHECK(training_selected IN (0, 1)),
                    reviewed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_inspections_product ON inspections(product);
                CREATE INDEX IF NOT EXISTS idx_inspections_equipment ON inspections(machine_id, station, camera_id);
                CREATE INDEX IF NOT EXISTS idx_inspections_model ON inspections(model_version);
                CREATE INDEX IF NOT EXISTS idx_inspections_review ON inspections(review_outcome, failure_category);
                CREATE INDEX IF NOT EXISTS idx_inspections_timestamp ON inspections(timestamp);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
