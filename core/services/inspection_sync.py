"""Offline-first synchronization of inspection records to a company API."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import requests

from core.services.inspection_database import InspectionDatabaseManager

_SYNC_PAYLOAD_SCHEMA_VERSION = 1


class InspectionSyncError(RuntimeError):
    """Base error for durable inspection synchronization."""


class InspectionSyncConfigurationError(InspectionSyncError):
    """The company synchronization settings are unsafe or incomplete."""


class InspectionSyncTransportError(InspectionSyncError):
    """The remote API did not accept an inspection revision."""


@dataclass(frozen=True)
class InspectionSyncClaim:
    inspection_id: str
    revision: int
    lease_token: str
    attempt_count: int


@dataclass(frozen=True)
class InspectionSyncPolicy:
    batch_size: int = 20
    lease_seconds: int = 120
    max_attempts: int = 12
    base_backoff_seconds: int = 5
    max_backoff_seconds: int = 3600

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 500:
            raise ValueError("Inspection sync batch size must be between 1 and 500.")
        if type(self.lease_seconds) is not int or self.lease_seconds < 10:
            raise ValueError("Inspection sync lease must be at least 10 seconds.")
        if type(self.max_attempts) is not int or self.max_attempts < 1:
            raise ValueError("Inspection sync max attempts must be positive.")
        if (
            type(self.base_backoff_seconds) is not int
            or self.base_backoff_seconds < 1
            or type(self.max_backoff_seconds) is not int
            or self.max_backoff_seconds < self.base_backoff_seconds
        ):
            raise ValueError("Inspection sync backoff settings are invalid.")


@dataclass(frozen=True)
class InspectionSyncStatus:
    pending: int = 0
    inflight: int = 0
    synced: int = 0
    dead: int = 0


class InspectionSyncTransport(Protocol):
    def send(
        self,
        payload: Mapping[str, Any],
        *,
        inspection_id: str,
        revision: int,
    ) -> None: ...

    def close(self) -> None: ...


class RequestsInspectionSyncTransport:
    """POST idempotent inspection revisions over authenticated HTTPS."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_token: str,
        timeout_seconds: float = 10.0,
        allow_insecure_http: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        normalized_endpoint = str(endpoint or "").strip()
        parsed = urlparse(normalized_endpoint)
        is_local_http = parsed.scheme == "http" and parsed.hostname in {
            "127.0.0.1",
            "localhost",
            "::1",
        }
        if (
            not parsed.hostname
            or parsed.scheme not in {"http", "https"}
            or (
                parsed.scheme != "https"
                and not is_local_http
                and not allow_insecure_http
            )
        ):
            raise InspectionSyncConfigurationError(
                "Inspection sync endpoint must be HTTPS. HTTP is allowed only "
                "for localhost unless explicitly enabled."
            )
        if not str(api_token or "").strip():
            raise InspectionSyncConfigurationError(
                "Inspection sync API token is empty."
            )
        if timeout_seconds <= 0:
            raise ValueError("Inspection sync timeout must be positive.")
        self.endpoint = normalized_endpoint
        self.timeout_seconds = float(timeout_seconds)
        self._session = session or requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {api_token.strip()}",
                "Content-Type": "application/json",
                "User-Agent": "yolo11-inference/inspection-sync-v1",
            }
        )

    def send(
        self,
        payload: Mapping[str, Any],
        *,
        inspection_id: str,
        revision: int,
    ) -> None:
        try:
            response = self._session.post(
                self.endpoint,
                json=dict(payload),
                headers={
                    "Idempotency-Key": inspection_id,
                    "X-Inspection-Revision": str(revision),
                },
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise InspectionSyncTransportError(
                f"Company API request failed: {exc}"
            ) from exc
        if not 200 <= response.status_code < 300:
            detail = str(response.text or "").strip().replace("\r", " ").replace(
                "\n", " "
            )[:300]
            raise InspectionSyncTransportError(
                "Company API rejected inspection "
                f"(HTTP {response.status_code}): {detail or 'no response body'}"
            )

    def close(self) -> None:
        self._session.close()


class InspectionSyncOutbox:
    """Lease and acknowledge inspection revisions without holding I/O locks."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._database = InspectionDatabaseManager(self.database_path)

    def claim(
        self,
        policy: InspectionSyncPolicy,
        *,
        now: datetime | None = None,
    ) -> tuple[InspectionSyncClaim, ...]:
        current = _utc(now)
        now_text = current.isoformat()
        lease_expires = (current + timedelta(seconds=policy.lease_seconds)).isoformat()
        lease_token = uuid.uuid4().hex
        with closing(self._database.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT inspection_id, revision, attempt_count
                FROM inspection_sync_outbox
                WHERE (
                    state = 'pending' AND next_attempt_at <= ?
                ) OR (
                    state = 'inflight' AND lease_expires_at <= ?
                )
                ORDER BY next_attempt_at, inspection_id
                LIMIT ?
                """,
                (now_text, now_text, policy.batch_size),
            ).fetchall()
            claims = tuple(
                InspectionSyncClaim(
                    inspection_id=str(row["inspection_id"]),
                    revision=int(row["revision"]),
                    lease_token=lease_token,
                    attempt_count=int(row["attempt_count"]),
                )
                for row in rows
            )
            connection.executemany(
                """
                UPDATE inspection_sync_outbox
                SET state='inflight', lease_token=?, lease_expires_at=?,
                    updated_at=?
                WHERE inspection_id=? AND revision=?
                """,
                [
                    (
                        claim.lease_token,
                        lease_expires,
                        now_text,
                        claim.inspection_id,
                        claim.revision,
                    )
                    for claim in claims
                ],
            )
        return claims

    def load_payload(self, claim: InspectionSyncClaim) -> dict[str, Any]:
        with closing(self._database.connect(readonly=True)) as connection:
            connection.execute("BEGIN")
            inspection = connection.execute(
                "SELECT * FROM inspections WHERE inspection_id=?",
                (claim.inspection_id,),
            ).fetchone()
            if inspection is None:
                raise InspectionSyncError(
                    f"Inspection disappeared before synchronization: "
                    f"{claim.inspection_id}"
                )
            predictions = connection.execute(
                """
                SELECT prediction_index, class_id, class_name, confidence,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2, mask_json
                FROM ai_predictions
                WHERE inspection_id=?
                ORDER BY prediction_index
                """,
                (claim.inspection_id,),
            ).fetchall()
            artifacts = connection.execute(
                """
                SELECT artifact_type, artifact_index, path
                FROM inspection_artifacts
                WHERE inspection_id=?
                ORDER BY artifact_type, artifact_index
                """,
                (claim.inspection_id,),
            ).fetchall()
            connection.execute("COMMIT")
        inspection_payload = dict(inspection)
        for key in (
            "decision_reasons_json",
            "predictions_json",
            "crop_paths_json",
            "mask_paths_json",
        ):
            inspection_payload[key.removesuffix("_json")] = _decode_json(
                inspection_payload.pop(key, None)
            )
        return {
            "schema_version": _SYNC_PAYLOAD_SCHEMA_VERSION,
            "idempotency_key": claim.inspection_id,
            "revision": claim.revision,
            "inspection": inspection_payload,
            "predictions": [dict(row) for row in predictions],
            "artifacts": [dict(row) for row in artifacts],
        }

    def mark_success(
        self,
        claim: InspectionSyncClaim,
        *,
        now: datetime | None = None,
    ) -> bool:
        timestamp = _utc(now).isoformat()
        with closing(self._database.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE inspection_sync_outbox
                SET state='synced', lease_token='', lease_expires_at='',
                    last_error='', synced_at=?, updated_at=?
                WHERE inspection_id=? AND revision=? AND state='inflight'
                    AND lease_token=?
                """,
                (
                    timestamp,
                    timestamp,
                    claim.inspection_id,
                    claim.revision,
                    claim.lease_token,
                ),
            )
        return cursor.rowcount == 1

    def mark_failure(
        self,
        claim: InspectionSyncClaim,
        error: str,
        policy: InspectionSyncPolicy,
        *,
        now: datetime | None = None,
    ) -> bool:
        current = _utc(now)
        next_attempt_count = claim.attempt_count + 1
        dead = next_attempt_count >= policy.max_attempts
        delay = min(
            policy.max_backoff_seconds,
            policy.base_backoff_seconds * (2 ** min(claim.attempt_count, 20)),
        )
        next_attempt = (
            current if dead else current + timedelta(seconds=delay)
        ).isoformat()
        message = str(error or "Unknown synchronization error")[:1000]
        with closing(self._database.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE inspection_sync_outbox
                SET state=?, attempt_count=?, next_attempt_at=?,
                    lease_token='', lease_expires_at='', last_error=?,
                    updated_at=?
                WHERE inspection_id=? AND revision=? AND state='inflight'
                    AND lease_token=?
                """,
                (
                    "dead" if dead else "pending",
                    next_attempt_count,
                    next_attempt,
                    message,
                    current.isoformat(),
                    claim.inspection_id,
                    claim.revision,
                    claim.lease_token,
                ),
            )
        return cursor.rowcount == 1

    def status(self) -> InspectionSyncStatus:
        if not self.database_path.is_file():
            return InspectionSyncStatus()
        with closing(self._database.connect(readonly=True)) as connection:
            rows = connection.execute(
                """
                SELECT state, COUNT(*) AS count
                FROM inspection_sync_outbox
                GROUP BY state
                """
            ).fetchall()
        counts = {str(row["state"]): int(row["count"]) for row in rows}
        return InspectionSyncStatus(
            pending=counts.get("pending", 0),
            inflight=counts.get("inflight", 0),
            synced=counts.get("synced", 0),
            dead=counts.get("dead", 0),
        )

    def retry_dead(self, *, now: datetime | None = None) -> int:
        """Explicitly return dead-letter rows to the retry queue."""
        timestamp = _utc(now).isoformat()
        with closing(self._database.connect()) as connection, connection:
            cursor = connection.execute(
                """
                UPDATE inspection_sync_outbox
                SET state='pending', attempt_count=0, next_attempt_at=?,
                    lease_token='', lease_expires_at='', last_error='',
                    updated_at=?
                WHERE state='dead'
                """,
                (timestamp, timestamp),
            )
        return max(0, cursor.rowcount)


class InspectionSyncWorker:
    """Wakeable single worker for eventually consistent company upload."""

    def __init__(
        self,
        outbox: InspectionSyncOutbox,
        transport: InspectionSyncTransport,
        *,
        policy: InspectionSyncPolicy | None = None,
        interval_seconds: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("Inspection sync interval must be positive.")
        self.outbox = outbox
        self.transport = transport
        self.policy = policy or InspectionSyncPolicy()
        self.interval_seconds = float(interval_seconds)
        self.logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._sync_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="inspection-company-sync",
            daemon=True,
        )
        self._thread.start()

    def notify(self) -> None:
        self._wake_event.set()

    def sync_once(self) -> int:
        with self._sync_lock:
            succeeded = 0
            for claim in self.outbox.claim(self.policy):
                if self._stop_event.is_set():
                    break
                try:
                    payload = self.outbox.load_payload(claim)
                    self.transport.send(
                        payload,
                        inspection_id=claim.inspection_id,
                        revision=claim.revision,
                    )
                except (InspectionSyncError, OSError, sqlite3.Error) as exc:
                    self.outbox.mark_failure(claim, str(exc), self.policy)
                    self.logger.warning(
                        "Inspection sync deferred: inspection_id=%s revision=%d "
                        "reason=%s",
                        claim.inspection_id,
                        claim.revision,
                        exc,
                    )
                    continue
                if self.outbox.mark_success(claim):
                    succeeded += 1
            return succeeded

    def close(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        self._thread.join(timeout=max(0.0, timeout))

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._wake_event.wait(self.interval_seconds)
                self._wake_event.clear()
                if self._stop_event.is_set():
                    break
                try:
                    self.sync_once()
                except (InspectionSyncError, OSError, sqlite3.Error) as exc:
                    self.logger.error(
                        "Inspection sync worker cycle failed: %s",
                        exc,
                    )
        finally:
            self.transport.close()


def build_inspection_sync_worker(
    database_path: str | Path,
    *,
    endpoint: str,
    api_token_env: str,
    timeout_seconds: float,
    interval_seconds: float,
    batch_size: int,
    max_attempts: int,
    allow_insecure_http: bool,
    logger: logging.Logger,
) -> InspectionSyncWorker:
    token_env = str(api_token_env or "").strip()
    if not token_env:
        raise InspectionSyncConfigurationError(
            "Inspection sync token environment variable name is empty."
        )
    token = os.environ.get(token_env, "")
    if not token:
        raise InspectionSyncConfigurationError(
            f"Inspection sync token environment variable is not set: {token_env}"
        )
    transport = RequestsInspectionSyncTransport(
        endpoint,
        api_token=token,
        timeout_seconds=timeout_seconds,
        allow_insecure_http=allow_insecure_http,
    )
    return InspectionSyncWorker(
        InspectionSyncOutbox(database_path),
        transport,
        policy=InspectionSyncPolicy(
            batch_size=batch_size,
            max_attempts=max_attempts,
        ),
        interval_seconds=interval_seconds,
        logger=logger,
    )


def _decode_json(value: object) -> Any:
    try:
        return json.loads(str(value or "null"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _utc(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise ValueError("Inspection sync clock must be timezone-aware.")
    return current.astimezone(timezone.utc)
