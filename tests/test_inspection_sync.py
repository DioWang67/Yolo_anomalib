from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from core.services.inspection_repository import InspectionRepository
from core.services.inspection_sync import (
    InspectionSyncConfigurationError,
    InspectionSyncOutbox,
    InspectionSyncPolicy,
    InspectionSyncTransportError,
    InspectionSyncWorker,
    RequestsInspectionSyncTransport,
    build_inspection_sync_worker,
)

_NOW = datetime(2026, 7, 29, 4, 0, tzinfo=timezone.utc)


def _insert(repository: InspectionRepository, root: Path) -> tuple[str, Path]:
    snapshot_path = root / "result.json"
    inspection_id = repository.upsert_snapshot(
        {
            "timestamp": "2026-07-29T12:00:00+08:00",
            "status": "DETECTION_FAIL",
            "detector": "yolo",
            "product": "Cable1",
            "equipment": {
                "station": "A",
                "machine_id": "LINE-01",
                "work_order": "WO-007",
                "camera_id": "CAM-A",
            },
            "model_info": {"model_version": "v3", "weights": "best.onnx"},
            "inference_time": 0.123,
            "fail_reasons": [{"code": "POSITION_SHIFT"}],
            "detections": [
                {
                    "class_id": 1,
                    "class_name": "Black",
                    "confidence": 0.99,
                    "bbox": [1, 2, 3, 4],
                }
            ],
            "artifacts": {
                "original_path": str(root / "original.jpg"),
                "annotated_path": str(root / "annotated.jpg"),
            },
        },
        snapshot_path=snapshot_path,
    )
    return inspection_id, snapshot_path


class _RecordingTransport:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.sent: list[tuple[dict, str, int]] = []
        self.closed = False

    def send(self, payload, *, inspection_id: str, revision: int) -> None:
        if self.error is not None:
            raise self.error
        self.sent.append((dict(payload), inspection_id, revision))

    def close(self) -> None:
        self.closed = True


def test_outbox_claim_payload_and_acknowledgement(tmp_path: Path) -> None:
    database_path = tmp_path / "inspection_records.sqlite3"
    repository = InspectionRepository(
        database_path,
        now_provider=lambda: _NOW,
    )
    inspection_id, _ = _insert(repository, tmp_path)
    outbox = InspectionSyncOutbox(database_path)

    claims = outbox.claim(InspectionSyncPolicy(), now=_NOW)

    assert len(claims) == 1
    claim = claims[0]
    assert claim.inspection_id == inspection_id
    payload = outbox.load_payload(claim)
    assert payload["schema_version"] == 1
    assert payload["idempotency_key"] == inspection_id
    assert payload["inspection"]["product"] == "Cable1"
    assert payload["inspection"]["decision_reasons"] == [
        {"code": "POSITION_SHIFT"}
    ]
    assert payload["predictions"][0]["class_name"] == "Black"
    assert len(payload["artifacts"]) == 2
    assert outbox.mark_success(claim, now=_NOW) is True
    assert outbox.status().synced == 1


def test_new_revision_cannot_be_acknowledged_by_stale_lease(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspection_records.sqlite3"
    repository = InspectionRepository(
        database_path,
        now_provider=lambda: _NOW,
    )
    _, snapshot_path = _insert(repository, tmp_path)
    outbox = InspectionSyncOutbox(database_path)
    stale_claim = outbox.claim(InspectionSyncPolicy(), now=_NOW)[0]

    repository.sync_review_row(
        {
            "config_snapshot_path": str(snapshot_path),
            "review_outcome": "confirmed_ng",
            "failure_category": "position",
        }
    )

    assert outbox.mark_success(stale_claim, now=_NOW) is False
    fresh_claim = outbox.claim(InspectionSyncPolicy(), now=_NOW)[0]
    assert fresh_claim.revision == stale_claim.revision + 1


def test_outbox_failure_uses_backoff_then_dead_letters(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspection_records.sqlite3"
    repository = InspectionRepository(
        database_path,
        now_provider=lambda: _NOW,
    )
    _insert(repository, tmp_path)
    outbox = InspectionSyncOutbox(database_path)
    policy = InspectionSyncPolicy(max_attempts=2, base_backoff_seconds=5)
    first = outbox.claim(policy, now=_NOW)[0]

    assert outbox.mark_failure(first, "offline", policy, now=_NOW) is True
    assert outbox.status().pending == 1
    assert outbox.claim(policy, now=_NOW + timedelta(seconds=4)) == ()
    second = outbox.claim(policy, now=_NOW + timedelta(seconds=5))[0]
    assert second.attempt_count == 1
    assert outbox.mark_failure(
        second,
        "still offline",
        policy,
        now=_NOW + timedelta(seconds=5),
    )
    assert outbox.status().dead == 1
    assert outbox.retry_dead(now=_NOW + timedelta(minutes=1)) == 1
    assert outbox.status().pending == 1


def test_worker_retries_transport_error_without_losing_record(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "inspection_records.sqlite3"
    repository = InspectionRepository(database_path)
    _insert(repository, tmp_path)
    transport = _RecordingTransport(
        error=InspectionSyncTransportError("network unavailable")
    )
    worker = InspectionSyncWorker(
        InspectionSyncOutbox(database_path),
        transport,
        policy=InspectionSyncPolicy(base_backoff_seconds=1),
        interval_seconds=3600,
    )
    try:
        assert worker.sync_once() == 0
        assert InspectionSyncOutbox(database_path).status().pending == 1
    finally:
        worker.close()
    assert transport.closed is True


class _FakeSession:
    def __init__(self, response) -> None:
        self.headers: dict[str, str] = {}
        self.response = response
        self.calls = []
        self.closed = False

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    def close(self) -> None:
        self.closed = True


def test_https_transport_sends_idempotency_contract() -> None:
    session = _FakeSession(SimpleNamespace(status_code=202, text=""))
    transport = RequestsInspectionSyncTransport(
        "https://company.example/api/v1/inspections",
        api_token="secret",
        session=session,
    )

    transport.send({"schema_version": 1}, inspection_id="abc", revision=3)
    transport.close()

    assert session.headers["Authorization"] == "Bearer secret"
    assert session.calls[0][1]["headers"] == {
        "Idempotency-Key": "abc",
        "X-Inspection-Revision": "3",
    }
    assert session.closed is True


def test_transport_rejects_insecure_endpoint_and_network_failure() -> None:
    with pytest.raises(InspectionSyncConfigurationError):
        RequestsInspectionSyncTransport(
            "http://company.example/inspections",
            api_token="secret",
        )

    transport = RequestsInspectionSyncTransport(
        "http://localhost:8000/inspections",
        api_token="secret",
        session=_FakeSession(requests.ConnectionError("offline")),
    )
    with pytest.raises(InspectionSyncTransportError, match="offline"):
        transport.send({}, inspection_id="abc", revision=1)


def test_worker_factory_requires_token_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("COMPANY_TOKEN", raising=False)

    with pytest.raises(
        InspectionSyncConfigurationError,
        match="COMPANY_TOKEN",
    ):
        build_inspection_sync_worker(
            tmp_path / "inspection_records.sqlite3",
            endpoint="https://company.example/inspections",
            api_token_env="COMPANY_TOKEN",
            timeout_seconds=10,
            interval_seconds=30,
            batch_size=20,
            max_attempts=12,
            allow_insecure_http=False,
            logger=logging.getLogger("test"),
        )
