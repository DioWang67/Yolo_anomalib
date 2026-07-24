from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tools.operator_job_control import (
    OperatorJobControlError,
    request_operator_job_cancel,
)


def _write_status(root: Path, *, state: str = "training") -> Path:
    status_path = root / "jobs" / "job-1" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps({"job_id": "job-1", "state": state}),
        encoding="utf-8",
    )
    return status_path


def test_cancel_request_is_atomic_read_back_verified_and_idempotent(
    tmp_path: Path,
) -> None:
    status_path = _write_status(tmp_path)

    first = request_operator_job_cancel(status_path, job_id="job-1")
    second = request_operator_job_cancel(status_path, job_id="job-1")

    assert first.request_id == second.request_id
    assert first.reused_existing is False
    assert second.reused_existing is True
    persisted = json.loads(first.control_path.read_text(encoding="utf-8"))
    assert persisted["action"] == "cancel"
    assert persisted["job_id"] == "job-1"
    assert not list(first.control_path.parent.glob("*.tmp"))


def test_acknowledged_cancel_can_be_followed_by_a_new_request(tmp_path: Path) -> None:
    status_path = _write_status(tmp_path)
    first = request_operator_job_cancel(status_path, job_id="job-1")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["handled_control_request_id"] = first.request_id
    status_path.write_text(json.dumps(status), encoding="utf-8")

    second = request_operator_job_cancel(status_path, job_id="job-1")

    assert second.request_id != first.request_id
    assert second.reused_existing is False


def test_terminal_or_mismatched_job_rejects_cancel_request(tmp_path: Path) -> None:
    terminal = _write_status(tmp_path, state="deployed")
    with pytest.raises(OperatorJobControlError, match="terminal"):
        request_operator_job_cancel(terminal, job_id="job-1")
    with pytest.raises(OperatorJobControlError, match="identity"):
        request_operator_job_cancel(terminal, job_id="other-job")


def test_cancel_request_burst_reuses_one_durable_request(tmp_path: Path) -> None:
    status_path = _write_status(tmp_path)

    started = time.perf_counter()
    receipts = [
        request_operator_job_cancel(status_path, job_id="job-1")
        for _iteration in range(500)
    ]
    elapsed_seconds = time.perf_counter() - started

    assert len({receipt.request_id for receipt in receipts}) == 1
    assert receipts[0].reused_existing is False
    assert all(receipt.reused_existing for receipt in receipts[1:])
    assert elapsed_seconds < 5.0
