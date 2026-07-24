from __future__ import annotations

import socket
from datetime import datetime, timedelta, timezone

from tools.process_liveness import (
    heartbeat_age_seconds,
    heartbeat_is_stale,
    is_process_active,
)


def test_invalid_process_ids_are_not_active() -> None:
    assert is_process_active(0) is False
    assert is_process_active(-1) is False


def test_remote_process_is_treated_as_active_to_prevent_duplicate_work() -> None:
    remote_host = f"{socket.gethostname()}-remote"

    assert is_process_active(12345, remote_host) is True


def test_heartbeat_staleness_uses_explicit_lease_timeout() -> None:
    now = datetime(2026, 7, 22, 12, 0, tzinfo=timezone.utc)
    recent = (now - timedelta(seconds=10)).isoformat()
    expired = (now - timedelta(seconds=46)).isoformat()

    assert heartbeat_age_seconds(recent, now=now) == 10
    assert heartbeat_is_stale(recent, 45, now=now) is False
    assert heartbeat_is_stale(expired, 45, now=now) is True
    assert heartbeat_is_stale("", 45, now=now) is False


def test_invalid_explicit_heartbeat_is_stale() -> None:
    assert heartbeat_is_stale("not-a-timestamp", 45) is True
