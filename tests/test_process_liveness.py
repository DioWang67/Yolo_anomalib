from __future__ import annotations

import socket

from tools.process_liveness import is_process_active


def test_invalid_process_ids_are_not_active() -> None:
    assert is_process_active(0) is False
    assert is_process_active(-1) is False


def test_remote_process_is_treated_as_active_to_prevent_duplicate_work() -> None:
    remote_host = f"{socket.gethostname()}-remote"

    assert is_process_active(12345, remote_host) is True
