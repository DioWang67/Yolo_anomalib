"""Safe cross-platform process liveness checks for operator workflows."""

from __future__ import annotations

import os
import socket
from datetime import datetime, timezone
from typing import Any

DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 45.0


def is_process_active(process_id: int, process_host: str = "") -> bool:
    """Return whether a recorded process is active without mutating it."""
    try:
        pid = int(process_id)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    host = str(process_host or "").strip().lower()
    if host and host != socket.gethostname().lower():
        return True
    if os.name == "nt":
        return _is_windows_process_active(pid)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _is_windows_process_active(process_id: int) -> bool:
    """Query a Windows process handle without sending it a signal."""
    import ctypes

    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information,
        False,
        process_id,
    )
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not ctypes.windll.kernel32.GetExitCodeProcess(
            handle,
            ctypes.byref(exit_code),
        ):
            return False
        return exit_code.value == still_active
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def heartbeat_age_seconds(
    heartbeat_at: Any,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return a non-negative heartbeat age, or ``None`` for legacy records."""
    text = str(heartbeat_at or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    return max(0.0, (reference.astimezone(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds())


def heartbeat_is_stale(
    heartbeat_at: Any,
    timeout_seconds: Any = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    *,
    now: datetime | None = None,
) -> bool:
    """Return whether an explicitly published heartbeat lease has expired."""
    age = heartbeat_age_seconds(heartbeat_at, now=now)
    if age is None:
        return False
    try:
        timeout = float(timeout_seconds)
    except (TypeError, ValueError):
        timeout = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    return age > max(5.0, timeout)
