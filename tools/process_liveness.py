"""Safe cross-platform process liveness checks for operator workflows."""

from __future__ import annotations

import os
import socket


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
