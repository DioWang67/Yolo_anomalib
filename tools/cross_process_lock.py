"""Small cross-platform file lock for short atomic metadata mutations."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}


class CrossProcessLockTimeoutError(TimeoutError):
    """Raised when a metadata mutation lock cannot be acquired in time."""


@contextmanager
def cross_process_file_lock(
    lock_path: str | Path,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Hold one byte-range lock across threads and operating-system processes."""

    raw = Path(lock_path).expanduser()
    # Checked before resolving, which is the only point where a symbolic link is
    # still visible: ``resolve()`` follows it, so the same test afterwards is
    # always False and would read as a guard while protecting nothing.
    if raw.is_symlink() or raw.parent.is_symlink():
        raise ValueError(f"Lock path is unsafe: {raw}")
    resolved = raw.resolve()
    with _LOCKS_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(resolved, threading.RLock())
    timeout_seconds = max(float(timeout), 0.0)
    started_at = time.monotonic()
    if not process_lock.acquire(timeout=timeout_seconds):
        raise CrossProcessLockTimeoutError(f"Timed out waiting for lock: {resolved}")
    handle: BinaryIO | None = None
    locked = False
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        handle = resolved.open("a+b")
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        while not locked:
            handle.seek(0)
            try:
                _lock_byte(handle)
                locked = True
            except OSError as exc:
                if time.monotonic() - started_at >= timeout_seconds:
                    raise CrossProcessLockTimeoutError(
                        f"Timed out waiting for lock: {resolved}"
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        if handle is not None:
            if locked:
                try:
                    handle.seek(0)
                    _unlock_byte(handle)
                except OSError:
                    pass
            try:
                handle.close()
            except OSError:
                pass
        process_lock.release()


def _lock_byte(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - production station is Windows
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_byte(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - production station is Windows
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
