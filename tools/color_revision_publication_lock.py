"""Cross-process lock shared by color activation and model publication."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

_LOCK_FILENAME = "deployment-publication.lock"
_PROCESS_LOCK = threading.Lock()


class ColorRevisionPublicationLockTimeoutError(TimeoutError):
    """Raised when another process is publishing a linked runtime contract."""


def color_revision_publication_lock_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / "locks" / _LOCK_FILENAME


@contextmanager
def color_revision_publication_lock(
    root: str | Path,
    *,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Hold the global color pointer/publication lock for a short mutation."""
    timeout_seconds = max(float(timeout), 0.0)
    started_at = time.monotonic()
    if not _PROCESS_LOCK.acquire(timeout=timeout_seconds):
        raise ColorRevisionPublicationLockTimeoutError(
            "Another thread is publishing a color-linked runtime contract."
        )
    handle: BinaryIO | None = None
    locked = False
    try:
        resolved_root = Path(root).expanduser().resolve()
        locks_root = resolved_root / "locks"
        if locks_root.is_symlink():
            raise ValueError(
                f"Color revision lock root cannot be a symlink: {locks_root}"
            )
        locks_root.mkdir(parents=True, exist_ok=True)
        if locks_root.is_symlink():
            raise ValueError(
                f"Color revision lock root cannot be a symlink: {locks_root}"
            )
        resolved_locks_root = locks_root.resolve()
        if not resolved_locks_root.is_relative_to(resolved_root):
            raise ValueError(
                f"Color revision lock root escapes its store: {locks_root}"
            )
        lock_path = resolved_locks_root / _LOCK_FILENAME
        if lock_path.is_symlink():
            raise ValueError(
                f"Color revision publication lock cannot be a symlink: {lock_path}"
            )
        handle = lock_path.open("a+b")
        if handle.tell() == 0:
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
                    raise ColorRevisionPublicationLockTimeoutError(
                        "Another process is publishing a color-linked runtime "
                        "contract."
                    ) from exc
                time.sleep(0.05)
        yield
    finally:
        try:
            if handle is not None and locked:
                try:
                    handle.seek(0)
                    _unlock_byte(handle)
                except OSError:
                    pass
        finally:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            _PROCESS_LOCK.release()


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
