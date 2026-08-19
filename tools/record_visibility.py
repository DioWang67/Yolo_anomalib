"""Concurrency-safe visibility controls for operator-facing history records."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

RECORD_CATEGORIES = frozenset({"model_update_jobs", "submission_history"})
_SCHEMA_VERSION = 1


class RecordVisibilityError(RuntimeError):
    """Raised when record visibility state cannot be updated safely."""


def load_hidden_record_ids(
    data_root: str | Path, category: str
) -> frozenset[str]:
    """Return record IDs hidden from one operator-facing history list."""
    normalized = _validate_category(category)
    payload = _read_payload(_visibility_path(data_root))
    values = payload.get(normalized)
    if not isinstance(values, list):
        return frozenset()
    return frozenset(str(value) for value in values if str(value).strip())


def hide_record(data_root: str | Path, category: str, record_id: str) -> None:
    """Atomically hide one history record without deleting training artifacts."""
    normalized = _validate_category(category)
    identifier = str(record_id or "").strip()
    if not identifier:
        raise ValueError("record_id must not be empty")
    path = _visibility_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _visibility_lock(path.parent):
        payload = _read_payload(path)
        hidden = {
            str(value)
            for value in payload.get(normalized, [])
            if str(value).strip()
        }
        hidden.add(identifier)
        payload[normalized] = sorted(hidden)
        _write_payload_atomic(path, payload)


def _validate_category(category: str) -> str:
    normalized = str(category or "").strip()
    if normalized not in RECORD_CATEGORIES:
        raise ValueError(f"Unsupported record category: {category}")
    return normalized


def _visibility_path(data_root: str | Path) -> Path:
    return (
        Path(data_root).expanduser().resolve()
        / ".operator_handoff"
        / "hidden_records.json"
    )


def _read_payload(path: Path) -> dict[str, object]:
    empty: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "model_update_jobs": [],
        "submission_history": [],
    }
    if not path.is_file():
        return empty
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return empty
    if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
        return empty
    for category in RECORD_CATEGORIES:
        if not isinstance(raw.get(category), list):
            raw[category] = []
    return raw


def _write_payload_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _visibility_lock(directory: Path) -> Iterator[None]:
    lock_path = directory / "hidden_records.lock"
    descriptor: int | None = None
    for _attempt in range(20):
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            break
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 10.0
            except FileNotFoundError:
                continue
            if stale:
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(0.05)
    if descriptor is None:
        raise RecordVisibilityError("Record cleanup is already running")
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
