"""Cross-process control requests for operator retraining jobs."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_JOB_STATES = frozenset({"deployed", "failed", "cancelled", "invalid"})


class OperatorJobControlError(RuntimeError):
    """Raised when a retraining control request cannot be published safely."""


@dataclass(frozen=True)
class OperatorJobControlReceipt:
    """Durable identity returned for an idempotent control request."""

    job_id: str
    request_id: str
    action: str
    requested_at: str
    control_path: Path
    reused_existing: bool


def request_operator_job_cancel(
    status_path: str | Path,
    *,
    job_id: str,
) -> OperatorJobControlReceipt:
    """Durably request cooperative cancellation without touching job status.

    Status has a single writer (the training process) and ``control.json`` has
    a single writer (the inference process). This avoids a lost-update race
    between heartbeat/status publication and operator control requests.
    """
    path = Path(status_path).expanduser().resolve()
    normalized_job_id = str(job_id or "").strip()
    if path.name != "status.json" or path.parent.name != normalized_job_id:
        raise OperatorJobControlError("Invalid retraining job status identity.")
    status = _read_json_mapping(path)
    if str(status.get("job_id") or "") != normalized_job_id:
        raise OperatorJobControlError("Retraining status does not match the selected job.")
    state = str(status.get("state") or "").strip().lower()
    if state in TERMINAL_JOB_STATES:
        raise OperatorJobControlError(f"Retraining job is already terminal ({state}).")

    control_path = path.with_name("control.json")
    existing = _read_json_mapping(control_path)
    existing_request_id = str(existing.get("request_id") or "").strip()
    if (
        str(existing.get("job_id") or "") == normalized_job_id
        and str(existing.get("action") or "").strip().lower() == "cancel"
        and existing_request_id
        and existing_request_id
        != str(status.get("handled_control_request_id") or "").strip()
    ):
        return _receipt(control_path, existing, reused_existing=True)

    payload = {
        "schema_version": 1,
        "job_id": normalized_job_id,
        "request_id": uuid.uuid4().hex,
        "action": "cancel",
        "requested_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _write_json_atomic(control_path, payload)
    except OSError as exc:
        raise OperatorJobControlError(
            f"Unable to write retraining cancellation request: {exc}"
        ) from exc
    persisted = _read_json_mapping(control_path)
    if persisted != payload:
        raise OperatorJobControlError(
            "Retraining cancellation request failed read-back verification."
        )
    return _receipt(control_path, payload, reused_existing=False)


def _receipt(
    control_path: Path,
    payload: dict[str, Any],
    *,
    reused_existing: bool,
) -> OperatorJobControlReceipt:
    return OperatorJobControlReceipt(
        job_id=str(payload.get("job_id") or ""),
        request_id=str(payload.get("request_id") or ""),
        action=str(payload.get("action") or ""),
        requested_at=str(payload.get("requested_at") or ""),
        control_path=control_path,
        reused_existing=reused_existing,
    )


def _read_json_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
