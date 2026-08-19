"""Persistent folder workspaces for independent retraining batches."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.training_batch_version import validate_training_batch_version
from tools.submission_history import (
    DuplicateTrainingBatchVersionError,
    claim_training_batch_version,
    ensure_training_batch_version_available,
)

WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_STATES = frozenset({"draft", "submitted"})


class RetrainingWorkspaceError(ValueError):
    """A retraining workspace folder is missing, invalid, or already exists."""


@dataclass(frozen=True)
class RetrainingWorkspace:
    """One folder that owns selection, review, and training state."""

    batch_version: str
    product: str
    area: str
    root: Path
    created_at: datetime
    state: str
    job_id: str = ""
    handoff_path: Path | None = None

    @property
    def manifest_path(self) -> Path:
        return self.root / "review_manifest.csv"

    @property
    def metadata_path(self) -> Path:
        return self.root / "workspace.json"


def create_retraining_workspace(
    data_root: str | Path,
    *,
    product: str,
    area: str,
    batch_version: str,
) -> RetrainingWorkspace:
    """Atomically create an empty target-scoped batch folder."""
    root = Path(data_root).expanduser().resolve()
    normalized_version = ensure_training_batch_version_available(
        root,
        product=product,
        area=area,
        batch_version=batch_version,
    )
    jobs_root = root / ".operator_handoff" / "jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    workspace_root = (jobs_root / normalized_version).resolve()
    if workspace_root.parent != jobs_root.resolve():
        raise RetrainingWorkspaceError("補訓資料夾不可離開訓練工作目錄。")
    try:
        workspace_root.mkdir()
    except FileExistsError as exc:
        raise RetrainingWorkspaceError(
            f"補訓資料夾已存在：{normalized_version}"
        ) from exc

    created_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "batch_version": normalized_version,
        "product": str(product).strip(),
        "area": str(area).strip(),
        "created_at": created_at.isoformat(),
        "updated_at": created_at.isoformat(),
        "state": "draft",
        "job_id": normalized_version,
        "handoff_path": "",
    }
    metadata_path = workspace_root / "workspace.json"
    try:
        _write_json_atomic(metadata_path, payload)
        claim_training_batch_version(
            root,
            batch_version=normalized_version,
            product=product,
            area=area,
            submission_hash=f"workspace:{normalized_version.casefold()}",
            owner_job_id=normalized_version,
            record_path=metadata_path,
        )
    except (OSError, ValueError, DuplicateTrainingBatchVersionError):
        metadata_path.unlink(missing_ok=True)
        try:
            workspace_root.rmdir()
        except OSError:
            pass
        raise
    return _workspace_from_payload(payload, workspace_root)


def list_retraining_workspaces(
    data_root: str | Path,
    *,
    product: str | None = None,
    area: str | None = None,
) -> list[RetrainingWorkspace]:
    """Return newest-first valid batch folders without mutating them."""
    jobs_root = (
        Path(data_root).expanduser().resolve()
        / ".operator_handoff"
        / "jobs"
    )
    if not jobs_root.is_dir():
        return []
    workspaces: list[RetrainingWorkspace] = []
    for metadata_path in jobs_root.glob("*/workspace.json"):
        try:
            workspace = load_retraining_workspace(metadata_path.parent)
        except (OSError, RetrainingWorkspaceError):
            continue
        if product and workspace.product != product:
            continue
        if area and workspace.area != area:
            continue
        workspaces.append(workspace)
    return sorted(
        workspaces,
        key=lambda item: (item.created_at.timestamp(), item.batch_version),
        reverse=True,
    )


def load_retraining_workspace(path: str | Path) -> RetrainingWorkspace:
    """Load and validate one existing retraining folder."""
    root = Path(path).expanduser().resolve()
    metadata_path = root / "workspace.json"
    payload = _read_json(metadata_path)
    if payload.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
        raise RetrainingWorkspaceError(f"補訓資料夾設定無效：{metadata_path}")
    workspace = _workspace_from_payload(payload, root)
    expected_version = validate_training_batch_version(
        workspace.batch_version,
        product=workspace.product,
        area=workspace.area,
    )
    if root.name.casefold() != expected_version.casefold():
        raise RetrainingWorkspaceError(
            "補訓資料夾名稱與批次版本不一致。"
        )
    return workspace


def mark_retraining_workspace_submitted(
    workspace_root: str | Path,
    *,
    job_id: str,
    handoff_path: str | Path,
) -> RetrainingWorkspace:
    """Atomically attach the resulting training job to its source folder."""
    workspace = load_retraining_workspace(workspace_root)
    normalized_job_id = str(job_id or "").strip()
    if normalized_job_id != workspace.batch_version:
        raise RetrainingWorkspaceError(
            "補訓任務編號必須與資料夾版本一致。"
        )
    normalized_handoff = Path(handoff_path).expanduser().resolve()
    if normalized_handoff.parent != workspace.root:
        raise RetrainingWorkspaceError(
            "補訓任務不在原始批次資料夾內。"
        )
    if not normalized_handoff.is_file():
        raise RetrainingWorkspaceError(
            f"找不到補訓任務設定：{normalized_handoff}"
        )
    payload = _read_json(workspace.metadata_path)
    payload.update(
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": "submitted",
            "job_id": normalized_job_id,
            "handoff_path": str(normalized_handoff),
        }
    )
    _write_json_atomic(workspace.metadata_path, payload)
    return _workspace_from_payload(payload, workspace.root)


def _workspace_from_payload(
    payload: dict[str, Any],
    root: Path,
) -> RetrainingWorkspace:
    product = str(payload.get("product") or "").strip()
    area = str(payload.get("area") or "").strip()
    batch_version = str(payload.get("batch_version") or "").strip()
    state = str(payload.get("state") or "").strip()
    if not product or not area or not batch_version or state not in WORKSPACE_STATES:
        raise RetrainingWorkspaceError("補訓資料夾缺少必要設定。")
    created_at = _parse_datetime(payload.get("created_at"))
    if created_at is None:
        raise RetrainingWorkspaceError("補訓資料夾建立時間無效。")
    handoff_text = str(payload.get("handoff_path") or "").strip()
    return RetrainingWorkspace(
        batch_version=batch_version,
        product=product,
        area=area,
        root=root.resolve(),
        created_at=created_at,
        state=state,
        job_id=str(payload.get("job_id") or ""),
        handoff_path=Path(handoff_text).resolve() if handoff_text else None,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_datetime(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
