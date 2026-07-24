"""Immutable color configuration revisions and atomic activation pointers."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.color_calibration_packages import ColorCalibrationPackage, _write_json_atomic
from tools.color_calibration_service import (
    COLOR_ALGORITHM_VERSION,
    COLOR_CONFIG_SCHEMA_VERSION,
    ColorCalibrationError,
    ColorCalibrationScope,
    canonical_sha256,
    sha256_file,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class ColorConfigurationRevision:
    revision_id: str
    package_id: str
    scope: ColorCalibrationScope
    created_at: datetime
    operator: str
    approval_reason: str
    parent_revision_id: str
    parent_config_sha256: str
    new_config_sha256: str
    proposal_sha256: str
    preview_sha256: str
    metrics: Mapping[str, Any]
    supersedes_revision_id: str
    root: Path

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"


class ColorConfigurationRevisionStore:
    """Commit revisions once; activate only through a short atomic pointer swap."""

    _locks_guard = threading.Lock()
    _scope_locks: dict[str, threading.Lock] = {}

    def __init__(
        self,
        *,
        root: str | Path,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        replace_file: Callable[[str | Path, str | Path], Any] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_generator = id_generator or (lambda: str(uuid4()))
        self.replace_file = replace_file or os.replace

    def commit(
        self,
        package: ColorCalibrationPackage,
        scope: ColorCalibrationScope,
        *,
        operator: str,
        reason: str,
        proposal_sha256: str,
        preview_sha256: str,
        proposed_config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        parent_revision_id: str = "",
        parent_config_sha256: str = "",
    ) -> ColorConfigurationRevision:
        if not operator.strip() or not reason.strip():
            raise ColorCalibrationError("COLOR_APPROVAL_REQUIRED", "Named operator and approval reason are required.")
        config_payload = dict(proposed_config)
        if int(config_payload.get("schema_version") or 0) != COLOR_CONFIG_SCHEMA_VERSION:
            raise ColorCalibrationError("COLOR_CONFIG_VERSION_INCOMPATIBLE", "Unsupported color config schema version.")
        if str(config_payload.get("scope", {}).get("scope_hash") or "") != scope.scope_hash:
            raise ColorCalibrationError("COLOR_CONFIG_SCOPE_MISMATCH", "Proposed config scope does not match approval.")
        config_sha = canonical_sha256(config_payload)
        existing = self._find_idempotent(package.package_id, scope, config_sha)
        if existing is not None:
            return existing
        revision_id = _safe_id(self.id_generator(), "revision")
        scope_root = self.root / scope.scope_hash
        destination = scope_root / revision_id
        scope_root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ColorCalibrationError("COLOR_REVISION_COLLISION", f"Revision already exists: {revision_id}")
        created_at = self.clock()
        if created_at.tzinfo is None:
            raise ValueError("Color revision clock must return timezone-aware time")
        staging = Path(tempfile.mkdtemp(prefix=f".{revision_id}.", dir=scope_root))
        revision = ColorConfigurationRevision(
            revision_id=revision_id, package_id=package.package_id, scope=scope,
            created_at=created_at, operator=operator.strip(), approval_reason=reason.strip(),
            parent_revision_id=parent_revision_id, parent_config_sha256=parent_config_sha256,
            new_config_sha256=config_sha, proposal_sha256=proposal_sha256,
            preview_sha256=preview_sha256, metrics=dict(metrics),
            supersedes_revision_id=parent_revision_id, root=destination,
        )
        try:
            _write_json_atomic(staging / "config.json", config_payload)
            metadata = {
                "schema_version": 1, "revision_id": revision_id, "package_id": package.package_id,
                "scope": {**asdict(scope), "scope_hash": scope.scope_hash},
                "created_at": created_at.isoformat(), "operator": revision.operator,
                "approval_reason": revision.approval_reason,
                "parent_revision_id": parent_revision_id or None,
                "parent_config_sha256": parent_config_sha256,
                "new_config_sha256": config_sha, "algorithm_version": COLOR_ALGORITHM_VERSION,
                "proposal_sha256": proposal_sha256, "preview_sha256": preview_sha256,
                "metrics": dict(metrics), "supersedes_revision_id": parent_revision_id or None,
            }
            _write_json_atomic(staging / "revision.json", metadata)
            _write_json_atomic(staging / "checksums.json", {
                "config.json": sha256_file(staging / "config.json"),
                "revision.json": sha256_file(staging / "revision.json"),
            })
            (staging / "revocations").mkdir()
            (staging / "activation_events").mkdir()
            os.replace(staging, destination)
            loaded = self.load(scope, revision_id)
            if loaded.new_config_sha256 != config_sha:
                raise ColorCalibrationError("COLOR_REVISION_VERIFY_FAILED", "Committed revision failed verification.")
            return loaded
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def activate(
        self,
        revision: ColorConfigurationRevision,
        *,
        operator: str,
        reason: str,
        expected_current_sha256: str,
        event_type: str = "COLOR_REVISION_ACTIVATED",
    ) -> Path:
        if not operator.strip() or not reason.strip():
            raise ColorCalibrationError("COLOR_ACTIVATION_REASON_REQUIRED", "Activation operator and reason are required.")
        if self.is_revoked(revision):
            raise ColorCalibrationError("COLOR_REVISION_REVOKED", "A revoked color revision cannot be activated.")
        self.load(revision.scope, revision.revision_id)
        lock = self._lock_for(revision.scope.scope_hash)
        with lock:
            active = self.read_active_pointer(revision.scope)
            current_sha = str(active.get("config_sha256") or "") if active else revision.parent_config_sha256
            if current_sha != expected_current_sha256:
                raise ColorCalibrationError("CURRENT_CONFIG_STALE", "Active color configuration changed after approval.", retryable=True)
            if active and str(active.get("revision_id")) == revision.revision_id:
                return self.active_pointer_path(revision.scope)
            pointer = {
                "schema_version": 1,
                "scope": {**asdict(revision.scope), "scope_hash": revision.scope.scope_hash},
                "revision_id": revision.revision_id,
                "config_sha256": revision.new_config_sha256,
                "activated_at": self.clock().isoformat(),
                "operator": operator.strip(),
                "previous_revision_id": str(active.get("revision_id") or "") if active else None,
                "activation_reason": reason.strip(),
            }
            pointer_path = self.active_pointer_path(revision.scope)
            self._atomic_pointer_write(pointer_path, pointer)
            event_id = _safe_id(self.id_generator(), "activation event")
            _write_json_atomic(revision.root / "activation_events" / f"{event_id}.json", {
                **pointer, "event_id": event_id, "event_type": event_type,
            })
            return pointer_path

    def rollback(
        self,
        scope: ColorCalibrationScope,
        target_revision_id: str,
        *,
        operator: str,
        reason: str,
    ) -> Path:
        if not reason.strip():
            raise ColorCalibrationError("COLOR_ROLLBACK_REASON_REQUIRED", "Rollback reason is required.")
        target = self.load(scope, target_revision_id)
        active = self.read_active_pointer(scope)
        if not active:
            raise ColorCalibrationError("COLOR_ACTIVE_REVISION_MISSING", "No active color revision exists.")
        return self.activate(
            target, operator=operator, reason=reason,
            expected_current_sha256=str(active["config_sha256"]),
            event_type="COLOR_REVISION_ROLLED_BACK",
        )

    def revoke(self, revision: ColorConfigurationRevision, *, operator: str, reason: str) -> Path:
        if not operator.strip() or not reason.strip():
            raise ColorCalibrationError("COLOR_REVOCATION_REASON_REQUIRED", "Revocation operator and reason are required.")
        active = self.read_active_pointer(revision.scope)
        if active and str(active.get("revision_id")) == revision.revision_id:
            raise ColorCalibrationError("COLOR_ACTIVE_REVISION_CANNOT_REVOKE", "Rollback to another revision before revoking the active revision.")
        revocations = revision.root / "revocations"
        event_id = _safe_id(self.id_generator(), "revocation")
        path = revocations / f"{event_id}.json"
        _write_json_atomic(path, {
            "schema_version": 1, "event_type": "COLOR_REVISION_REVOKED", "event_id": event_id,
            "revision_id": revision.revision_id, "scope_hash": revision.scope.scope_hash,
            "operator": operator.strip(), "reason": reason.strip(), "created_at": self.clock().isoformat(),
        })
        return path

    def load(self, scope: ColorCalibrationScope, revision_id: str) -> ColorConfigurationRevision:
        revision_root = self.root / scope.scope_hash / _safe_id(revision_id, "revision")
        try:
            checksums = _read_json(revision_root / "checksums.json")
            for relative, expected in checksums.items():
                path = revision_root / relative
                if path.is_symlink() or sha256_file(path) != expected:
                    raise ColorCalibrationError("COLOR_REVISION_SHA_MISMATCH", f"Revision artifact changed: {relative}")
            raw = _read_json(revision_root / "revision.json")
            config = _read_json(revision_root / "config.json")
            if canonical_sha256(config) != str(raw["new_config_sha256"]):
                raise ColorCalibrationError("COLOR_REVISION_SHA_MISMATCH", "Revision config checksum mismatch.")
            loaded_scope = ColorCalibrationScope(*(str(raw["scope"][key]) for key in ("product", "area", "model_type", "checker_type", "threshold_key")))
            if loaded_scope != scope:
                raise ColorCalibrationError("COLOR_CONFIG_SCOPE_MISMATCH", "Revision scope mismatch.")
            return ColorConfigurationRevision(
                revision_id=str(raw["revision_id"]), package_id=str(raw["package_id"]), scope=scope,
                created_at=datetime.fromisoformat(str(raw["created_at"])), operator=str(raw["operator"]),
                approval_reason=str(raw["approval_reason"]), parent_revision_id=str(raw.get("parent_revision_id") or ""),
                parent_config_sha256=str(raw["parent_config_sha256"]), new_config_sha256=str(raw["new_config_sha256"]),
                proposal_sha256=str(raw["proposal_sha256"]), preview_sha256=str(raw["preview_sha256"]),
                metrics=dict(raw.get("metrics") or {}), supersedes_revision_id=str(raw.get("supersedes_revision_id") or ""), root=revision_root,
            )
        except ColorCalibrationError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ColorCalibrationError("COLOR_REVISION_INVALID", f"Color revision is unreadable: {revision_root}") from exc

    def active_pointer_path(self, scope: ColorCalibrationScope) -> Path:
        return self.root / "active" / f"{scope.scope_hash}.json"

    def read_active_pointer(self, scope: ColorCalibrationScope) -> Mapping[str, Any] | None:
        path = self.active_pointer_path(scope)
        if not path.is_file():
            return None
        try:
            payload = _read_json(path)
            if str(payload.get("scope", {}).get("scope_hash") or "") != scope.scope_hash:
                raise ColorCalibrationError("COLOR_ACTIVE_POINTER_INVALID", "Active pointer scope mismatch.")
            return payload
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ColorCalibrationError("COLOR_ACTIVE_POINTER_INVALID", f"Active pointer is unreadable: {path}") from exc

    @staticmethod
    def is_revoked(revision: ColorConfigurationRevision) -> bool:
        return any((revision.root / "revocations").glob("*.json"))

    def _find_idempotent(self, package_id: str, scope: ColorCalibrationScope, config_sha: str) -> ColorConfigurationRevision | None:
        scope_root = self.root / scope.scope_hash
        if not scope_root.is_dir():
            return None
        for path in sorted(scope_root.glob("*/revision.json")):
            raw = _read_json(path)
            if str(raw.get("package_id")) == package_id and str(raw.get("new_config_sha256")) == config_sha:
                return self.load(scope, str(raw["revision_id"]))
        return None

    def _atomic_pointer_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.replace_file(temporary, path)
        except OSError as exc:
            raise ColorCalibrationError("COLOR_ACTIVATION_FAILED", f"Could not activate color revision: {exc}", retryable=True) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _lock_for(cls, scope_hash: str) -> threading.Lock:
        with cls._locks_guard:
            return cls._scope_locks.setdefault(scope_hash, threading.Lock())


def _safe_id(value: str, kind: str) -> str:
    result = str(value).strip()
    if not _SAFE_ID.fullmatch(result):
        raise ColorCalibrationError("COLOR_REVISION_PATH_ESCAPE", f"Unsafe {kind} identifier.")
    return result


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
