"""Immutable color configuration revisions and atomic activation pointers."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
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
from tools.color_revision_publication_lock import (
    ColorRevisionPublicationLockTimeoutError,
    color_revision_publication_lock,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SCOPE_HASH = re.compile(r"^[0-9a-f]{24}$")
_DISPLAY_VERSION = re.compile(r"^color-v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
_ACTIVATION_EVENT_TYPES = frozenset(
    {
        "COLOR_REVISION_ACTIVATED",
        "COLOR_REVISION_ROLLED_BACK",
        "COLOR_BASELINE_CAPTURED",
        "COLOR_OK_ONLY_CANDIDATE_ACTIVATED",
    }
)
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColorConfigurationRevision:
    revision_id: str
    display_version: str
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


@dataclass(frozen=True)
class ColorConfigurationRevisionSummary:
    scope_hash: str
    scope_label: str
    revision_id: str
    display_version: str
    parent_display_version: str
    active: bool
    created_at: datetime
    operator: str
    evidence_level: str
    changes: tuple[str, ...]


class ColorConfigurationRevisionStore:
    """Commit revisions once; activate only through a short atomic pointer swap."""

    _locks_guard = threading.Lock()
    _scope_locks: dict[str, threading.Lock] = {}
    _commit_locks: dict[str, threading.Lock] = {}

    def __init__(
        self,
        *,
        root: str | Path,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        replace_file: Callable[[str | Path, str | Path], Any] | None = None,
        publication_lock_timeout: float = 30.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_generator = id_generator or (lambda: str(uuid4()))
        self.replace_file = replace_file or os.replace
        self.publication_lock_timeout = max(float(publication_lock_timeout), 0.0)

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
        display_version: str = "",
    ) -> ColorConfigurationRevision:
        """Serialize short revision commits so one scope cannot reuse a version."""
        with self._commit_lock_for(scope.scope_hash):
            return self._commit(
                package.package_id,
                scope,
                operator=operator,
                reason=reason,
                proposal_sha256=proposal_sha256,
                preview_sha256=preview_sha256,
                proposed_config=proposed_config,
                metrics=metrics,
                parent_revision_id=parent_revision_id,
                parent_config_sha256=parent_config_sha256,
                display_version=display_version,
            )

    def commit_configuration(
        self,
        source_id: str,
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
        display_version: str = "",
    ) -> ColorConfigurationRevision:
        """Commit a version from a non-package workflow such as OK-only review."""
        package_id = _safe_id(source_id, "color revision source")
        with self._commit_lock_for(scope.scope_hash):
            return self._commit(
                package_id,
                scope,
                operator=operator,
                reason=reason,
                proposal_sha256=proposal_sha256,
                preview_sha256=preview_sha256,
                proposed_config=proposed_config,
                metrics=metrics,
                parent_revision_id=parent_revision_id,
                parent_config_sha256=parent_config_sha256,
                display_version=display_version,
            )

    def _commit(
        self,
        package_id: str,
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
        display_version: str = "",
    ) -> ColorConfigurationRevision:
        if not operator.strip() or not reason.strip():
            raise ColorCalibrationError("COLOR_APPROVAL_REQUIRED", "Named operator and approval reason are required.")
        config_payload = dict(proposed_config)
        if int(config_payload.get("schema_version") or 0) != COLOR_CONFIG_SCHEMA_VERSION:
            raise ColorCalibrationError("COLOR_CONFIG_VERSION_INCOMPATIBLE", "Unsupported color config schema version.")
        if str(config_payload.get("scope", {}).get("scope_hash") or "") != scope.scope_hash:
            raise ColorCalibrationError("COLOR_CONFIG_SCOPE_MISMATCH", "Proposed config scope does not match approval.")
        config_sha = canonical_sha256(config_payload)
        existing = self._find_idempotent(package_id, scope, config_sha)
        if existing is not None:
            return existing
        display_version = (
            _validate_display_version(display_version)
            if display_version
            else self.next_display_version(scope)
        )
        if any(
            item.display_version == display_version
            for item in self.list_revisions(scope)
        ):
            raise ColorCalibrationError(
                "COLOR_VERSION_COLLISION",
                f"Color version already exists for this scope: {display_version}",
                retryable=True,
            )
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
            revision_id=revision_id, display_version=display_version,
            package_id=package_id, scope=scope,
            created_at=created_at, operator=operator.strip(), approval_reason=reason.strip(),
            parent_revision_id=parent_revision_id, parent_config_sha256=parent_config_sha256,
            new_config_sha256=config_sha, proposal_sha256=proposal_sha256,
            preview_sha256=preview_sha256, metrics=dict(metrics),
            supersedes_revision_id=parent_revision_id, root=destination,
        )
        try:
            _write_json_atomic(staging / "config.json", config_payload)
            metadata = {
                "schema_version": 2, "revision_id": revision_id,
                "display_version": display_version, "package_id": package_id,
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
        if event_type not in _ACTIVATION_EVENT_TYPES:
            raise ColorCalibrationError(
                "COLOR_ACTIVATION_EVENT_TYPE_INVALID",
                "Color activation event type is unsupported.",
            )
        lock = self._lock_for(revision.scope.scope_hash)
        try:
            with color_revision_publication_lock(
                self.root,
                timeout=self.publication_lock_timeout,
            ):
                with lock:
                    revision = self.load(
                        revision.scope,
                        revision.revision_id,
                    )
                    if self.is_revoked(revision):
                        raise ColorCalibrationError(
                            "COLOR_REVISION_REVOKED",
                            "A revoked color revision cannot be activated.",
                        )
                    active = self.read_active_pointer(revision.scope)
                    current_sha = (
                        str(active.get("config_sha256") or "")
                        if active
                        else revision.parent_config_sha256
                    )
                    if current_sha != expected_current_sha256:
                        raise ColorCalibrationError(
                            "CURRENT_CONFIG_STALE",
                            "Active color configuration changed after approval.",
                            retryable=True,
                        )
                    if (
                        active
                        and str(active.get("revision_id"))
                        == revision.revision_id
                    ):
                        return self.active_pointer_path(revision.scope)
                    event_id = _safe_id(
                        self.id_generator(),
                        "activation event",
                    )
                    pointer = {
                        "schema_version": 2,
                        "scope": {
                            **asdict(revision.scope),
                            "scope_hash": revision.scope.scope_hash,
                        },
                        "revision_id": revision.revision_id,
                        "display_version": revision.display_version,
                        "config_sha256": revision.new_config_sha256,
                        "activated_at": self.clock().isoformat(),
                        "operator": operator.strip(),
                        "previous_revision_id": (
                            str(active.get("revision_id") or "")
                            if active
                            else None
                        ),
                        "activation_reason": reason.strip(),
                        "event_id": event_id,
                        "event_type": event_type,
                    }
                    pointer_path = self.active_pointer_path(revision.scope)
                    event_path = (
                        revision.root
                        / "activation_events"
                        / f"{event_id}.json"
                    )
                    if (
                        revision.root.is_symlink()
                        or event_path.parent.is_symlink()
                        or not event_path.parent.is_dir()
                        or not event_path.parent.resolve().is_relative_to(
                            self.root
                        )
                    ):
                        raise ColorCalibrationError(
                            "COLOR_REVISION_PATH_ESCAPE",
                            "Color activation event directory is unsafe.",
                        )
                    if event_path.exists() or event_path.is_symlink():
                        raise ColorCalibrationError(
                            "COLOR_ACTIVATION_EVENT_COLLISION",
                            "Color activation event identity already exists.",
                            retryable=True,
                        )
                    try:
                        _write_json_atomic(
                            event_path,
                            pointer,
                        )
                    except (OSError, TypeError, ValueError) as event_error:
                        if _activation_event_matches(event_path, pointer):
                            _LOGGER.warning(
                                "Color activation event was committed but its "
                                "durability sync reported an error: path=%s "
                                "error=%s",
                                event_path,
                                event_error,
                            )
                        else:
                            raise ColorCalibrationError(
                                "COLOR_ACTIVATION_EVENT_FAILED",
                                "Color activation event could not be recorded; "
                                "the active pointer was not published.",
                                retryable=True,
                            ) from event_error
                    try:
                        self._atomic_pointer_write(pointer_path, pointer)
                    except ColorCalibrationError as pointer_error:
                        evidence_note = (
                            "Completed activation event retained without an active "
                            f"pointer commit: {event_path}"
                        )
                        pointer_error.add_note(evidence_note)
                        _LOGGER.error("%s", evidence_note)
                        raise
                    return pointer_path
        except ColorRevisionPublicationLockTimeoutError as exc:
            raise ColorCalibrationError(
                "COLOR_REVISION_PUBLICATION_BUSY",
                "A model deployment is publishing the active color contract.",
                retryable=True,
            ) from exc

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
        target = self.resolve_revision(scope, target_revision_id)
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
        lock = self._lock_for(revision.scope.scope_hash)
        try:
            with color_revision_publication_lock(
                self.root,
                timeout=self.publication_lock_timeout,
            ):
                with lock:
                    revision = self.load(
                        revision.scope,
                        revision.revision_id,
                    )
                    active = self.read_active_pointer(revision.scope)
                    if (
                        active
                        and str(active.get("revision_id"))
                        == revision.revision_id
                    ):
                        raise ColorCalibrationError(
                            "COLOR_ACTIVE_REVISION_CANNOT_REVOKE",
                            "Rollback to another revision before revoking the "
                            "active revision.",
                        )
                    revocations = revision.root / "revocations"
                    event_id = _safe_id(self.id_generator(), "revocation")
                    path = revocations / f"{event_id}.json"
                    if (
                        revision.root.is_symlink()
                        or revocations.is_symlink()
                        or not revocations.is_dir()
                        or not revocations.resolve().is_relative_to(self.root)
                    ):
                        raise ColorCalibrationError(
                            "COLOR_REVISION_PATH_ESCAPE",
                            "Color revocation event directory is unsafe.",
                        )
                    if path.exists() or path.is_symlink():
                        raise ColorCalibrationError(
                            "COLOR_REVOCATION_EVENT_COLLISION",
                            "Color revocation event identity already exists.",
                            retryable=True,
                        )
                    payload = {
                        "schema_version": 1,
                        "event_type": "COLOR_REVISION_REVOKED",
                        "event_id": event_id,
                        "revision_id": revision.revision_id,
                        "scope_hash": revision.scope.scope_hash,
                        "operator": operator.strip(),
                        "reason": reason.strip(),
                        "created_at": self.clock().isoformat(),
                    }
                    try:
                        _write_json_atomic(path, payload)
                    except (OSError, TypeError, ValueError) as event_error:
                        if _activation_event_matches(path, payload):
                            _LOGGER.warning(
                                "Color revocation event was committed but its "
                                "durability sync reported an error: path=%s "
                                "error=%s",
                                path,
                                event_error,
                            )
                        else:
                            raise ColorCalibrationError(
                                "COLOR_REVOCATION_EVENT_FAILED",
                                "Color revocation event could not be recorded.",
                                retryable=True,
                            ) from event_error
                    return path
        except ColorRevisionPublicationLockTimeoutError as exc:
            raise ColorCalibrationError(
                "COLOR_REVISION_PUBLICATION_BUSY",
                "A model deployment is publishing the active color contract.",
                retryable=True,
            ) from exc

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
                revision_id=str(raw["revision_id"]),
                display_version=str(raw.get("display_version") or ""),
                package_id=str(raw["package_id"]), scope=scope,
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
        if path.parent.is_symlink() or path.is_symlink():
            raise ColorCalibrationError(
                "COLOR_ACTIVE_POINTER_INVALID",
                f"Active color pointer cannot be a symbolic link: {path}",
            )
        if path.parent.exists() and not path.parent.resolve().is_relative_to(
            self.root
        ):
            raise ColorCalibrationError(
                "COLOR_ACTIVE_POINTER_INVALID",
                f"Active color pointer escapes its store: {path}",
            )
        if not path.is_file():
            return None
        try:
            payload = _read_json(path)
            if not isinstance(payload, Mapping):
                raise ColorCalibrationError(
                    "COLOR_ACTIVE_POINTER_INVALID",
                    "Active color pointer must contain an object.",
                )
            raw_scope = payload.get("scope")
            if not isinstance(raw_scope, Mapping) or (
                str(raw_scope.get("scope_hash") or "") != scope.scope_hash
            ):
                raise ColorCalibrationError("COLOR_ACTIVE_POINTER_INVALID", "Active pointer scope mismatch.")
            schema_version = payload.get("schema_version")
            if type(schema_version) is not int or schema_version not in {1, 2}:
                raise ColorCalibrationError(
                    "COLOR_ACTIVE_POINTER_INVALID",
                    "Active color pointer schema version is unsupported.",
                )
            if schema_version == 2:
                if not _has_valid_activation_evidence(
                    self.root,
                    scope,
                    payload,
                ):
                    raise ColorCalibrationError(
                        "COLOR_ACTIVATION_EVIDENCE_MISSING",
                        "Schema 2 active color pointer has no exact activation "
                        "event.",
                    )
            return payload
        except ColorCalibrationError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ColorCalibrationError("COLOR_ACTIVE_POINTER_INVALID", f"Active pointer is unreadable: {path}") from exc

    def scope_for_hash(self, scope_hash: str) -> ColorCalibrationScope:
        """Resolve one persisted scope without consulting its active pointer."""
        safe_scope_hash = _safe_id(scope_hash, "scope hash")
        scope_root = self.root / safe_scope_hash
        revision_paths = sorted(scope_root.glob("*/revision.json"))
        if not revision_paths:
            raise ColorCalibrationError(
                "COLOR_SCOPE_NOT_FOUND",
                f"No color revisions exist for scope: {safe_scope_hash}",
            )
        try:
            raw = _read_json(revision_paths[0])
            raw_scope = raw["scope"]
            scope = ColorCalibrationScope(
                *(
                    str(raw_scope[key])
                    for key in (
                        "product",
                        "area",
                        "model_type",
                        "checker_type",
                        "threshold_key",
                    )
                )
            )
            if scope.scope_hash != safe_scope_hash:
                raise ColorCalibrationError(
                    "COLOR_CONFIG_SCOPE_MISMATCH",
                    "Persisted color scope hash does not match its directory.",
                )
            self.load(scope, str(raw["revision_id"]))
            return scope
        except ColorCalibrationError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ColorCalibrationError(
                "COLOR_REVISION_INVALID",
                f"Color scope metadata is unreadable: {scope_root}",
            ) from exc

    def iter_scopes(self) -> tuple[ColorCalibrationScope, ...]:
        """Return verified revision scopes, excluding store infrastructure."""
        if not self.root.is_dir():
            return ()
        scope_hashes = sorted(
            path.name
            for path in self.root.iterdir()
            if (
                not path.is_symlink()
                and path.is_dir()
                and _SCOPE_HASH.fullmatch(path.name)
            )
        )
        return tuple(self.scope_for_hash(scope_hash) for scope_hash in scope_hashes)

    def list_revisions(
        self, scope: ColorCalibrationScope
    ) -> tuple[ColorConfigurationRevision, ...]:
        """Return verified revisions with stable labels for legacy metadata."""
        scope_root = self.root / scope.scope_hash
        if not scope_root.is_dir():
            return ()
        revisions = sorted(
            (
                self.load(scope, path.parent.name)
                for path in scope_root.glob("*/revision.json")
            ),
            key=lambda item: (item.created_at, item.revision_id),
        )
        assigned = {
            item.display_version for item in revisions if item.display_version
        }
        next_patch = 1
        compatible: list[ColorConfigurationRevision] = []
        for revision in revisions:
            if revision.display_version:
                compatible.append(revision)
                continue
            while f"color-v1.0.{next_patch}" in assigned:
                next_patch += 1
            legacy_version = f"color-v1.0.{next_patch}"
            assigned.add(legacy_version)
            next_patch += 1
            compatible.append(replace(revision, display_version=legacy_version))
        return tuple(compatible)

    def next_display_version(self, scope: ColorCalibrationScope) -> str:
        patches = [
            int(match.group("patch"))
            for revision in self.list_revisions(scope)
            if (match := _DISPLAY_VERSION.fullmatch(revision.display_version))
            and match.group("major") == "1"
            and match.group("minor") == "0"
        ]
        return f"color-v1.0.{max(patches, default=0) + 1}"

    def resolve_revision(
        self, scope: ColorCalibrationScope, revision_reference: str
    ) -> ColorConfigurationRevision:
        reference = str(revision_reference).strip()
        if not reference:
            raise ColorCalibrationError(
                "COLOR_REVISION_REFERENCE_REQUIRED",
                "A color version or revision ID is required.",
            )
        for revision in self.list_revisions(scope):
            if reference in {revision.revision_id, revision.display_version}:
                return revision
        raise ColorCalibrationError(
            "COLOR_REVISION_NOT_FOUND",
            f"Unknown color version or revision ID: {reference}",
        )

    def revision_history(
        self, scope: ColorCalibrationScope
    ) -> tuple[ColorConfigurationRevisionSummary, ...]:
        revisions = self.list_revisions(scope)
        active = self.read_active_pointer(scope)
        active_id = str(active.get("revision_id") or "") if active else ""
        by_id = {item.revision_id: item for item in revisions}
        summaries: list[ColorConfigurationRevisionSummary] = []
        for revision in reversed(revisions):
            parent = by_id.get(revision.parent_revision_id)
            config = _read_json(revision.config_path)
            parent_config = _read_json(parent.config_path) if parent else None
            evidence = revision.metrics.get("evidence")
            evidence_level = (
                str(evidence.get("level") or "UNKNOWN")
                if isinstance(evidence, Mapping)
                else "UNKNOWN"
            )
            summaries.append(
                ColorConfigurationRevisionSummary(
                    scope_hash=scope.scope_hash,
                    scope_label=scope.key,
                    revision_id=revision.revision_id,
                    display_version=revision.display_version,
                    parent_display_version=parent.display_version if parent else "",
                    active=revision.revision_id == active_id,
                    created_at=revision.created_at,
                    operator=revision.operator,
                    evidence_level=evidence_level,
                    changes=_configuration_changes(parent_config, config),
                )
            )
        return tuple(summaries)

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
        if path.parent.is_symlink() or path.is_symlink():
            raise ColorCalibrationError(
                "COLOR_REVISION_PATH_ESCAPE",
                "Color activation pointer destination is unsafe.",
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        if (
            path.parent.is_symlink()
            or not path.parent.resolve().is_relative_to(self.root)
        ):
            raise ColorCalibrationError(
                "COLOR_REVISION_PATH_ESCAPE",
                "Color activation pointer destination is unsafe.",
            )
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.replace_file(temporary, path)
            _fsync_directory(path.parent)
        except OSError as exc:
            if _activation_event_matches(path, payload):
                _LOGGER.warning(
                    "Color activation pointer was committed but its durability "
                    "sync reported an error: path=%s error=%s",
                    path,
                    exc,
                )
                return
            raise ColorCalibrationError("COLOR_ACTIVATION_FAILED", f"Could not activate color revision: {exc}", retryable=True) from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @classmethod
    def _lock_for(cls, scope_hash: str) -> threading.Lock:
        with cls._locks_guard:
            return cls._scope_locks.setdefault(scope_hash, threading.Lock())

    @classmethod
    def _commit_lock_for(cls, scope_hash: str) -> threading.Lock:
        with cls._locks_guard:
            return cls._commit_locks.setdefault(scope_hash, threading.Lock())


def _safe_id(value: str, kind: str) -> str:
    result = str(value).strip()
    if not _SAFE_ID.fullmatch(result):
        raise ColorCalibrationError("COLOR_REVISION_PATH_ESCAPE", f"Unsafe {kind} identifier.")
    return result


def _validate_display_version(value: str) -> str:
    result = str(value).strip()
    if not _DISPLAY_VERSION.fullmatch(result):
        raise ColorCalibrationError(
            "COLOR_VERSION_INVALID",
            "Color version must use the format color-v<major>.<minor>.<patch>.",
        )
    return result


def _configuration_changes(
    parent: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> tuple[str, ...]:
    if parent is None:
        return ("baseline snapshot",)
    changes: list[str] = []
    for key in ("public_threshold", "config_value"):
        before = parent.get(key)
        after = current.get(key)
        if before != after:
            changes.append(f"{key}: {_display_value(before)} -> {_display_value(after)}")
    return tuple(changes or ("no threshold change",))


def _display_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _activation_event_matches(
    path: Path,
    pointer: Mapping[str, Any],
) -> bool:
    try:
        return _read_json(path) == dict(pointer)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _has_valid_activation_evidence(
    root: Path,
    scope: ColorCalibrationScope,
    pointer: Mapping[str, Any],
) -> bool:
    """Validate new exact references or one uniquely matching legacy event."""
    try:
        revision_id = _safe_id(
            str(pointer.get("revision_id") or ""),
            "revision",
        )
        events_root = root / scope.scope_hash / revision_id / "activation_events"
        event_id_value = pointer.get("event_id")
        event_type_value = pointer.get("event_type")
        if event_id_value is not None or event_type_value is not None:
            event_id = _safe_id(str(event_id_value or ""), "activation event")
            event_type = str(event_type_value or "")
            if event_type not in _ACTIVATION_EVENT_TYPES:
                return False
            event_path = events_root / f"{event_id}.json"
            return _safe_event_payload(event_path, root=root) == dict(pointer)
        return _legacy_activation_event_count(
            events_root,
            root=root,
            pointer=pointer,
        ) == 1
    except (OSError, TypeError, ValueError, ColorCalibrationError):
        return False


def _legacy_activation_event_count(
    events_root: Path,
    *,
    root: Path,
    pointer: Mapping[str, Any],
) -> int:
    if events_root.is_symlink() or not events_root.is_dir():
        return 0
    matches = 0
    for event_path in sorted(events_root.glob("*.json")):
        event = _safe_event_payload(event_path, root=root)
        if event is None:
            return 0
        event_id = str(event.get("event_id") or "")
        event_type = str(event.get("event_type") or "")
        if (
            _safe_id(event_id, "activation event") != event_path.stem
            or event_path.name != f"{event_id}.json"
            or event_type not in _ACTIVATION_EVENT_TYPES
            or set(event) != set(pointer) | {"event_id", "event_type"}
        ):
            return 0
        legacy_pointer = dict(event)
        legacy_pointer.pop("event_id")
        legacy_pointer.pop("event_type")
        if legacy_pointer == dict(pointer):
            matches += 1
    return matches


def _safe_event_payload(
    path: Path,
    *,
    root: Path,
) -> Mapping[str, Any] | None:
    if (
        path.parent.is_symlink()
        or path.parent.parent.is_symlink()
        or path.parent.parent.parent.is_symlink()
        or path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(root)
    ):
        return None
    payload = _read_json(path)
    return payload if isinstance(payload, Mapping) else None


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
