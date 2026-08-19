"""Checksum-verified storage and atomic activation for inspection releases."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from core.services.inspection_release_models import (
    ActivationMode,
    InspectionRelease,
    InspectionReleaseConflictError,
    InspectionReleaseError,
    InspectionReleasePolicyError,
    InspectionScope,
    ReleaseStatus,
    ReleaseValidationAttestation,
    template_for_inference_type,
)
from core.station_data import StationDataPaths, load_station_data_paths


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(_canonical_json(payload))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_file_matches(path: Path, payload: Mapping[str, Any]) -> bool:
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")) == dict(payload))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


class InspectionReleasePolicy:
    """Central risk classification shared by GUI and store.

    Validation findings are warnings. Only an explicitly blocked release or
    invalid immutable evidence removes operator choice.
    """

    def allowed_modes(self, release: InspectionRelease) -> tuple[ActivationMode, ...]:
        if release.status is ReleaseStatus.BLOCKED:
            return ()
        operator_modes = (
            ActivationMode.LIMITED_TRIAL,
            ActivationMode.RISK_ACCEPTED,
        )
        return (
            (ActivationMode.FULL,) + operator_modes
            if not self.validation_warnings(release)
            else operator_modes
        )

    def validation_warnings(self, release: InspectionRelease) -> tuple[str, ...]:
        warnings: list[str] = []
        evidence = release.validation
        errors = evidence.metric("errors")
        false_negatives = evidence.metric("fn")
        if release.status is ReleaseStatus.DRAFT:
            warnings.append("Release is a draft and has not completed validation.")
        if errors != 0:
            warnings.append(f"Validation inference errors: {errors!r}.")
        if false_negatives != 0:
            warnings.append(f"Validation false negatives: {false_negatives!r}.")
        if (
            release.component_for_role("color_check") is not None
            and not evidence.color_escape_known
        ):
            warnings.append("Color NG truth is missing; color escape rate is unknown.")
        return tuple(warnings)

    def require_allowed(self, release: InspectionRelease, mode: ActivationMode) -> None:
        allowed = self.allowed_modes(release)
        if mode in allowed:
            return
        if release.status is ReleaseStatus.BLOCKED:
            reason = "The release is explicitly blocked."
        elif mode is ActivationMode.FULL and self.validation_warnings(release):
            reason = (
                "Standard full activation requires complete validation. "
                "Choose limited trial or risk-accepted activation."
            )
        else:
            reason = "The release is not eligible for activation."
        raise InspectionReleasePolicyError(reason)


class InspectionReleaseStore:
    """Append-only releases with one compare-and-swap pointer per scope."""

    def __init__(
        self,
        root: str | Path,
        *,
        policy: InspectionReleasePolicy | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.policy = policy or InspectionReleasePolicy()
        self._locks_guard = threading.Lock()
        self._scope_locks: dict[str, threading.Lock] = {}
        discovered_paths = load_station_data_paths(self.root)
        self._station_paths: StationDataPaths | None = (
            discovered_paths
            if discovered_paths.inspection_releases == self.root
            else None
        )

    def _lock_for(self, scope_hash: str) -> threading.Lock:
        with self._locks_guard:
            return self._scope_locks.setdefault(scope_hash, threading.Lock())

    @contextmanager
    def _cross_process_lock(self, scope_hash: str) -> Iterator[None]:
        lock_path = self.root / "locks" / f"{scope_hash}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                msvcrt: Any = importlib.import_module("msvcrt")

                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError as exc:
                    raise InspectionReleaseConflictError(
                        "Another process is changing this inspection release."
                    ) from exc
            else:  # pragma: no cover - production station is Windows
                fcntl_api: Any = importlib.import_module("fcntl")
                try:
                    fcntl_api.flock(
                        handle.fileno(),
                        fcntl_api.LOCK_EX | fcntl_api.LOCK_NB,
                    )
                except OSError as exc:
                    raise InspectionReleaseConflictError(
                        "Another process is changing this inspection release."
                    ) from exc
            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt_unlock: Any = importlib.import_module("msvcrt")

                    msvcrt_unlock.locking(
                        handle.fileno(),
                        msvcrt_unlock.LK_UNLCK,
                        1,
                    )
                else:  # pragma: no cover - production station is Windows
                    fcntl_unlock_api: Any = importlib.import_module("fcntl")
                    fcntl_unlock_api.flock(
                        handle.fileno(),
                        fcntl_unlock_api.LOCK_UN,
                    )
            except OSError:
                pass
            handle.close()

    def release_dir(self, release: InspectionRelease) -> Path:
        return (
            self.root
            / "releases"
            / str(release.scope.scope_hash)
            / str(release.release_id)
        )

    def commit(self, release: InspectionRelease) -> InspectionRelease:
        """Commit one immutable release after validating all referenced files."""
        self._verify_external_evidence(release)
        destination = self.release_dir(release)
        if destination.exists():
            loaded = self._relocate_external_paths(
                self._load_base(release.scope, release.release_id)
            )
            if loaded.to_dict() != release.to_dict():
                raise InspectionReleaseConflictError(
                    "Release ID already exists with different content."
                )
            return self.load(release.scope, release.release_id)
        staging = destination.with_name(f".{release.release_id}.{uuid4().hex}.tmp")
        staging.parent.mkdir(parents=True, exist_ok=True)
        try:
            staging.mkdir()
            release_path = staging / "release.json"
            release_path.write_bytes(_canonical_json(release.to_dict()))
            checksums = {"release.json": sha256_file(release_path)}
            (staging / "checksums.json").write_bytes(_canonical_json(checksums))
            os.replace(staging, destination)
            return self.load(release.scope, release.release_id)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def commit_validation(
        self,
        validated_release: InspectionRelease,
        *,
        validator: str,
        reason: str,
    ) -> InspectionRelease:
        """Append validation evidence and project it onto an immutable draft."""
        if not validator.strip() or not reason.strip():
            raise InspectionReleasePolicyError(
                "Validation operator and reason are required."
            )
        if validated_release.status not in {
            ReleaseStatus.TESTED,
            ReleaseStatus.BLOCKED,
        }:
            raise InspectionReleasePolicyError(
                "Only a TESTED or BLOCKED validation result can be attested."
            )
        self._verify_external_evidence(validated_release)
        scope_hash = validated_release.scope.scope_hash
        lock = self._lock_for(scope_hash)
        with lock, self._cross_process_lock(scope_hash):
            base = self._load_base(
                validated_release.scope,
                validated_release.release_id,
            )
            if base.status is not ReleaseStatus.DRAFT:
                raise InspectionReleasePolicyError(
                    "Only an immutable DRAFT release can receive validation evidence."
                )
            expected_projection = replace(
                self._relocate_external_paths(base),
                status=validated_release.status,
                validation=validated_release.validation,
            )
            if expected_projection != validated_release:
                raise InspectionReleaseConflictError(
                    "Validation result does not match the selected draft components."
                )
            base_sha256 = _canonical_sha256(base.to_dict())
            identity_payload = {
                "release_id": base.release_id,
                "report_sha256": validated_release.validation.report_sha256,
                "combination_id": validated_release.validation.combination_id,
            }
            attestation_id = _canonical_sha256(identity_payload)
            destination = self.validation_attestation_path(
                base.scope,
                base.release_id,
                attestation_id,
            )
            if destination.exists():
                existing = self._read_validation_attestation(destination, base)
                if (
                    existing.validation.report_sha256
                    != validated_release.validation.report_sha256
                    or existing.validation.combination_id
                    != validated_release.validation.combination_id
                ):
                    raise InspectionReleaseConflictError(
                        "Validation attestation ID already exists with different evidence."
                    )
                return self.load(base.scope, base.release_id)
            active = self.active_pointer(base.scope)
            if active and str(active.get("release_id") or "") == base.release_id:
                raise InspectionReleasePolicyError(
                    "An active inspection release cannot receive new validation "
                    "evidence. Activate another release or roll back before "
                    "recording the new validation result."
                )
            attestation = ReleaseValidationAttestation(
                attestation_id=attestation_id,
                release_id=base.release_id,
                scope_hash=base.scope.scope_hash,
                base_release_sha256=base_sha256,
                status=validated_release.status,
                validated_at=datetime.now(timezone.utc).isoformat(),
                validator=validator.strip(),
                reason=reason.strip(),
                validation=validated_release.validation,
            )
            body = attestation.to_dict()
            _write_json_atomic(
                destination,
                {
                    **body,
                    "payload_sha256": _canonical_sha256(body),
                },
            )
            return self.load(base.scope, base.release_id)

    def validation_attestation_path(
        self,
        scope: InspectionScope,
        release_id: str,
        attestation_id: str,
    ) -> Path:
        return (
            self.root
            / "validations"
            / str(scope.scope_hash)
            / str(release_id)
            / f"{str(attestation_id)}.json"
        )

    def load(self, scope: InspectionScope, release_id: str) -> InspectionRelease:
        base = self._load_base(scope, release_id)
        attestation = self._latest_validation_attestation(base)
        if attestation is None:
            return self._relocate_external_paths(base)
        projected = replace(
            base,
            status=attestation.status,
            validation=attestation.validation,
        )
        self._verify_external_evidence(projected)
        return self._relocate_external_paths(projected)

    def _load_base(self, scope: InspectionScope, release_id: str) -> InspectionRelease:
        release_dir = self.root / "releases" / scope.scope_hash / release_id
        if release_dir.is_symlink() or not release_dir.is_dir():
            raise InspectionReleaseError(
                f"Inspection release is unavailable: {release_id}"
            )
        release_path = release_dir / "release.json"
        checksums_path = release_dir / "checksums.json"
        try:
            checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
            expected = str(checksums["release.json"]).lower()
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("Release checksums are invalid.") from exc
        if release_path.is_symlink() or sha256_file(release_path) != expected:
            raise InspectionReleaseError("Release metadata checksum mismatch.")
        try:
            payload = json.loads(release_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("Release metadata is unreadable.") from exc
        release = InspectionRelease.from_dict(payload)
        if release.scope != scope or release.release_id != release_id:
            raise InspectionReleaseError("Release identity does not match its path.")
        self._verify_external_evidence(release)
        return release

    def _latest_validation_attestation(
        self,
        base: InspectionRelease,
    ) -> ReleaseValidationAttestation | None:
        validation_root = (
            self.root / "validations" / base.scope.scope_hash / base.release_id
        )
        if not validation_root.is_dir() or validation_root.is_symlink():
            return None
        attestations = tuple(
            self._read_validation_attestation(path, base)
            for path in sorted(validation_root.glob("*.json"))
            if path.is_file() and not path.is_symlink()
        )
        if not attestations:
            return None
        return max(
            attestations,
            key=lambda item: (item.validated_at, item.attestation_id),
        )

    @staticmethod
    def _read_validation_attestation(
        path: Path,
        base: InspectionRelease,
    ) -> ReleaseValidationAttestation:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError(
                "Validation attestation is unreadable."
            ) from exc
        if not isinstance(payload, Mapping):
            raise InspectionReleaseError("Validation attestation is invalid.")
        expected_payload_sha256 = str(payload.get("payload_sha256") or "").lower()
        body = dict(payload)
        body.pop("payload_sha256", None)
        if expected_payload_sha256 != _canonical_sha256(body):
            raise InspectionReleaseError("Validation attestation checksum mismatch.")
        attestation = ReleaseValidationAttestation.from_dict(body)
        if (
            attestation.release_id != base.release_id
            or attestation.scope_hash != base.scope.scope_hash
            or attestation.base_release_sha256 != _canonical_sha256(base.to_dict())
            or path.stem != attestation.attestation_id
        ):
            raise InspectionReleaseError(
                "Validation attestation does not match its immutable draft."
            )
        return attestation

    def list_releases(
        self, *, product: str | None = None, area: str | None = None
    ) -> tuple[InspectionRelease, ...]:
        releases_root = self.root / "releases"
        if not releases_root.is_dir():
            return ()
        releases: list[InspectionRelease] = []
        for scope_dir in releases_root.iterdir():
            if scope_dir.is_symlink() or not scope_dir.is_dir():
                continue
            for release_dir in scope_dir.iterdir():
                if release_dir.is_symlink() or not release_dir.is_dir():
                    continue
                try:
                    payload = json.loads(
                        (release_dir / "release.json").read_text(encoding="utf-8")
                    )
                    scope_payload = payload["scope"]
                    scope = InspectionScope(
                        scope_payload["product"],
                        scope_payload["area"],
                        scope_payload["template_id"],
                    )
                    if product and scope.product != product:
                        continue
                    if area and scope.area != area:
                        continue
                    releases.append(self.load(scope, release_dir.name))
                except (
                    InspectionReleaseError,
                    OSError,
                    KeyError,
                    json.JSONDecodeError,
                ):
                    continue
        return tuple(
            sorted(
                releases,
                key=lambda item: (item.created_at, item.display_version),
                reverse=True,
            )
        )

    def active_pointer(self, scope: InspectionScope) -> dict[str, Any] | None:
        path = self.root / "active" / f"{scope.scope_hash}.json"
        if not path.is_file() or path.is_symlink():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("Active release pointer is invalid.") from exc
        if not isinstance(payload, Mapping):
            raise InspectionReleaseError("Active release pointer is invalid.")
        if payload.get("scope_hash") != scope.scope_hash:
            raise InspectionReleaseError("Active release scope mismatch.")
        schema_version = payload.get("schema_version")
        if type(schema_version) is not int or schema_version not in {1, 2}:
            raise InspectionReleaseError("Active release pointer schema is invalid.")
        if schema_version == 2:
            event = self._activation_event(scope, str(payload.get("event_id") or ""))
            if event != dict(payload):
                raise InspectionReleaseError(
                    "Active release activation event is missing or invalid."
                )
        return dict(payload)

    def resolve_active(
        self, product: str, area: str, inference_type: str
    ) -> InspectionRelease | None:
        template = template_for_inference_type(inference_type)
        scope = InspectionScope(product, area, template.template_id)
        pointer = self.active_pointer(scope)
        if pointer is None:
            return None
        release = self.load(scope, str(pointer.get("release_id") or ""))
        self.policy.require_allowed(release, ActivationMode(pointer["mode"]))
        return release

    def activate(
        self,
        release: InspectionRelease,
        *,
        mode: ActivationMode,
        operator: str,
        reason: str,
        expected_release_id: str | None,
    ) -> dict[str, Any]:
        if not operator.strip() or not reason.strip():
            raise InspectionReleasePolicyError(
                "Activation operator and reason are required."
            )
        lock = self._lock_for(release.scope.scope_hash)
        with lock, self._cross_process_lock(release.scope.scope_hash):
            committed = self.load(release.scope, release.release_id)
            self.policy.require_allowed(committed, mode)
            self._verify_external_evidence(committed)
            current = self.active_pointer(release.scope)
            current_id = str(current.get("release_id") or "") if current else None
            if current_id != expected_release_id:
                raise InspectionReleaseConflictError(
                    "Active inspection release changed; refresh and retry."
                )
            event_id = str(uuid4())
            now = datetime.now(timezone.utc).isoformat()
            pointer = {
                "schema_version": 2,
                "scope_hash": release.scope.scope_hash,
                "release_id": release.release_id,
                "display_version": committed.display_version,
                "mode": mode.value,
                "activated_at": now,
                "operator": operator.strip(),
                "reason": reason.strip(),
                "previous_release_id": current_id,
                "previous_event_id": (
                    str(current.get("event_id") or "") if current else None
                ),
                "event_id": event_id,
            }
            event_path = self._activation_event_path(release.scope, event_id)
            pointer_path = self.root / "active" / f"{release.scope.scope_hash}.json"
            if event_path.exists() or event_path.is_symlink():
                raise InspectionReleaseConflictError(
                    "Activation event identity already exists; retry activation."
                )
            # The immutable event is the write-ahead record; replacing the
            # active pointer is the commit point. Unreachable schema 2 events
            # are aborted attempts and are excluded from rollback history.
            try:
                _write_json_atomic(event_path, pointer)
            except (OSError, TypeError, ValueError) as exc:
                if not _json_file_matches(event_path, pointer):
                    raise InspectionReleaseError(
                        "Activation event could not be recorded; the active "
                        "release pointer was not changed."
                    ) from exc
            try:
                _write_json_atomic(pointer_path, pointer)
            except (OSError, TypeError, ValueError) as exc:
                if not _json_file_matches(pointer_path, pointer):
                    raise InspectionReleaseError(
                        "Active release pointer could not be recorded; the "
                        "previous release remains active."
                    ) from exc
            return pointer

    def rollback(
        self,
        scope: InspectionScope,
        *,
        operator: str,
        reason: str,
    ) -> dict[str, Any]:
        current = self.active_pointer(scope)
        if current is None or not current.get("previous_release_id"):
            raise InspectionReleasePolicyError(
                "No previous inspection release to restore."
            )
        previous = self.load(scope, str(current["previous_release_id"]))
        previous_pointer = self._pointer_for_release(scope, previous.release_id)
        previous_mode = (
            ActivationMode(previous_pointer["mode"])
            if previous_pointer is not None
            else next(
                iter(self.policy.allowed_modes(previous)), ActivationMode.LIMITED_TRIAL
            )
        )
        return self.activate(
            previous,
            mode=previous_mode,
            operator=operator,
            reason=reason,
            expected_release_id=str(current["release_id"]),
        )

    def _pointer_for_release(
        self, scope: InspectionScope, release_id: str
    ) -> dict[str, Any] | None:
        current = self.active_pointer(scope)
        event = current
        visited: set[str] = set()
        while event is not None and event.get("schema_version") == 2:
            event_id = str(event.get("event_id") or "")
            if not event_id or event_id in visited:
                raise InspectionReleaseError("Activation event history is invalid.")
            visited.add(event_id)
            if event.get("release_id") == release_id:
                return event
            previous_event_id = str(event.get("previous_event_id") or "")
            if not previous_event_id:
                break
            event = self._activation_event(scope, previous_event_id)

        # Schema 1 events predate the committed-event chain. They remain
        # readable for rollback compatibility, while unreachable schema 2
        # events are ignored as aborted activation attempts.
        event_root = self.root / "events" / str(scope.scope_hash)
        if not event_root.is_dir():
            return None
        for path in sorted(event_root.glob("*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                isinstance(payload, Mapping)
                and payload.get("schema_version") == 1
                and payload.get("release_id") == release_id
            ):
                return dict(payload)
        return None

    def _activation_event_path(
        self,
        scope: InspectionScope,
        event_id: str,
    ) -> Path:
        try:
            canonical_event_id = str(UUID(event_id))
        except (TypeError, ValueError) as exc:
            raise InspectionReleaseError("Activation event identity is invalid.") from exc
        if canonical_event_id != event_id.lower():
            raise InspectionReleaseError("Activation event identity is invalid.")
        event_root = self.root / "events" / str(scope.scope_hash)
        if event_root.parent.is_symlink() or event_root.is_symlink():
            raise InspectionReleaseError("Activation event directory is unsafe.")
        return event_root / f"{canonical_event_id}.json"

    def _activation_event(
        self,
        scope: InspectionScope,
        event_id: str,
    ) -> dict[str, Any]:
        exact_path = self._activation_event_path(scope, event_id)
        candidates = [exact_path] if exact_path.is_file() else []
        if not candidates:
            candidates = sorted(exact_path.parent.glob(f"*-{event_id}.json"))
        if len(candidates) != 1:
            raise InspectionReleaseError(
                "Active release activation event is missing or ambiguous."
            )
        path = candidates[0]
        if path.is_symlink():
            raise InspectionReleaseError("Activation event cannot be a symlink.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise InspectionReleaseError("Activation event is invalid.") from exc
        if (
            not isinstance(payload, Mapping)
            or str(payload.get("event_id") or "").lower() != event_id.lower()
            or payload.get("scope_hash") != scope.scope_hash
        ):
            raise InspectionReleaseError("Activation event is invalid.")
        return dict(payload)

    def _resolve_external_path(self, path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if path.exists() or self._station_paths is None:
            return path.resolve()
        return Path(self._station_paths.relocate_legacy_path(path))

    def _relocate_external_paths(
        self,
        release: InspectionRelease,
    ) -> InspectionRelease:
        if self._station_paths is None:
            return release
        components = tuple(
            replace(
                component,
                artifact_path=(
                    str(self._resolve_external_path(component.artifact_path))
                    if component.artifact_path
                    else ""
                ),
                config_path=(
                    str(self._resolve_external_path(component.config_path))
                    if component.config_path
                    else ""
                ),
            )
            for component in release.components
        )
        validation = replace(
            release.validation,
            report_path=(
                str(self._resolve_external_path(release.validation.report_path))
                if release.validation.report_path
                else ""
            ),
        )
        return replace(release, components=components, validation=validation)

    def _verify_file(
        self,
        path_value: str,
        expected_sha256: str,
        label: str,
    ) -> Path:
        path = self._resolve_external_path(path_value)
        if path.is_symlink() or not path.is_file():
            raise InspectionReleaseError(f"{label} is unavailable: {path}")
        if sha256_file(path) != expected_sha256.lower():
            raise InspectionReleaseError(f"{label} checksum mismatch: {path}")
        return path

    def _verify_external_evidence(self, release: InspectionRelease) -> None:
        if release.validation.report_path:
            self._verify_file(
                release.validation.report_path,
                release.validation.report_sha256,
                "Validation report",
            )
        for component in release.components:
            if component.artifact_path:
                self._verify_file(
                    component.artifact_path,
                    component.artifact_sha256,
                    f"{component.component_id} artifact",
                )
            if component.config_path:
                self._verify_file(
                    component.config_path,
                    component.config_sha256,
                    f"{component.component_id} config",
                )
            if (
                component.kind == "stats_color"
                and component.artifact_path
                and dict(component.metadata).get("package_id")
            ):
                from core.services.color_profile_store import ColorProfileStore

                manifest = self._resolve_external_path(component.config_path)
                ColorProfileStore(manifest.parents[1]).load(manifest)


class InspectionReleaseResolver:
    """Resolve one active release with stat-based checksum cache invalidation.

    The active pointer and external artifacts are hashed on first use. Ordinary
    inspections only perform inexpensive ``stat`` calls; any path, size, or
    timestamp change forces a full checksum verification before inference.
    """

    def __init__(self, store: InspectionReleaseStore) -> None:
        self.store = store
        self._lock = threading.Lock()
        self._cache: dict[
            str,
            tuple[tuple[int, int], tuple[tuple[str, int, int], ...], InspectionRelease],
        ] = {}

    def resolve(
        self, product: str, area: str, inference_type: str
    ) -> InspectionRelease | None:
        template = template_for_inference_type(inference_type)
        scope = InspectionScope(product, area, template.template_id)
        pointer_path = self.store.root / "active" / f"{scope.scope_hash}.json"
        try:
            pointer_stat = pointer_path.stat()
        except FileNotFoundError:
            with self._lock:
                self._cache.pop(scope.scope_hash, None)
            return None
        if pointer_path.is_symlink():
            raise InspectionReleaseError("Active release pointer cannot be a symlink.")
        pointer_identity = (pointer_stat.st_mtime_ns, pointer_stat.st_size)
        with self._lock:
            cached = self._cache.get(scope.scope_hash)
        if cached and cached[0] == pointer_identity:
            current_external = self._external_identity(cached[2])
            if current_external == cached[1]:
                return cached[2]
        release = self.store.resolve_active(product, area, inference_type)
        if release is None:
            return None
        external_identity = self._external_identity(release)
        with self._lock:
            self._cache[scope.scope_hash] = (
                pointer_identity,
                external_identity,
                release,
            )
        return release

    def _external_identity(
        self,
        release: InspectionRelease,
    ) -> tuple[tuple[str, int, int], ...]:
        paths = (
            [release.validation.report_path] if release.validation.report_path else []
        )
        for component in release.components:
            paths.extend(
                value
                for value in (component.artifact_path, component.config_path)
                if value
            )
            if (
                component.kind == "stats_color"
                and dict(component.metadata).get("package_id")
                and component.config_path
            ):
                try:
                    payload = json.loads(
                        Path(component.config_path).read_text(encoding="utf-8")
                    )
                    paths.extend(
                        str(self.store._resolve_external_path(str(item["config_path"])))
                        for item in payload.get("revisions") or ()
                    )
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    paths.append(component.config_path)
        identities: list[tuple[str, int, int]] = []
        for value in paths:
            path = Path(value).expanduser().resolve()
            stat = path.stat()
            identities.append((str(path), stat.st_mtime_ns, stat.st_size))
        return tuple(identities)
