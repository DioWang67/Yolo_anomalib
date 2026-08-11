"""Immutable acceptance contract for production-active color revisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from core.config import DetectionConfig
from tools.color_calibration_service import (
    ColorCalibrationError,
    ColorCalibrationScope,
    canonical_sha256,
    sha256_file,
)
from tools.color_configuration_revisions import ColorConfigurationRevisionStore


class ColorRevisionContractError(RuntimeError):
    """Raised when an active color-revision snapshot is unsafe or stale."""


_CONTRACT_FIELDS = {
    "schema_version",
    "enabled",
    "checker_type",
    "target",
    "entries",
    "identity_sha256",
}
_TARGET_FIELDS = {"product", "area", "inference_type"}
_ENTRY_FIELDS = {
    "scope_hash",
    "threshold_key",
    "revision_id",
    "config_sha256",
    "config_file_sha256",
    "pointer_sha256",
}


def capture_candidate_color_revision_contract(
    *,
    revisions_root: str | Path,
    candidate_config_path: str | Path,
    global_config_path: str | Path | None = None,
    color_model_present: bool,
    product: str,
    area: str,
    inference_type: str,
) -> dict[str, Any]:
    """Capture the active revisions that the candidate runtime would apply."""
    config_path = Path(candidate_config_path).expanduser().resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ColorRevisionContractError(
            f"Candidate color config is unreadable: {config_path}"
        ) from exc
    if not isinstance(config, Mapping):
        raise ColorRevisionContractError(
            f"Candidate color config must be a mapping: {config_path}"
        )
    try:
        normalized_config = DetectionConfig.normalize_model_dict(
            dict(config),
            str(config_path),
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ColorRevisionContractError(
            "Candidate color config does not satisfy the runtime schema."
        ) from exc
    base_enabled = False
    base_checker = "color_qc"
    if global_config_path is not None:
        base_enabled, base_checker = _load_global_color_defaults(
            Path(global_config_path).expanduser().resolve()
        )

    configured_enabled = normalized_config.get("enable_color_check")
    if configured_enabled is None:
        configured_enabled = base_enabled
    if not isinstance(configured_enabled, bool):
        raise ColorRevisionContractError(
            "Candidate enable_color_check must be a boolean."
        )
    configured_checker = normalized_config.get(
        "color_checker_type",
        base_checker,
    )
    if configured_checker is not None and not isinstance(configured_checker, str):
        raise ColorRevisionContractError(
            "Candidate color_checker_type must be a string."
        )
    enabled = configured_enabled and color_model_present
    checker_type = (
        str(configured_checker or "color_qc").strip().lower()
        if enabled
        else ""
    )
    if enabled and not checker_type:
        raise ColorRevisionContractError(
            "Enabled color checking requires a checker type."
        )
    return capture_active_color_revision_contract(
        revisions_root=revisions_root,
        product=product,
        area=area,
        inference_type=inference_type,
        enabled=enabled,
        checker_type=checker_type,
    )


def capture_active_color_revision_contract(
    *,
    revisions_root: str | Path,
    product: str,
    area: str,
    inference_type: str,
    enabled: bool,
    checker_type: str,
) -> dict[str, Any]:
    """Return a stable, canonical snapshot of matching active pointers."""
    normalized_checker = str(checker_type).strip().lower() if enabled else ""
    first = _capture_once(
        revisions_root=Path(revisions_root).expanduser().resolve(),
        product=product,
        area=area,
        inference_type=inference_type,
        enabled=enabled,
        checker_type=normalized_checker,
    )
    second = _capture_once(
        revisions_root=Path(revisions_root).expanduser().resolve(),
        product=product,
        area=area,
        inference_type=inference_type,
        enabled=enabled,
        checker_type=normalized_checker,
    )
    if first != second:
        raise ColorRevisionContractError(
            "Active color revisions changed while their contract was captured."
        )
    return first


def verify_active_color_revision_contract(
    expected: Mapping[str, Any],
    *,
    revisions_root: str | Path,
) -> dict[str, Any]:
    """Recompute and compare one report-owned active revision contract."""
    validated = validate_color_revision_contract(expected)
    target = validated["target"]
    current = capture_active_color_revision_contract(
        revisions_root=revisions_root,
        product=target["product"],
        area=target["area"],
        inference_type=target["inference_type"],
        enabled=validated["enabled"],
        checker_type=validated["checker_type"],
    )
    if validated != current:
        raise ColorRevisionContractError(
            "Active color revision contract changed: "
            f"expected={validated.get('identity_sha256')} "
            f"current={current.get('identity_sha256')}"
        )
    return current


def validate_color_revision_contract(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact report schema and its canonical identity."""
    payload = dict(contract)
    if set(payload) != _CONTRACT_FIELDS:
        raise ColorRevisionContractError(
            "Color revision contract has unexpected or missing fields."
        )
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ColorRevisionContractError(
            "Color revision contract schema version is unsupported."
        )
    enabled = payload["enabled"]
    checker_type = payload["checker_type"]
    if not isinstance(enabled, bool) or not isinstance(checker_type, str):
        raise ColorRevisionContractError(
            "Color revision contract runtime fields are invalid."
        )
    if checker_type != checker_type.strip().lower():
        raise ColorRevisionContractError(
            "Color revision checker type is not canonical."
        )
    if (enabled and not checker_type) or (not enabled and checker_type):
        raise ColorRevisionContractError(
            "Color revision checker type does not match its enabled state."
        )

    target = payload["target"]
    if not isinstance(target, Mapping) or set(target) != _TARGET_FIELDS:
        raise ColorRevisionContractError(
            "Color revision contract has no exact target."
        )
    normalized_target: dict[str, str] = {}
    for field in sorted(_TARGET_FIELDS):
        value = target[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ColorRevisionContractError(
                f"Color revision target field is invalid: {field}."
            )
        normalized_target[field] = value
    payload["target"] = normalized_target

    entries = payload["entries"]
    if not isinstance(entries, list) or (not enabled and entries):
        raise ColorRevisionContractError(
            "Color revision contract entries do not match its enabled state."
        )
    normalized_entries: list[dict[str, str]] = []
    seen_scopes: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != _ENTRY_FIELDS:
            raise ColorRevisionContractError(
                "Color revision contract entry has invalid fields."
            )
        normalized_entry: dict[str, str] = {}
        for field in sorted(_ENTRY_FIELDS):
            value = entry[field]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ColorRevisionContractError(
                    f"Color revision contract entry field is invalid: {field}."
                )
            normalized_entry[field] = value
        scope_hash = normalized_entry["scope_hash"]
        if not _is_lower_hex_digest(scope_hash, length=24):
            raise ColorRevisionContractError(
                "Color revision contract entry has an invalid scope hash."
            )
        for field in (
            "config_sha256",
            "config_file_sha256",
            "pointer_sha256",
        ):
            if not _is_lower_hex_digest(normalized_entry[field], length=64):
                raise ColorRevisionContractError(
                    f"Color revision contract entry has an invalid {field}."
                )
        if scope_hash in seen_scopes:
            raise ColorRevisionContractError(
                "Color revision contract contains a duplicate scope."
            )
        seen_scopes.add(scope_hash)
        normalized_entries.append(normalized_entry)
    if normalized_entries != sorted(
        normalized_entries,
        key=lambda entry: entry["scope_hash"],
    ):
        raise ColorRevisionContractError(
            "Color revision contract entries are not canonical."
        )
    payload["entries"] = normalized_entries

    identity = payload["identity_sha256"]
    if not isinstance(identity, str) or not _is_lower_hex_digest(identity, length=64):
        raise ColorRevisionContractError(
            "Color revision contract has an invalid identity."
        )
    identity_payload = dict(payload)
    identity_payload.pop("identity_sha256")
    if canonical_sha256(identity_payload) != identity:
        raise ColorRevisionContractError(
            "Color revision contract identity does not match its content."
        )
    return payload


def color_revision_overrides(
    contract: Mapping[str, Any],
) -> dict[str, str]:
    """Convert a verified snapshot into immutable resolver overrides."""
    raw_entries = validate_color_revision_contract(contract)["entries"]
    overrides: dict[str, str] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ColorRevisionContractError(
                "Color revision contract entry must be an object."
            )
        scope_hash = str(raw_entry.get("scope_hash") or "")
        revision_id = str(raw_entry.get("revision_id") or "")
        if not scope_hash or not revision_id or scope_hash in overrides:
            raise ColorRevisionContractError(
                "Color revision contract contains an invalid or duplicate entry."
            )
        overrides[scope_hash] = revision_id
    return overrides


def _capture_once(
    *,
    revisions_root: Path,
    product: str,
    area: str,
    inference_type: str,
    enabled: bool,
    checker_type: str,
) -> dict[str, Any]:
    target = {
        "product": product,
        "area": area,
        "inference_type": inference_type,
    }
    entries: list[dict[str, str]] = []
    if enabled:
        active_root = revisions_root / "active"
        if active_root.is_symlink():
            raise ColorRevisionContractError(
                f"Active color revision root cannot be a symbolic link: {active_root}"
            )
        pointer_paths = (
            sorted(active_root.glob("*.json")) if active_root.is_dir() else ()
        )
        store = ColorConfigurationRevisionStore(root=revisions_root)
        for pointer_path in pointer_paths:
            entries.extend(
                _matching_pointer_entry(
                    pointer_path,
                    store=store,
                    product=product,
                    area=area,
                    inference_type=inference_type,
                    checker_type=checker_type,
                )
            )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "enabled": bool(enabled),
        "checker_type": checker_type,
        "target": target,
        "entries": sorted(entries, key=lambda entry: entry["scope_hash"]),
    }
    payload["identity_sha256"] = canonical_sha256(payload)
    return payload


def _load_global_color_defaults(config_path: Path) -> tuple[bool, str]:
    """Apply the same global schema and local overlay used by DetectionSystem."""
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise TypeError("global config must be a mapping")
        overlaid = DetectionConfig._apply_local_overlay(raw, config_path)
        normalized = DetectionConfig.normalize_global_dict(
            overlaid,
            str(config_path),
        )
        return (
            bool(normalized.get("enable_color_check", False)),
            str(normalized.get("color_checker_type") or "color_qc"),
        )
    except (
        OSError,
        UnicodeDecodeError,
        RuntimeError,
        TypeError,
        ValueError,
        yaml.YAMLError,
    ) as exc:
        raise ColorRevisionContractError(
            "Global runtime config is invalid for color revision capture."
        ) from exc


def _matching_pointer_entry(
    pointer_path: Path,
    *,
    store: ColorConfigurationRevisionStore,
    product: str,
    area: str,
    inference_type: str,
    checker_type: str,
) -> list[dict[str, str]]:
    if pointer_path.is_symlink():
        raise ColorRevisionContractError(
            f"Active color pointer cannot be a symbolic link: {pointer_path}"
        )
    try:
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes.decode("utf-8"))
        raw_scope = pointer["scope"]
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
        if str(raw_scope.get("scope_hash") or "") != scope.scope_hash:
            raise ColorRevisionContractError(
                f"Active color pointer has a stale scope hash: {pointer_path}"
            )
        if pointer_path.name != f"{scope.scope_hash}.json":
            raise ColorRevisionContractError(
                f"Active color pointer filename has the wrong scope: {pointer_path}"
            )
    except ColorRevisionContractError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ColorCalibrationError,
    ) as exc:
        raise ColorRevisionContractError(
            f"Active color pointer is invalid: {pointer_path}"
        ) from exc
    if (
        scope.product,
        scope.area,
        scope.model_type,
        scope.checker_type,
    ) != (product, area, inference_type, checker_type):
        return []
    try:
        verified_pointer = store.read_active_pointer(scope)
        if verified_pointer is None or dict(verified_pointer) != pointer:
            raise ColorRevisionContractError(
                f"Active color pointer changed while it was read: {pointer_path}"
            )
        revision_id = str(pointer.get("revision_id") or "")
        revision = store.load(scope, revision_id)
        if store.is_revoked(revision):
            raise ColorRevisionContractError(
                f"Active color revision is revoked: {revision_id}"
            )
        pointer_config_sha256 = str(pointer.get("config_sha256") or "")
        if pointer_config_sha256 != revision.new_config_sha256:
            raise ColorRevisionContractError(
                f"Active color pointer has a stale config checksum: {pointer_path}"
            )
        return [
            {
                "scope_hash": scope.scope_hash,
                "threshold_key": scope.threshold_key,
                "revision_id": revision.revision_id,
                "config_sha256": revision.new_config_sha256,
                "config_file_sha256": sha256_file(revision.config_path),
                "pointer_sha256": _sha256_bytes(pointer_bytes),
            }
        ]
    except ColorRevisionContractError:
        raise
    except (OSError, ValueError, ColorCalibrationError) as exc:
        raise ColorRevisionContractError(
            f"Active color revision is invalid: {pointer_path}"
        ) from exc


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_lower_hex_digest(value: str, *, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )
