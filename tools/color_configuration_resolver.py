"""Resolve activated color revisions without touching annotation canonical data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.color_calibration_service import ColorCalibrationError, ColorCalibrationScope, sha256_file
from tools.color_configuration_revisions import ColorConfigurationRevisionStore


@dataclass(frozen=True)
class ResolvedColorConfiguration:
    scope: ColorCalibrationScope
    revision_id: str
    config_sha256: str
    config: Mapping[str, Any]
    source_path: Path


class ColorConfigurationResolver:
    def __init__(
        self,
        *,
        models_root: str | Path,
        revisions_root: str | Path,
        revision_overrides: Mapping[str, str] | None = None,
        include_active_revisions: bool = True,
    ) -> None:
        self.models_root = Path(models_root).resolve()
        self.revision_store = ColorConfigurationRevisionStore(root=revisions_root)
        self.revision_overrides = {
            str(scope_hash).strip(): str(revision).strip()
            for scope_hash, revision in (revision_overrides or {}).items()
        }
        if any(
            not scope_hash or not revision
            for scope_hash, revision in self.revision_overrides.items()
        ):
            raise ColorCalibrationError(
                "COLOR_REVISION_REFERENCE_REQUIRED",
                "Color revision overrides require a scope hash and revision.",
            )
        self.include_active_revisions = bool(include_active_revisions)

    def resolve(self, scope: ColorCalibrationScope) -> ResolvedColorConfiguration:
        selected_reference = self.revision_overrides.get(scope.scope_hash)
        if selected_reference is not None:
            revision = self.revision_store.resolve_revision(
                scope, selected_reference
            )
            if self.revision_store.is_revoked(revision):
                raise ColorCalibrationError(
                    "COLOR_REVISION_REVOKED",
                    "A revoked color revision cannot be used for acceptance.",
                )
            config = json.loads(revision.config_path.read_text(encoding="utf-8"))
            return ResolvedColorConfiguration(
                scope,
                revision.revision_id,
                revision.new_config_sha256,
                config,
                revision.config_path,
            )
        pointer = (
            self.revision_store.read_active_pointer(scope)
            if self.include_active_revisions
            else None
        )
        if pointer:
            revision = self.revision_store.load(scope, str(pointer["revision_id"]))
            if self.revision_store.is_revoked(revision):
                raise ColorCalibrationError("COLOR_ACTIVE_REVISION_REVOKED", "Active color revision is revoked.")
            if revision.new_config_sha256 != str(pointer.get("config_sha256") or ""):
                raise ColorCalibrationError("COLOR_ACTIVE_POINTER_INVALID", "Active pointer config SHA mismatch.")
            config = json.loads(revision.config_path.read_text(encoding="utf-8"))
            return ResolvedColorConfiguration(scope, revision.revision_id, revision.new_config_sha256, config, revision.config_path)
        base = self._base_config_path(scope)
        return ResolvedColorConfiguration(scope, "", sha256_file(base), {}, base)

    def active_overrides(
        self,
        *,
        product: str,
        area: str,
        model_type: str,
        checker_type: str,
    ) -> tuple[dict[str, float], float | None, tuple[str, ...]]:
        overrides: dict[str, float] = {}
        global_value: float | None = None
        revision_ids: list[str] = []
        scopes: dict[str, ColorCalibrationScope] = {}
        active_root = self.revision_store.root / "active"
        pointer_paths = (
            sorted(active_root.glob("*.json"))
            if self.include_active_revisions and active_root.is_dir()
            else ()
        )
        for pointer_path in pointer_paths:
            try:
                pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                raw = pointer["scope"]
                scope = ColorCalibrationScope(*(str(raw[key]) for key in ("product", "area", "model_type", "checker_type", "threshold_key")))
                scopes[scope.scope_hash] = scope
            except ColorCalibrationError:
                raise
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ColorCalibrationError("COLOR_ACTIVE_POINTER_INVALID", f"Invalid active color pointer: {pointer_path}") from exc
        for scope_hash in self.revision_overrides:
            scopes[scope_hash] = self.revision_store.scope_for_hash(scope_hash)
        for scope in sorted(scopes.values()):
            if (scope.product, scope.area, scope.model_type, scope.checker_type) != (product, area, model_type, checker_type):
                continue
            try:
                resolved = self.resolve(scope)
                value = float(resolved.config["config_value"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError("active threshold is outside 0..1")
                if scope.threshold_key == "global":
                    global_value = value
                else:
                    overrides[scope.threshold_key] = value
                revision_ids.append(resolved.revision_id)
            except ColorCalibrationError:
                raise
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ColorCalibrationError(
                    "COLOR_REVISION_INVALID",
                    f"Invalid selected color revision: {scope.scope_hash}",
                ) from exc
        return overrides, global_value, tuple(revision_ids)

    def _base_config_path(self, scope: ColorCalibrationScope) -> Path:
        path = (self.models_root / scope.product / scope.area / scope.model_type / "config.yaml").resolve()
        try:
            path.relative_to(self.models_root)
        except ValueError as exc:
            raise ColorCalibrationError("CALIBRATION_SCOPE_INVALID", "Base config path escapes models root.") from exc
        if not path.is_file() or path.is_symlink():
            raise ColorCalibrationError("CURRENT_CONFIG_MISSING", f"Base color config is unavailable: {path}")
        return path
