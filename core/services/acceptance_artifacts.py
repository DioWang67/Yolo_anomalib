"""Immutable artifact binding for every model-acceptance entry point.

An acceptance result is only meaningful when the files named by its report are
the files loaded by the inference engine.  This module owns that boundary: it
resolves the model config exactly once, binds it to one weight file, pins the
optional color model, and gives the resulting evidence a canonical identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

ColorModelMode = Literal["disabled", "embedded", "override"]


class AcceptanceArtifactError(ValueError):
    """Raised when acceptance artifacts are missing, unsafe, or mismatched."""


def color_scope_model_type(inference_type: str) -> str:
    """Return the model type that color artifacts are registered under.

    Fusion's color stage *is* the YOLO one, so fusion shares the YOLO scope;
    every other type keeps its own. This lived as six separate copies across the
    window, the gate, the matrix, and the matrix dialog, and they had already
    drifted -- one of them skipped the lowercasing its own caller then re-applied
    -- so a scope lookup could miss artifacts that existed.
    """

    normalized = inference_type.strip().lower()
    return "yolo" if normalized == "fusion" else normalized


@dataclass(frozen=True)
class ArtifactRef:
    """One immutable file reference used by an acceptance run."""

    path: Path
    sha256: str
    size_bytes: int

    def report_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class AcceptanceArtifactBundle:
    """A complete, content-addressed acceptance inference combination."""

    product: str
    area: str
    inference_type: str
    version: str
    global_config: ArtifactRef
    model_config: ArtifactRef
    model_weight: ArtifactRef
    color_model: ArtifactRef | None
    color_model_mode: ColorModelMode
    color_revision_overrides: tuple[tuple[str, str], ...]
    include_active_color_revisions: bool
    color_revision_contract_sha256: str
    bundle_sha256: str

    def report_payload(self) -> dict[str, Any]:
        return {
            "product": self.product,
            "area": self.area,
            "inference_type": self.inference_type,
            "version": self.version,
            "global_config": self.global_config.report_payload(),
            "model_config": self.model_config.report_payload(),
            "model_weight": self.model_weight.report_payload(),
            "color_model": (
                self.color_model.report_payload()
                if self.color_model is not None
                else None
            ),
            "color_model_mode": self.color_model_mode,
            "color_revision_overrides": dict(self.color_revision_overrides),
            "include_active_color_revisions": self.include_active_color_revisions,
            "color_revision_contract_sha256": self.color_revision_contract_sha256,
            "bundle_sha256": self.bundle_sha256,
        }


def build_acceptance_artifact_bundle(
    *,
    product: str,
    area: str,
    inference_type: str,
    version: str,
    global_config_path: str | Path,
    model_config_path: str | Path,
    models_root: str | Path,
    model_weight_path: str | Path | None = None,
    color_model_path: str | Path | None = None,
    color_model_is_override: bool = False,
    color_revision_overrides: Mapping[str, str] | None = None,
    include_active_color_revisions: bool = False,
    color_revision_contract: Mapping[str, Any] | None = None,
) -> AcceptanceArtifactBundle:
    """Resolve and bind one exact inference combination.

    A caller may select a weight explicitly, but the model config must already
    resolve to that same file.  Silently rewriting a mismatched config would
    hide a packaging defect and make the source config hash misleading.
    """

    normalized_product = _required_text(product, "product")
    normalized_area = _required_text(area, "area")
    normalized_type = _required_text(inference_type, "inference type").lower()
    normalized_version = _required_text(version, "model version")
    resolved_models_root = Path(models_root).expanduser().resolve()

    global_config = artifact_ref(global_config_path, "global config")
    model_config = artifact_ref(model_config_path, "model config")
    configured_weight = resolve_configured_model_weight(
        model_config.path,
        models_root=resolved_models_root,
    )
    if model_weight_path is not None:
        selected_weight = artifact_ref(model_weight_path, "model weight")
        if configured_weight.path != selected_weight.path:
            raise AcceptanceArtifactError(
                "Model config weights do not resolve to the selected acceptance "
                f"weight: config={configured_weight.path} selected={selected_weight.path}"
            )
        model_weight = selected_weight
    else:
        model_weight = configured_weight

    if color_model_path is not None:
        color_model = artifact_ref(color_model_path, "color model")
        color_mode: ColorModelMode = (
            "override" if color_model_is_override else "embedded"
        )
    else:
        color_model = resolve_effective_color_model(
            model_config_path=model_config.path,
            global_config_path=global_config.path,
            models_root=resolved_models_root,
        )
        color_mode = "embedded" if color_model is not None else "disabled"

    normalized_overrides = tuple(
        sorted(
            (
                _required_text(str(scope_hash), "color revision scope"),
                _required_text(str(revision_id), "color revision id"),
            )
            for scope_hash, revision_id in (color_revision_overrides or {}).items()
        )
    )
    contract_sha256 = _canonical_sha256(dict(color_revision_contract or {}))
    identity_payload = {
        "schema_version": 1,
        "target": {
            "product": normalized_product,
            "area": normalized_area,
            "inference_type": normalized_type,
        },
        "version": normalized_version,
        "global_config": global_config.identity_payload(),
        "model_config": model_config.identity_payload(),
        "model_weight": model_weight.identity_payload(),
        "color_model": (
            color_model.identity_payload() if color_model is not None else None
        ),
        "color_model_mode": color_mode,
        "color_revision_overrides": dict(normalized_overrides),
        "include_active_color_revisions": bool(include_active_color_revisions),
        "color_revision_contract_sha256": contract_sha256,
    }
    return AcceptanceArtifactBundle(
        product=normalized_product,
        area=normalized_area,
        inference_type=normalized_type,
        version=normalized_version,
        global_config=global_config,
        model_config=model_config,
        model_weight=model_weight,
        color_model=color_model,
        color_model_mode=color_mode,
        color_revision_overrides=normalized_overrides,
        include_active_color_revisions=bool(include_active_color_revisions),
        color_revision_contract_sha256=contract_sha256,
        bundle_sha256=_canonical_sha256(identity_payload),
    )


def verify_acceptance_artifact_bundle(
    bundle: AcceptanceArtifactBundle,
    *,
    models_root: str | Path,
) -> None:
    """Fail if any pinned artifact or config-to-weight binding has changed."""

    for label, expected in (
        ("global config", bundle.global_config),
        ("model config", bundle.model_config),
        ("model weight", bundle.model_weight),
    ):
        _verify_artifact_ref(expected, label)
    if bundle.color_model is not None:
        _verify_artifact_ref(bundle.color_model, "color model")

    configured_weight = resolve_configured_model_weight(
        bundle.model_config.path,
        models_root=models_root,
    )
    if configured_weight.path != bundle.model_weight.path:
        raise AcceptanceArtifactError(
            "Model config-to-weight binding changed during acceptance."
        )


def resolve_configured_model_weight(
    model_config_path: str | Path,
    *,
    models_root: str | Path,
) -> ArtifactRef:
    """Resolve ``weights`` with the same project/config-relative precedence."""

    config_path = artifact_ref(model_config_path, "model config").path
    config = _load_yaml_mapping(config_path, "model config")
    raw_weight = str(config.get("weights") or "").strip()
    if not raw_weight:
        raise AcceptanceArtifactError(
            f"Model config is missing weights: {config_path}"
        )
    return _resolve_configured_artifact(
        raw_weight,
        config_path=config_path,
        models_root=Path(models_root).expanduser().resolve(),
        label="configured model weight",
    )


def resolve_configured_color_model(
    model_config_path: str | Path,
    *,
    models_root: str | Path,
) -> ArtifactRef | None:
    """Resolve the optional embedded color model referenced by model config."""

    config_path = artifact_ref(model_config_path, "model config").path
    config = _load_yaml_mapping(config_path, "model config")
    raw_color = str(config.get("color_model_path") or "").strip()
    if not raw_color:
        return None
    return _resolve_configured_artifact(
        raw_color,
        config_path=config_path,
        models_root=Path(models_root).expanduser().resolve(),
        label="configured color model",
    )


def resolve_effective_color_model(
    *,
    model_config_path: str | Path,
    global_config_path: str | Path,
    models_root: str | Path,
) -> ArtifactRef | None:
    """Resolve the model-level color path, falling back to the global config."""

    model_config = _load_yaml_mapping(
        artifact_ref(model_config_path, "model config").path,
        "model config",
    )
    global_config = _load_yaml_mapping(
        artifact_ref(global_config_path, "global config").path,
        "global config",
    )
    enabled = model_config.get(
        "enable_color_check",
        global_config.get("enable_color_check", False),
    )
    if not isinstance(enabled, bool):
        raise AcceptanceArtifactError(
            "Effective enable_color_check must be a boolean."
        )
    if not enabled:
        return None

    model_color = resolve_configured_color_model(
        model_config_path,
        models_root=models_root,
    )
    if model_color is not None:
        return model_color
    return resolve_configured_color_model(
        global_config_path,
        models_root=models_root,
    )


def artifact_ref(raw_path: str | Path, label: str) -> ArtifactRef:
    """Create a content-addressed reference for one regular, non-symlink file."""

    candidate = Path(raw_path).expanduser()
    # Checked before resolving, which is the only point where it can be seen:
    # ``resolve()`` follows the link, so the same test afterwards is always
    # False and would read as a guard while protecting nothing.
    if candidate.is_symlink():
        raise AcceptanceArtifactError(f"{label} cannot be a symbolic link: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise AcceptanceArtifactError(f"{label} is missing or unsafe: {resolved}")
    size_bytes = resolved.stat().st_size
    if size_bytes <= 0:
        raise AcceptanceArtifactError(f"{label} is empty: {resolved}")
    return ArtifactRef(
        path=resolved,
        sha256=_sha256_file(resolved),
        size_bytes=size_bytes,
    )


def _resolve_configured_artifact(
    raw_path: str,
    *,
    config_path: Path,
    models_root: Path,
    label: str,
) -> ArtifactRef:
    configured = Path(raw_path).expanduser()
    candidates = (
        (configured,)
        if configured.is_absolute()
        else (
            models_root.resolve().parent / configured,
            config_path.parent / configured,
        )
    )
    seen: set[Path] = set()
    for candidate in candidates:
        absolute = candidate.absolute()
        if absolute in seen:
            continue
        seen.add(absolute)
        if candidate.is_file() and not candidate.is_symlink():
            return artifact_ref(candidate, label)
    raise AcceptanceArtifactError(
        f"{label} does not resolve to a regular file: {raw_path}"
    )


def _verify_artifact_ref(expected: ArtifactRef, label: str) -> None:
    actual = artifact_ref(expected.path, label)
    if (
        actual.path != expected.path
        or actual.sha256 != expected.sha256
        or actual.size_bytes != expected.size_bytes
    ):
        raise AcceptanceArtifactError(f"{label} changed during acceptance: {expected.path}")


def _load_yaml_mapping(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AcceptanceArtifactError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, Mapping):
        raise AcceptanceArtifactError(f"{label} must be a YAML mapping: {path}")
    return payload


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise AcceptanceArtifactError(f"{label} is required")
    return normalized


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
