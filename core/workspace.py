"""Discover inference and training paths inside a relocatable workspace."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.path_utils import project_root

WORKSPACE_FILENAME = "workspace.yaml"
WORKSPACE_SCHEMA_VERSION = 1
WORKSPACE_ROOT_ENV = "YOLO11_WORKSPACE_ROOT"


class WorkspaceConfigurationError(ValueError):
    """Raised when a discovered workspace manifest violates its contract."""


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved, immutable paths shared by inference and training projects."""

    root: Path
    training_project: Path
    inference_project: Path
    training_data: Path
    inference_models: Path
    manifest_path: Path | None = None


def load_workspace_paths(start: str | Path | None = None) -> WorkspacePaths:
    """Load the nearest workspace manifest or return legacy sibling paths.

    Discovery order is deterministic: ``YOLO11_WORKSPACE_ROOT``, ancestors of
    *start*, then the historical sibling-project layout.  A manifest that is
    found but invalid is never silently ignored.
    """
    anchor = _directory_anchor(start)
    environment_manifest = _environment_workspace_manifest()
    if environment_manifest is not None:
        if not environment_manifest.is_file():
            raise WorkspaceConfigurationError(
                f"{WORKSPACE_ROOT_ENV} does not contain {WORKSPACE_FILENAME}: "
                f"{environment_manifest.parent}"
            )
        return _load_manifest(environment_manifest)

    manifest_candidates = [
        parent / WORKSPACE_FILENAME for parent in _ancestors(anchor)
    ]

    seen: set[Path] = set()
    for candidate in manifest_candidates:
        resolved_candidate = candidate.resolve()
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        if resolved_candidate.is_file():
            return _load_manifest(resolved_candidate)

    return _legacy_workspace_paths(anchor)


def _directory_anchor(start: str | Path | None) -> Path:
    path = Path(start).expanduser() if start is not None else project_root()
    resolved = path.resolve()
    return resolved.parent if resolved.is_file() else resolved


def _environment_workspace_manifest() -> Path | None:
    raw_root = os.getenv(WORKSPACE_ROOT_ENV, "").strip()
    if not raw_root:
        return None
    root = Path(raw_root).expanduser().resolve()
    if root.name.lower() == WORKSPACE_FILENAME:
        return root
    return root / WORKSPACE_FILENAME


def _ancestors(path: Path) -> tuple[Path, ...]:
    return (path, *path.parents)


def _load_manifest(manifest_path: Path) -> WorkspacePaths:
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise WorkspaceConfigurationError(
            f"Unable to read workspace manifest {manifest_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise WorkspaceConfigurationError("workspace.yaml must contain a mapping")
    if raw.get("schema_version") != WORKSPACE_SCHEMA_VERSION or isinstance(
        raw.get("schema_version"), bool
    ):
        raise WorkspaceConfigurationError(
            f"workspace.yaml schema_version must be {WORKSPACE_SCHEMA_VERSION}"
        )

    projects = _require_mapping(raw, "projects")
    paths = _require_mapping(raw, "paths")
    root = manifest_path.parent.resolve()
    return WorkspacePaths(
        root=root,
        training_project=_resolve_workspace_member(root, projects, "training"),
        inference_project=_resolve_workspace_member(root, projects, "inference"),
        training_data=_resolve_workspace_member(root, paths, "training_data"),
        inference_models=_resolve_workspace_member(root, paths, "inference_models"),
        manifest_path=manifest_path.resolve(),
    )


def _require_mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise WorkspaceConfigurationError(f"workspace.yaml {key!r} must be a mapping")
    return value


def _resolve_workspace_member(
    root: Path, values: Mapping[str, Any], key: str
) -> Path:
    raw_path = values.get(key)
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise WorkspaceConfigurationError(
            f"workspace.yaml path {key!r} must be a non-empty string"
        )
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise WorkspaceConfigurationError(
            f"workspace.yaml path {key!r} must be relative to the workspace"
        )
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WorkspaceConfigurationError(
            f"workspace.yaml path {key!r} escapes the workspace: {raw_path}"
        ) from exc
    return resolved


def _legacy_workspace_paths(
    anchor: Path,
) -> WorkspacePaths:
    inference_project = _find_legacy_inference_project(anchor)
    workspace_root = inference_project.parent
    training_project = workspace_root / "Yolo11_auto_train"
    return WorkspacePaths(
        root=workspace_root.resolve(),
        training_project=training_project.resolve(),
        inference_project=inference_project.resolve(),
        training_data=(training_project / "data").resolve(),
        inference_models=(inference_project / "models").resolve(),
    )


def _find_legacy_inference_project(anchor: Path) -> Path:
    for candidate in _ancestors(anchor):
        if candidate.name.casefold() == "yolo11_inference":
            return candidate
    fallback = project_root().resolve()
    if fallback.name.casefold() == "yolo11_inference":
        return fallback
    return anchor
