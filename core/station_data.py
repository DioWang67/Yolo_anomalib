"""Centralized station-local paths separated from the source repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.workspace import WorkspacePaths, load_workspace_paths

_MUTABLE_TOP_LEVEL_NAMES = frozenset(
    {
        ".color_baselines",
        ".color_profiles",
        ".color_revisions",
        ".inspection_releases",
        ".processing_runs",
        ".review_repairs",
        "acceptance",
        "acceptance_reports",
        "logs",
    }
)


@dataclass(frozen=True)
class StationDataPaths:
    """All mutable inference-station locations derived from one workspace."""

    source_root: Path
    root: Path
    artifacts_root: Path
    models: Path
    results: Path
    logs: Path
    acceptance: Path
    acceptance_reports: Path
    color_baselines: Path
    color_profiles: Path
    color_revisions: Path
    inspection_releases: Path
    processing_runs: Path
    review_repairs: Path
    review_root: Path

    @property
    def default_review_manifest(self) -> Path:
        return self.review_root / "review_manifest.csv"

    def relocate_legacy_path(self, value: str | Path) -> Path:
        """Resolve one persisted pre-migration path without mutating evidence."""
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
        if not path.is_absolute():
            return path
        try:
            relative = path.resolve().relative_to(self.source_root)
        except (OSError, ValueError):
            return path
        if not relative.parts:
            return path

        top_level = relative.parts[0]
        if top_level == "Result":
            return self.relocate_legacy_result_path(path)
        elif top_level in _MUTABLE_TOP_LEVEL_NAMES:
            candidate = self.root / relative
        elif top_level == "dist":
            candidate = self.artifacts_root / relative
        elif len(relative.parts) == 1 and top_level.lstrip(".").startswith(
            "review_manifest"
        ):
            candidate = self.review_root / relative.name
        else:
            return path
        return candidate.resolve() if candidate.exists() else path

    def relocate_legacy_result_path(self, value: str | Path) -> Path:
        """Map source- or station-era Result paths to the configured root."""
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
        if not path.is_absolute():
            return path
        legacy_roots = (
            self.source_root / "Result",
            self.root / "Result",
        )
        for legacy_root in legacy_roots:
            try:
                relative = path.resolve().relative_to(legacy_root.resolve())
            except (OSError, ValueError):
                continue
            candidate = self.results / relative
            return candidate.resolve() if candidate.exists() else path
        return path


@dataclass(frozen=True)
class InferencePathContract:
    """Canonical inference paths passed across GUI and training boundaries."""

    models_dir: Path
    station_data_dir: Path
    project_root: Path


def station_data_paths_from_workspace(workspace: WorkspacePaths) -> StationDataPaths:
    root = workspace.station_data.resolve()
    return StationDataPaths(
        source_root=workspace.inference_project.resolve(),
        root=root,
        artifacts_root=workspace.inference_artifacts.resolve(),
        models=workspace.inference_models.resolve(),
        results=workspace.inference_results.resolve(),
        logs=root / "logs",
        acceptance=root / "acceptance",
        acceptance_reports=root / "acceptance_reports",
        color_baselines=root / ".color_baselines",
        color_profiles=root / ".color_profiles",
        color_revisions=root / ".color_revisions",
        inspection_releases=root / ".inspection_releases",
        processing_runs=root / ".processing_runs",
        review_repairs=root / ".review_repairs",
        review_root=root,
    )


def load_station_data_paths(start: str | Path | None = None) -> StationDataPaths:
    """Discover station-local paths with legacy in-project fallback."""
    discovery_start = (
        Path(start) if start is not None else Path(__file__).resolve().parents[1]
    )
    return station_data_paths_from_workspace(load_workspace_paths(discovery_start))


def resolve_inference_path_contract(
    *,
    models_dir: str | Path | None = None,
    station_data_dir: str | Path | None = None,
    project_root: str | Path | None = None,
    start: str | Path | None = None,
) -> InferencePathContract:
    """Resolve missing paths without discovery when every path is injected."""
    fallback_paths = None
    if any(value is None for value in (models_dir, station_data_dir, project_root)):
        fallback_paths = load_station_data_paths(start)
    if models_dir is None:
        assert fallback_paths is not None
        models_dir = fallback_paths.models
    if station_data_dir is None:
        assert fallback_paths is not None
        station_data_dir = fallback_paths.root
    if project_root is None:
        assert fallback_paths is not None
        project_root = fallback_paths.source_root
    return InferencePathContract(
        models_dir=Path(models_dir).expanduser().resolve(),
        station_data_dir=Path(station_data_dir).expanduser().resolve(),
        project_root=Path(project_root).expanduser().resolve(),
    )


def resolve_result_root(
    value: str | Path | None,
    *,
    start: str | Path | None = None,
) -> Path:
    """Resolve an explicit result root or the workspace-canonical default."""
    if value is not None:
        return Path(value).expanduser().resolve()
    return load_station_data_paths(start).results


def resolve_review_manifest(
    value: str | Path | None,
    *,
    start: str | Path | None = None,
    default_name: str = "review_manifest.csv",
) -> Path:
    """Resolve an explicit review manifest or its station-local default."""
    if value is not None:
        return Path(value).expanduser().resolve()
    if Path(default_name).name != default_name:
        raise ValueError("Default review manifest name must be a file name.")
    return load_station_data_paths(start).review_root / default_name
