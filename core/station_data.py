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
        "Result",
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
        if top_level in _MUTABLE_TOP_LEVEL_NAMES:
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


def station_data_paths_from_workspace(workspace: WorkspacePaths) -> StationDataPaths:
    root = workspace.station_data.resolve()
    return StationDataPaths(
        source_root=workspace.inference_project.resolve(),
        root=root,
        artifacts_root=workspace.inference_artifacts.resolve(),
        models=workspace.inference_models.resolve(),
        results=root / "Result",
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
    return station_data_paths_from_workspace(load_workspace_paths(start))
