"""Centralized station-local paths separated from the source repository."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.workspace import WorkspacePaths, load_workspace_paths


@dataclass(frozen=True)
class StationDataPaths:
    """All mutable inference-station locations derived from one workspace."""

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


def station_data_paths_from_workspace(workspace: WorkspacePaths) -> StationDataPaths:
    root = workspace.station_data.resolve()
    return StationDataPaths(
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
