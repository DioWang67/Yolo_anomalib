"""Move mutable inference data out of the source repository.

The migration is conservative: only an allowlist of runtime-owned paths is
moved, model bundles remain in place, and every successful move is recorded so
the operation can be rolled back.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.path_utils import project_root
from core.station_data import station_data_paths_from_workspace
from core.workspace import load_workspace_paths

DATA_DIRECTORIES = (
    ".color_baselines",
    ".color_profiles",
    ".color_revisions",
    ".inspection_releases",
    ".processing_runs",
    ".review_repairs",
    "acceptance",
    "acceptance_reports",
    "logs",
    "Result",
)
REVIEW_FILE_PATTERNS = ("review_manifest*", ".review_manifest*")
ARTIFACT_DIRECTORIES = ("dist",)
MIGRATION_MANIFEST_NAME = "migration_manifest.json"


class StationDataMigrationError(RuntimeError):
    """Raised when a safe, reversible migration cannot be completed."""


@dataclass(frozen=True)
class MigrationEntry:
    kind: str
    source: Path
    destination: Path

    def to_manifest_dict(self) -> dict[str, str]:
        payload = asdict(self)
        return {key: str(value) for key, value in payload.items()}


def build_migration_plan(
    source_root: str | Path,
    station_root: str | Path,
    artifacts_root: str | Path,
) -> tuple[MigrationEntry, ...]:
    """Return deterministic allowlisted moves without changing the filesystem."""
    source = Path(source_root).expanduser().resolve()
    station = Path(station_root).expanduser().resolve()
    artifacts = Path(artifacts_root).expanduser().resolve()
    if source in {station, artifacts}:
        raise StationDataMigrationError(
            "Station data and release artifacts must be outside the source root."
        )

    entries: list[MigrationEntry] = []
    for name in DATA_DIRECTORIES:
        candidate = source / name
        if candidate.exists():
            entries.append(MigrationEntry("station_data", candidate, station / name))

    review_files: set[Path] = set()
    for pattern in REVIEW_FILE_PATTERNS:
        review_files.update(path for path in source.glob(pattern) if path.is_file())
    entries.extend(
        MigrationEntry("station_data", path, station / path.name)
        for path in sorted(review_files, key=lambda item: item.name.casefold())
    )

    for name in ARTIFACT_DIRECTORIES:
        candidate = source / name
        if candidate.exists():
            entries.append(MigrationEntry("release_artifact", candidate, artifacts / name))
    return tuple(entries)


def execute_migration(
    entries: Sequence[MigrationEntry],
    *,
    manifest_path: str | Path,
) -> Path:
    """Move every entry and write a rollback manifest, reverting on failure."""
    manifest = Path(manifest_path).expanduser().resolve()
    normalized = tuple(entries)
    _validate_plan(normalized)
    if manifest.exists():
        raise StationDataMigrationError(
            f"Migration manifest already exists; rollback or archive it first: {manifest}"
        )

    moved: list[MigrationEntry] = []
    try:
        for entry in normalized:
            entry.destination.parent.mkdir(parents=True, exist_ok=True)
            entry.source.replace(entry.destination)
            moved.append(entry)
        _write_manifest(manifest, moved)
    except OSError as exc:
        rollback_errors = _rollback_entries(moved)
        detail = f"; rollback errors: {'; '.join(rollback_errors)}" if rollback_errors else ""
        raise StationDataMigrationError(f"Migration failed: {exc}{detail}") from exc
    return manifest


def rollback_migration(manifest_path: str | Path) -> tuple[MigrationEntry, ...]:
    """Restore all entries recorded by a completed migration manifest."""
    manifest = Path(manifest_path).expanduser().resolve()
    entries = _read_manifest(manifest)
    errors = _rollback_entries(entries)
    if errors:
        raise StationDataMigrationError("Rollback failed: " + "; ".join(errors))
    manifest.unlink()
    return entries


def _validate_plan(entries: Sequence[MigrationEntry]) -> None:
    destinations: set[Path] = set()
    for entry in entries:
        if not entry.source.exists():
            raise StationDataMigrationError(f"Migration source is missing: {entry.source}")
        if entry.destination.exists():
            raise StationDataMigrationError(
                f"Migration destination already exists: {entry.destination}"
            )
        if entry.destination in destinations:
            raise StationDataMigrationError(
                f"Duplicate migration destination: {entry.destination}"
            )
        destinations.add(entry.destination)


def _write_manifest(path: Path, entries: Sequence[MigrationEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entries": [entry.to_manifest_dict() for entry in entries],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_manifest(path: Path) -> tuple[MigrationEntry, ...]:
    if not path.is_file():
        raise StationDataMigrationError(f"Migration manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StationDataMigrationError(f"Migration manifest is invalid: {path}") from exc
    if payload.get("schema_version") != 1 or not isinstance(payload.get("entries"), list):
        raise StationDataMigrationError(f"Unsupported migration manifest: {path}")
    try:
        return tuple(
            MigrationEntry(
                kind=str(item["kind"]),
                source=Path(item["source"]).resolve(),
                destination=Path(item["destination"]).resolve(),
            )
            for item in payload["entries"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StationDataMigrationError(f"Migration manifest entries are invalid: {path}") from exc


def _rollback_entries(entries: Sequence[MigrationEntry]) -> list[str]:
    errors: list[str] = []
    for entry in reversed(tuple(entries)):
        if not entry.destination.exists():
            errors.append(f"destination missing: {entry.destination}")
            continue
        if entry.source.exists():
            errors.append(f"source already exists: {entry.source}")
            continue
        try:
            entry.source.parent.mkdir(parents=True, exist_ok=True)
            entry.destination.replace(entry.source)
        except OSError as exc:
            errors.append(f"{entry.destination}: {exc}")
    return errors


@contextmanager
def migration_lock(workspace_root: str | Path) -> Iterator[None]:
    """Prevent two station migrations from moving the same paths concurrently."""
    lock_path = Path(workspace_root).expanduser().resolve() / ".station-data-migration.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise StationDataMigrationError(f"Another station migration is active: {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply the displayed migration plan.")
    mode.add_argument("--rollback", action="store_true", help="Restore paths from the migration manifest.")
    parser.add_argument("--source-root", type=Path, default=project_root())
    return parser


def main() -> int:
    args = _parser().parse_args()
    source_root = args.source_root.expanduser().resolve()
    workspace = load_workspace_paths(source_root)
    paths = station_data_paths_from_workspace(workspace)
    manifest = paths.root / MIGRATION_MANIFEST_NAME
    with migration_lock(workspace.root):
        if args.rollback:
            entries = rollback_migration(manifest)
            print(f"Rolled back {len(entries)} path(s).")
            return 0
        entries = build_migration_plan(source_root, paths.root, paths.artifacts_root)
        for entry in entries:
            print(f"[{entry.kind}] {entry.source} -> {entry.destination}")
        if not args.apply:
            print(f"Dry run only: {len(entries)} path(s). Use --apply to migrate.")
            return 0
        execute_migration(entries, manifest_path=manifest)
        print(f"Migrated {len(entries)} path(s). Rollback manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
