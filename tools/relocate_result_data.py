"""Relocate Result data to the workspace-configured root with rollback support."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from core.path_utils import project_root
from core.workspace import load_workspace_paths
from tools.migrate_station_data import migration_lock

RESULT_RELOCATION_SCHEMA_VERSION = 1
RESULT_RELOCATION_MANIFEST_NAME = "result_relocation_manifest.json"
_DATABASE_PATH_COLUMNS = {
    "inspections": (
        "snapshot_path",
        "original_path",
        "preprocessed_path",
        "annotated_path",
        "heatmap_path",
        "crop_paths_json",
        "mask_paths_json",
    ),
    "inspection_artifacts": ("path",),
}
_DATABASE_SIDECAR_SUFFIXES = ("-shm", "-wal")
_ACTIVE_DATABASE_SIDECAR_NAMES = frozenset(
    f"inspection_records.sqlite3{suffix}" for suffix in _DATABASE_SIDECAR_SUFFIXES
)


class ResultRelocationError(RuntimeError):
    """Raised when Result relocation cannot finish or roll back safely."""


@dataclass(frozen=True)
class ResultFileEntry:
    relative_path: str
    size: int
    source_sha256: str

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True)
class MetadataBackup:
    original_path: str
    backup_path: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_result_relocation_plan(
    source_root: str | Path,
    destination_root: str | Path,
) -> tuple[ResultFileEntry, ...]:
    """Hash every source file and return a deterministic relocation plan."""
    source, destination = _validated_roots(source_root, destination_root)
    if not source.is_dir():
        raise ResultRelocationError(f"Result source does not exist: {source}")
    if destination.exists() and any(destination.iterdir()):
        raise ResultRelocationError(f"Result destination is not empty: {destination}")
    entries: list[ResultFileEntry] = []
    for path in sorted(
        (
            candidate
            for candidate in source.rglob("*")
            if candidate.is_file()
            and not (
                candidate.parent == source
                and candidate.name in _ACTIVE_DATABASE_SIDECAR_NAMES
            )
        ),
        key=lambda candidate: str(candidate.relative_to(source)).casefold(),
    ):
        relative = path.relative_to(source)
        entries.append(
            ResultFileEntry(
                relative_path=relative.as_posix(),
                size=path.stat().st_size,
                source_sha256=_sha256_file(path),
            )
        )
    return tuple(entries)


def apply_result_relocation(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    station_root: str | Path,
    manifest_path: str | Path,
    metadata_paths: tuple[str | Path, ...] = (),
) -> Path:
    """Move Result files, rewrite operational paths, and retain rollback data."""
    source, destination = _validated_roots(source_root, destination_root)
    station = Path(station_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.exists():
        raise ResultRelocationError(
            f"Result relocation manifest already exists: {manifest}"
        )
    _checkpoint_and_remove_database_sidecars(source / "inspection_records.sqlite3")
    entries = build_result_relocation_plan(source, destination)
    if not entries:
        raise ResultRelocationError(f"Result source contains no files: {source}")

    workspace = load_workspace_paths(source)
    migration_id = uuid4().hex
    backup_root = station / ".result_relocation_backups" / migration_id
    metadata = _metadata_candidates(source, metadata_paths)
    backups = _backup_metadata(metadata, backup_root)
    payload: dict[str, Any] = {
        "schema_version": RESULT_RELOCATION_SCHEMA_VERSION,
        "migration_id": migration_id,
        "status": "IN_PROGRESS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source),
        "destination_root": str(destination),
        "legacy_roots": [
            str(source),
            str(workspace.inference_project / "Result"),
        ],
        "entries": [entry.to_dict() for entry in entries],
        "metadata_backups": [backup.to_dict() for backup in backups],
    }
    try:
        _write_relocation_manifest(manifest, payload)
        _move_entries(entries, source, destination)
        _verify_entries(entries, destination)
        legacy_roots = tuple(Path(value) for value in payload["legacy_roots"])
        database_path = destination / "inspection_records.sqlite3"
        database_updates = _rewrite_database_paths(
            database_path,
            legacy_roots=legacy_roots,
            destination_root=destination,
        )
        _checkpoint_and_remove_database_sidecars(database_path)
        manifest_updates = sum(
            _rewrite_text_metadata(
                Path(backup.original_path),
                legacy_roots=legacy_roots,
                destination_root=destination,
            )
            for backup in backups
            if Path(backup.original_path) != source / "inspection_records.sqlite3"
        )
        _remove_empty_directories(source)
        payload.update(
            {
                "status": "COMPLETE",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "file_count": len(entries),
                "total_bytes": sum(entry.size for entry in entries),
                "database_path_updates": database_updates,
                "manifest_path_updates": manifest_updates,
            }
        )
        _write_relocation_manifest(manifest, payload)
    except (OSError, ResultRelocationError, sqlite3.Error, UnicodeError) as exc:
        rollback_errors = _rollback_from_payload(payload)
        if not rollback_errors:
            manifest.unlink(missing_ok=True)
            shutil.rmtree(backup_root, ignore_errors=True)
        detail = (
            f"; rollback errors: {'; '.join(rollback_errors)}"
            if rollback_errors
            else ""
        )
        raise ResultRelocationError(f"Result relocation failed: {exc}{detail}") from exc
    return manifest


def rollback_result_relocation(manifest_path: str | Path) -> int:
    """Restore files and operational metadata recorded by one migration."""
    manifest = Path(manifest_path).expanduser().resolve()
    payload = _read_manifest(manifest)
    errors = _rollback_from_payload(payload)
    if errors:
        raise ResultRelocationError("Result rollback failed: " + "; ".join(errors))
    backup_paths = [
        Path(item["backup_path"]).resolve()
        for item in payload.get("metadata_backups") or ()
    ]
    manifest.unlink()
    if backup_paths:
        shutil.rmtree(backup_paths[0].parent, ignore_errors=True)
    return len(payload["entries"])


def _validated_roots(
    source_root: str | Path,
    destination_root: str | Path,
) -> tuple[Path, Path]:
    source = Path(source_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ResultRelocationError(
            "Result source and destination must be separate, non-nested roots."
        )
    return source, destination


def _metadata_candidates(
    source: Path,
    metadata_paths: tuple[str | Path, ...],
) -> tuple[Path, ...]:
    candidates = {Path(value).expanduser().resolve() for value in metadata_paths}
    database_path = source / "inspection_records.sqlite3"
    if database_path.is_file():
        candidates.add(database_path.resolve())
    return tuple(sorted((path for path in candidates if path.is_file()), key=str))


def _backup_metadata(
    paths: tuple[Path, ...],
    backup_root: Path,
) -> tuple[MetadataBackup, ...]:
    backups: list[MetadataBackup] = []
    try:
        for index, path in enumerate(paths):
            backup = backup_root / f"{index:04d}-{path.name}"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            backups.append(MetadataBackup(str(path), str(backup)))
    except OSError:
        shutil.rmtree(backup_root, ignore_errors=True)
        raise
    return tuple(backups)


def _move_entries(
    entries: tuple[ResultFileEntry, ...],
    source: Path,
    destination: Path,
) -> None:
    for entry in entries:
        source_path = source / entry.relative_path
        destination_path = destination / entry.relative_path
        if destination_path.exists():
            raise ResultRelocationError(
                f"Result destination already exists: {destination_path}"
            )
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(destination_path)


def _verify_entries(
    entries: tuple[ResultFileEntry, ...],
    destination: Path,
) -> None:
    for entry in entries:
        path = destination / entry.relative_path
        if not path.is_file() or path.stat().st_size != entry.size:
            raise ResultRelocationError(f"Relocated Result file is incomplete: {path}")
        if _sha256_file(path) != entry.source_sha256:
            raise ResultRelocationError(f"Relocated Result checksum mismatch: {path}")


def _rewrite_database_paths(
    database_path: Path,
    *,
    legacy_roots: tuple[Path, ...],
    destination_root: Path,
) -> int:
    if not database_path.is_file():
        return 0
    updates = 0
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute("PRAGMA foreign_keys = ON")
            for table, columns in _DATABASE_PATH_COLUMNS.items():
                available = {
                    str(row[1])
                    for row in connection.execute(f'PRAGMA table_info("{table}")')
                }
                for column in columns:
                    if column not in available:
                        continue
                    rows = connection.execute(
                        f'SELECT rowid, "{column}" FROM "{table}" '
                        f'WHERE "{column}" IS NOT NULL AND "{column}" != \'\''
                    ).fetchall()
                    for rowid, raw_value in rows:
                        rewritten = _replace_legacy_roots(
                            str(raw_value),
                            legacy_roots=legacy_roots,
                            destination_root=destination_root,
                        )
                        if rewritten == raw_value:
                            continue
                        connection.execute(
                            f'UPDATE "{table}" SET "{column}"=? WHERE rowid=?',
                            (rewritten, rowid),
                        )
                        updates += 1
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]).casefold() != "ok":
                raise ResultRelocationError(
                    f"Relocated inspection database integrity failed: {integrity}"
                )
    return updates


def _checkpoint_and_remove_database_sidecars(database_path: Path) -> None:
    if not database_path.is_file():
        return
    with closing(sqlite3.connect(database_path)) as connection:
        checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        if checkpoint is not None and int(checkpoint[0]) != 0:
            raise ResultRelocationError(
                f"Inspection database is busy and cannot be moved: {database_path}"
            )
    for suffix in _DATABASE_SIDECAR_SUFFIXES:
        database_path.with_name(database_path.name + suffix).unlink(missing_ok=True)


def _rewrite_text_metadata(
    path: Path,
    *,
    legacy_roots: tuple[Path, ...],
    destination_root: Path,
) -> int:
    if not path.is_file():
        return 0
    original = path.read_text(encoding="utf-8-sig")
    rewritten = _replace_legacy_roots(
        original,
        legacy_roots=legacy_roots,
        destination_root=destination_root,
    )
    if rewritten == original:
        return 0
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(rewritten, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return 1


def _replace_legacy_roots(
    value: str,
    *,
    legacy_roots: tuple[Path, ...],
    destination_root: Path,
) -> str:
    rewritten = value
    destination_values = {
        "native": str(destination_root),
        "posix": destination_root.as_posix(),
        "json": json.dumps(str(destination_root))[1:-1],
    }
    for legacy_root in legacy_roots:
        replacements = {
            str(legacy_root): destination_values["native"],
            legacy_root.as_posix(): destination_values["posix"],
            json.dumps(str(legacy_root))[1:-1]: destination_values["json"],
        }
        for old, new in replacements.items():
            rewritten = rewritten.replace(old, new)
    return rewritten


def _rollback_from_payload(payload: dict[str, Any]) -> list[str]:
    source = Path(payload["source_root"]).resolve()
    destination = Path(payload["destination_root"]).resolve()
    entries = _entries_from_payload(payload)
    errors: list[str] = []
    for entry in reversed(entries):
        source_path = source / entry.relative_path
        destination_path = destination / entry.relative_path
        if source_path.exists():
            continue
        if not destination_path.exists():
            errors.append(f"relocated file missing: {destination_path}")
            continue
        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.replace(source_path)
        except OSError as exc:
            errors.append(f"{destination_path}: {exc}")
    for item in payload.get("metadata_backups") or ():
        original = Path(item["original_path"]).resolve()
        backup = Path(item["backup_path"]).resolve()
        if not backup.is_file():
            errors.append(f"metadata backup missing: {backup}")
            continue
        try:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, original)
        except OSError as exc:
            errors.append(f"{backup}: {exc}")
    _remove_empty_directories(destination)
    return errors


def _remove_empty_directories(root: Path) -> None:
    if not root.is_dir():
        return
    for directory, _children, _files in os.walk(root, topdown=False):
        path = Path(directory)
        try:
            path.rmdir()
        except OSError:
            continue


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ResultRelocationError(f"Result relocation manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResultRelocationError(
            f"Result relocation manifest is invalid: {path}"
        ) from exc
    if payload.get(
        "schema_version"
    ) != RESULT_RELOCATION_SCHEMA_VERSION or not isinstance(
        payload.get("entries"), list
    ):
        raise ResultRelocationError(f"Unsupported Result relocation manifest: {path}")
    expected_sha256 = str(payload.get("payload_sha256") or "")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if expected_sha256 != _canonical_sha256(body):
        raise ResultRelocationError(
            f"Result relocation manifest checksum mismatch: {path}"
        )
    _validated_roots(payload["source_root"], payload["destination_root"])
    _entries_from_payload(payload)
    return payload


def _entries_from_payload(payload: dict[str, Any]) -> tuple[ResultFileEntry, ...]:
    entries: list[ResultFileEntry] = []
    try:
        for item in payload["entries"]:
            relative = Path(str(item["relative_path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ResultRelocationError(
                    f"Unsafe Result relocation entry: {relative}"
                )
            size = int(item["size"])
            digest = str(item["source_sha256"])
            if size < 0 or len(digest) != 64:
                raise ResultRelocationError(
                    f"Invalid Result relocation entry: {relative}"
                )
            entries.append(ResultFileEntry(relative.as_posix(), size, digest))
    except (KeyError, TypeError, ValueError) as exc:
        raise ResultRelocationError("Result relocation entries are invalid") from exc
    return tuple(entries)


def _write_relocation_manifest(path: Path, payload: dict[str, Any]) -> None:
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    _write_json_atomic(
        path,
        {
            **body,
            "payload_sha256": _canonical_sha256(body),
        },
    )


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _review_metadata_paths(station_root: Path) -> tuple[Path, ...]:
    paths = set(station_root.glob("review_manifest*.csv"))
    paths.update(station_root.glob("review_manifest*.json"))
    paths.update(station_root.glob(".review_manifest*.json"))
    return tuple(sorted((path for path in paths if path.is_file()), key=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    parser.add_argument("--start", type=Path, default=project_root())
    args = parser.parse_args()

    workspace = load_workspace_paths(args.start)
    source = workspace.station_data / "Result"
    destination = workspace.inference_results
    manifest = workspace.station_data / RESULT_RELOCATION_MANIFEST_NAME
    with migration_lock(workspace.root):
        if args.rollback:
            count = rollback_result_relocation(manifest)
            print(f"Rolled back {count} Result file(s).")
            return 0
        if not args.apply:
            plan = build_result_relocation_plan(source, destination)
            print(f"Result relocation: {source} -> {destination}")
            print(
                f"Dry run only: {len(plan)} file(s), "
                f"{sum(item.size for item in plan)} byte(s)."
            )
            return 0
        apply_result_relocation(
            source,
            destination,
            station_root=workspace.station_data,
            manifest_path=manifest,
            metadata_paths=_review_metadata_paths(workspace.station_data),
        )
        print(f"Result relocation complete: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
