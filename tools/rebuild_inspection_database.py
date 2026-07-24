"""Backfill the SQLite inspection index from existing result snapshots."""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

from core.services.inspection_repository import InspectionRepository


def rebuild_inspection_database(
    result_root: str | Path,
    *,
    database_path: str | Path | None = None,
) -> tuple[int, list[str]]:
    """Index every readable snapshot without deleting current review records."""
    root = Path(result_root).resolve()
    repository = InspectionRepository(
        database_path or root / "inspection_records.sqlite3"
    )
    indexed_count = 0
    errors: list[str] = []
    for directory, child_dirs, filenames in os.walk(
        root, onerror=lambda error: errors.append(str(error))
    ):
        child_dirs.sort()
        for filename in sorted(filenames):
            if not filename.endswith("_config_snapshot.json"):
                continue
            snapshot_path = Path(directory) / filename
            try:
                repository.upsert_snapshot_file(snapshot_path)
                indexed_count += 1
            except (OSError, ValueError, sqlite3.Error) as exc:
                errors.append(f"{snapshot_path}: {exc}")
    return indexed_count, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="Result")
    parser.add_argument("--database")
    args = parser.parse_args(argv)
    indexed_count, errors = rebuild_inspection_database(
        args.result_root,
        database_path=args.database,
    )
    print(f"Indexed {indexed_count} inspection snapshot(s).")
    for error in errors:
        print(f"WARNING: {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
