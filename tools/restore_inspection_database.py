"""Verify or restore an inspection database backup with an explicit safety gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.services.inspection_database import InspectionDatabaseManager


def restore_database(
    *,
    result_root: Path,
    backup_path: Path,
    confirm_application_closed: bool,
) -> dict[str, object]:
    result_root = result_root.resolve()
    backup_path = backup_path.resolve()
    database_path = result_root / "inspection_records.sqlite3"

    InspectionDatabaseManager(backup_path).check_integrity()
    if not confirm_application_closed:
        return {
            "mode": "verify_only",
            "backup_path": str(backup_path),
            "target_path": str(database_path),
            "backup_integrity": "ok",
        }

    manager = InspectionDatabaseManager(database_path)
    safety_backup = manager.restore(backup_path)
    manager.check_integrity()
    return {
        "mode": "restored",
        "backup_path": str(backup_path),
        "target_path": str(database_path),
        "target_integrity": "ok",
        "pre_restore_backup": (
            str(safety_backup.path) if safety_backup is not None else ""
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="Result")
    parser.add_argument("--backup", required=True)
    parser.add_argument(
        "--confirm-application-closed",
        action="store_true",
        help=(
            "Perform the restore. Without this flag the command only verifies "
            "the backup and prints the planned target."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = restore_database(
        result_root=Path(args.result_root),
        backup_path=Path(args.backup),
        confirm_application_closed=args.confirm_application_closed,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
