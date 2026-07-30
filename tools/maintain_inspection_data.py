"""Back up the inspection database and enforce approved evidence retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.services.inspection_database import InspectionDatabaseManager
from core.services.inspection_maintenance import (
    InspectionMaintenanceService,
    InspectionRetentionPolicy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="Result")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete eligible image artifacts; default is a read-only plan.",
    )
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="Create and verify one SQLite backup without retention cleanup.",
    )
    parser.add_argument("--pass-days", type=int, default=30)
    parser.add_argument("--fail-preprocessed-days", type=int, default=90)
    parser.add_argument("--fail-all-days", type=int, default=180)
    parser.add_argument("--output-json", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_root = Path(args.result_root).resolve()
    database_path = result_root / "inspection_records.sqlite3"
    if args.backup_only:
        backup = InspectionDatabaseManager(database_path).backup(reason="manual_cli")
        payload = {
            "mode": "backup",
            "backup_path": str(backup.path),
            "schema_version": backup.source_schema_version,
        }
    else:
        policy = InspectionRetentionPolicy(
            pass_image_days=args.pass_days,
            fail_preprocessed_days=args.fail_preprocessed_days,
            fail_all_image_days=args.fail_all_days,
        )
        report = InspectionMaintenanceService(
            result_root,
            database_path=database_path,
            policy=policy,
        ).run(dry_run=not args.apply)
        payload = {
            "mode": "apply" if args.apply else "dry_run",
            "candidate_count": len(report.candidates),
            "deleted_files": report.deleted_files,
            "missing_files": report.missing_files,
            "reclaimed_bytes": report.reclaimed_bytes,
            "backup_path": str(report.backup_path or ""),
            "candidates": [
                {
                    "path": str(candidate.path),
                    "artifact_types": list(candidate.artifact_types),
                    "inspection_ids": list(candidate.inspection_ids),
                    "stored_paths": list(candidate.stored_paths),
                    "size_bytes": candidate.size_bytes,
                }
                for candidate in report.candidates
            ],
        }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
