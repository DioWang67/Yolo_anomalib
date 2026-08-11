"""Inspect or explicitly requeue the company synchronization outbox."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.services.inspection_sync import InspectionSyncOutbox
from core.station_data import resolve_result_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default=None)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--retry-dead",
        action="store_true",
        help="Requeue all dead-letter rows after the cause has been fixed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database_path = resolve_result_root(args.result_root) / "inspection_records.sqlite3"
    if not database_path.is_file():
        print(f"Inspection database does not exist: {database_path}")
        return 1
    outbox = InspectionSyncOutbox(database_path)
    retried = outbox.retry_dead() if args.retry_dead else 0
    status = outbox.status()
    payload = {"retried": retried, **asdict(status)}
    if args.as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(
            "Inspection sync: "
            f"pending={status.pending} inflight={status.inflight} "
            f"synced={status.synced} dead={status.dead}"
        )
        if args.retry_dead:
            print(f"Requeued dead-letter records: {retried}")
    return 2 if status.dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
