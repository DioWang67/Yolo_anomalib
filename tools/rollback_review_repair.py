#!/usr/bin/env python3
"""Rollback an applied review repair using its exact-byte backup."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    repair = importlib.import_module("tools.review_repair")
    try:
        report = repair.rollback_repair(
            args.report,
            audit_log_path=args.audit_log,
        )
    except repair.RepairPlanBlockedError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    except repair.ReviewRepairError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
