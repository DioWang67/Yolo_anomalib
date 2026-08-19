#!/usr/bin/env python3
"""Generate a draft, non-mutating repair plan for one review manifest."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--conflict-report",
        action="append",
        type=Path,
        default=[],
        help="Optional Phase 1A conflict report; may be repeated",
    )
    parser.add_argument("--output", type=Path, help="Draft plan output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    repair = importlib.import_module("tools.review_repair")
    try:
        plan = repair.generate_repair_plan(
            args.manifest,
            conflict_report_paths=args.conflict_report,
        )
        destination = args.output or repair.default_plan_path(plan)
        path = repair.write_repair_plan(plan, destination)
    except repair.ReviewRepairError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"plan_path": str(path), **plan}, ensure_ascii=False, indent=2))
    return 2 if plan["proposal_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
