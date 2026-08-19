#!/usr/bin/env python3
"""Read-only RC-1 historical cleanup audit CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.historical_cleanup import (
        HistoricalCleanupAnalyzer,
        HistoricalCleanupError,
        cleanup_audit_exit_code,
        write_cleanup_audit,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from historical_cleanup import (  # type: ignore[no-redef]
        HistoricalCleanupAnalyzer,
        HistoricalCleanupError,
        cleanup_audit_exit_code,
        write_cleanup_audit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify historical review blockers without modifying source data."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--conflict-report",
        action="append",
        default=[],
        type=Path,
        help="Optional Phase 1A conflict report; repeat for multiple reports.",
    )
    parser.add_argument("--output", type=Path, help="Optional atomic JSON report path.")
    parser.add_argument("--operator", default="dry-run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        analysis = HistoricalCleanupAnalyzer().analyze(
            args.manifest,
            conflict_report_paths=args.conflict_report,
            operator=args.operator,
        )
        payload = analysis.to_dict()
        if args.output:
            report_path = write_cleanup_audit(analysis, args.output)
            payload["report_path"] = str(report_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return cleanup_audit_exit_code(analysis)
    except (HistoricalCleanupError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"error": str(exc), "mutation_performed": False},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
