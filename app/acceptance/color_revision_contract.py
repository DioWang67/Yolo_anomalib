"""CLI verifier for an acceptance report's active color-revision contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from core.services.color_revision_contract import (
    ColorRevisionContractError,
    verify_active_color_revision_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that active color revisions still match an acceptance report."
    )
    parser.add_argument("--revisions-root", required=True)
    parser.add_argument("--report", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_report_path = Path(args.report).expanduser()
    try:
        if raw_report_path.is_symlink():
            raise ColorRevisionContractError(
                f"Acceptance report cannot be a symbolic link: {raw_report_path}"
            )
        report_path = raw_report_path.resolve()
        if not report_path.is_file():
            raise ColorRevisionContractError(
                f"Acceptance report is missing or unsafe: {report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ColorRevisionContractError(
                "Acceptance report must contain an object."
            )
        expected = report.get("color_revisions")
        if not isinstance(expected, dict):
            raise ColorRevisionContractError(
                "Acceptance report has no color revision contract."
            )
        verified = verify_active_color_revision_contract(
            expected,
            revisions_root=args.revisions_root,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ColorRevisionContractError,
    ) as exc:
        print(f"[color-revisions] BLOCKED {exc}", file=sys.stderr, flush=True)
        return 2
    print(
        "[color-revisions] VERIFIED "
        f"identity={verified['identity_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
