from __future__ import annotations

"""Single-entry PCBA pilot helper for operators.

This wrapper keeps the production tools available behind short commands:

    python tools/pcba_pilot.py readiness A
    python tools/pcba_pilot.py collect --include-pass
    python tools/pcba_pilot.py summary A
    python tools/pcba_pilot.py pilot A --include-pass
    python tools/pcba_pilot.py metrics
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.collect_review_cases import collect_review_cases, write_manifest
from tools.inspection_metrics import (
    compute_report,
    load_manifest_rows,
    render_console,
    write_report as write_metrics_report,
)
from tools.pilot_acceptance_report import build_acceptance_summary, write_summary
from tools.production_readiness_check import (
    has_blocking_failures,
    run_readiness_checks,
    write_report,
)


DEFAULT_PRODUCT = "PCBA1"


def default_config_path(product: str, area: str) -> Path:
    """Return the conventional model config path for one product/area."""
    return Path("models") / product / area / "yolo" / "config.yaml"


def default_readiness_report_path(area: str) -> Path:
    """Return the conventional readiness report output path."""
    return Path(f"readiness_report_{area}.json")


def default_summary_json_path(area: str) -> Path:
    """Return the conventional pilot summary JSON output path."""
    return Path(f"pilot_acceptance_summary_{area}.json")


def default_summary_md_path(area: str) -> Path:
    """Return the conventional pilot summary Markdown output path."""
    return Path(f"pilot_acceptance_summary_{area}.md")


def run_readiness_command(args: argparse.Namespace) -> int:
    """Run the readiness gate with operator-friendly defaults."""
    product = args.product
    area = _resolve_area(args)
    config_path = Path(args.config) if args.config else default_config_path(product, area)
    output_json = Path(args.output_json) if args.output_json else default_readiness_report_path(area)

    checks = run_readiness_checks(config_path, product=product, area=area)
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.message}")
    write_report(checks, output_json)
    print(f"Wrote readiness report to {output_json}")
    return 1 if has_blocking_failures(checks) else 0


def run_collect_command(args: argparse.Namespace) -> int:
    """Collect review cases with operator-friendly defaults."""
    cases = collect_review_cases(args.result_root, include_pass=args.include_pass)
    write_manifest(cases, args.output_csv, args.output_json)
    print(f"Wrote {len(cases)} review cases to {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON manifest to {args.output_json}")
    if not cases:
        print("No review cases found. Run inference first, or use --include-pass for golden board review.")
    return 0


def run_summary_command(args: argparse.Namespace) -> int:
    """Build the pilot acceptance summary with operator-friendly defaults."""
    product = args.product
    area = _resolve_area(args)
    readiness_json = Path(args.readiness_json) if args.readiness_json else default_readiness_report_path(area)
    output_json = Path(args.output_json) if args.output_json else default_summary_json_path(area)
    output_md = Path(args.output_md) if args.output_md else default_summary_md_path(area)

    summary = build_acceptance_summary(
        product=product,
        area=area,
        readiness_json=readiness_json,
        review_manifest_csv=args.review_manifest_csv,
    )
    write_summary(summary, output_json=output_json, output_md=output_md)
    print(f"Recommendation: {summary.recommendation}")
    print(f"Wrote JSON summary to {output_json}")
    print(f"Wrote Markdown summary to {output_md}")
    return 0


def run_pilot_command(args: argparse.Namespace) -> int:
    """Run readiness, collect review cases, then build the summary."""
    area = _resolve_area(args)
    readiness_json = args.readiness_json or str(default_readiness_report_path(area))
    review_manifest_csv = args.output_csv

    readiness_args = argparse.Namespace(
        product=args.product,
        area=args.area,
        area_option=args.area_option,
        config=args.config,
        output_json=readiness_json,
    )
    collect_args = argparse.Namespace(
        result_root=args.result_root,
        output_csv=review_manifest_csv,
        output_json=args.review_manifest_json,
        include_pass=args.include_pass,
    )
    summary_args = argparse.Namespace(
        product=args.product,
        area=args.area,
        area_option=args.area_option,
        readiness_json=readiness_json,
        review_manifest_csv=review_manifest_csv,
        output_json=args.summary_json,
        output_md=args.summary_md,
    )

    readiness_code = run_readiness_command(readiness_args)
    run_collect_command(collect_args)
    try:
        run_summary_command(summary_args)
    except FileNotFoundError as exc:
        print(f"Summary skipped: {exc}")
    return readiness_code


def run_metrics_command(args: argparse.Namespace) -> int:
    """Compute trust metrics (confusion matrix, escape/overkill) from labels."""
    rows = load_manifest_rows(args.review_manifest_csv)
    report = compute_report(rows)
    print(render_console(report))
    output_json = None if args.no_json else args.output_json
    write_metrics_report(report, output_json)
    if output_json:
        print(f"\nWrote metrics report to {output_json}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    readiness = subparsers.add_parser("readiness", help="Run readiness gate")
    _add_product_area_args(readiness)
    readiness.add_argument("--config", default=None, help="Override config path")
    readiness.add_argument("--output-json", default=None, help="Readiness report path")
    readiness.set_defaults(func=run_readiness_command)

    collect = subparsers.add_parser("collect", help="Collect review manifest from Result/")
    collect.add_argument("--result-root", default="Result", help="Root result directory")
    collect.add_argument("--output-csv", default="review_manifest.csv", help="Review manifest CSV path")
    collect.add_argument("--output-json", default="review_manifest.json", help="Review manifest JSON path")
    collect.add_argument("--include-pass", action="store_true", help="Include PASS cases for golden board review")
    collect.set_defaults(func=run_collect_command)

    summary = subparsers.add_parser("summary", help="Build pilot acceptance summary")
    _add_product_area_args(summary)
    summary.add_argument("--readiness-json", default=None, help="Readiness report JSON path")
    summary.add_argument("--review-manifest-csv", default="review_manifest.csv", help="Review manifest CSV path")
    summary.add_argument("--output-json", default=None, help="Pilot summary JSON path")
    summary.add_argument("--output-md", default=None, help="Pilot summary Markdown path")
    summary.set_defaults(func=run_summary_command)

    pilot = subparsers.add_parser("pilot", help="Run readiness, collect, and summary")
    _add_product_area_args(pilot)
    pilot.add_argument("--config", default=None, help="Override config path")
    pilot.add_argument("--readiness-json", default=None, help="Readiness report path")
    pilot.add_argument("--result-root", default="Result", help="Root result directory")
    pilot.add_argument("--output-csv", default="review_manifest.csv", help="Review manifest CSV path")
    pilot.add_argument("--review-manifest-json", default="review_manifest.json", help="Review manifest JSON path")
    pilot.add_argument("--summary-json", default=None, help="Pilot summary JSON path")
    pilot.add_argument("--summary-md", default=None, help="Pilot summary Markdown path")
    pilot.add_argument("--include-pass", action="store_true", help="Include PASS cases for golden board review")
    pilot.set_defaults(func=run_pilot_command)

    metrics = subparsers.add_parser(
        "metrics", help="Confusion matrix + escape/overkill from labeled manifest"
    )
    metrics.add_argument(
        "--review-manifest-csv", default="review_manifest.csv", help="Labeled review manifest CSV path"
    )
    metrics.add_argument(
        "--output-json", default="inspection_metrics.json", help="Metrics report JSON path"
    )
    metrics.add_argument("--no-json", action="store_true", help="Do not write the JSON report")
    metrics.set_defaults(func=run_metrics_command)

    return parser


def _add_product_area_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("area", nargs="?", help="Area, for example A or B")
    parser.add_argument("--area", dest="area_option", default=None, help="Area override")
    parser.add_argument("--product", default=DEFAULT_PRODUCT, help=f"Product name, default {DEFAULT_PRODUCT}")


def _resolve_area(args: argparse.Namespace) -> str:
    area = str(args.area_option or args.area or "").strip()
    if not area:
        raise SystemExit("area is required, for example: readiness A")
    return area


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
