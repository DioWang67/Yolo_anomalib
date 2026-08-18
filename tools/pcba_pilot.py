"""Single-entry PCBA pilot helper for operators.

This wrapper keeps the production tools available behind short commands:

    python tools/pcba_pilot.py readiness A
    python tools/pcba_pilot.py collect --include-pass
    python tools/pcba_pilot.py summary A
    python tools/pcba_pilot.py pilot A --include-pass \
        --start-time <ISO-8601> --end-time <ISO-8601>
    python tools/pcba_pilot.py metrics
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.station_data import resolve_result_root, resolve_review_manifest  # noqa: E402
from tools.collect_review_cases import (  # noqa: E402
    collect_review_cases,
    normalize_time_bound,
    validate_manifest_destinations,
    write_manifest,
)
from tools.inspection_metrics import (  # noqa: E402
    compute_report,
    load_manifest_rows,
    render_console,
)
from tools.inspection_metrics import (  # noqa: E402
    write_report as write_metrics_report,
)
from tools.pilot_acceptance_report import (  # noqa: E402
    build_acceptance_summary,
    is_pre_pilot_ready,
    write_summary,
)
from tools.production_readiness_check import (  # noqa: E402
    ReadinessCheck,
    has_blocking_failures,
    run_readiness_checks,
    write_report,
)

DEFAULT_PRODUCT = "PCBA1"


@dataclass(frozen=True)
class ReadinessEvaluation:
    """One immutable readiness evaluation shared by pilot validation/write."""

    checks: tuple[ReadinessCheck, ...]
    source_paths: frozenset[Path]


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


def _scope_identifier(
    *,
    product: str | None,
    area: str | None,
    start_time: str | None,
    end_time: str | None,
) -> str:
    values = tuple(str(value or "").strip() for value in (product, area, start_time, end_time))
    digest = hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()[:12]
    labels = [re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")[:32] for value in values[:2] if value]
    scope = "_".join(labels) or "filtered"
    return f"{scope}_{digest}"


def _scoped_manifest_name(args: argparse.Namespace, *, suffix: str) -> str:
    scope_identifier = _scope_identifier(
        product=getattr(args, "product", None),
        area=getattr(args, "area", None),
        start_time=getattr(args, "start_time", None),
        end_time=getattr(args, "end_time", None),
    )
    return f"review_manifest_{scope_identifier}.{suffix}"


def _resolve_collect_outputs(
    args: argparse.Namespace,
    *,
    result_root: Path,
) -> tuple[Path, Path]:
    """Resolve a paired manifest destination without touching result evidence."""
    filters_present = any(
        str(getattr(args, name, None) or "").strip() for name in ("product", "area", "start_time", "end_time")
    )
    raw_csv = getattr(args, "output_csv", None)
    raw_json = getattr(args, "output_json", None)
    for raw_path in (raw_csv, raw_json):
        if raw_path is not None and Path(raw_path).expanduser().is_symlink():
            raise ValueError("review manifest destination cannot be a symbolic link")

    csv_default = _scoped_manifest_name(args, suffix="csv") if filters_present else "review_manifest.csv"
    output_csv = resolve_review_manifest(raw_csv, default_name=csv_default)
    output_json = resolve_review_manifest(raw_json) if raw_json is not None else output_csv.with_suffix(".json")
    validated_csv, validated_json = validate_manifest_destinations(
        output_csv,
        output_json,
        result_root=result_root,
    )
    if validated_json is None:  # pragma: no cover - output_json is always paired above
        raise AssertionError("paired review JSON destination was not resolved")
    return validated_csv, validated_json


def _validate_pilot_output_destinations(
    outputs: dict[str, str | Path],
    *,
    result_root: Path,
    protected_sources: frozenset[Path],
) -> dict[str, Path]:
    """Resolve every pilot output before writing and reject unsafe overlap."""
    expected_suffixes = {
        "readiness JSON": ".json",
        "review manifest CSV": ".csv",
        "review manifest JSON": ".json",
        "summary JSON": ".json",
        "summary Markdown": ".md",
    }
    canonical_outputs: dict[str, Path] = {}
    protected_root = result_root.expanduser().resolve(strict=False)
    canonical_sources = {source.expanduser().resolve(strict=False) for source in protected_sources}
    for label, raw_path in outputs.items():
        candidate = Path(raw_path).expanduser()
        if candidate.is_symlink():
            raise ValueError(f"{label} destination cannot be a symbolic link")
        destination = candidate.resolve(strict=False)
        expected_suffix = expected_suffixes[label]
        if destination.suffix.lower() != expected_suffix:
            raise ValueError(f"{label} destination must use a {expected_suffix} suffix")
        try:
            destination.relative_to(protected_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"pilot outputs must be outside the result root: {protected_root}")
        if destination in canonical_sources:
            raise ValueError(f"{label} destination cannot overwrite an active readiness source")
        canonical_outputs[label] = destination

    collisions: dict[Path, list[str]] = {}
    for label, destination in canonical_outputs.items():
        collisions.setdefault(destination, []).append(label)
    duplicate_labels = [labels for labels in collisions.values() if len(labels) > 1]
    if duplicate_labels:
        raise ValueError(
            "pilot output destinations must be unique: " + "; ".join(" = ".join(labels) for labels in duplicate_labels)
        )
    return canonical_outputs


def _evaluate_readiness(
    *,
    product: str,
    area: str,
    config_path: Path,
) -> ReadinessEvaluation:
    source_paths: set[Path] = set()
    checks = run_readiness_checks(
        config_path,
        product=product,
        area=area,
        source_paths=source_paths,
    )
    return ReadinessEvaluation(
        checks=tuple(checks),
        source_paths=frozenset(source_paths),
    )


def run_readiness_command(args: argparse.Namespace) -> int:
    """Run the readiness gate with operator-friendly defaults."""
    product = args.product
    area = _resolve_area(args)
    config_path = Path(args.config) if args.config else default_config_path(product, area)
    output_json = Path(args.output_json) if args.output_json else default_readiness_report_path(area)

    evaluation = getattr(args, "readiness_evaluation", None)
    if not isinstance(evaluation, ReadinessEvaluation):
        evaluation = _evaluate_readiness(
            product=product,
            area=area,
            config_path=config_path,
        )
    checks = list(evaluation.checks)
    source_paths = evaluation.source_paths
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.message}")
    write_report(checks, output_json, protected_sources=source_paths)
    print(f"Wrote readiness report to {output_json}")
    return 1 if has_blocking_failures(checks) else 0


def run_collect_command(args: argparse.Namespace) -> int:
    """Collect review cases with operator-friendly defaults."""
    result_root = resolve_result_root(args.result_root)
    output_csv, output_json = _resolve_collect_outputs(
        args,
        result_root=result_root,
    )
    cases = collect_review_cases(
        result_root,
        include_pass=args.include_pass,
        start_time=getattr(args, "start_time", None),
        end_time=getattr(args, "end_time", None),
        product=getattr(args, "product", None),
        area=getattr(args, "area", None),
        strict_evidence=bool(getattr(args, "strict_evidence", False)),
    )
    write_manifest(cases, output_csv, output_json)
    print(f"Wrote {len(cases)} review cases to {output_csv}")
    print(f"Wrote JSON manifest to {output_json}")
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
        review_manifest_csv=resolve_review_manifest(args.review_manifest_csv),
    )
    write_summary(summary, output_json=output_json, output_md=output_md)
    print(f"Recommendation: {summary.recommendation}")
    print(f"Wrote JSON summary to {output_json}")
    print(f"Wrote Markdown summary to {output_md}")
    return 0 if is_pre_pilot_ready(summary) else 1


def run_pilot_command(args: argparse.Namespace) -> int:
    """Run readiness, collect review cases, then build the summary."""
    area = _resolve_area(args)
    normalized_start = normalize_time_bound(args.start_time, field_name="start_time")
    normalized_end = normalize_time_bound(args.end_time, field_name="end_time")
    if normalized_start is None or normalized_end is None:
        raise ValueError("pilot requires both start_time and end_time")
    if normalized_start > normalized_end:
        raise ValueError("start_time must not be later than end_time")
    result_root = resolve_result_root(args.result_root)
    config_path = Path(args.config) if args.config else default_config_path(args.product, area)
    readiness_evaluation = _evaluate_readiness(
        product=args.product,
        area=area,
        config_path=config_path,
    )
    scope_identifier = _scope_identifier(
        product=args.product,
        area=area,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    readiness_json = args.readiness_json or f"readiness_report_{scope_identifier}.json"
    summary_json = args.summary_json or f"pilot_acceptance_summary_{scope_identifier}.json"
    summary_md = args.summary_md or f"pilot_acceptance_summary_{scope_identifier}.md"
    manifest_args = argparse.Namespace(
        output_csv=args.output_csv,
        output_json=args.review_manifest_json,
        product=args.product,
        area=area,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    review_manifest_csv, review_manifest_json = _resolve_collect_outputs(
        manifest_args,
        result_root=result_root,
    )
    outputs = _validate_pilot_output_destinations(
        {
            "readiness JSON": readiness_json,
            "review manifest CSV": review_manifest_csv,
            "review manifest JSON": review_manifest_json,
            "summary JSON": summary_json,
            "summary Markdown": summary_md,
        },
        result_root=result_root,
        protected_sources=readiness_evaluation.source_paths,
    )

    readiness_args = argparse.Namespace(
        product=args.product,
        area=args.area,
        area_option=args.area_option,
        config=args.config,
        output_json=outputs["readiness JSON"],
        readiness_evaluation=readiness_evaluation,
    )
    collect_args = argparse.Namespace(
        result_root=result_root,
        output_csv=outputs["review manifest CSV"],
        output_json=outputs["review manifest JSON"],
        include_pass=args.include_pass,
        product=args.product,
        area=area,
        start_time=args.start_time,
        end_time=args.end_time,
        strict_evidence=True,
    )
    summary_args = argparse.Namespace(
        product=args.product,
        area=args.area,
        area_option=args.area_option,
        readiness_json=outputs["readiness JSON"],
        review_manifest_csv=outputs["review manifest CSV"],
        output_json=outputs["summary JSON"],
        output_md=outputs["summary Markdown"],
    )

    readiness_code = run_readiness_command(readiness_args)
    run_collect_command(collect_args)
    try:
        summary_code = run_summary_command(summary_args)
    except FileNotFoundError as exc:
        print(f"Summary skipped: {exc}")
        return 1
    return 1 if readiness_code or summary_code else 0


def run_metrics_command(args: argparse.Namespace) -> int:
    """Compute trust metrics (confusion matrix, escape/overkill) from labels."""
    rows = load_manifest_rows(resolve_review_manifest(args.review_manifest_csv))
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

    collect = subparsers.add_parser("collect", help="Collect the station review manifest")
    collect.add_argument("--result-root", default=None, help="Root result directory")
    collect.add_argument("--output-csv", default=None, help="Review manifest CSV path")
    collect.add_argument("--output-json", default=None, help="Review manifest JSON path")
    collect.add_argument("--include-pass", action="store_true", help="Include PASS cases for golden board review")
    collect.add_argument("--product", default=None, help="Exact product filter")
    collect.add_argument("--area", default=None, help="Exact area filter")
    collect.add_argument("--start-time", default=None, help="Inclusive ISO-8601 lower timestamp bound")
    collect.add_argument("--end-time", default=None, help="Inclusive ISO-8601 upper timestamp bound")
    collect.add_argument(
        "--strict-evidence",
        action="store_true",
        help="Fail when any snapshot or active scope field cannot be verified",
    )
    collect.set_defaults(func=run_collect_command)

    summary = subparsers.add_parser("summary", help="Build pilot acceptance summary")
    _add_product_area_args(summary)
    summary.add_argument("--readiness-json", default=None, help="Readiness report JSON path")
    summary.add_argument("--review-manifest-csv", default=None, help="Review manifest CSV path")
    summary.add_argument("--output-json", default=None, help="Pilot summary JSON path")
    summary.add_argument("--output-md", default=None, help="Pilot summary Markdown path")
    summary.set_defaults(func=run_summary_command)

    pilot = subparsers.add_parser("pilot", help="Run readiness, collect, and summary")
    _add_product_area_args(pilot)
    pilot.add_argument("--config", default=None, help="Override config path")
    pilot.add_argument("--readiness-json", default=None, help="Readiness report path")
    pilot.add_argument("--result-root", default=None, help="Root result directory")
    pilot.add_argument("--output-csv", default=None, help="Review manifest CSV path")
    pilot.add_argument("--review-manifest-json", default=None, help="Review manifest JSON path")
    pilot.add_argument("--summary-json", default=None, help="Pilot summary JSON path")
    pilot.add_argument("--summary-md", default=None, help="Pilot summary Markdown path")
    pilot.add_argument("--include-pass", action="store_true", help="Include PASS cases for golden board review")
    pilot.add_argument(
        "--start-time",
        required=True,
        help="Inclusive ISO-8601 lower timestamp bound",
    )
    pilot.add_argument(
        "--end-time",
        required=True,
        help="Inclusive ISO-8601 upper timestamp bound",
    )
    pilot.set_defaults(func=run_pilot_command)

    metrics = subparsers.add_parser("metrics", help="Confusion matrix + escape/overkill from labeled manifest")
    metrics.add_argument("--review-manifest-csv", default=None, help="Labeled review manifest CSV path")
    metrics.add_argument("--output-json", default="inspection_metrics.json", help="Metrics report JSON path")
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
