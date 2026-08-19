"""Build a fail-closed pre-pilot screening summary from existing evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.collect_review_cases import FAIL_STATUSES
from tools.inspection_metrics import build_confusion_matrix

PRE_PILOT_EVIDENCE_KIND = "pre_pilot_screening"
READY_RECOMMENDATIONS = frozenset({"READY_TO_START_SUPERVISED_PILOT"})
REQUIRED_READINESS_CHECK_NAMES = frozenset(
    {
        "config_exists",
        "model_config_loaded",
        "product_area",
        "weights_configured",
        "weights_exists",
        "expected_items",
        "position_check_enabled",
        "expected_boxes",
        "expected_box_coverage",
        "conf_threshold_range",
        "iou_threshold_range",
        "alignment_quality_gate",
        "defect_coverage_declared",
        "save_original",
        "save_annotated",
        "save_crops",
        "output_dir",
        "fail_on_unexpected",
        "color_check_enabled",
        "missing_slot_check",
    }
)
POSITION_TOLERANCE_CHECK_NAMES = frozenset(
    {"position_iou_tolerance", "position_tolerance_percent", "position_tolerance_pixel"}
)


@dataclass(frozen=True)
class PilotAcceptanceSummary:
    """Pre-pilot screening summary; never a completed-pilot approval.

    Args:
        product: Product name being evaluated.
        area: Inspection area being evaluated.
        readiness_path: Source readiness report JSON path.
        readiness_sha256: SHA-256 of the exact parsed readiness bytes.
        review_manifest_path: Source review manifest CSV path.
        review_manifest_sha256: SHA-256 of the exact parsed manifest bytes.
        readiness_fail_count: Number of blocking readiness failures.
        readiness_warn_count: Number of readiness warnings.
        readiness_warnings: Warning names and messages.
        total_review_cases: Number of review manifest rows.
        reviewed_case_count: Rows with a recognized, internally consistent label.
        unreviewed_case_count: Rows requiring review or evidence repair.
        uncertain_case_count: Rows explicitly marked uncertain or image-quality issue.
        unknown_review_label_count: Rows with an unsupported review label.
        inconsistent_review_count: Rows whose label contradicts machine status.
        review_label_counts: Count by operator review label.
        status_counts: Count by inspection status.
        decision_reason_counts: Count by decision reason code.
        recommendation: Conservative next action.
    """

    schema_version: int
    evidence_kind: str
    product: str
    area: str
    readiness_path: str
    readiness_sha256: str
    review_manifest_path: str
    review_manifest_sha256: str
    readiness_fail_count: int
    readiness_warn_count: int
    readiness_warnings: list[str]
    total_review_cases: int
    reviewed_case_count: int
    unreviewed_case_count: int
    uncertain_case_count: int
    unknown_review_label_count: int
    inconsistent_review_count: int
    review_label_counts: dict[str, int]
    status_counts: dict[str, int]
    decision_reason_counts: dict[str, int]
    recommendation: str
    operational_acceptance_status: str
    merge_eligible: bool


def build_acceptance_summary(
    *,
    product: str,
    area: str,
    readiness_json: str | Path,
    review_manifest_csv: str | Path,
) -> PilotAcceptanceSummary:
    """Build a pilot acceptance summary from existing evidence files.

    Args:
        product: Product name being evaluated.
        area: Inspection area being evaluated.
        readiness_json: JSON written by ``production_readiness_check.py``.
        review_manifest_csv: CSV written by ``collect_review_cases.py``.

    Returns:
        A structured summary suitable for JSON or Markdown export.

    Raises:
        FileNotFoundError: If either source file does not exist.
        ValueError: If a source file has an invalid format.
    """
    readiness_path = Path(readiness_json)
    review_path = Path(review_manifest_csv)
    product = product.strip()
    area = area.strip()
    if not product or not area:
        raise ValueError("product and area are required for scoped pilot evidence")
    readiness, readiness_sha256 = _load_readiness(
        readiness_path,
        product=product,
        area=area,
    )
    review_rows, review_manifest_sha256 = _load_review_rows(
        review_path,
        product=product,
        area=area,
    )

    fail_items = [item for item in readiness if str(item.get("status") or "").upper() == "FAIL"]
    warn_items = [item for item in readiness if str(item.get("status") or "").upper() == "WARN"]

    review_label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    decision_reason_counts: Counter[str] = Counter()

    for row in review_rows:
        label = str(row.get("review_label") or "").strip().lower()
        if label:
            review_label_counts[label] += 1
        else:
            review_label_counts["unreviewed"] += 1

        status = str(row.get("status") or "UNKNOWN").strip().upper() or "UNKNOWN"
        status_counts[status] += 1

        for reason in str(row.get("decision_reasons") or "").split("|"):
            reason = reason.strip()
            if reason:
                decision_reason_counts[reason] += 1

    metric_rows: list[dict[str, Any]] = [dict(row) for row in review_rows]
    matrix = build_confusion_matrix(metric_rows)
    reviewed_count = matrix.labeled_total - matrix.inconsistent
    unreviewed_count = len(review_rows) - reviewed_count
    recommendation = _recommend(
        fail_count=len(fail_items),
        warn_count=len(warn_items),
        total_cases=len(review_rows),
        unreviewed_count=unreviewed_count,
        false_negative_count=matrix.fn,
        uncertain_count=matrix.uncertain,
        unknown_label_count=matrix.unknown_label,
        inconsistent_count=matrix.inconsistent,
    )

    return PilotAcceptanceSummary(
        schema_version=1,
        evidence_kind=PRE_PILOT_EVIDENCE_KIND,
        product=product,
        area=area,
        readiness_path=str(readiness_path.resolve(strict=False)),
        readiness_sha256=readiness_sha256,
        review_manifest_path=str(review_path.resolve(strict=False)),
        review_manifest_sha256=review_manifest_sha256,
        readiness_fail_count=len(fail_items),
        readiness_warn_count=len(warn_items),
        readiness_warnings=[f"{item.get('name', '')}: {item.get('message', '')}".strip(": ") for item in warn_items],
        total_review_cases=len(review_rows),
        reviewed_case_count=reviewed_count,
        unreviewed_case_count=unreviewed_count,
        uncertain_case_count=matrix.uncertain,
        unknown_review_label_count=matrix.unknown_label,
        inconsistent_review_count=matrix.inconsistent,
        review_label_counts=dict(sorted(review_label_counts.items())),
        status_counts=dict(sorted(status_counts.items())),
        decision_reason_counts=dict(sorted(decision_reason_counts.items())),
        recommendation=recommendation,
        operational_acceptance_status="NOT_CAPTURED",
        merge_eligible=False,
    )


def write_summary(
    summary: PilotAcceptanceSummary,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> None:
    """Atomically write a pre-pilot summary as JSON and/or Markdown."""
    _validate_summary_sources(summary)
    destinations = _validate_summary_destinations(
        summary,
        output_json=output_json,
        output_md=output_md,
    )
    if output_json is not None:
        json_path = destinations["json"]
        _write_text_atomic(
            json_path,
            json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        )

    if output_md is not None:
        md_path = destinations["markdown"]
        _write_text_atomic(md_path, _summary_to_markdown(summary))


def is_pre_pilot_ready(summary: PilotAcceptanceSummary) -> bool:
    """Return whether screening permits starting a supervised pilot."""
    return summary.recommendation in READY_RECOMMENDATIONS


def _load_readiness(
    path: Path,
    *,
    product: str,
    area: str,
) -> tuple[list[dict[str, Any]], str]:
    if not path.exists():
        raise FileNotFoundError(f"readiness report not found: {path}")
    if path.is_symlink():
        raise ValueError("readiness report cannot be a symbolic link")
    raw_report = path.read_bytes()
    data = json.loads(raw_report.decode("utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("readiness report must be a JSON list")
    if not data:
        raise ValueError("readiness report cannot be empty")
    checks: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"readiness item {index} must be a JSON object")
        name = str(item.get("name") or "").strip()
        status = str(item.get("status") or "").strip().upper()
        if not name:
            raise ValueError(f"readiness item {index} requires a name")
        if name in names:
            raise ValueError(f"readiness report contains duplicate check: {name}")
        if status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError(f"readiness check {name} has invalid status: {status or '<empty>'}")
        names.add(name)
        checks.append({**item, "name": name, "status": status})
    missing_checks = sorted(REQUIRED_READINESS_CHECK_NAMES - names)
    if missing_checks:
        raise ValueError("readiness report is incomplete; missing check(s): " + ", ".join(missing_checks))
    tolerance_checks = names & POSITION_TOLERANCE_CHECK_NAMES
    if len(tolerance_checks) != 1:
        raise ValueError("readiness report must contain exactly one position tolerance check")
    by_name = {str(item["name"]): item for item in checks}
    expected_scope = f"product={product}, area={area}"
    if str(by_name["product_area"].get("message") or "").strip() != expected_scope:
        raise ValueError(f"readiness product_area does not match requested scope: expected {expected_scope}")
    if by_name["alignment_quality_gate"]["status"] == "PASS" and ("alignment_shift_limits" not in names):
        raise ValueError(
            "readiness report is incomplete; alignment_shift_limits is required when alignment_quality_gate passes"
        )
    if by_name["defect_coverage_declared"]["status"] == "PASS" and ("defect_coverage_limitations" not in names):
        raise ValueError(
            "readiness report is incomplete; defect_coverage_limitations is required when defect coverage is declared"
        )
    color_enabled_message = str(by_name["color_check_enabled"].get("message") or "").strip()
    if color_enabled_message not in {"enabled=true", "enabled=false"}:
        raise ValueError("readiness color_check_enabled marker is invalid")
    if color_enabled_message == "enabled=true":
        color_checks = {
            "color_model_configured",
            "color_model_exists",
            "color_fail_closed",
        }
        missing_color_checks = sorted(color_checks - names)
        if missing_color_checks:
            raise ValueError(
                "readiness report is incomplete; missing enabled color check(s): " + ", ".join(missing_color_checks)
            )
    return checks, hashlib.sha256(raw_report).hexdigest()


def _load_review_rows(
    path: Path,
    *,
    product: str,
    area: str,
) -> tuple[list[dict[str, str]], str]:
    if not path.exists():
        raise FileNotFoundError(f"review manifest not found: {path}")
    if path.is_symlink():
        raise ValueError("review manifest cannot be a symbolic link")
    raw_manifest = path.read_bytes()
    with io.StringIO(raw_manifest.decode("utf-8-sig"), newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        required = {"product", "area", "status", "decision_reasons", "review_label"}
        raw_fieldnames = list(reader.fieldnames or ())
        if not raw_fieldnames:
            raise ValueError("review manifest must contain a CSV header")
        if len(set(raw_fieldnames)) != len(raw_fieldnames):
            raise ValueError("review manifest contains duplicate CSV columns")
        fieldnames = set(raw_fieldnames)
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError("review manifest is missing required column(s): " + ", ".join(missing))
        rows = []
        for index, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"review manifest row {index} contains unexpected extra columns")
            rows.append(dict(row))
    valid_statuses = {"PASS", *FAIL_STATUSES}
    for index, row in enumerate(rows, start=2):
        row_product = str(row.get("product") or "").strip()
        row_area = str(row.get("area") or "").strip()
        if row_product != product or row_area != area:
            raise ValueError(
                f"review manifest row {index} scope mismatch: "
                f"expected product={product}, area={area}; "
                f"got product={row_product or '<empty>'}, area={row_area or '<empty>'}"
            )
        status = str(row.get("status") or "").strip().upper()
        if status not in valid_statuses:
            raise ValueError(f"review manifest row {index} has invalid status: {status or '<empty>'}")
        row["status"] = status
    return rows, hashlib.sha256(raw_manifest).hexdigest()


def _recommend(
    *,
    fail_count: int,
    warn_count: int,
    total_cases: int,
    unreviewed_count: int,
    false_negative_count: int,
    uncertain_count: int,
    unknown_label_count: int,
    inconsistent_count: int,
) -> str:
    if fail_count > 0:
        return "NO_GO_FIX_READINESS_FAILS"
    if unknown_label_count > 0:
        return "NO_GO_FIX_UNKNOWN_REVIEW_LABELS"
    if inconsistent_count > 0:
        return "NO_GO_FIX_INCONSISTENT_REVIEW_LABELS"
    if false_negative_count > 0:
        return "NO_GO_INVESTIGATE_FALSE_NEGATIVES"
    if total_cases == 0:
        return "NO_GO_COLLECT_DRY_RUN_EVIDENCE"
    if uncertain_count > 0:
        return "HOLD_RESOLVE_UNCERTAIN_OPERATOR_REVIEWS"
    if unreviewed_count > 0:
        return "HOLD_COMPLETE_OPERATOR_REVIEW"
    if warn_count > 0:
        return "HOLD_DOCUMENT_READINESS_WARNING_ACCEPTANCE"
    return "READY_TO_START_SUPERVISED_PILOT"


def _summary_to_markdown(summary: PilotAcceptanceSummary) -> str:
    lines = [
        "# PCBA Pre-Pilot Screening Summary",
        "",
        "> This report does not prove completed operational acceptance and is never merge approval.",
        "",
        f"- Product: {summary.product}",
        f"- Area: {summary.area}",
        f"- Readiness report: {summary.readiness_path}",
        f"- Readiness SHA-256: {summary.readiness_sha256}",
        f"- Review manifest: {summary.review_manifest_path}",
        f"- Review manifest SHA-256: {summary.review_manifest_sha256}",
        f"- Recommendation: {summary.recommendation}",
        f"- Operational acceptance: {summary.operational_acceptance_status}",
        f"- Merge eligible: {str(summary.merge_eligible).lower()}",
        "",
        "## Readiness",
        "",
        f"- Blocking FAIL count: {summary.readiness_fail_count}",
        f"- WARN count: {summary.readiness_warn_count}",
    ]
    if summary.readiness_warnings:
        lines.append("- WARN items:")
        lines.extend(f"  - {item}" for item in summary.readiness_warnings)

    lines.extend(
        [
            "",
            "## Review",
            "",
            f"- Total review cases: {summary.total_review_cases}",
            f"- Reviewed cases: {summary.reviewed_case_count}",
            f"- Unreviewed cases: {summary.unreviewed_case_count}",
            f"- Uncertain cases: {summary.uncertain_case_count}",
            f"- Unknown review labels: {summary.unknown_review_label_count}",
            f"- Inconsistent review rows: {summary.inconsistent_review_count}",
            "",
            "### Review Labels",
            "",
        ]
    )
    lines.extend(_counter_lines(summary.review_label_counts))
    lines.extend(["", "### Inspection Status", ""])
    lines.extend(_counter_lines(summary.status_counts))
    lines.extend(["", "### Decision Reasons", ""])
    lines.extend(_counter_lines(summary.decision_reason_counts))
    lines.append("")
    return "\n".join(lines)


def _counter_lines(values: dict[str, int]) -> list[str]:
    if not values:
        return ["- none: 0"]
    return [f"- {name}: {count}" for name, count in values.items()]


def _validate_summary_destinations(
    summary: PilotAcceptanceSummary,
    *,
    output_json: str | Path | None,
    output_md: str | Path | None,
) -> dict[str, Path]:
    """Reject output collisions before either evidence file is replaced."""
    destinations: dict[str, Path] = {}
    if output_json is not None:
        json_candidate = Path(output_json).expanduser()
        if json_candidate.is_symlink():
            raise ValueError("summary JSON destination cannot be a symbolic link")
        destinations["json"] = json_candidate.resolve(strict=False)
    if output_md is not None:
        markdown_candidate = Path(output_md).expanduser()
        if markdown_candidate.is_symlink():
            raise ValueError("summary Markdown destination cannot be a symbolic link")
        destinations["markdown"] = markdown_candidate.resolve(strict=False)
    if len(set(destinations.values())) != len(destinations):
        raise ValueError("summary JSON and Markdown destinations must be different")
    if "json" in destinations and destinations["json"].suffix.lower() != ".json":
        raise ValueError("summary JSON destination must use a .json suffix")
    if "markdown" in destinations and destinations["markdown"].suffix.lower() != ".md":
        raise ValueError("summary Markdown destination must use a .md suffix")
    sources = {
        Path(summary.readiness_path).expanduser().resolve(strict=False),
        Path(summary.review_manifest_path).expanduser().resolve(strict=False),
    }
    for kind, destination in destinations.items():
        if destination in sources:
            raise ValueError(f"summary {kind} destination cannot overwrite source evidence")
    return destinations


def _validate_summary_sources(summary: PilotAcceptanceSummary) -> None:
    """Ensure source evidence is unchanged between parsing and summary write."""
    sources = (
        (Path(summary.readiness_path), summary.readiness_sha256, "readiness report"),
        (
            Path(summary.review_manifest_path),
            summary.review_manifest_sha256,
            "review manifest",
        ),
    )
    for path, expected_sha256, label in sources:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} became missing or unsafe before summary write")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError(f"{label} changed before summary write")


def _write_text_atomic(path: Path, text: str) -> None:
    if path.is_symlink():
        raise ValueError("summary destination cannot be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, help="Product name")
    parser.add_argument("--area", required=True, help="Area name")
    parser.add_argument("--readiness-json", required=True, help="Readiness report JSON path")
    parser.add_argument("--review-manifest-csv", required=True, help="Review manifest CSV path")
    parser.add_argument("--output-json", default="pilot_acceptance_summary.json", help="Output JSON path")
    parser.add_argument("--output-md", default="pilot_acceptance_summary.md", help="Output Markdown path")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    summary = build_acceptance_summary(
        product=args.product,
        area=args.area,
        readiness_json=args.readiness_json,
        review_manifest_csv=args.review_manifest_csv,
    )
    write_summary(summary, output_json=args.output_json, output_md=args.output_md)
    print(f"Recommendation: {summary.recommendation}")
    print(f"Wrote JSON summary to {args.output_json}")
    print(f"Wrote Markdown summary to {args.output_md}")
    return 0 if is_pre_pilot_ready(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
