from __future__ import annotations

"""Build a concise pilot acceptance summary from readiness and review outputs."""

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PilotAcceptanceSummary:
    """Summarized evidence for one controlled pilot decision.

    Args:
        product: Product name being evaluated.
        area: Inspection area being evaluated.
        readiness_path: Source readiness report JSON path.
        review_manifest_path: Source review manifest CSV path.
        readiness_fail_count: Number of blocking readiness failures.
        readiness_warn_count: Number of readiness warnings.
        readiness_warnings: Warning names and messages.
        total_review_cases: Number of review manifest rows.
        reviewed_case_count: Rows with a non-empty review label.
        unreviewed_case_count: Rows without a review label.
        review_label_counts: Count by operator review label.
        status_counts: Count by inspection status.
        decision_reason_counts: Count by decision reason code.
        recommendation: Conservative next action.
    """

    product: str
    area: str
    readiness_path: str
    review_manifest_path: str
    readiness_fail_count: int
    readiness_warn_count: int
    readiness_warnings: list[str]
    total_review_cases: int
    reviewed_case_count: int
    unreviewed_case_count: int
    review_label_counts: dict[str, int]
    status_counts: dict[str, int]
    decision_reason_counts: dict[str, int]
    recommendation: str


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
    readiness = _load_readiness(readiness_path)
    review_rows = _load_review_rows(review_path)

    fail_items = [item for item in readiness if str(item.get("status") or "").upper() == "FAIL"]
    warn_items = [item for item in readiness if str(item.get("status") or "").upper() == "WARN"]

    review_label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    decision_reason_counts: Counter[str] = Counter()
    reviewed_count = 0

    for row in review_rows:
        label = str(row.get("review_label") or "").strip()
        if label:
            reviewed_count += 1
            review_label_counts[label] += 1
        else:
            review_label_counts["unreviewed"] += 1

        status = str(row.get("status") or "UNKNOWN").strip().upper() or "UNKNOWN"
        status_counts[status] += 1

        for reason in str(row.get("decision_reasons") or "").split("|"):
            reason = reason.strip()
            if reason:
                decision_reason_counts[reason] += 1

    unreviewed_count = len(review_rows) - reviewed_count
    recommendation = _recommend(
        fail_count=len(fail_items),
        warn_count=len(warn_items),
        total_cases=len(review_rows),
        unreviewed_count=unreviewed_count,
        false_negative_count=review_label_counts.get("false_negative", 0),
    )

    return PilotAcceptanceSummary(
        product=product,
        area=area,
        readiness_path=str(readiness_path),
        review_manifest_path=str(review_path),
        readiness_fail_count=len(fail_items),
        readiness_warn_count=len(warn_items),
        readiness_warnings=[
            f"{item.get('name', '')}: {item.get('message', '')}".strip(": ")
            for item in warn_items
        ],
        total_review_cases=len(review_rows),
        reviewed_case_count=reviewed_count,
        unreviewed_case_count=unreviewed_count,
        review_label_counts=dict(sorted(review_label_counts.items())),
        status_counts=dict(sorted(status_counts.items())),
        decision_reason_counts=dict(sorted(decision_reason_counts.items())),
        recommendation=recommendation,
    )


def write_summary(
    summary: PilotAcceptanceSummary,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> None:
    """Write a pilot summary as JSON and/or Markdown."""
    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(summary), handle, ensure_ascii=False, indent=2)

    if output_md is not None:
        md_path = Path(output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_summary_to_markdown(summary), encoding="utf-8")


def _load_readiness(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"readiness report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("readiness report must be a JSON list")
    return [item for item in data if isinstance(item, dict)]


def _load_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"review manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _recommend(
    *,
    fail_count: int,
    warn_count: int,
    total_cases: int,
    unreviewed_count: int,
    false_negative_count: int,
) -> str:
    if fail_count > 0:
        return "NO_GO_FIX_READINESS_FAILS"
    if false_negative_count > 0:
        return "NO_GO_INVESTIGATE_FALSE_NEGATIVES"
    if total_cases == 0:
        return "NO_GO_COLLECT_DRY_RUN_EVIDENCE"
    if unreviewed_count > 0:
        return "HOLD_COMPLETE_OPERATOR_REVIEW"
    if warn_count > 0:
        return "SUPERVISED_PILOT_WITH_ACCEPTED_WARNINGS"
    return "SUPERVISED_PILOT_READY"


def _summary_to_markdown(summary: PilotAcceptanceSummary) -> str:
    lines = [
        "# PCBA Pilot Acceptance Summary",
        "",
        f"- Product: {summary.product}",
        f"- Area: {summary.area}",
        f"- Readiness report: {summary.readiness_path}",
        f"- Review manifest: {summary.review_manifest_path}",
        f"- Recommendation: {summary.recommendation}",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
