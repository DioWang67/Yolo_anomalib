from __future__ import annotations

"""Quantify inspection trustworthiness from a labeled review manifest.

This turns the human review labels produced by ``collect_review_cases.py``
(``confirmed_ng`` / ``false_positive`` / ``false_negative`` / optional
``confirmed_ok``) into a confusion matrix and the metrics that actually
gate production use:

* **escape_rate (漏檢率)** = FN / (TP + FN) — defective units passed as OK.
  This is the safety-critical number for QC.
* **recall (檢出率)** = TP / (TP + FN)
* **precision** = TP / (TP + FP)
* **overkill_rate (過殺率)** = FP / (FP + TN) — only available once OK units
  are also labeled (see note below).

Confusion mapping (label encodes both ground truth and machine outcome):

    confirmed_ng    -> TP  (machine NG, truly NG)
    false_positive  -> FP  (machine NG, truly OK)  == 過殺
    false_negative  -> FN  (machine PASS, truly NG) == 漏檢
    confirmed_ok    -> TN  (machine PASS, truly OK)
    uncertain/blank -> excluded (reported separately)

NOTE on 漏檢: a false_negative case has machine status PASS, so it only
appears in the manifest when it was collected with ``--include-pass``.
Without PASS cases in the manifest you cannot measure escape rate.
"""

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.collect_review_cases import FAIL_STATUSES

# Label -> confusion cell. Extra TN aliases are accepted so the review
# vocabulary can grow to capture confirmed-OK boards without code changes.
_LABEL_TO_CELL: dict[str, str] = {
    "confirmed_ng": "tp",
    "false_positive": "fp",
    "false_negative": "fn",
    "confirmed_ok": "tn",
    "confirmed_pass": "tn",
    "true_negative": "tn",
}
_EXCLUDED_LABELS = {"uncertain"}

# Cells whose machine decision was "flagged NG"; used for status cross-checks.
_MACHINE_POSITIVE_CELLS = {"tp", "fp"}


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion counts plus excluded/contradictory bookkeeping."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    uncertain: int = 0
    unlabeled: int = 0
    unknown_label: int = 0
    inconsistent: int = 0

    @property
    def labeled_total(self) -> int:
        """Cases that landed in a confusion cell (tp+fp+fn+tn)."""
        return self.tp + self.fp + self.fn + self.tn


@dataclass(frozen=True)
class InspectionMetrics:
    """Trust metrics for one scope (overall or a single product/area)."""

    scope: str
    product: str
    area: str
    matrix: ConfusionMatrix
    escape_rate: float | None
    recall: float | None
    precision: float | None
    overkill_rate: float | None
    f1: float | None
    notes: list[str] = field(default_factory=list)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    """Return ``numerator / denominator`` rounded, or None when undefined."""
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _classify(label: str, status: str) -> tuple[str, bool]:
    """Map a review label to a confusion cell.

    Returns ``(cell, is_inconsistent)`` where *cell* is one of
    ``tp/fp/fn/tn/uncertain/unlabeled/unknown_label`` and *is_inconsistent*
    flags a label whose machine status contradicts the label semantics
    (e.g. ``confirmed_ng`` on a PASS row).
    """
    normalized = label.strip().lower()
    if not normalized:
        return "unlabeled", False
    if normalized in _EXCLUDED_LABELS:
        return "uncertain", False
    cell = _LABEL_TO_CELL.get(normalized)
    if cell is None:
        return "unknown_label", False

    machine_flagged = status.strip().upper() in FAIL_STATUSES
    expected_flagged = cell in _MACHINE_POSITIVE_CELLS
    is_inconsistent = machine_flagged != expected_flagged
    return cell, is_inconsistent


def build_confusion_matrix(rows: list[dict[str, Any]]) -> ConfusionMatrix:
    """Aggregate manifest rows into a single confusion matrix."""
    counts: dict[str, int] = defaultdict(int)
    inconsistent = 0
    for row in rows:
        cell, bad = _classify(
            str(row.get("review_label") or ""),
            str(row.get("status") or ""),
        )
        counts[cell] += 1
        if bad:
            inconsistent += 1
    return ConfusionMatrix(
        tp=counts["tp"],
        fp=counts["fp"],
        fn=counts["fn"],
        tn=counts["tn"],
        uncertain=counts["uncertain"],
        unlabeled=counts["unlabeled"],
        unknown_label=counts["unknown_label"],
        inconsistent=inconsistent,
    )


def metrics_from_matrix(
    matrix: ConfusionMatrix, *, scope: str, product: str = "", area: str = ""
) -> InspectionMetrics:
    """Derive trust metrics from a confusion matrix, with caveats as notes."""
    escape_rate = _safe_ratio(matrix.fn, matrix.tp + matrix.fn)
    recall = _safe_ratio(matrix.tp, matrix.tp + matrix.fn)
    precision = _safe_ratio(matrix.tp, matrix.tp + matrix.fp)
    overkill_rate = _safe_ratio(matrix.fp, matrix.fp + matrix.tn)

    f1: float | None = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    notes: list[str] = []
    if matrix.fn == 0 and matrix.tn == 0:
        notes.append(
            "No PASS cases labeled: escape_rate/overkill_rate cannot be trusted. "
            "Collect the manifest with --include-pass and label confirmed_ok / "
            "false_negative boards."
        )
    if matrix.tn == 0 and matrix.fp > 0:
        notes.append(
            "overkill_rate needs confirmed_ok (TN) labels; only false_positive "
            "(FP) counts are available."
        )
    if matrix.inconsistent > 0:
        notes.append(
            f"{matrix.inconsistent} row(s) have a label contradicting their "
            "machine status (data quality issue)."
        )
    if matrix.unknown_label > 0:
        notes.append(
            f"{matrix.unknown_label} row(s) have an unrecognized review_label."
        )

    return InspectionMetrics(
        scope=scope,
        product=product,
        area=area,
        matrix=matrix,
        escape_rate=escape_rate,
        recall=recall,
        precision=precision,
        overkill_rate=overkill_rate,
        f1=f1,
        notes=notes,
    )


def compute_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build overall + per (product, area) metrics from manifest rows."""
    overall = metrics_from_matrix(build_confusion_matrix(rows), scope="overall")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("product") or ""), str(row.get("area") or ""))
        grouped[key].append(row)

    per_station = [
        metrics_from_matrix(
            build_confusion_matrix(group),
            scope=f"{product or '?'}/{area or '?'}",
            product=product,
            area=area,
        )
        for (product, area), group in sorted(grouped.items())
    ]
    return {
        "total_rows": len(rows),
        "overall": _metrics_to_dict(overall),
        "per_station": [_metrics_to_dict(m) for m in per_station],
    }


def _metrics_to_dict(metrics: InspectionMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["matrix"]["labeled_total"] = metrics.matrix.labeled_total
    return payload


def load_manifest_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a review manifest CSV into a list of row dicts."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"review manifest not found: {manifest_path}")
    # utf-8-sig strips the BOM Excel adds when operators edit/save the manifest.
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def render_console(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary."""
    lines: list[str] = []
    lines.append(f"Inspection trust metrics (rows={report['total_rows']})")
    blocks = [report["overall"], *report["per_station"]]
    for block in blocks:
        m = block["matrix"]
        lines.append("")
        lines.append(f"[{block['scope']}]")
        lines.append(
            f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} "
            f"(uncertain={m['uncertain']}, unlabeled={m['unlabeled']})"
        )
        lines.append(
            f"  escape_rate={_fmt_rate(block['escape_rate'])}  "
            f"recall={_fmt_rate(block['recall'])}  "
            f"precision={_fmt_rate(block['precision'])}  "
            f"overkill_rate={_fmt_rate(block['overkill_rate'])}"
        )
        for note in block["notes"]:
            lines.append(f"  ! {note}")
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_json: str | Path | None) -> None:
    """Persist the metrics report as JSON when a path is given."""
    if output_json is None:
        return
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-manifest-csv",
        default="review_manifest.csv",
        help="Labeled review manifest CSV from collect_review_cases.py",
    )
    parser.add_argument(
        "--output-json",
        default="inspection_metrics.json",
        help="Output metrics JSON path (omit with --no-json to skip)",
    )
    parser.add_argument(
        "--no-json", action="store_true", help="Do not write the JSON report"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    rows = load_manifest_rows(args.review_manifest_csv)
    report = compute_report(rows)
    print(render_console(report))
    output_json = None if args.no_json else args.output_json
    write_report(report, output_json)
    if output_json:
        print(f"\nWrote metrics report to {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
