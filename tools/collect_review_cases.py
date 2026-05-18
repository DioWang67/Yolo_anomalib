from __future__ import annotations

"""Collect production review cases from saved inspection metadata.

The tool scans Result/**/metadata/*_config_snapshot.json files and writes a
CSV/JSON manifest for human review. It intentionally stays filesystem-based so
the workflow remains easy to run on an offline inspection machine.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REVIEW_LABELS = ("", "confirmed_ng", "false_positive", "false_negative", "uncertain")
FAIL_STATUSES = {"FAIL", "DETECTION_FAIL", "ERROR", "INFERENCE_ERROR"}


@dataclass(frozen=True)
class ReviewCase:
    """One inspection case prepared for manual review.

    Args:
        timestamp: Inspection timestamp from the config snapshot.
        product: Product name.
        area: Area name.
        status: Final inspection status.
        detector: Detector name.
        decision_reasons: Pipe-delimited decision reason codes.
        model_version: Parsed model version when available.
        weights: Model weights path.
        inference_time: Inference time in seconds, if recorded.
        config_snapshot_path: Path to the snapshot JSON.
        annotated_path: Best-effort path to the annotated image.
        failure_crop_paths: Pipe-delimited NG crop paths.
        review_label: Empty field for human labeling.
        review_note: Empty field for human notes.
    """

    timestamp: str
    product: str
    area: str
    status: str
    detector: str
    decision_reasons: str
    model_version: str
    weights: str
    inference_time: str
    config_snapshot_path: str
    annotated_path: str
    failure_crop_paths: str
    review_label: str = ""
    review_note: str = ""


def collect_review_cases(
    result_root: str | Path,
    *,
    include_pass: bool = False,
) -> list[ReviewCase]:
    """Collect review cases from result metadata snapshots.

    Args:
        result_root: Root directory containing dated inspection outputs.
        include_pass: Include PASS cases as well as failure cases.

    Returns:
        Sorted list of ReviewCase records.
    """
    root = Path(result_root)
    if not root.exists():
        return []

    cases: list[ReviewCase] = []
    for snapshot_path in sorted(root.rglob("*_config_snapshot.json")):
        snapshot = _load_json(snapshot_path)
        if snapshot is None:
            continue
        status = str(snapshot.get("status") or "").upper()
        if not include_pass and status not in FAIL_STATUSES:
            continue

        detector = str(snapshot.get("detector") or "")
        product = str(snapshot.get("product") or "")
        area = str(snapshot.get("area") or "")
        decision = snapshot.get("decision") if isinstance(snapshot.get("decision"), dict) else {}
        model_info = snapshot.get("model_info") if isinstance(snapshot.get("model_info"), dict) else {}

        cases.append(
            ReviewCase(
                timestamp=str(snapshot.get("timestamp") or ""),
                product=product,
                area=area,
                status=status,
                detector=detector,
                decision_reasons="|".join(str(item) for item in decision.get("reasons", []) or []),
                model_version=str(model_info.get("model_version") or ""),
                weights=str(model_info.get("weights") or ""),
                inference_time=_format_inference_time(snapshot.get("inference_time")),
                config_snapshot_path=str(snapshot_path),
                annotated_path=str(_find_annotated_path(snapshot_path, detector, product, area)),
                failure_crop_paths="|".join(
                    str(path) for path in _find_failure_crop_paths(snapshot_path, detector)
                ),
            )
        )

    cases.sort(key=lambda item: (item.timestamp, item.product, item.area, item.config_snapshot_path))
    return cases


def write_manifest(
    cases: list[ReviewCase],
    output_csv: str | Path,
    output_json: str | Path | None = None,
) -> None:
    """Write review cases to CSV and optionally JSON."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(case) for case in cases]
    fieldnames = list(ReviewCase.__dataclass_fields__.keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if output_json is not None:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _format_inference_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def _find_annotated_path(
    snapshot_path: Path,
    detector: str,
    product: str,
    area: str,
) -> Path | str:
    base_path = _inspection_base_path(snapshot_path)
    if base_path is None:
        return ""
    detector_prefix = detector.lower()
    stem = snapshot_path.name.removesuffix("_config_snapshot.json")
    annotated_dir = base_path / "annotated" / detector_prefix
    candidates = [
        annotated_dir / f"{stem}.jpg",
        annotated_dir / f"{stem}.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    prefix = f"{detector_prefix}_{product}_{area}_"
    matches = sorted(annotated_dir.glob(f"{prefix}*.jpg")) + sorted(
        annotated_dir.glob(f"{prefix}*.png")
    )
    return matches[0] if matches else ""


def _find_failure_crop_paths(snapshot_path: Path, detector: str) -> list[Path]:
    base_path = _inspection_base_path(snapshot_path)
    if base_path is None:
        return []
    crop_dir = base_path / "cropped" / detector.lower()
    if not crop_dir.exists():
        return []
    return sorted(crop_dir.glob("*_NG_*.png"))


def _inspection_base_path(snapshot_path: Path) -> Path | None:
    # .../<status>/metadata/<detector>/<file> -> .../<status>
    try:
        return snapshot_path.parents[2]
    except IndexError:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", default="Result", help="Root result directory")
    parser.add_argument("--output-csv", default="review_manifest.csv", help="Output CSV path")
    parser.add_argument("--output-json", default="review_manifest.json", help="Output JSON path")
    parser.add_argument(
        "--include-pass",
        action="store_true",
        help="Include PASS cases in addition to failure cases",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    cases = collect_review_cases(args.result_root, include_pass=args.include_pass)
    write_manifest(cases, args.output_csv, args.output_json)
    print(f"Wrote {len(cases)} review cases to {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON manifest to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
