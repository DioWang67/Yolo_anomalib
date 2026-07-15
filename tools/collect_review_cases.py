"""Collect production review cases from saved inspection metadata.

The tool scans Result/**/metadata/*_config_snapshot.json files and writes a
CSV/JSON manifest for human review. It intentionally stays filesystem-based so
the workflow remains easy to run on an offline inspection machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REVIEW_LABELS = (
    "",
    "confirmed_ng",
    "verified_empty",
    "false_positive",
    "false_negative",
    "wrong_class",
    "uncertain",
    "image_quality_issue",
)
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
        training_selected: ``1`` when included in the next training submission.
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
    original_path: str
    preprocessed_path: str
    annotated_path: str
    detections_json: str
    class_names_json: str
    class_map_json: str
    failure_crop_paths: str
    review_label: str = ""
    review_note: str = ""
    training_selected: str = "1"


def collect_review_cases(
    result_root: str | Path,
    *,
    include_pass: bool = False,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
) -> list[ReviewCase]:
    """Collect review cases from result metadata snapshots.

    Args:
        result_root: Root directory containing dated inspection outputs.
        include_pass: Include PASS cases as well as failure cases.
        start_time: Optional inclusive lower timestamp bound.
        end_time: Optional inclusive upper timestamp bound.

    Returns:
        Sorted list of ReviewCase records.
    """
    root = Path(result_root)
    if not root.exists():
        return []
    normalized_start = normalize_time_bound(start_time, field_name="start_time")
    normalized_end = normalize_time_bound(end_time, field_name="end_time")
    if normalized_start and normalized_end and normalized_start > normalized_end:
        raise ValueError("start_time must not be later than end_time")

    cases: list[ReviewCase] = []
    for snapshot_path in _iter_snapshot_paths(root):
        snapshot = _load_json(snapshot_path)
        if snapshot is None:
            continue
        status = str(snapshot.get("status") or "").upper()
        if not include_pass and status not in FAIL_STATUSES:
            continue
        timestamp = str(snapshot.get("timestamp") or "")
        if not timestamp_in_range(
            timestamp, start_time=normalized_start, end_time=normalized_end
        ):
            continue

        detector = str(snapshot.get("detector") or "")
        product = str(snapshot.get("product") or "")
        area = str(snapshot.get("area") or "")
        decision = snapshot.get("decision") if isinstance(snapshot.get("decision"), dict) else {}
        model_info = snapshot.get("model_info") if isinstance(snapshot.get("model_info"), dict) else {}
        detections = snapshot.get("detections")
        if not isinstance(detections, list):
            detections = []
        preprocessed_path = _artifact_or_fallback(
            snapshot,
            "preprocessed_path",
            _find_evidence_image(
                snapshot_path,
                detector,
                product,
                area,
                directory_name="preprocessed",
            ),
        )
        detections = _attach_image_dimensions(detections, preprocessed_path)
        class_names = _normalize_class_names(model_info.get("class_names"))
        observed_class_map = _observed_class_map(detections)

        cases.append(
            ReviewCase(
                timestamp=timestamp,
                product=product,
                area=area,
                status=status,
                detector=detector,
                decision_reasons="|".join(str(item) for item in decision.get("reasons", []) or []),
                model_version=str(model_info.get("model_version") or ""),
                weights=str(model_info.get("weights") or ""),
                inference_time=_format_inference_time(snapshot.get("inference_time")),
                config_snapshot_path=str(snapshot_path),
                original_path=str(
                    _artifact_or_fallback(
                        snapshot,
                        "original_path",
                        _find_original_path(snapshot_path, detector, product, area),
                    )
                ),
                preprocessed_path=str(preprocessed_path),
                annotated_path=str(
                    _artifact_or_fallback(
                        snapshot,
                        "annotated_path",
                        _find_annotated_path(snapshot_path, detector, product, area),
                    )
                ),
                detections_json=json.dumps(
                    detections, ensure_ascii=False, separators=(",", ":")
                ),
                class_names_json=json.dumps(
                    class_names, ensure_ascii=False, separators=(",", ":")
                ),
                class_map_json=json.dumps(
                    observed_class_map, ensure_ascii=False, separators=(",", ":")
                ),
                failure_crop_paths="|".join(
                    str(path) for path in _find_failure_crop_paths(snapshot_path, detector)
                ),
            )
        )

    cases.sort(key=lambda item: (item.timestamp, item.product, item.area, item.config_snapshot_path))
    return cases


def _normalize_class_names(value: Any) -> list[str]:
    """Return an ordered class-name list from model metadata."""
    if isinstance(value, list):
        return [str(name) for name in value]
    if isinstance(value, dict):
        indexed: list[tuple[int, str]] = []
        for raw_index, raw_name in value.items():
            try:
                indexed.append((int(raw_index), str(raw_name)))
            except (TypeError, ValueError):
                return []
        return [name for _index, name in sorted(indexed)]
    return []


def _observed_class_map(detections: list[Any]) -> dict[str, str]:
    """Build the observed class-ID/name mapping for legacy snapshots."""
    observed: dict[str, str] = {}
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        try:
            class_id = int(detection["class_id"])
        except (KeyError, TypeError, ValueError):
            continue
        class_name = str(
            detection.get("class") or detection.get("class_name") or ""
        ).strip()
        if class_id >= 0 and class_name:
            observed[str(class_id)] = class_name
    return observed


def normalize_time_bound(
    value: datetime | str | None, *, field_name: str
) -> datetime | None:
    """Normalize a local/ISO timestamp bound to a timezone-naive local value.

    Args:
        value: A datetime, ISO-8601 string, or ``None``.
        field_name: Field name used in validation errors.

    Returns:
        A local timezone-naive datetime, or ``None``.

    Raises:
        ValueError: If a provided string is not a valid ISO-8601 timestamp.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid {field_name}: {value!r}") from exc
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone().replace(tzinfo=None)
    return parsed


def timestamp_in_range(
    timestamp: str,
    *,
    start_time: datetime | None,
    end_time: datetime | None,
) -> bool:
    """Return whether one inspection timestamp is inside inclusive bounds."""
    if start_time is None and end_time is None:
        return True
    try:
        observed = normalize_time_bound(timestamp, field_name="timestamp")
    except ValueError:
        return False
    if observed is None:
        return False
    if start_time is not None and observed < start_time:
        return False
    if end_time is not None and observed > end_time:
        return False
    return True


def _iter_snapshot_paths(root: Path) -> list[Path]:
    """Return snapshots while skipping directories that cannot be read."""
    paths: list[Path] = []
    for directory, child_dirs, filenames in os.walk(root, onerror=lambda _error: None):
        child_dirs.sort()
        for filename in sorted(filenames):
            if filename.endswith("_config_snapshot.json"):
                paths.append(Path(directory) / filename)
    return paths


def _artifact_or_fallback(
    snapshot: dict[str, Any], key: str, fallback: Path | str
) -> Path | str:
    """Prefer the schema-v2 artifact path when it still exists."""
    artifacts = snapshot.get("artifacts")
    if isinstance(artifacts, dict):
        value = str(artifacts.get(key) or "")
        if value and Path(value).exists():
            return Path(value)
    return fallback


def _attach_image_dimensions(
    detections: list[Any], image_path: Path | str
) -> list[dict[str, Any]]:
    """Add image dimensions needed to convert pixel boxes into YOLO labels.

    Existing schema-v2 records did not always include these values. Reading the
    saved preprocessed image keeps historical production cases exportable while
    failing safely when that evidence image is unavailable or corrupt.
    """
    normalized = [dict(item) for item in detections if isinstance(item, dict)]
    if not normalized or all(
        item.get("image_width") and item.get("image_height") for item in normalized
    ):
        return normalized
    dimensions = _read_image_dimensions(image_path)
    if dimensions is None:
        return normalized
    width, height = dimensions
    for item in normalized:
        item.setdefault("image_width", width)
        item.setdefault("image_height", height)
    return normalized


def _read_image_dimensions(image_path: Path | str) -> tuple[int, int] | None:
    """Read width and height using the inference project's OpenCV dependency."""
    path = Path(image_path)
    if not path.is_file():
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    try:
        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error):
        return None
    if image is None or image.ndim < 2:
        return None
    height, width = image.shape[:2]
    return int(width), int(height)


def write_manifest(
    cases: list[ReviewCase],
    output_csv: str | Path,
    output_json: str | Path | None = None,
) -> None:
    """Write review cases while preserving existing human review fields."""
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_reviews = _load_existing_reviews(csv_path)
    rows: list[dict[str, Any]] = []
    for case in cases:
        row = asdict(case)
        previous = existing_reviews.get(case.config_snapshot_path)
        if previous:
            row["review_label"] = previous[0]
            row["review_note"] = previous[1]
            row["training_selected"] = previous[2]
        rows.append(row)
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


def _load_existing_reviews(csv_path: Path) -> dict[str, tuple[str, str, str]]:
    """Load operator-entered fields so regenerating a manifest is idempotent."""
    if not csv_path.exists():
        return {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reviews: dict[str, tuple[str, str, str]] = {}
            for row in csv.DictReader(handle):
                key = str(row.get("config_snapshot_path") or "")
                if not key:
                    continue
                review_label = str(row.get("review_label") or "")
                reviews[key] = (
                    review_label,
                    str(row.get("review_note") or ""),
                    "0"
                    if review_label in {"uncertain", "image_quality_issue"}
                    or str(row.get("training_selected") or "1") == "0"
                    else "1",
                )
            return reviews
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}


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


def _find_original_path(
    snapshot_path: Path,
    detector: str,
    product: str,
    area: str,
) -> Path | str:
    """Return the clean source image saved for one inspection."""
    return _find_evidence_image(
        snapshot_path, detector, product, area, directory_name="original"
    )


def _find_evidence_image(
    snapshot_path: Path,
    detector: str,
    product: str,
    area: str,
    *,
    directory_name: str,
) -> Path | str:
    """Find an inspection image without assuming JPEG or PNG output."""
    base_path = _inspection_base_path(snapshot_path)
    if base_path is None:
        return ""
    detector_prefix = detector.lower()
    stem = snapshot_path.name.removesuffix("_config_snapshot.json")
    evidence_dir = base_path / directory_name / detector_prefix
    for extension in (".jpg", ".png", ".bmp", ".jpeg"):
        candidate = evidence_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    prefix = f"{detector_prefix}_{product}_{area}_"
    matches = sorted(
        path
        for path in evidence_dir.glob(f"{prefix}*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
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
    parser.add_argument("--start-time", help="Inclusive ISO-8601 lower bound")
    parser.add_argument("--end-time", help="Inclusive ISO-8601 upper bound")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = build_arg_parser().parse_args(argv)
    cases = collect_review_cases(
        args.result_root,
        include_pass=args.include_pass,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    write_manifest(cases, args.output_csv, args.output_json)
    print(f"Wrote {len(cases)} review cases to {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON manifest to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
