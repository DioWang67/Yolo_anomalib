"""Collect production review cases from saved inspection metadata.

The tool scans Result/**/metadata/*_config_snapshot.json files and writes a
CSV/JSON manifest for human review. It intentionally stays filesystem-based so
the workflow remains easy to run on an offline inspection machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REVIEW_LABELS = (
    "",
    "confirmed_ng",
    "confirmed_ok",
    "verified_empty",
    "false_positive",
    "false_negative",
    "wrong_box",
    "wrong_class",
    "uncertain",
    "image_quality_issue",
    "color_confirmed_ng",
    "color_false_reject",
)
FAIL_STATUSES = {"FAIL", "DETECTION_FAIL", "ERROR", "INFERENCE_ERROR"}
logger = logging.getLogger(__name__)


class ReviewManifestReadError(RuntimeError):
    """Raised when an existing review manifest cannot be read safely."""


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
        failure_category: Operator-selected failure cause, independent of routing.
        failure_source: Extensible subsystem identifier for the failure cause.
        review_label: Empty field for human labeling.
        review_note: Empty field for human notes.
        review_selected: ``1`` when selected in the failure overview.
        training_selected: ``1`` when included in the next training submission.
    """

    timestamp: str
    product: str
    area: str
    machine_id: str
    work_order: str
    camera_id: str
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
    detected_box_count: str
    detection_evidence_source: str
    class_names_json: str
    class_map_json: str
    failure_crop_paths: str
    crop_paths: str = ""
    mask_paths: str = ""
    color_result_json: str = "{}"
    color_checker_type: str = ""
    color_failure_count: str = "0"
    product_verdict: str = ""
    detection_verdict: str = ""
    color_verdict: str = ""
    action_route: str = ""
    failure_category: str = ""
    failure_source: str = ""
    failure_note: str = ""
    review_outcome: str = ""
    skip_reason: str = ""
    review_label: str = ""
    review_note: str = ""
    review_selected: str = "0"
    training_selected: str = "1"


def collect_review_cases(
    result_root: str | Path,
    *,
    include_pass: bool = False,
    start_time: datetime | str | None = None,
    end_time: datetime | str | None = None,
    product: str | None = None,
    area: str | None = None,
) -> list[ReviewCase]:
    """Collect review cases from result metadata snapshots.

    Args:
        result_root: Root directory containing dated inspection outputs.
        include_pass: Include PASS cases as well as failure cases.
        start_time: Optional inclusive lower timestamp bound.
        end_time: Optional inclusive upper timestamp bound.
        product: Optional exact product filter applied before artifact discovery.
        area: Optional exact station/area filter applied before artifact discovery.

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
    product_filter = str(product or "").strip()
    area_filter = str(area or "").strip()
    legacy_crop_cache: dict[Path, tuple[Path, ...]] = {}

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
        product = str(snapshot.get("product") or "")
        area = str(snapshot.get("area") or "")
        if product_filter and product != product_filter:
            continue
        if area_filter and area != area_filter:
            continue

        detector = str(snapshot.get("detector") or "")
        decision = snapshot.get("decision") if isinstance(snapshot.get("decision"), dict) else {}
        model_info = snapshot.get("model_info") if isinstance(snapshot.get("model_info"), dict) else {}
        color_result = (
            snapshot.get("color_result")
            if isinstance(snapshot.get("color_result"), dict)
            else {}
        )
        runtime_config = (
            snapshot.get("config")
            if isinstance(snapshot.get("config"), dict)
            else {}
        )
        equipment = (
            snapshot.get("equipment")
            if isinstance(snapshot.get("equipment"), dict)
            else {}
        )
        artifacts = (
            snapshot.get("artifacts")
            if isinstance(snapshot.get("artifacts"), dict)
            else {}
        )
        raw_detections = snapshot.get("detections")
        has_detection_contract = isinstance(raw_detections, list)
        detections = raw_detections if has_detection_contract else []
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
        legacy_detection_crops = (
            []
            if has_detection_contract
            else _find_detection_crop_paths(
                snapshot_path,
                detector,
                directory_cache=legacy_crop_cache,
            )
        )
        if has_detection_contract:
            detected_box_count = _usable_detection_count(detections)
            detection_evidence_source = "snapshot"
        elif legacy_detection_crops:
            detected_box_count = len(legacy_detection_crops)
            detection_evidence_source = "saved_crops"
        else:
            detected_box_count = None
            detection_evidence_source = "unknown"
        class_names = _normalize_class_names(model_info.get("class_names"))
        observed_class_map = _observed_class_map(detections)
        artifact_crop_values = artifacts.get("cropped_paths")
        has_artifact_crop_contract = isinstance(
            artifact_crop_values, (list, tuple)
        )
        artifact_crop_paths = [
            Path(str(path))
            for path in _path_list(artifact_crop_values)
            if str(path or "").strip()
        ]
        failure_crop_paths = (
            [path for path in artifact_crop_paths if "_NG_" in path.name]
            if has_artifact_crop_contract
            else _find_failure_crop_paths(
                snapshot_path,
                detector,
                directory_cache=legacy_crop_cache,
            )
        )

        cases.append(
            ReviewCase(
                timestamp=timestamp,
                product=product,
                area=area,
                machine_id=str(equipment.get("machine_id") or ""),
                work_order=str(equipment.get("work_order") or ""),
                camera_id=str(equipment.get("camera_id") or ""),
                status=status,
                detector=detector,
                decision_reasons="|".join(
                    str(item)
                    for item in _snapshot_fail_reasons(snapshot, decision)
                ),
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
                detected_box_count=(
                    "" if detected_box_count is None else str(detected_box_count)
                ),
                detection_evidence_source=detection_evidence_source,
                class_names_json=json.dumps(
                    class_names, ensure_ascii=False, separators=(",", ":")
                ),
                class_map_json=json.dumps(
                    observed_class_map, ensure_ascii=False, separators=(",", ":")
                ),
                failure_crop_paths="|".join(
                    str(path) for path in failure_crop_paths
                ),
                crop_paths="|".join(str(path) for path in artifact_crop_paths),
                mask_paths="|".join(
                    str(path)
                    for path in _path_list(artifacts.get("mask_paths"))
                    if str(path or "").strip()
                ),
                color_result_json=json.dumps(
                    color_result, ensure_ascii=False, separators=(",", ":")
                ),
                color_checker_type=str(
                    runtime_config.get("color_checker_type") or ""
                ),
                color_failure_count=str(_color_failure_count(color_result)),
            )
        )

    cases.sort(key=lambda item: (item.timestamp, item.product, item.area, item.config_snapshot_path))
    return cases


def _snapshot_fail_reasons(
    snapshot: dict[str, Any], decision: dict[str, Any]
) -> list[str]:
    """Return stable failure reason codes from new or legacy snapshots."""
    raw_reasons = snapshot.get("fail_reasons")
    if not isinstance(raw_reasons, list):
        raw_reasons = decision.get("reasons")
    if not isinstance(raw_reasons, list):
        return []
    return [str(reason) for reason in raw_reasons if str(reason).strip()]


def _path_list(value: Any) -> list[Any]:
    """Return artifact path entries without treating a string as a sequence."""
    return list(value) if isinstance(value, (list, tuple)) else []


def _color_failure_count(color_result: dict[str, Any]) -> int:
    """Count color-check items that failed their runtime threshold."""
    items = color_result.get("items")
    if not isinstance(items, list):
        return 0
    return sum(
        isinstance(item, dict) and item.get("is_ok") is False
        for item in items
    )


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


def _usable_detection_count(detections: list[Any]) -> int:
    """Count structured detections that carry a usable bounding box."""
    return sum(
        isinstance(detection, dict)
        and isinstance(detection.get("bbox"), (list, tuple))
        and len(detection["bbox"]) >= 4
        for detection in detections
    )


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
            for field in (
                "product_verdict",
                "detection_verdict",
                "color_verdict",
                "action_route",
                "failure_category",
                "failure_source",
                "failure_note",
                "review_outcome",
                "skip_reason",
                "review_label",
                "review_note",
                "review_selected",
                "training_selected",
            ):
                row[field] = previous.get(field, row[field])
        rows.append(row)
    fieldnames = list(ReviewCase.__dataclass_fields__.keys())
    _write_csv_atomic(csv_path, fieldnames, rows)

    if output_json is not None:
        json_path = Path(output_json)
        _write_json_atomic(json_path, rows)


def _load_existing_reviews(csv_path: Path) -> dict[str, dict[str, str]]:
    """Load operator-entered fields so regenerating a manifest is idempotent."""
    if not csv_path.exists():
        return {}
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reviews: dict[str, dict[str, str]] = {}
            reader = csv.DictReader(handle, strict=True)
            if not reader.fieldnames or "config_snapshot_path" not in reader.fieldnames:
                raise ReviewManifestReadError(
                    "Existing review manifest is missing the "
                    f"config_snapshot_path column: {csv_path}"
                )
            for row in reader:
                key = str(row.get("config_snapshot_path") or "")
                if not key:
                    continue
                review_label = str(row.get("review_label") or "")
                reviews[key] = {
                    "product_verdict": str(row.get("product_verdict") or ""),
                    "detection_verdict": str(row.get("detection_verdict") or ""),
                    "color_verdict": str(row.get("color_verdict") or ""),
                    "action_route": str(row.get("action_route") or ""),
                    "failure_category": str(row.get("failure_category") or ""),
                    "failure_source": str(row.get("failure_source") or ""),
                    "failure_note": str(row.get("failure_note") or ""),
                    "review_outcome": str(row.get("review_outcome") or ""),
                    "skip_reason": str(row.get("skip_reason") or ""),
                    "review_label": review_label,
                    "review_note": str(row.get("review_note") or ""),
                    "review_selected": (
                        "1" if str(row.get("review_selected") or "0") == "1" else "0"
                    ),
                    "training_selected": (
                        "0"
                        if review_label
                        in {"confirmed_ok", "uncertain", "image_quality_issue"}
                        or str(row.get("training_selected") or "1") == "0"
                        else "1"
                    ),
                }
            return reviews
    except ReviewManifestReadError:
        logger.error(
            "Existing review manifest is invalid; refusing to overwrite it: %s",
            csv_path,
            exc_info=True,
        )
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.error(
            "Unable to read existing review manifest; refusing to overwrite it: %s",
            csv_path,
            exc_info=True,
        )
        raise ReviewManifestReadError(
            "Unable to read existing review manifest; the original file was not "
            f"changed: {csv_path}: {exc}"
        ) from exc


def _write_csv_atomic(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Durably replace one CSV without exposing a partial destination file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError:
        logger.error("Atomic review manifest write failed: %s", path, exc_info=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_json_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Durably replace the optional JSON companion in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(rows, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except OSError:
        logger.error("Atomic review JSON write failed: %s", path, exc_info=True)
        raise
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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


def _find_failure_crop_paths(
    snapshot_path: Path,
    detector: str,
    *,
    directory_cache: dict[Path, tuple[Path, ...]] | None = None,
) -> list[Path]:
    base_path = _inspection_base_path(snapshot_path)
    if base_path is None:
        return []
    crop_dir = base_path / "cropped" / detector.lower()
    return [
        path
        for path in _legacy_crop_files(crop_dir, directory_cache=directory_cache)
        if "_NG_" in path.name
    ]


def _find_detection_crop_paths(
    snapshot_path: Path,
    detector: str,
    *,
    directory_cache: dict[Path, tuple[Path, ...]] | None = None,
) -> list[Path]:
    """Find legacy per-detection crops belonging to exactly one snapshot.

    Schema-v1 snapshots did not persist detections.  Their normal crop names
    still share the snapshot stem and provide reliable evidence that YOLO drew
    boxes.  Reason-based ``_NG_`` crops are excluded because a missing-item
    crop does not prove that a detection existed.
    """
    base_path = _inspection_base_path(snapshot_path)
    if base_path is None:
        return []
    crop_dir = base_path / "cropped" / detector.lower()
    stem = snapshot_path.name.removesuffix("_config_snapshot.json")
    return [
        path
        for path in _legacy_crop_files(crop_dir, directory_cache=directory_cache)
        if path.name.startswith(f"{stem}_") and "_NG_" not in path.name
    ]


def _legacy_crop_files(
    crop_dir: Path,
    *,
    directory_cache: dict[Path, tuple[Path, ...]] | None,
) -> tuple[Path, ...]:
    """List one legacy crop directory once per collection run."""
    if directory_cache is not None and crop_dir in directory_cache:
        return directory_cache[crop_dir]
    files = tuple(sorted(crop_dir.glob("*.png"))) if crop_dir.is_dir() else ()
    if directory_cache is not None:
        directory_cache[crop_dir] = files
    return files


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
