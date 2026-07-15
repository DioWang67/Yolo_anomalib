"""Export reviewed production cases into a dataset curation folder.

This tool does not invent YOLO labels. It copies reviewed evidence images into a
raw/images folder and optionally creates empty label placeholders so operators
can annotate them before running Yolo11_auto_train.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from tools.process_liveness import is_process_active

DEFAULT_LABELS = {
    "confirmed_ng",
    "verified_empty",
    "false_positive",
    "false_negative",
    "wrong_class",
}

HOLD_LABELS = {"uncertain", "image_quality_issue"}
MISSING_LABEL_BASELINE = "missing"
SNAPSHOT_DUPLICATE_IOU_THRESHOLD = 0.90


@dataclass(frozen=True)
class ExportedReviewItem:
    """One copied review image prepared for annotation or retraining."""

    source_manifest: str
    review_label: str
    review_note: str
    source_image: str
    output_image: str
    output_label: str
    annotation_status: str
    sample_id: str
    image_sha256: str
    product: str
    area: str
    timestamp: str
    status: str
    decision_reasons: str
    model_version: str
    class_names_json: str
    class_map_json: str
    class_schema_hash: str


@dataclass(frozen=True)
class OperatorHandoffReport:
    """Summary returned after an OP sends reviewed cases to training."""

    handoff_path: Path
    ready_count: int
    pending_count: int
    skipped_count: int
    targets: tuple[tuple[str, str], ...]
    total_ready_count: int = 0
    total_pending_count: int = 0
    job_id: str = ""
    status_path: Path | None = None
    reused_existing: bool = False


@dataclass(frozen=True)
class _SnapshotDetection:
    """One validated snapshot detection in stored-image pixel coordinates."""

    source_index: int
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    image_width: float
    image_height: float


def export_review_dataset(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    include_labels: set[str] | None = None,
    source_kind: str = "original",
    create_label_placeholders: bool = False,
    group_by_target: bool = True,
) -> list[ExportedReviewItem]:
    """Export reviewed cases into ``raw/images`` and ``raw/labels``.

    Args:
        manifest_csv: Review manifest with filled ``review_label`` values.
        output_dir: Destination dataset curation directory.
        include_labels: Review labels to export.
        source_kind: ``original``, ``failure_crops``, ``annotated``, or ``both``.
        create_label_placeholders: Create empty YOLO label files for annotation.
        group_by_target: Write ``<product>/<area>/raw`` dataset roots.

    Returns:
        Exported item records.
    """
    include = include_labels or set(DEFAULT_LABELS)
    manifest_path = Path(manifest_csv)
    output_root = Path(output_dir)
    exported: list[ExportedReviewItem] = []
    seen_hashes: set[tuple[str, str, str]] = set()
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            review_label = str(row.get("review_label") or "").strip()
            if review_label in HOLD_LABELS or str(
                row.get("training_selected") or "1"
            ) == "0":
                continue
            if review_label not in include:
                continue
            product = str(row.get("product") or "")
            area = str(row.get("area") or "")
            for source_image in _source_images(row, source_kind):
                source_path = Path(source_image)
                if not source_path.exists():
                    continue
                image_sha256 = _sha256_file(source_path)
                content_key = (product, area, image_sha256)
                if content_key in seen_hashes:
                    continue
                seen_hashes.add(content_key)
                target_root = output_root
                if group_by_target:
                    target_root = (
                        output_root
                        / _safe_name(product or "unknown")
                        / _safe_name(area or "unknown")
                    )
                images_dir = target_root / "raw" / "images"
                labels_dir = target_root / "raw" / "labels"
                images_dir.mkdir(parents=True, exist_ok=True)
                labels_dir.mkdir(parents=True, exist_ok=True)
                output_name = _build_output_name(
                    row, row_index, review_label, source_path, image_sha256
                )
                output_image = images_dir / output_name
                shutil.copy2(source_path, output_image)
                output_label = labels_dir / f"{output_image.stem}.txt"
                if create_label_placeholders and not output_label.exists():
                    output_label.write_text("", encoding="utf-8")
                exported.append(
                    ExportedReviewItem(
                        source_manifest=str(manifest_path),
                        review_label=review_label,
                        review_note=str(row.get("review_note") or ""),
                        source_image=str(source_path),
                        output_image=str(output_image),
                        output_label=str(output_label) if create_label_placeholders else "",
                        annotation_status="pending",
                        sample_id=_sample_id(product, area, image_sha256),
                        image_sha256=image_sha256,
                        product=str(row.get("product") or ""),
                        area=str(row.get("area") or ""),
                        timestamp=str(row.get("timestamp") or ""),
                        status=str(row.get("status") or ""),
                        decision_reasons=str(row.get("decision_reasons") or ""),
                        model_version=str(row.get("model_version") or ""),
                        class_names_json=str(row.get("class_names_json") or "[]"),
                        class_map_json=str(row.get("class_map_json") or "{}"),
                        class_schema_hash=_class_schema_hash(
                            _json_string_list(row.get("class_names_json"))
                        ),
                    )
                )

    _write_manifests(exported, output_root, group_by_target=group_by_target)
    return exported


def export_operator_handoff(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    inference_models_dir: str | Path | None = None,
) -> OperatorHandoffReport:
    """Export OP-confirmed boxes and route unsafe cases to an annotation queue.

    Only ``confirmed_ng`` means every displayed box and class was confirmed by
    the OP and can be converted from the saved schema-v2 detections. A false
    positive, missed box, or uncertain case requires annotation and never
    enters training automatically.
    """
    manifest_path = Path(manifest_csv)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    updates: list[tuple[str, ExportedReviewItem | dict[str, str]]] = []
    skipped_count = 0

    with _handoff_export_lock(output_root):
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        review_rows = _enrich_legacy_class_contracts(review_rows, output_root)
        _preflight_operator_class_contracts(review_rows)
        for row_index, row in enumerate(review_rows, start=1):
            review_label = str(row.get("review_label") or "").strip()
            if not review_label:
                skipped_count += 1
                continue
            if review_label in HOLD_LABELS or str(
                row.get("training_selected") or "1"
            ) == "0":
                updates.append(("excluded", _excluded_state_row(row)))
                continue
            if review_label == "confirmed_ng":
                item, reason = _export_snapshot_verified_row(
                    row, row_index, manifest_path, output_root
                )
                if item is not None:
                    updates.append(("ready", item))
                else:
                    updates.append(
                        ("pending", _export_pending_row(row, output_root, reason))
                    )
                continue
            if review_label == "verified_empty":
                item, reason = _export_verified_empty_row(
                    row, manifest_path, output_root
                )
                if item is not None:
                    updates.append(("ready", item))
                else:
                    updates.append(
                        ("pending", _export_pending_row(row, output_root, reason))
                    )
                continue
            if review_label in {
                "false_positive",
                "false_negative",
                "wrong_class",
            }:
                pending_reason = {
                    "false_positive": "false_detection_requires_correction",
                    "false_negative": "missed_detection_requires_box_annotation",
                    "wrong_class": "wrong_class_requires_correction",
                }[review_label]
                updates.append(
                    (
                        "pending",
                        _export_pending_row(row, output_root, pending_reason),
                    )
                )
                continue
            skipped_count += 1

        # The last decision for identical image content wins.  This makes a
        # re-review reversible and prevents duplicate files from entering the
        # splitter when a different time range is submitted later.
        latest_updates: dict[
            tuple[str, str, str],
            tuple[str, ExportedReviewItem | dict[str, str]],
        ] = {}
        for state, payload in updates:
            key = _state_key(payload)
            latest_updates[key] = (state, payload)
        final_updates = list(latest_updates.values())

        ready_items = [
            payload
            for state, payload in final_updates
            if state == "ready" and isinstance(payload, ExportedReviewItem)
        ]
        pending_rows = [
            payload
            for state, payload in final_updates
            if state == "pending" and isinstance(payload, dict)
        ]
        targets = sorted(
            {
                (_state_key(payload)[0], _state_key(payload)[1])
                for state, payload in final_updates
                if state != "excluded"
            }
        )
        target_contracts = _build_target_class_contracts(final_updates)
        for product, area in targets:
            has_pending_annotation = any(
                state == "pending"
                and _state_key(payload)[:2] == (product, area)
                for state, payload in final_updates
            )
            contract = target_contracts.get((product, area), {})
            if has_pending_annotation and not contract.get("class_names"):
                raise ValueError(
                    "補標案件缺少部署模型的完整類別順序，已停止送訓。"
                    f"請先用目前模型重新檢測影像：{product}/{area}"
                )
        _materialize_pending_rows(pending_rows)
        _apply_operator_state_updates(final_updates, output_root)
        total_ready_by_target = {
            target: len(_read_export_manifest(_target_manifest(output_root, *target)))
            for target in targets
        }
        total_pending_by_target = {
            target: len(_read_pending_manifest(_pending_manifest(output_root, *target)))
            for target in targets
        }
        total_ready_count = sum(total_ready_by_target.values())
        total_pending_count = sum(total_pending_by_target.values())
        submission_hash = _operator_submission_hash(final_updates)
        existing_handoff = _find_active_operator_job(output_root, submission_hash)
        if existing_handoff is not None:
            existing_payload = _read_json_mapping(existing_handoff)
            existing_status_path = Path(
                str(existing_payload.get("status_path") or "")
            ).resolve()
            return OperatorHandoffReport(
                handoff_path=existing_handoff,
                ready_count=len(ready_items),
                pending_count=len(pending_rows),
                skipped_count=skipped_count,
                targets=tuple(targets),
                total_ready_count=total_ready_count,
                total_pending_count=total_pending_count,
                job_id=str(existing_payload.get("job_id") or ""),
                status_path=existing_status_path,
                reused_existing=True,
            )

        job_id = _new_operator_job_id()
        job_dir = output_root / ".operator_handoff" / "jobs" / job_id
        handoff_path = job_dir / "handoff.json"
        status_path = job_dir / "status.json"
        handoff_payload = {
            "schema_version": 3,
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_hash": submission_hash,
            "source_manifest": str(manifest_path.resolve()),
            "data_root": str(output_root.resolve()),
            "status_path": str(status_path.resolve()),
            "inference_models_dir": (
                str(Path(inference_models_dir).resolve())
                if inference_models_dir is not None
                else ""
            ),
            "ready_count": len(ready_items),
            "total_ready_count": total_ready_count,
            "pending_count": len(pending_rows),
            "skipped_count": skipped_count,
            "targets": [
                {
                    "product": product,
                    "area": area,
                    "dataset_root": str(
                        (output_root / _safe_name(product) / _safe_name(area)).resolve()
                    ),
                    "ready_count": sum(
                        item.product == product and item.area == area
                        for item in ready_items
                    ),
                    "total_ready_count": total_ready_by_target[(product, area)],
                    "pending_count": sum(
                        _state_key(payload)[:2] == (product, area)
                        for state, payload in final_updates
                        if state == "pending"
                    ),
                    "sample_ids": sorted(
                        _state_key(payload)[2]
                        for state, payload in final_updates
                        if state != "excluded"
                        and _state_key(payload)[:2] == (product, area)
                    ),
                    "pending_sample_ids": sorted(
                        _state_key(payload)[2]
                        for state, payload in final_updates
                        if state == "pending"
                        and _state_key(payload)[:2] == (product, area)
                    ),
                    **target_contracts.get((product, area), {}),
                }
                for product, area in targets
            ],
        }
        _write_json_atomic(handoff_path, handoff_payload)
        # ``latest.json`` remains a compatibility view only. The launched
        # training process always receives the immutable job-specific path.
        _write_json_atomic(
            output_root / ".operator_handoff" / "latest.json", handoff_payload
        )
        initial_state = "waiting_annotation" if pending_rows else "queued"
        update_operator_job_status(
            status_path,
            state=initial_state,
            message=(
                "等待完成補標"
                if pending_rows
                else "已排入模型更新流程"
            ),
            job_id=job_id,
            created_at=handoff_payload["created_at"],
            product=targets[0][0] if len(targets) == 1 else "",
            area=targets[0][1] if len(targets) == 1 else "",
            ready_count=total_ready_count,
            pending_count=len(pending_rows),
            progress=0,
        )

    return OperatorHandoffReport(
        handoff_path=handoff_path,
        ready_count=len(ready_items),
        pending_count=len(pending_rows),
        skipped_count=skipped_count,
        targets=tuple(targets),
        total_ready_count=total_ready_count,
        total_pending_count=total_pending_count,
        job_id=job_id,
        status_path=status_path,
    )


def _export_snapshot_verified_row(
    row: dict[str, str],
    row_index: int,
    manifest_path: Path,
    output_root: Path,
) -> tuple[ExportedReviewItem | None, str]:
    source_path = Path(str(row.get("preprocessed_path") or ""))
    if not source_path.is_file():
        return None, "preprocessed_image_missing"
    class_names = _json_string_list(row.get("class_names_json"))
    label_lines = _snapshot_yolo_label_lines(
        row.get("detections_json") or "",
        image_size=_read_image_size(source_path),
        class_names=class_names,
    )
    if not label_lines:
        return None, "verified_snapshot_has_no_valid_detections"

    product = str(row.get("product") or "unknown")
    area = str(row.get("area") or "unknown")
    review_label = str(row.get("review_label") or "")
    image_sha256 = _sha256_file(source_path)
    sample_id = _sample_id(product, area, image_sha256)
    target_root = output_root / _safe_name(product) / _safe_name(area)
    images_dir = target_root / "raw" / "images"
    labels_dir = target_root / "raw" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    output_name = _stable_output_name(sample_id, source_path)
    output_image = images_dir / output_name
    output_label = labels_dir / f"{output_image.stem}.txt"
    if not output_image.exists():
        shutil.copy2(source_path, output_image)
    _write_text_atomic(output_label, "\n".join(label_lines) + "\n")
    return (
        ExportedReviewItem(
            source_manifest=str(manifest_path),
            review_label=review_label,
            review_note=str(row.get("review_note") or ""),
            source_image=str(source_path),
            output_image=str(output_image),
            output_label=str(output_label),
            annotation_status="verified_snapshot",
            sample_id=sample_id,
            image_sha256=image_sha256,
            product=product,
            area=area,
            timestamp=str(row.get("timestamp") or ""),
            status=str(row.get("status") or ""),
            decision_reasons=str(row.get("decision_reasons") or ""),
            model_version=str(row.get("model_version") or ""),
            class_names_json=json.dumps(
                class_names, ensure_ascii=False, separators=(",", ":")
            ),
            class_map_json=_normalized_class_map_json(row),
            class_schema_hash=_class_schema_hash(class_names),
        ),
        "",
    )


def _snapshot_yolo_label_lines(
    detections_json: str,
    *,
    image_size: tuple[int, int] | None = None,
    class_names: list[str] | None = None,
) -> list[str]:
    """Convert snapshot boxes using the dimensions of the stored review image.

    Detection metadata can describe the camera frame even when ``bbox`` values
    belong to a resized, letterboxed review image.  ``image_size`` therefore
    takes precedence whenever the stored image can be inspected.
    """
    try:
        detections = json.loads(detections_json)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(detections, list):
        return []

    actual_width: float | None = None
    actual_height: float | None = None
    if image_size is not None:
        actual_width, actual_height = map(float, image_size)
        if actual_width <= 0 or actual_height <= 0:
            return []

    ordered_class_names = [str(name).strip() for name in class_names or []]
    class_id_by_name = {
        name: class_id for class_id, name in enumerate(ordered_class_names) if name
    }
    candidates: list[_SnapshotDetection] = []
    for source_index, raw in enumerate(detections):
        if not isinstance(raw, dict):
            continue
        bbox = raw.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            class_id = int(raw["class_id"])
            x1, y1, x2, y2 = [float(value) for value in bbox]
            confidence = float(raw.get("confidence", 0.0))
            width = (
                actual_width
                if actual_width is not None
                else float(raw["image_width"])
            )
            height = (
                actual_height
                if actual_height is not None
                else float(raw["image_height"])
            )
        except (KeyError, TypeError, ValueError):
            continue
        numeric_values = (x1, y1, x2, y2, confidence, width, height)
        if not all(math.isfinite(value) for value in numeric_values):
            continue
        if class_id < 0 or width <= 0 or height <= 0 or x2 <= x1 or y2 <= y1:
            continue
        verified_class = str(raw.get("verified_class") or "").strip()
        if verified_class in class_id_by_name:
            class_id = class_id_by_name[verified_class]
        if ordered_class_names and class_id >= len(ordered_class_names):
            continue
        x1, x2 = max(0.0, x1), min(width, x2)
        y1, y2 = max(0.0, y1), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append(
            _SnapshotDetection(
                source_index=source_index,
                class_id=class_id,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                image_width=width,
                image_height=height,
            )
        )

    selected: list[_SnapshotDetection] = []
    for candidate in sorted(
        candidates, key=lambda item: (-item.confidence, item.source_index)
    ):
        if any(
            _bbox_iou(candidate.bbox, existing.bbox)
            >= SNAPSHOT_DUPLICATE_IOU_THRESHOLD
            for existing in selected
        ):
            continue
        selected.append(candidate)

    lines: list[str] = []
    for detection in sorted(selected, key=lambda item: item.source_index):
        x1, y1, x2, y2 = detection.bbox
        center_x = ((x1 + x2) / 2.0) / detection.image_width
        center_y = ((y1 + y2) / 2.0) / detection.image_height
        box_width = (x2 - x1) / detection.image_width
        box_height = (y2 - y1) / detection.image_height
        lines.append(
            f"{detection.class_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f}"
        )
    return lines


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    """Return intersection-over-union for two validated XYXY boxes."""
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection_area = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def _export_verified_empty_row(
    row: dict[str, str], manifest_path: Path, output_root: Path
) -> tuple[ExportedReviewItem | None, str]:
    """Export an explicitly verified background image with an empty label."""
    source_path = Path(
        str(row.get("preprocessed_path") or row.get("original_path") or "")
    )
    if not source_path.is_file():
        return None, "verified_empty_image_missing"
    product = str(row.get("product") or "unknown")
    area = str(row.get("area") or "unknown")
    image_sha256 = _sha256_file(source_path)
    sample_id = _sample_id(product, area, image_sha256)
    target_root = output_root / _safe_name(product) / _safe_name(area)
    output_image = target_root / "raw" / "images" / _stable_output_name(
        sample_id, source_path
    )
    output_label = target_root / "raw" / "labels" / f"review_{sample_id}.txt"
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    if not output_image.exists():
        shutil.copy2(source_path, output_image)
    _write_text_atomic(output_label, "")
    class_names = _json_string_list(row.get("class_names_json"))
    return (
        ExportedReviewItem(
            source_manifest=str(manifest_path),
            review_label="verified_empty",
            review_note=str(row.get("review_note") or ""),
            source_image=str(source_path),
            output_image=str(output_image),
            output_label=str(output_label),
            annotation_status="verified_empty",
            sample_id=sample_id,
            image_sha256=image_sha256,
            product=product,
            area=area,
            timestamp=str(row.get("timestamp") or ""),
            status=str(row.get("status") or ""),
            decision_reasons=str(row.get("decision_reasons") or ""),
            model_version=str(row.get("model_version") or ""),
            class_names_json=json.dumps(
                class_names, ensure_ascii=False, separators=(",", ":")
            ),
            class_map_json=_normalized_class_map_json(row),
            class_schema_hash=_class_schema_hash(class_names),
        ),
        "",
    )


def _export_pending_row(
    row: dict[str, str], output_root: Path, reason: str
) -> dict[str, str]:
    product = str(row.get("product") or "unknown")
    area = str(row.get("area") or "unknown")
    processed_source = Path(str(row.get("preprocessed_path") or ""))
    original_source = Path(str(row.get("original_path") or ""))
    source = processed_source if processed_source.is_file() else original_source
    output_image = ""
    output_label = ""
    image_sha256 = ""
    sample_id = ""
    label_baseline_sha256 = MISSING_LABEL_BASELINE
    if source.is_file():
        image_sha256 = _sha256_file(source)
        sample_id = _sample_id(product, area, image_sha256)
        pending_dir = (
            output_root
            / _safe_name(product)
            / _safe_name(area)
            / "review_pending"
            / "images"
        )
        destination = pending_dir / _stable_output_name(sample_id, source)
        output_image = str(destination)
        labels_dir = pending_dir.parent / "labels"
        label_path = labels_dir / f"{destination.stem}.txt"
        output_label = str(label_path)
        if label_path.is_file():
            label_baseline_sha256 = _sha256_file(label_path)
    if not sample_id:
        identity = str(row.get("config_snapshot_path") or source)
        fallback_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        sample_id = _sample_id(product, area, fallback_hash)
    return {
        "sample_id": sample_id,
        "image_sha256": image_sha256,
        "product": product,
        "area": area,
        "timestamp": str(row.get("timestamp") or ""),
        "review_label": str(row.get("review_label") or ""),
        "reason": reason,
        "annotation_status": "pending",
        "review_note": str(row.get("review_note") or ""),
        "status": str(row.get("status") or ""),
        "decision_reasons": str(row.get("decision_reasons") or ""),
        "model_version": str(row.get("model_version") or ""),
        "class_names_json": json.dumps(
            _json_string_list(row.get("class_names_json")),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "class_map_json": _normalized_class_map_json(row),
        "class_schema_hash": _class_schema_hash(
            _json_string_list(row.get("class_names_json"))
        ),
        "detections_json": str(row.get("detections_json") or "[]"),
        "source_image": str(source),
        "output_image": output_image,
        "output_label": output_label,
        "label_baseline_sha256": label_baseline_sha256,
        "config_snapshot_path": str(row.get("config_snapshot_path") or ""),
    }


def _materialize_pending_rows(pending_rows: list[dict[str, str]]) -> None:
    """Copy pending images only after the complete class contract is valid.

    Newly created files are removed if materialization fails. Existing operator
    work is never deleted or overwritten.
    """
    created_paths: list[Path] = []
    try:
        for row in pending_rows:
            source = Path(str(row.get("source_image") or ""))
            output_image = Path(str(row.get("output_image") or ""))
            output_label = Path(str(row.get("output_label") or ""))
            if (
                not source.is_file()
                or not str(row.get("output_image") or "")
                or not str(row.get("output_label") or "")
            ):
                raise ValueError(
                    "補標案件的已保存影像不存在，已停止送訓。"
                    "請回到推理系統重新檢測並保存結果。"
                )
            output_image.parent.mkdir(parents=True, exist_ok=True)
            if not output_image.exists():
                shutil.copy2(source, output_image)
                created_paths.append(output_image)

            reason = str(row.get("reason") or "")
            if reason in {
                "false_detection_requires_correction",
                "wrong_class_requires_correction",
                "operator_uncertain_requires_review",
            } and not output_label.exists():
                draft_lines = _snapshot_yolo_label_lines(
                    str(row.get("detections_json") or ""),
                    image_size=_read_image_size(output_image),
                    class_names=_json_string_list(row.get("class_names_json")),
                )
                if draft_lines:
                    output_label.parent.mkdir(parents=True, exist_ok=True)
                    _write_text_atomic(output_label, "\n".join(draft_lines) + "\n")
                    created_paths.append(output_label)
            if output_label.is_file():
                row["label_baseline_sha256"] = _sha256_file(output_label)
    except (OSError, shutil.Error, ValueError):
        for path in reversed(created_paths):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _preflight_operator_class_contracts(
    review_rows: list[dict[str, str]],
) -> None:
    """Reject conflicting or incomplete contracts before copying any file."""
    contract_rows: list[tuple[str, ExportedReviewItem | dict[str, str]]] = []
    pending_targets: set[tuple[str, str]] = set()
    for row in review_rows:
        review_label = str(row.get("review_label") or "").strip()
        if (
            not review_label
            or review_label in HOLD_LABELS
            or str(row.get("training_selected") or "1") == "0"
        ):
            continue
        if review_label not in {
            "confirmed_ng",
            "verified_empty",
            "false_positive",
            "false_negative",
            "wrong_class",
        }:
            continue
        contract_rows.append(("pending", row))
        if review_label in {"false_positive", "false_negative", "wrong_class"}:
            pending_targets.add(
                (
                    str(row.get("product") or "unknown"),
                    str(row.get("area") or "unknown"),
                )
            )
    contracts = _build_target_class_contracts(contract_rows)
    for product, area in sorted(pending_targets):
        if not contracts.get((product, area), {}).get("class_names"):
            raise ValueError(
                "補標案件缺少部署模型的完整類別順序，已停止送訓。"
                f"請先用目前模型重新檢測影像：{product}/{area}"
            )


def _enrich_legacy_class_contracts(
    review_rows: list[dict[str, str]], output_root: Path
) -> list[dict[str, str]]:
    """Recover a legacy batch only from a signed, fully observed contract.

    Old review manifests did not persist ``class_names_json``. They are safe to
    migrate when the target already has exactly one checksum-valid class
    contract and this batch collectively observes every class ID with matching
    names under one weights path. No source CSV is modified.
    """
    rows = [dict(row) for row in review_rows]
    selected_by_target: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        review_label = str(row.get("review_label") or "").strip()
        if (
            review_label not in DEFAULT_LABELS
            or str(row.get("training_selected") or "1") == "0"
        ):
            continue
        target = (
            str(row.get("product") or "unknown"),
            str(row.get("area") or "unknown"),
        )
        selected_by_target.setdefault(target, []).append(row)

    for (product, area), target_rows in selected_by_target.items():
        legacy_rows = [
            row
            for row in target_rows
            if not _json_string_list(row.get("class_names_json"))
        ]
        if not legacy_rows:
            continue
        contract = _load_persisted_target_class_contract(
            output_root, product, area
        )
        if not contract:
            continue
        weights = {
            str(row.get("weights") or "").strip()
            for row in target_rows
            if str(row.get("weights") or "").strip()
        }
        if len(weights) != 1:
            continue

        observed: dict[int, str] = {}
        for row in target_rows:
            raw_mapping = _json_string_map(
                _normalized_class_map_json(row)
            )
            for raw_class_id, class_name in raw_mapping.items():
                try:
                    class_id = int(raw_class_id)
                except ValueError as exc:
                    raise ValueError(
                        f"舊版類別 ID 無效：{product}/{area}={raw_class_id!r}"
                    ) from exc
                previous = observed.get(class_id)
                if previous is not None and previous != class_name:
                    raise ValueError(
                        "舊版檢測紀錄的類別對應互相衝突，已停止送訓："
                        f"{product}/{area} id={class_id}"
                    )
                if class_id < 0 or class_id >= len(contract):
                    raise ValueError(
                        "舊版檢測紀錄含有超出類別契約的 ID，已停止送訓："
                        f"{product}/{area} id={class_id}"
                    )
                if contract[class_id] != class_name:
                    raise ValueError(
                        "舊版檢測紀錄與既有類別順序不一致，已停止送訓："
                        f"{product}/{area} id={class_id}"
                    )
                observed[class_id] = class_name

        if set(observed) != set(range(len(contract))):
            continue
        serialized_contract = json.dumps(
            contract, ensure_ascii=False, separators=(",", ":")
        )
        for row in legacy_rows:
            row["class_names_json"] = serialized_contract
    return rows


def _load_persisted_target_class_contract(
    output_root: Path, product: str, area: str
) -> list[str]:
    """Load the target's one checksum-valid class contract, if available."""
    candidates: set[tuple[str, ...]] = set()
    ready_items = _read_export_manifest(
        _target_manifest(output_root, product, area)
    )
    pending_rows = _read_pending_manifest(
        _pending_manifest(output_root, product, area)
    )
    raw_contracts = [
        (item.class_names_json, item.class_schema_hash)
        for item in ready_items
    ] + [
        (
            str(row.get("class_names_json") or "[]"),
            str(row.get("class_schema_hash") or ""),
        )
        for row in pending_rows
    ]
    for names_value, stored_hash in raw_contracts:
        names = _json_string_list(names_value)
        if not names or not stored_hash:
            continue
        expected_hash = _class_schema_hash(names)
        if stored_hash != expected_hash:
            raise ValueError(
                "既有訓練資料的類別 checksum 不正確，已停止送訓："
                f"{product}/{area}"
            )
        candidates.add(tuple(names))
    if len(candidates) > 1:
        raise ValueError(
            "既有訓練資料包含多種不同類別順序，已停止送訓："
            f"{product}/{area}"
        )
    return list(next(iter(candidates))) if candidates else []


def _excluded_state_row(row: dict[str, str]) -> dict[str, str]:
    """Build a non-copying state update that revokes a prior exported sample."""
    product = str(row.get("product") or "unknown")
    area = str(row.get("area") or "unknown")
    source = Path(
        str(row.get("preprocessed_path") or row.get("original_path") or "")
    )
    if source.is_file():
        image_sha256 = _sha256_file(source)
    else:
        identity = str(row.get("config_snapshot_path") or source)
        image_sha256 = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return {
        "sample_id": _sample_id(product, area, image_sha256),
        "image_sha256": image_sha256,
        "product": product,
        "area": area,
        "class_names_json": str(row.get("class_names_json") or "[]"),
        "class_map_json": str(row.get("class_map_json") or "{}"),
        "config_snapshot_path": str(row.get("config_snapshot_path") or ""),
    }


def _source_images(row: dict[str, str], source_kind: str) -> list[str]:
    values: list[str] = []
    if source_kind in {"original", "both"}:
        original = str(row.get("original_path") or "")
        if original:
            values.append(original)
    if source_kind in {"failure_crops", "both"}:
        crops = str(row.get("failure_crop_paths") or "")
        values.extend([item for item in crops.split("|") if item])
    if source_kind in {"annotated", "both"}:
        annotated = str(row.get("annotated_path") or "")
        if annotated:
            values.append(annotated)
    return values


def _sha256_file(path: Path) -> str:
    """Return a stable content identity used to prevent duplicate exports."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_output_name(
    row: dict[str, str],
    row_index: int,
    review_label: str,
    source_path: Path,
    image_sha256: str,
) -> str:
    product = _safe_name(row.get("product") or "unknown")
    area = _safe_name(row.get("area") or "unknown")
    reason = _safe_name(row.get("decision_reasons") or "review")
    return (
        f"{row_index:06d}_{review_label}_{product}_{area}_{reason}_"
        f"{image_sha256[:12]}_{source_path.name}"
    )


def _sample_id(product: str, area: str, image_sha256: str) -> str:
    """Return a stable target-local identity for identical image content."""
    identity = f"{product}\0{area}\0{image_sha256}".encode()
    return hashlib.sha256(identity).hexdigest()[:24]


def _stable_output_name(sample_id: str, source_path: Path) -> str:
    suffix = source_path.suffix.lower() or ".png"
    return f"review_{sample_id}{suffix}"


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    """Read only the stored image dimensions; return ``None`` if unreadable."""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError:
        return None
    if width <= 0 or height <= 0:
        return None
    return int(width), int(height)


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value or "[]")
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _json_string_map(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _normalized_class_map_json(row: dict[str, str]) -> str:
    observed = _json_string_map(row.get("class_map_json"))
    if not observed:
        try:
            detections = json.loads(str(row.get("detections_json") or "[]"))
        except json.JSONDecodeError:
            detections = []
        if isinstance(detections, list):
            for detection in detections:
                if not isinstance(detection, dict):
                    continue
                try:
                    class_id = str(int(detection["class_id"]))
                except (KeyError, TypeError, ValueError):
                    continue
                class_name = str(
                    detection.get("class") or detection.get("class_name") or ""
                ).strip()
                if class_name:
                    observed[class_id] = class_name
    return json.dumps(observed, ensure_ascii=False, separators=(",", ":"))


def _class_schema_hash(class_names: list[str]) -> str:
    if not class_names:
        return ""
    serialized = json.dumps(
        class_names, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _safe_name(value: str) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _write_export_manifest(items: list[ExportedReviewItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ExportedReviewItem.__dataclass_fields__.keys())
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(item) for item in items])
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _state_key(
    payload: ExportedReviewItem | dict[str, str],
) -> tuple[str, str, str]:
    """Return the target-local identity used for reversible state updates."""
    if isinstance(payload, ExportedReviewItem):
        return payload.product, payload.area, payload.sample_id
    return (
        str(payload.get("product") or "unknown"),
        str(payload.get("area") or "unknown"),
        str(payload.get("sample_id") or payload.get("config_snapshot_path") or ""),
    )


def _target_manifest(output_root: Path, product: str, area: str) -> Path:
    return (
        output_root
        / _safe_name(product)
        / _safe_name(area)
        / "metadata"
        / "review_dataset_manifest.csv"
    )


def _pending_manifest(output_root: Path, product: str, area: str) -> Path:
    return (
        output_root
        / _safe_name(product)
        / _safe_name(area)
        / "review_pending"
        / "manifest.csv"
    )


def _apply_operator_state_updates(
    updates: list[tuple[str, ExportedReviewItem | dict[str, str]]],
    output_root: Path,
) -> None:
    """Apply latest ready/pending decisions and remove the superseded state."""
    grouped: dict[
        tuple[str, str], list[tuple[str, ExportedReviewItem | dict[str, str]]]
    ] = {}
    for state, payload in updates:
        product, area, _sample_id_value = _state_key(payload)
        grouped.setdefault((product, area), []).append((state, payload))

    for (product, area), target_updates in grouped.items():
        target_root = output_root / _safe_name(product) / _safe_name(area)
        ready_path = _target_manifest(output_root, product, area)
        pending_path = _pending_manifest(output_root, product, area)
        ready_by_id = {
            item.sample_id or _sample_id(item.product, item.area, item.image_sha256): item
            for item in _read_export_manifest(ready_path)
        }
        pending_by_id = {
            str(row.get("sample_id") or row.get("config_snapshot_path") or ""): row
            for row in _read_pending_manifest(pending_path)
            if str(row.get("sample_id") or row.get("config_snapshot_path") or "")
        }

        for state, payload in target_updates:
            _product, _area, sample_id = _state_key(payload)
            if state == "ready" and isinstance(payload, ExportedReviewItem):
                old_pending = pending_by_id.pop(sample_id, None)
                if old_pending:
                    _remove_pending_files(old_pending, target_root)
                old_ready = ready_by_id.get(sample_id)
                if old_ready and old_ready.output_image != payload.output_image:
                    _remove_ready_files(old_ready, target_root)
                ready_by_id[sample_id] = payload
                continue

            if state == "pending" and isinstance(payload, dict):
                old_ready = ready_by_id.pop(sample_id, None)
                if old_ready:
                    _remove_ready_files(old_ready, target_root)
                old_pending = pending_by_id.get(sample_id)
                if old_pending and old_pending.get("output_image") != payload.get(
                    "output_image"
                ):
                    _remove_pending_files(old_pending, target_root)
                pending_by_id[sample_id] = payload
                continue

            if state == "excluded":
                old_ready = ready_by_id.pop(sample_id, None)
                if old_ready:
                    _remove_ready_files(old_ready, target_root)
                old_pending = pending_by_id.pop(sample_id, None)
                if old_pending:
                    _remove_pending_files(old_pending, target_root)

        ready_items = sorted(ready_by_id.values(), key=lambda item: item.sample_id)
        pending_rows = sorted(
            pending_by_id.values(), key=lambda row: str(row.get("sample_id") or "")
        )
        _write_export_manifest(ready_items, ready_path)
        _write_pending_manifest(pending_path, pending_rows)
        _cleanup_legacy_review_files(target_root, ready_items)

    all_ready: list[ExportedReviewItem] = []
    all_pending: list[dict[str, str]] = []
    for path in sorted(output_root.glob("*/*/metadata/review_dataset_manifest.csv")):
        all_ready.extend(_read_export_manifest(path))
    for path in sorted(output_root.glob("*/*/review_pending/manifest.csv")):
        all_pending.extend(_read_pending_manifest(path))
    _write_export_manifest(
        _merge_export_items([], all_ready),
        output_root / "metadata" / "review_dataset_manifest.csv",
    )
    _write_pending_manifest(
        output_root / ".operator_handoff" / "pending.csv",
        _merge_pending_rows([], all_pending),
    )


def _remove_ready_files(item: ExportedReviewItem, target_root: Path) -> None:
    for value in (item.output_image, item.output_label):
        _unlink_managed_path(value, target_root)


def _remove_pending_files(row: dict[str, str], target_root: Path) -> None:
    for field in ("output_image", "output_label"):
        _unlink_managed_path(str(row.get(field) or ""), target_root)


def _unlink_managed_path(value: str, target_root: Path) -> None:
    if not value:
        return
    path = Path(value)
    try:
        resolved = path.resolve()
        allowed = target_root.resolve()
    except OSError:
        return
    if not resolved.is_relative_to(allowed):
        return
    try:
        resolved.unlink()
    except FileNotFoundError:
        pass


def _cleanup_legacy_review_files(
    target_root: Path, ready_items: list[ExportedReviewItem]
) -> None:
    """Remove orphan files created by the previous row-index naming scheme."""
    referenced = {
        Path(value).resolve()
        for item in ready_items
        for value in (item.output_image, item.output_label)
        if value
    }
    for folder in (target_root / "raw" / "images", target_root / "raw" / "labels"):
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file() and "_confirmed_ng_" in path.name:
                if path.resolve() not in referenced:
                    path.unlink()


def _build_target_class_contracts(
    updates: list[tuple[str, ExportedReviewItem | dict[str, str]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Build and validate ordered class contracts for each handoff target."""
    contracts: dict[tuple[str, str], dict[str, Any]] = {}
    for state, payload in updates:
        if state == "excluded":
            continue
        product, area, _sample_id_value = _state_key(payload)
        if isinstance(payload, ExportedReviewItem):
            names = _json_string_list(payload.class_names_json)
            observed = _json_string_map(payload.class_map_json)
        else:
            names = _json_string_list(payload.get("class_names_json"))
            observed = _json_string_map(payload.get("class_map_json"))
        current = contracts.setdefault(
            (product, area), {"class_names": [], "observed_class_map": {}}
        )
        if names:
            if any(not name.strip() for name in names) or len(set(names)) != len(names):
                raise ValueError(
                    f"Invalid ordered class contract for {product}/{area}: {names!r}"
                )
            existing_names = current["class_names"]
            if existing_names and existing_names != names:
                raise ValueError(
                    f"Conflicting class order for {product}/{area}: "
                    f"{existing_names!r} != {names!r}"
                )
            current["class_names"] = names
        for class_id, class_name in observed.items():
            previous = current["observed_class_map"].get(class_id)
            if previous and previous != class_name:
                raise ValueError(
                    f"Conflicting class mapping for {product}/{area}: "
                    f"{class_id}={previous!r}/{class_name!r}"
                )
            current["observed_class_map"][class_id] = class_name

    for (product, area), contract in contracts.items():
        names = contract["class_names"]
        for raw_id, class_name in contract["observed_class_map"].items():
            class_id = int(raw_id)
            if names and (
                class_id >= len(names) or names[class_id] != class_name
            ):
                raise ValueError(
                    f"Class metadata mismatch for {product}/{area}: "
                    f"id {class_id} is {class_name!r}"
                )
        contract["class_schema_hash"] = _class_schema_hash(names)
        contract["class_contract_required"] = bool(
            names or contract["observed_class_map"]
        )
    return contracts


@contextmanager
def _handoff_export_lock(output_root: Path, timeout_seconds: float = 10.0):
    """Serialize review exports so two OP windows cannot lose updates."""
    lock_path = output_root / ".operator_handoff" / "export.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(descriptor, str(os.getpid()).encode("ascii"))
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 300.0
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError("Training-data export is already running.") from None
            time.sleep(0.1)
    try:
        yield
    finally:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _merge_ready_manifests(
    new_items: list[ExportedReviewItem], output_root: Path
) -> None:
    grouped: dict[tuple[str, str], list[ExportedReviewItem]] = {}
    for item in new_items:
        grouped.setdefault((item.product, item.area), []).append(item)
    for (product, area), items in grouped.items():
        manifest_path = (
            output_root
            / _safe_name(product)
            / _safe_name(area)
            / "metadata"
            / "review_dataset_manifest.csv"
        )
        merged = _merge_export_items(_read_export_manifest(manifest_path), items)
        _write_export_manifest(merged, manifest_path)

    all_items: list[ExportedReviewItem] = []
    for manifest_path in sorted(
        output_root.glob("*/*/metadata/review_dataset_manifest.csv")
    ):
        all_items.extend(_read_export_manifest(manifest_path))
    _write_export_manifest(
        _merge_export_items([], all_items),
        output_root / "metadata" / "review_dataset_manifest.csv",
    )


def _read_export_manifest(path: Path) -> list[ExportedReviewItem]:
    if not path.exists():
        return []
    fieldnames = ExportedReviewItem.__dataclass_fields__.keys()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                ExportedReviewItem(
                    **{field: str(row.get(field) or "") for field in fieldnames}
                )
                for row in csv.DictReader(handle)
            ]
    except (OSError, UnicodeDecodeError, csv.Error, TypeError):
        return []


def _merge_export_items(
    existing: list[ExportedReviewItem], new_items: list[ExportedReviewItem]
) -> list[ExportedReviewItem]:
    merged: dict[tuple[str, str, str], ExportedReviewItem] = {}
    for item in [*existing, *new_items]:
        key = (
            item.product,
            item.area,
            item.sample_id or item.image_sha256 or item.output_image,
        )
        merged[key] = item
    return sorted(merged.values(), key=lambda item: item.output_image)


def _write_pending_manifests(
    new_rows: list[dict[str, str]], output_root: Path
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in new_rows:
        grouped.setdefault((row["product"], row["area"]), []).append(row)
    all_rows: list[dict[str, str]] = []
    for (product, area), rows in grouped.items():
        path = (
            output_root
            / _safe_name(product)
            / _safe_name(area)
            / "review_pending"
            / "manifest.csv"
        )
        existing = _read_pending_manifest(path)
        merged = _merge_pending_rows(existing, rows)
        _write_pending_manifest(path, merged)
        all_rows.extend(merged)
    _write_pending_manifest(
        output_root / ".operator_handoff" / "pending.csv",
        _merge_pending_rows([], all_rows),
    )


def _read_pending_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _merge_pending_rows(
    existing: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in [*existing, *new_rows]:
        key = str(
            row.get("sample_id")
            or row.get("config_snapshot_path")
            or row.get("output_image")
            or ""
        )
        if key:
            merged[key] = row
    return sorted(merged.values(), key=lambda row: row.get("output_image", ""))


def _write_pending_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "image_sha256",
        "product",
        "area",
        "timestamp",
        "review_label",
        "reason",
        "annotation_status",
        "review_note",
        "status",
        "decision_reasons",
        "model_version",
        "class_names_json",
        "class_map_json",
        "class_schema_hash",
        "detections_json",
        "source_image",
        "output_image",
        "output_label",
        "label_baseline_sha256",
        "config_snapshot_path",
    ]
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [
                    {field: str(row.get(field) or "") for field in fields}
                    for row in rows
                ]
            )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


TERMINAL_OPERATOR_JOB_STATES = {"deployed", "failed", "cancelled"}
JOB_START_GRACE_SECONDS = 5 * 60


def update_operator_job_status(
    status_path: str | Path,
    *,
    state: str,
    message: str,
    **values: Any,
) -> Path:
    """Atomically update the shared inference/training job status.

    Args:
        status_path: Job-specific ``status.json`` path.
        state: Stable machine-readable lifecycle state.
        message: Concise operator-facing status text.
        **values: Additional serializable fields such as progress or counts.

    Returns:
        Resolved status path.
    """
    path = Path(status_path).expanduser().resolve()
    payload = _read_json_mapping(path)
    payload.update({key: value for key, value in values.items() if value is not None})
    payload.update(
        {
            "schema_version": 1,
            "state": str(state),
            "message": str(message),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_json_atomic(path, payload)
    return path


def _new_operator_job_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:10]}"


def _operator_submission_hash(
    updates: list[tuple[str, ExportedReviewItem | dict[str, str]]],
) -> str:
    fingerprint: list[dict[str, str]] = []
    for state, payload in updates:
        values = asdict(payload) if isinstance(payload, ExportedReviewItem) else payload
        product, area, sample_id = _state_key(payload)
        fingerprint.append(
            {
                "state": state,
                "product": product,
                "area": area,
                "sample_id": sample_id,
                "image_sha256": str(values.get("image_sha256") or ""),
                "review_label": str(values.get("review_label") or ""),
                "annotation_status": str(values.get("annotation_status") or ""),
                "class_schema_hash": str(values.get("class_schema_hash") or ""),
            }
        )
    serialized = json.dumps(
        sorted(
            fingerprint,
            key=lambda item: (
                item["product"], item["area"], item["sample_id"], item["state"]
            ),
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _find_active_operator_job(
    output_root: Path, submission_hash: str
) -> Path | None:
    jobs_root = output_root / ".operator_handoff" / "jobs"
    if not jobs_root.is_dir():
        return None
    for job_dir in sorted(jobs_root.iterdir(), reverse=True):
        if not job_dir.is_dir():
            continue
        handoff_path = job_dir / "handoff.json"
        handoff = _read_json_mapping(handoff_path)
        if str(handoff.get("submission_hash") or "") != submission_hash:
            continue
        status = _read_json_mapping(job_dir / "status.json")
        state = str(status.get("state") or "").strip()
        if (
            state
            and state not in TERMINAL_OPERATOR_JOB_STATES
            and _job_is_alive(status)
        ):
            return handoff_path.resolve()
    return None


def _job_is_alive(status: dict[str, Any]) -> bool:
    """Return whether a non-terminal job still has a live local launcher."""
    try:
        process_id = int(status.get("training_process_id", 0))
    except (TypeError, ValueError):
        process_id = 0
    process_host = str(status.get("training_process_host") or "")
    if process_id > 0:
        return is_process_active(process_id, process_host)

    # The inference GUI writes the initial status immediately before launching.
    # Keep a short grace period so a rapid duplicate click cannot create a
    # second job before the detached-process PID is published.
    timestamp = str(status.get("updated_at") or "")
    try:
        updated_at = datetime.fromisoformat(timestamp)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - updated_at).total_seconds()
    except (TypeError, ValueError):
        return False
    return age_seconds < JOB_START_GRACE_SECONDS


def _read_json_mapping(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_manifests(
    items: list[ExportedReviewItem], output_root: Path, *, group_by_target: bool
) -> None:
    """Write a master manifest and target-local manifests for training."""
    _write_export_manifest(items, output_root / "metadata" / "review_dataset_manifest.csv")
    if not group_by_target:
        return
    targets: dict[tuple[str, str], list[ExportedReviewItem]] = {}
    for item in items:
        targets.setdefault((item.product, item.area), []).append(item)
    for (product, area), target_items in targets.items():
        _write_export_manifest(
            target_items,
            output_root
            / _safe_name(product or "unknown")
            / _safe_name(area or "unknown")
            / "metadata"
            / "review_dataset_manifest.csv",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, help="Reviewed review_manifest.csv path")
    parser.add_argument("--output-dir", required=True, help="Destination dataset curation directory")
    parser.add_argument(
        "--source-kind",
        choices=["original", "failure_crops", "annotated", "both"],
        default="original",
        help="Which image evidence to export",
    )
    parser.add_argument(
        "--include-label",
        action="append",
        default=None,
        help="Review label to include. Repeatable. Defaults to all non-empty review labels.",
    )
    parser.add_argument(
        "--create-label-placeholders",
        action="store_true",
        help="Create empty pending YOLO labels for annotation tools",
    )
    parser.add_argument(
        "--flat-layout",
        action="store_true",
        help="Write directly to output-dir/raw instead of product/area roots",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    include = set(args.include_label) if args.include_label else None
    items = export_review_dataset(
        args.manifest_csv,
        args.output_dir,
        include_labels=include,
        source_kind=args.source_kind,
        create_label_placeholders=args.create_label_placeholders,
        group_by_target=not args.flat_layout,
    )
    print(f"Exported {len(items)} review images to {args.output_dir}")
    print("Annotate generated raw/images and raw/labels before Yolo11_auto_train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
