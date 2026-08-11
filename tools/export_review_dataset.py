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
import logging
import math
import os
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from core.retraining_options import RetrainingOptions
from core.station_data import resolve_inference_path_contract
from core.training_batch_version import validate_training_batch_version
from tools.color_feedback import export_color_feedback
from tools.process_liveness import is_process_active
from tools.retraining_workspaces import load_retraining_workspace
from tools.review_routing import action_route
from tools.review_workflow import (
    blocking_violations,
    record_identity,
    validate_record_consistency,
)
from tools.submission_history import (
    claim_training_batch_version,
    ensure_training_batch_version_available,
)

DEFAULT_LABELS = {
    "confirmed_ng",
    "verified_empty",
    "false_positive",
    "false_negative",
    "wrong_box",
    "wrong_class",
    "position_false_reject",
}

HOLD_LABELS = {"confirmed_ok", "uncertain", "image_quality_issue"}
MISSING_LABEL_BASELINE = "missing"
SNAPSHOT_DUPLICATE_IOU_THRESHOLD = 0.90
logger = logging.getLogger(__name__)


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
class DeduplicationAuditRecord:
    """One explicit canonical selection for byte-identical image content."""

    image_sha256: str
    kept_sample: str
    excluded_sample: str
    reason: str
    kept_label_sha256: str
    excluded_label_sha256: str
    kept_source_type: str
    excluded_source_type: str


class CanonicalLabelConflictError(ValueError):
    """Raised when duplicate labels have no auditable canonical winner."""

    def __init__(self, first: ExportedReviewItem, second: ExportedReviewItem) -> None:
        reason = (
            "conflicting_equally_authoritative_labels"
            if _ready_item_priority(first) == _ready_item_priority(second)
            else "conflicting_labels_without_canonical_rule"
        )
        first_label_sha = _label_sha256(first.output_label)
        second_label_sha = _label_sha256(second.output_label)
        original_sample_ids = [first.sample_id, second.sample_id]
        sample_ids = original_sample_ids
        if first.sample_id == second.sample_id:
            sample_ids = [
                f"{first.sample_id}@label-{first_label_sha[:12]}",
                f"{second.sample_id}@label-{second_label_sha[:12]}",
            ]
        self.conflict = {
            "image_sha256": first.image_sha256,
            "sample_ids": sample_ids,
            "original_sample_ids": original_sample_ids,
            "label_sha256s": [
                first_label_sha,
                second_label_sha,
            ],
            "normalized_labels": [
                list(_label_signature(first.output_label)),
                list(_label_signature(second.output_label)),
            ],
            "source_types": [
                _ready_source_type(first),
                _ready_source_type(second),
            ],
            "label_paths": [first.output_label, second.output_label],
            "reason": reason,
        }
        super().__init__(
            "Identical image content has conflicting labels without an applicable "
            "canonical rule; "
            "automatic canonical selection is forbidden: "
            f"image_sha={first.image_sha256}, "
            f"samples={','.join(sample_ids)}, "
            f"label_sha={','.join(self.conflict['label_sha256s'])}, "
            f"source_type={','.join(self.conflict['source_types'])}"
        )


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
    color_feedback_count: int = 0
    color_case_count: int = 0
    color_manifest_paths: tuple[Path, ...] = ()
    batch_version: str = ""


@dataclass(frozen=True)
class _SnapshotDetection:
    """One validated snapshot detection in stored-image pixel coordinates."""

    source_index: int
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    image_width: float
    image_height: float


@dataclass(frozen=True)
class _TrainingImageSource:
    """Image selected for YOLO training and the canvas that owns its boxes."""

    path: Path
    image_size: tuple[int, int]
    detection_image_size: tuple[int, int]


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
    inference_station_data_dir: str | Path | None = None,
    inference_project_root: str | Path | None = None,
    training_options: dict[str, Any] | None = None,
    batch_version: str = "",
    batch_workspace_dir: str | Path | None = None,
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
    inference_paths = resolve_inference_path_contract(
        models_dir=inference_models_dir,
        station_data_dir=inference_station_data_dir,
        project_root=inference_project_root,
        start=Path(__file__).resolve().parents[1],
    )
    resolved_inference_models_dir = inference_paths.models_dir
    resolved_inference_station_data_dir = inference_paths.station_data_dir
    resolved_inference_project_root = inference_paths.project_root
    selected_training_options = RetrainingOptions.from_mapping(training_options)
    requested_batch_version = str(batch_version or "").strip()
    requested_workspace_dir = (
        Path(batch_workspace_dir).expanduser().resolve()
        if batch_workspace_dir is not None
        else None
    )
    selected_workspace = None
    updates: list[tuple[str, ExportedReviewItem | dict[str, str]]] = []
    skipped_count = 0

    with _handoff_export_lock(output_root):
        approved_annotation_selections = _load_approved_annotation_selections(
            manifest_path
        )
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            review_rows = list(csv.DictReader(handle))
        review_rows = _enrich_legacy_class_contracts(review_rows, output_root)
        _preflight_review_workflow(review_rows)
        _preflight_operator_class_contracts(review_rows)
        normalized_batch_version = ""
        version_target: tuple[str, str] | None = None
        if requested_batch_version:
            version_targets = {
                (
                    str(row.get("product") or "").strip(),
                    str(row.get("area") or "").strip(),
                )
                for row in review_rows
                if str(row.get("review_label") or "").strip()
                and str(row.get("training_selected") or "1") != "0"
                and action_route(row) != "color"
            }
            if len(version_targets) != 1:
                raise ValueError(
                    "具名補訓批次只能包含一個產品與工位。請分開送訓。"
                )
            version_target = next(iter(version_targets))
            if not all(version_target):
                raise ValueError(
                    "具名補訓批次的產品與工位不可為空白。"
                )
            version_product, version_area = version_target
            if requested_workspace_dir is not None:
                workspace = load_retraining_workspace(requested_workspace_dir)
                selected_workspace = workspace
                expected_workspace_parent = (
                    output_root / ".operator_handoff" / "jobs"
                ).resolve()
                if workspace.root.parent != expected_workspace_parent:
                    raise ValueError(
                        "補訓資料夾不在目前訓練中心內。"
                    )
                if (workspace.product, workspace.area) != version_target:
                    raise ValueError(
                        "補訓資料夾與所選照片的產品／工位不一致。"
                    )
                normalized_batch_version = validate_training_batch_version(
                    requested_batch_version,
                    product=workspace.product,
                    area=workspace.area,
                )
                if normalized_batch_version != workspace.batch_version:
                    raise ValueError(
                        "補訓資料夾名稱與送訓版本不一致。"
                    )
                claim_training_batch_version(
                    output_root,
                    batch_version=normalized_batch_version,
                    product=workspace.product,
                    area=workspace.area,
                    submission_hash=f"workspace:{normalized_batch_version.casefold()}",
                    owner_job_id=normalized_batch_version,
                    record_path=workspace.metadata_path,
                )
            else:
                normalized_batch_version = ensure_training_batch_version_available(
                    output_root,
                    product=version_product,
                    area=version_area,
                    batch_version=requested_batch_version,
                )
        allowed_ready_rows, deduplication_records = (
            _preflight_ready_canonical_selection(
                review_rows,
                manifest_path=manifest_path,
                output_root=output_root,
                approved_selections=approved_annotation_selections,
            )
        )
        color_feedback = export_color_feedback(
            review_rows,
            source_manifest=manifest_path,
            output_root=output_root,
        )
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
            route = action_route(row)
            if route == "color":
                continue
            if route == "both":
                review_label = str(row.get("detection_verdict") or "").strip()
            if review_label in {"confirmed_ng", "position_false_reject"}:
                if row_index not in allowed_ready_rows:
                    continue
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
                if row_index not in allowed_ready_rows:
                    continue
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
                "wrong_box",
                "wrong_class",
            }:
                pending_reason = {
                    "false_positive": "false_detection_requires_correction",
                    "false_negative": "missed_detection_requires_box_annotation",
                    "wrong_box": "box_geometry_requires_correction",
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

        if not final_updates and color_feedback.item_count:
            return OperatorHandoffReport(
                handoff_path=color_feedback.manifest_paths[0],
                ready_count=0,
                pending_count=0,
                skipped_count=skipped_count,
                targets=color_feedback.targets,
                color_feedback_count=color_feedback.item_count,
                color_case_count=color_feedback.case_count,
                color_manifest_paths=color_feedback.manifest_paths,
            )

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
        _apply_operator_state_updates(
            final_updates,
            output_root,
            approved_selections=approved_annotation_selections,
        )
        deduplication_audit_path = _write_deduplication_audit(
            output_root,
            source_manifest=manifest_path,
            records=deduplication_records,
        )
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
        if normalized_batch_version and targets != [version_target]:
            raise ValueError(
                "補訓批次目標與可訓練資料不一致，已停止送訓。"
            )
        submission_hash = _operator_submission_hash(
            final_updates,
            selected_training_options,
            normalized_batch_version,
            inference_models_dir=resolved_inference_models_dir,
            inference_station_data_dir=resolved_inference_station_data_dir,
            inference_project_root=resolved_inference_project_root,
        )
        existing_handoff = _find_active_operator_job(
            output_root,
            submission_hash,
            minimum_schema_version=6,
        )
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
                color_feedback_count=color_feedback.item_count,
                color_case_count=color_feedback.case_count,
                color_manifest_paths=color_feedback.manifest_paths,
                batch_version=str(existing_payload.get("batch_version") or ""),
            )

        if selected_workspace is not None and selected_workspace.state != "draft":
            raise ValueError(
                f"補訓資料夾已送出：{selected_workspace.batch_version}。"
                "請建立下一個版本資料夾。"
            )

        job_id = (
            normalized_batch_version
            if requested_workspace_dir is not None
            else _new_operator_job_id()
        )
        job_dir = (
            requested_workspace_dir
            if requested_workspace_dir is not None
            else output_root / ".operator_handoff" / "jobs" / job_id
        )
        handoff_path = job_dir / "handoff.json"
        status_path = job_dir / "status.json"
        handoff_payload = {
            "schema_version": 6,
            "job_id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "submission_hash": submission_hash,
            "batch_version": normalized_batch_version,
            "batch_workspace_dir": str(requested_workspace_dir or ""),
            "source_manifest": str(manifest_path.resolve()),
            "data_root": str(output_root.resolve()),
            "status_path": str(status_path.resolve()),
            "inference_models_dir": str(resolved_inference_models_dir),
            "inference_station_data_dir": str(
                resolved_inference_station_data_dir
            ),
            "inference_project_root": str(resolved_inference_project_root),
            "ready_count": len(ready_items),
            "total_ready_count": total_ready_count,
            "pending_count": len(pending_rows),
            "skipped_count": skipped_count,
            "training_options": selected_training_options.to_dict(),
            "deduplication_audit_path": str(deduplication_audit_path or ""),
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
        if normalized_batch_version:
            claim_training_batch_version(
                output_root,
                batch_version=normalized_batch_version,
                product=targets[0][0],
                area=targets[0][1],
                submission_hash=submission_hash,
                owner_job_id=job_id,
                record_path=handoff_path,
            )
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
            batch_version=normalized_batch_version,
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
        color_feedback_count=color_feedback.item_count,
        color_case_count=color_feedback.case_count,
        color_manifest_paths=color_feedback.manifest_paths,
        batch_version=normalized_batch_version,
    )


def _original_first_path(row: dict[str, str]) -> Path:
    """Return the camera frame when available, otherwise the inference canvas."""
    original = Path(str(row.get("original_path") or ""))
    if original.is_file():
        return original
    return Path(str(row.get("preprocessed_path") or ""))


def _select_training_image(
    row: dict[str, str], *, require_detection_canvas: bool
) -> _TrainingImageSource | None:
    """Select an original-first training image with an explicit box canvas."""
    source_path = _original_first_path(row)
    source_size = _read_image_size(source_path) if source_path.is_file() else None
    if source_size is None:
        return None

    processed_path = Path(str(row.get("preprocessed_path") or ""))
    processed_size = (
        _read_image_size(processed_path) if processed_path.is_file() else None
    )
    if source_path == processed_path:
        processed_size = source_size
    if require_detection_canvas and processed_size is None:
        return None
    return _TrainingImageSource(
        path=source_path,
        image_size=source_size,
        detection_image_size=processed_size or source_size,
    )


def prepare_annotation_draft(row: Mapping[str, Any]) -> tuple[Path, str]:
    """Return the original-first image and current YOLO evidence without writes.

    Phase 3C2 uses this narrow adapter so annotation packaging retains the same
    snapshot-to-original coordinate conversion as the Phase 1A exporter.
    An existing verified operator label is preferred over the AI snapshot.
    """
    values = {str(key): str(value or "") for key, value in row.items()}
    source = _select_training_image(values, require_detection_canvas=True)
    if source is None:
        raise ValueError("training_image_or_detection_canvas_missing")
    output_label = Path(values.get("output_label") or "")
    if (
        values.get("annotation_status", "").strip() == "verified_annotation"
        and output_label.is_file()
    ):
        return source.path, output_label.read_text(encoding="utf-8")
    class_names = _json_string_list(values.get("class_names_json"))
    lines = _snapshot_yolo_label_lines(
        values.get("detections_json") or "",
        image_size=source.image_size,
        detection_image_size=source.detection_image_size,
        class_names=class_names,
    )
    return source.path, ("\n".join(lines) + "\n" if lines else "")


def _export_snapshot_verified_row(
    row: dict[str, str],
    row_index: int,
    manifest_path: Path,
    output_root: Path,
    *,
    materialize: bool = True,
    copy_image: bool = True,
) -> tuple[ExportedReviewItem | None, str]:
    source = _select_training_image(row, require_detection_canvas=True)
    if source is None:
        return None, "training_image_or_detection_canvas_missing"
    source_path = source.path
    class_names = _json_string_list(row.get("class_names_json"))
    label_lines = _snapshot_yolo_label_lines(
        row.get("detections_json") or "",
        image_size=source.image_size,
        detection_image_size=source.detection_image_size,
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
    output_name = _stable_output_name(sample_id, source_path)
    output_image = images_dir / output_name
    output_label = labels_dir / f"{output_image.stem}.txt"
    if materialize:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        if copy_image and not output_image.exists():
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
    detection_image_size: tuple[int, int] | None = None,
    class_names: list[str] | None = None,
) -> list[str]:
    """Convert snapshot boxes into labels for the selected training image.

    ``bbox`` values are produced on the resized/letterboxed inference canvas.
    When the selected training image is the original camera frame,
    ``detection_image_size`` is used to reverse that letterbox transform before
    YOLO normalization.  Legacy callers that omit it keep same-canvas behavior.
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
    detection_width: float | None = None
    detection_height: float | None = None
    if detection_image_size is not None:
        detection_width, detection_height = map(float, detection_image_size)
        if detection_width <= 0 or detection_height <= 0:
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
            width = detection_width or actual_width or float(raw["image_width"])
            height = detection_height or actual_height or float(raw["image_height"])
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
        if (
            actual_width is not None
            and actual_height is not None
            and detection_width is not None
            and detection_height is not None
            and (actual_width, actual_height) != (detection_width, detection_height)
        ):
            projected = _reverse_letterbox_bbox(
                (x1, y1, x2, y2),
                original_size=(actual_width, actual_height),
                letterbox_size=(detection_width, detection_height),
            )
            if projected is None:
                continue
            x1, y1, x2, y2 = projected
            width, height = actual_width, actual_height
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


def _reverse_letterbox_bbox(
    bbox: tuple[float, float, float, float],
    *,
    original_size: tuple[float, float],
    letterbox_size: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    """Project one XYXY box from ``ImageUtils.letterbox`` back to the source."""
    original_width, original_height = original_size
    canvas_width, canvas_height = letterbox_size
    if min(original_width, original_height, canvas_width, canvas_height) <= 0:
        return None
    ratio = min(canvas_width / original_width, canvas_height / original_height)
    resized_width = int(original_width * ratio)
    resized_height = int(original_height * ratio)
    if resized_width <= 0 or resized_height <= 0:
        return None
    left = int((canvas_width - resized_width) // 2)
    top = int((canvas_height - resized_height) // 2)
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    x1, y1, x2, y2 = bbox
    projected = (
        (x1 - left) / scale_x,
        (y1 - top) / scale_y,
        (x2 - left) / scale_x,
        (y2 - top) / scale_y,
    )
    px1, py1, px2, py2 = projected
    px1, px2 = max(0.0, px1), min(original_width, px2)
    py1, py2 = max(0.0, py1), min(original_height, py2)
    if px2 <= px1 or py2 <= py1:
        return None
    return px1, py1, px2, py2


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
    row: dict[str, str],
    manifest_path: Path,
    output_root: Path,
    *,
    materialize: bool = True,
    copy_image: bool = True,
) -> tuple[ExportedReviewItem | None, str]:
    """Export an explicitly verified background image with an empty label."""
    source_path = _original_first_path(row)
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
    if materialize:
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        if copy_image and not output_image.exists():
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


def prepare_ready_review_item(
    row: dict[str, str],
    *,
    row_index: int,
    manifest_path: str | Path,
    workspace: str | Path,
    copy_image: bool = False,
) -> tuple[ExportedReviewItem | None, str]:
    """Build one READY item using the legacy handoff conversion rules.

    The workspace receives only the normalized label when ``copy_image`` is
    false. This lets dry-run callers exercise the exact Phase 1A label and
    canonical logic without creating a dataset or handoff job.
    """
    values = {str(key): str(value or "") for key, value in row.items()}
    review_label = str(values.get("review_label") or "").strip()
    if review_label in {"confirmed_ng", "position_false_reject"}:
        return _export_snapshot_verified_row(
            values,
            row_index,
            Path(manifest_path),
            Path(workspace),
            materialize=True,
            copy_image=copy_image,
        )
    if review_label == "verified_empty":
        return _export_verified_empty_row(
            values,
            Path(manifest_path),
            Path(workspace),
            materialize=True,
            copy_image=copy_image,
        )
    return None, "review_label_is_not_dataset_ready"


def _export_pending_row(
    row: dict[str, str], output_root: Path, reason: str
) -> dict[str, str]:
    product = str(row.get("product") or "unknown")
    area = str(row.get("area") or "unknown")
    source = _original_first_path(row)
    detection_source = Path(str(row.get("preprocessed_path") or ""))
    if not detection_source.is_file():
        detection_source = source
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
        "detection_source_image": str(detection_source),
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
                "box_geometry_requires_correction",
                "wrong_class_requires_correction",
                "operator_uncertain_requires_review",
            } and not output_label.exists():
                draft_lines = _snapshot_yolo_label_lines(
                    str(row.get("detections_json") or ""),
                    image_size=_read_image_size(output_image),
                    detection_image_size=_read_image_size(
                        Path(str(row.get("detection_source_image") or output_image))
                    ),
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
        route = action_route(row)
        if route == "color":
            continue
        if route == "both":
            review_label = str(row.get("detection_verdict") or "").strip()
        if review_label not in {
            "confirmed_ng",
            "verified_empty",
            "false_positive",
            "false_negative",
            "wrong_box",
            "wrong_class",
            "position_false_reject",
        }:
            continue
        contract_rows.append(("pending", row))
        if review_label in {"false_positive", "false_negative", "wrong_box", "wrong_class"}:
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
        route = action_route(row)
        if route == "color":
            continue
        if route == "both":
            review_label = str(row.get("detection_verdict") or "").strip()
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
    source = _original_first_path(row)
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
    temporary: Path | None = None
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
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([asdict(item) for item in items])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _state_key(
    payload: ExportedReviewItem | dict[str, str],
) -> tuple[str, str, str]:
    """Return the target-local identity used for reversible state updates."""
    if isinstance(payload, ExportedReviewItem):
        product = payload.product
        area = payload.area
        image_sha256 = payload.image_sha256
        fallback = payload.sample_id
    else:
        product = str(payload.get("product") or "unknown")
        area = str(payload.get("area") or "unknown")
        image_sha256 = str(payload.get("image_sha256") or "")
        fallback = str(
            payload.get("sample_id") or payload.get("config_snapshot_path") or ""
        )
    identity = (
        _sample_id(product, area, image_sha256)
        if image_sha256
        else fallback
    )
    return product, area, identity


def _index_ready_items_by_content(
    items: list[ExportedReviewItem],
    *,
    audit_records: list[DeduplicationAuditRecord] | None = None,
    approved_selections: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, ExportedReviewItem], list[ExportedReviewItem]]:
    """Collapse legacy sample IDs that point to identical target-local pixels."""
    indexed: dict[str, ExportedReviewItem] = {}
    superseded: list[ExportedReviewItem] = []
    for item in items:
        identity = _state_key(item)[2]
        existing = indexed.get(identity)
        if existing is None:
            indexed[identity] = item
            continue
        if (
            existing.sample_id == item.sample_id
            and existing.annotation_status == item.annotation_status
            and _label_signature(existing.output_label)
            == _label_signature(item.output_label)
        ):
            # Idempotent refresh of the same canonical record is not a
            # deduplication event and must remain part of the submission.
            indexed[identity] = item
            continue
        approved_preferred = _approved_annotation_preference(
            existing,
            item,
            approved_selections or {},
        )
        preferred = approved_preferred or _preferred_ready_item(existing, item)
        excluded = item if preferred is existing else existing
        superseded.append(excluded)
        if audit_records is not None:
            audit_records.append(
                _deduplication_audit_record(
                    preferred,
                    excluded,
                    reason_override=(
                        "approved_human_canonical_selection"
                        if approved_preferred is not None
                        else ""
                    ),
                )
            )
        indexed[identity] = preferred
    return indexed, superseded


def select_canonical_ready_items(
    items: list[ExportedReviewItem],
    *,
    audit_records: list[DeduplicationAuditRecord] | None = None,
    approved_selections: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, ExportedReviewItem], list[ExportedReviewItem]]:
    """Public, side-effect-free Phase 1A canonical-selection service.

    Dataset preparation and the legacy handoff must share this exact rule set.
    Callers own any temporary label files referenced by ``items``.
    """
    return _index_ready_items_by_content(
        items,
        audit_records=audit_records,
        approved_selections=approved_selections,
    )


def _preferred_ready_item(
    first: ExportedReviewItem,
    second: ExportedReviewItem,
) -> ExportedReviewItem:
    """Apply only explicit, auditable canonical selection rules."""
    first_rank = _ready_item_priority(first)
    second_rank = _ready_item_priority(second)
    labels_match = _label_signature(first.output_label) == _label_signature(
        second.output_label
    )
    if labels_match:
        if first_rank != second_rank:
            return first if first_rank > second_rank else second
        first_canonical = _is_canonical_ready_item(first)
        second_canonical = _is_canonical_ready_item(second)
        if first_canonical != second_canonical:
            return first if first_canonical else second
        return min((first, second), key=lambda item: item.sample_id)
    first_source = _ready_source_type(first)
    second_source = _ready_source_type(second)
    if first_source.startswith("human_") and second_source == "ai_snapshot":
        return first
    if second_source.startswith("human_") and first_source == "ai_snapshot":
        return second
    raise CanonicalLabelConflictError(first, second)


def _ready_item_priority(item: ExportedReviewItem) -> int:
    return {
        "verified_annotation": 3,
        "verified_empty": 3,
        "verified_snapshot": 2,
    }.get(item.annotation_status, 1)


def _is_canonical_ready_item(item: ExportedReviewItem) -> bool:
    canonical_id = _sample_id(item.product, item.area, item.image_sha256)
    return bool(item.image_sha256) and item.sample_id == canonical_id


def _ready_source_type(item: ExportedReviewItem) -> str:
    return {
        "verified_annotation": "human_annotation",
        "verified_empty": "human_empty",
        "verified_snapshot": "ai_snapshot",
    }.get(item.annotation_status, "legacy_unknown")


def _deduplication_audit_record(
    kept: ExportedReviewItem,
    excluded: ExportedReviewItem,
    *,
    reason_override: str = "",
) -> DeduplicationAuditRecord:
    kept_label_sha = _label_sha256(kept.output_label)
    excluded_label_sha = _label_sha256(excluded.output_label)
    kept_source = _ready_source_type(kept)
    excluded_source = _ready_source_type(excluded)
    if reason_override:
        reason = reason_override
    elif kept_source.startswith("human_") and excluded_source == "ai_snapshot":
        reason = (
            "identical_label_prefer_human_over_ai"
            if kept_label_sha == excluded_label_sha
            else "human_annotation_over_ai_snapshot"
        )
    elif kept_label_sha == excluded_label_sha:
        reason = (
            "identical_label_prefer_canonical_sample"
            if _is_canonical_ready_item(kept)
            else "identical_label_deterministic_sample"
        )
    else:
        reason = "higher_authority_source"
    return DeduplicationAuditRecord(
        image_sha256=kept.image_sha256,
        kept_sample=kept.sample_id,
        excluded_sample=excluded.sample_id,
        reason=reason,
        kept_label_sha256=kept_label_sha,
        excluded_label_sha256=excluded_label_sha,
        kept_source_type=kept_source,
        excluded_source_type=excluded_source,
    )


def _label_sha256(value: str) -> str:
    signature = _label_signature(value)
    serialized = "\n".join(signature).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def normalized_label_sha256(value: str | Path) -> str:
    """Return the normalized label SHA used by Phase 1A conflict checks."""
    return _label_sha256(str(value))


def _load_approved_annotation_selections(
    manifest_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load active Phase 1C canonical selections without mutating repair data."""
    root = manifest_path.resolve().parent / ".review_repairs" / "annotation_resolutions"
    if not root.is_dir():
        return {}
    selected_by_image: dict[str, dict[str, Any]] = {}
    resolution_documents: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*.json")):
        if path.name.endswith(".rollback.json"):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Approved annotation resolution is unreadable: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Approved annotation resolution is invalid: {path}")
        resolution_documents.append((path, payload))
    superseded_ids = {
        str(resolution.get("supersedes_revision_id") or "")
        for _path, payload in resolution_documents
        if isinstance((resolution := payload.get("resolution")), dict)
    }
    for path, payload in resolution_documents:
        resolution = payload.get("resolution")
        if not isinstance(resolution, dict) or resolution.get("mode") not in {
            "selected_sample", "new_annotation_revision"
        }:
            continue
        resolution_id = str(payload.get("resolution_id") or payload.get("proposal_id") or "")
        if resolution_id and resolution_id in superseded_ids:
            continue
        if resolution_id and any(
            str(revocation.get("resolution_id") or "") == resolution_id
            for revocation in (
                _read_json_mapping(candidate)
                for candidate in sorted((root / "revocations").glob("*.json"))
            )
        ):
            continue
        revocation_path = path.with_suffix(f"{path.suffix}.rollback.json")
        if revocation_path.exists():
            revocation = _read_json_mapping(revocation_path)
            expected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if (
                revocation.get("event") != "annotation_resolution_revoked"
                or revocation.get("resolution_sha256") != expected_sha
            ):
                raise ValueError(
                    f"Annotation resolution rollback record is invalid: {revocation_path}"
                )
            continue
        image_sha = str(payload.get("image_sha256") or "")
        selected_sample = str(resolution.get("selected_sample_id") or "")
        if resolution.get("mode") == "new_annotation_revision":
            new_path = Path(str(resolution.get("new_annotation_path") or ""))
            new_sha = str(resolution.get("new_annotation_sha256") or "")
            if (
                not image_sha or not selected_sample or not new_path.is_file()
                or _label_sha256(str(new_path)) != new_sha
            ):
                raise ValueError(f"Approved annotation revision is stale: {path}")
            candidate = {
                "mode": "new_annotation_revision",
                "selected_sample_id": selected_sample,
                "label_sha256_by_sample": {selected_sample: new_sha},
                "resolution_path": str(path),
                "revision_path": str(new_path),
            }
            existing = selected_by_image.get(image_sha)
            if existing is not None and existing != candidate:
                raise ValueError(
                    "Multiple active annotation resolutions disagree for image SHA "
                    f"{image_sha}"
                )
            selected_by_image[image_sha] = candidate
            continue
        sample_ids = [str(value) for value in payload.get("conflicting_sample_ids", [])]
        label_hashes = [str(value) for value in payload.get("old_label_sha256s", [])]
        if (
            not image_sha
            or selected_sample not in sample_ids
            or len(sample_ids) != len(label_hashes)
        ):
            raise ValueError(f"Approved annotation resolution is incomplete: {path}")
        candidate = {
            "mode": "selected_sample",
            "selected_sample_id": selected_sample,
            "label_sha256_by_sample": dict(zip(sample_ids, label_hashes, strict=True)),
            "resolution_path": str(path),
        }
        existing = selected_by_image.get(image_sha)
        if existing is not None and existing != candidate:
            raise ValueError(
                "Multiple active annotation resolutions disagree for image SHA "
                f"{image_sha}"
            )
        selected_by_image[image_sha] = candidate
    return selected_by_image


def load_approved_annotation_selections(
    manifest_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """Read active Phase 1C selections without mutating repair artifacts."""
    return _load_approved_annotation_selections(Path(manifest_path))


def _approved_annotation_preference(
    first: ExportedReviewItem,
    second: ExportedReviewItem,
    approved_selections: dict[str, dict[str, Any]],
) -> ExportedReviewItem | None:
    """Apply one approved human selection only to its exact conflict evidence."""
    if _label_signature(first.output_label) == _label_signature(second.output_label):
        return None
    selection = approved_selections.get(first.image_sha256)
    if selection is None or second.image_sha256 != first.image_sha256:
        return None
    expected_by_sample = selection["label_sha256_by_sample"]
    if selection.get("mode") == "new_annotation_revision":
        for item in (first, second):
            if (
                item.sample_id == selection["selected_sample_id"]
                and _label_sha256(item.output_label)
                == expected_by_sample[item.sample_id]
            ):
                return item
        return None
    candidate_by_item: dict[int, str] = {}
    for item in (first, second):
        label_sha = _label_sha256(item.output_label)
        direct = item.sample_id
        qualified = f"{item.sample_id}@label-{label_sha[:12]}"
        candidate_id = direct if direct in expected_by_sample else qualified
        candidate_by_item[id(item)] = candidate_id
    pair = set(candidate_by_item.values())
    if not pair.issubset(expected_by_sample):
        return None
    for item in (first, second):
        candidate_id = candidate_by_item[id(item)]
        if _label_sha256(item.output_label) != expected_by_sample[candidate_id]:
            raise ValueError(
                "Approved annotation resolution is stale for sample "
                f"{item.sample_id}: {selection['resolution_path']}"
            )
    selected = selection["selected_sample_id"]
    if candidate_by_item[id(first)] == selected:
        return first
    if candidate_by_item[id(second)] == selected:
        return second
    return None


def _preflight_ready_canonical_selection(
    review_rows: list[dict[str, str]],
    *,
    manifest_path: Path,
    output_root: Path,
    approved_selections: dict[str, dict[str, Any]],
) -> tuple[set[int], list[DeduplicationAuditRecord]]:
    """Resolve ready duplicates before any production label is overwritten."""
    preflight_root = (
        output_root
        / ".operator_handoff"
        / f".canonical-preflight-{os.getpid()}-{uuid.uuid4().hex}"
    )
    prospective: dict[
        tuple[str, str], list[tuple[int, ExportedReviewItem]]
    ] = {}
    affected_targets: set[tuple[str, str]] = set()
    pending_replacements: dict[tuple[str, str], set[str]] = {}
    try:
        for row_index, row in enumerate(review_rows, start=1):
            row_preflight_root = preflight_root / f"row-{row_index}"
            review_label = str(row.get("review_label") or "").strip()
            if not review_label or str(row.get("training_selected") or "1") == "0":
                continue
            route = action_route(row)
            if route == "color":
                continue
            if route == "both":
                review_label = str(row.get("detection_verdict") or "").strip()
            if review_label in {
                "false_positive",
                "false_negative",
                "wrong_box",
                "wrong_class",
            }:
                pending = _export_pending_row(
                    row,
                    output_root,
                    "canonical_preflight_pending_replacement",
                )
                identity = _state_key(pending)[2]
                if identity and str(pending.get("image_sha256") or ""):
                    target = (
                        str(pending.get("product") or "unknown"),
                        str(pending.get("area") or "unknown"),
                    )
                    pending_replacements.setdefault(target, set()).add(identity)
                continue
            item: ExportedReviewItem | None = None
            if review_label in {"confirmed_ng", "position_false_reject"}:
                item, _reason = _export_snapshot_verified_row(
                    row,
                    row_index,
                    manifest_path,
                    row_preflight_root,
                    materialize=True,
                )
            elif review_label == "verified_empty":
                item, _reason = _export_verified_empty_row(
                    row,
                    manifest_path,
                    row_preflight_root,
                    materialize=True,
                )
            if item is not None:
                affected_targets.add((item.product, item.area))
                prospective.setdefault((item.product, item.area), []).append(
                    (row_index, item)
                )

        allowed_rows: set[int] = set()
        audit_records: list[DeduplicationAuditRecord] = []
        for product, area in sorted(affected_targets):
            candidates = prospective.get((product, area), [])
            existing = _read_export_manifest(
                _target_manifest(output_root, product, area)
            )
            replaced_identities = pending_replacements.get((product, area), set())
            existing = [
                item
                for item in existing
                if _state_key(item)[2] not in replaced_identities
            ]
            combined = [*existing, *(item for _index, item in candidates)]
            indexed, _superseded = _index_ready_items_by_content(
                combined,
                audit_records=audit_records,
                approved_selections=approved_selections,
            )
            retained_object_ids = {id(item) for item in indexed.values()}
            allowed_rows.update(
                row_index
                for row_index, item in candidates
                if id(item) in retained_object_ids
            )
        return allowed_rows, audit_records
    except CanonicalLabelConflictError as exc:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        report_path = (
            output_root
            / ".operator_handoff"
            / "conflict_reports"
            / f"{timestamp}-{uuid.uuid4().hex}.json"
        )
        _write_json_atomic(
            report_path,
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_manifest": str(manifest_path.resolve()),
                "mode": "blocked_before_handoff",
                "mutation_performed": False,
                "conflicts": [exc.conflict],
                "recommended_action": (
                    "Review the conflicting authoritative labels and create an "
                    "explicit corrected review revision before resubmitting."
                ),
            },
        )
        raise ValueError(f"{exc} Conflict report: {report_path}") from exc
    finally:
        shutil.rmtree(preflight_root, ignore_errors=True)


def _preflight_review_workflow(review_rows: list[dict[str, str]]) -> None:
    """Block contradictory selected rows before any handoff artifact is written."""
    invalid: list[tuple[str, tuple[str, ...], dict[str, str]]] = []
    for row in review_rows:
        if str(row.get("training_selected") or "1").strip() == "0":
            continue
        if not str(row.get("review_label") or "").strip():
            continue
        candidate = {**row, "handoff_selected": "1"}
        violations = blocking_violations(validate_record_consistency(candidate))
        if not violations:
            continue
        sample_id = record_identity(candidate)
        codes = tuple(violation.code for violation in violations)
        fields = {
            field: str(candidate.get(field) or "")
            for field in (
                "review_selected",
                "review_outcome",
                "review_label",
                "failure_category",
                "skip_reason",
                "product_verdict",
                "detection_verdict",
                "color_verdict",
                "action_route",
                "training_selected",
            )
        }
        invalid.append((sample_id, codes, fields))
        logger.error(
            "Handoff workflow validation rejected sample=%s fields=%s "
            "violations=%s",
            sample_id,
            fields,
            list(codes),
        )
    if invalid:
        preview = "; ".join(
            f"sample={sample_id} rules={','.join(codes)}"
            for sample_id, codes, _fields in invalid[:10]
        )
        raise ValueError(
            "Review workflow validation failed before handoff; no handoff was "
            f"created: {preview}"
        )


def _write_deduplication_audit(
    output_root: Path,
    *,
    source_manifest: Path,
    records: list[DeduplicationAuditRecord],
) -> Path | None:
    """Persist one append-only audit document for every actual canonicalization."""
    if not records:
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = (
        output_root
        / ".operator_handoff"
        / "deduplication_audits"
        / f"{timestamp}-{uuid.uuid4().hex}.json"
    )
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest": str(source_manifest.resolve()),
            "records": [asdict(record) for record in records],
        },
    )
    for record in records:
        logger.info(
            "Canonical duplicate selection image_sha=%s kept=%s excluded=%s "
            "reason=%s kept_label_sha=%s excluded_label_sha=%s "
            "kept_source=%s excluded_source=%s",
            record.image_sha256,
            record.kept_sample,
            record.excluded_sample,
            record.reason,
            record.kept_label_sha256,
            record.excluded_label_sha256,
            record.kept_source_type,
            record.excluded_source_type,
        )
    return path


def _label_signature(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        return tuple(
            line.strip()
            for line in Path(value).read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeDecodeError):
        return (f"missing:{Path(value)}",)


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
    *,
    approved_selections: dict[str, dict[str, Any]] | None = None,
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
        replacement_states = {
            _state_key(payload)[2]: state
            for state, payload in target_updates
            if _state_key(payload)[2] and state in {"pending", "excluded"}
        }
        existing_ready = _read_export_manifest(ready_path)
        replaced_ready = [
            item
            for item in existing_ready
            if _state_key(item)[2] in replacement_states
        ]
        retained_ready = [
            item
            for item in existing_ready
            if _state_key(item)[2] not in replacement_states
        ]
        archive_path, archived_files = _archive_superseded_ready_items(
            replaced_ready,
            target_root,
            replacement_states=replacement_states,
        )
        try:
            for item in replaced_ready:
                _remove_ready_files(item, target_root)
            ready_by_id, superseded_ready = _index_ready_items_by_content(
                retained_ready,
                approved_selections=approved_selections,
            )
            for item in superseded_ready:
                if all(
                    item.output_image != retained.output_image
                    for retained in ready_by_id.values()
                ):
                    _remove_ready_files(item, target_root)
            pending_by_id = {
                _state_key(row)[2]: row
                for row in _read_pending_manifest(pending_path)
                if _state_key(row)[2]
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
        except (OSError, UnicodeError, ValueError, csv.Error):
            _restore_archived_ready_files(archived_files)
            _update_supersession_archive_status(archive_path, "rolled_back")
            raise
        _update_supersession_archive_status(archive_path, "committed")

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


def _archive_superseded_ready_items(
    items: list[ExportedReviewItem],
    target_root: Path,
    *,
    replacement_states: dict[str, str],
) -> tuple[Path | None, tuple[tuple[Path, Path], ...]]:
    """Preserve old ready evidence before a current review sends it elsewhere."""
    if not items:
        return None, ()
    event_id = (
        f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S.%fZ}-"
        f"{uuid.uuid4().hex}"
    )
    archive_parent = target_root / ".operator_handoff" / "superseded_ready"
    archive_path = archive_parent / event_id
    staging = archive_parent / f".{event_id}.tmp"
    archived_files: list[tuple[Path, Path]] = []
    records: list[dict[str, Any]] = []
    archive_parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir()
        for item_index, item in enumerate(items):
            identity = _state_key(item)[2]
            archived: dict[str, str] = {}
            for field, folder in (
                ("output_image", "images"),
                ("output_label", "labels"),
            ):
                source = Path(str(getattr(item, field) or ""))
                if not source.is_file() or not _is_managed_path(source, target_root):
                    continue
                relative = Path(folder) / f"{item_index:04d}-{source.name}"
                staged_destination = staging / relative
                staged_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, staged_destination)
                if _sha256_file(source) != _sha256_file(staged_destination):
                    raise OSError(f"Superseded evidence copy verification failed: {source}")
                archived[field] = str(relative)
                archived_files.append((source, archive_path / relative))
            records.append(
                {
                    "sample_id": item.sample_id,
                    "image_sha256": item.image_sha256,
                    "label_sha256": _label_sha256(item.output_label),
                    "source_type": _ready_source_type(item),
                    "replacement_state": replacement_states.get(identity, ""),
                    "reason": (
                        "current_review_requires_reannotation"
                        if replacement_states.get(identity) == "pending"
                        else "current_review_excluded"
                    ),
                    "original_output_image": item.output_image,
                    "original_output_label": item.output_label,
                    "archived_files": archived,
                }
            )
        _write_json_atomic(
            staging / "archive.json",
            {
                "schema_version": 1,
                "event_id": event_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "prepared",
                "old_annotations_preserved": True,
                "records": records,
            },
        )
        os.replace(staging, archive_path)
        return archive_path, tuple(archived_files)
    except (OSError, ValueError):
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _restore_archived_ready_files(
    archived_files: tuple[tuple[Path, Path], ...],
) -> None:
    """Best-effort rollback for evidence removed before manifest replacement."""
    for original, archived in archived_files:
        if original.exists() or not archived.is_file():
            continue
        try:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(archived, original)
        except OSError:
            logger.exception(
                "Failed to restore superseded ready evidence original=%s archive=%s",
                original,
                archived,
            )


def _update_supersession_archive_status(
    archive_path: Path | None,
    status: str,
) -> None:
    """Record whether the surrounding manifest transition committed."""
    if archive_path is None:
        return
    audit_path = archive_path / "archive.json"
    try:
        payload = _read_json_mapping(audit_path)
        payload["status"] = status
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(audit_path, payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        logger.exception(
            "Failed to update superseded-ready archive status path=%s status=%s",
            archive_path,
            status,
        )


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


def _is_managed_path(path: Path, target_root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(target_root.resolve())
    except OSError:
        return False


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
            if path.is_file() and (
                "_confirmed_ng_" in path.name
                or "_position_false_reject_" in path.name
            ):
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
    grouped: dict[tuple[str, str], list[ExportedReviewItem]] = {}
    for item in [*existing, *new_items]:
        grouped.setdefault((item.product, item.area), []).append(item)
    merged: list[ExportedReviewItem] = []
    for items in grouped.values():
        indexed, _superseded = _index_ready_items_by_content(items)
        merged.extend(indexed.values())
    return sorted(merged, key=lambda item: item.output_image)


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
        key = _state_key(row)[2] or str(row.get("output_image") or "")
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
        "detection_source_image",
        "output_image",
        "output_label",
        "label_baseline_sha256",
        "config_snapshot_path",
    ]
    temporary: Path | None = None
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
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                [
                    {field: str(row.get(field) or "") for field in fields}
                    for row in rows
                ]
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    training_options: RetrainingOptions | None = None,
    batch_version: str = "",
    *,
    inference_models_dir: Path | None = None,
    inference_station_data_dir: Path | None = None,
    inference_project_root: Path | None = None,
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
    submission_payload: dict[str, Any] = {
        "samples": sorted(
            fingerprint,
            key=lambda item: (
                item["product"],
                item["area"],
                item["sample_id"],
                item["state"],
            ),
        ),
        "training_options": (training_options or RetrainingOptions()).to_dict(),
    }
    if inference_models_dir is not None:
        submission_payload["inference_models_dir"] = str(inference_models_dir)
    if inference_station_data_dir is not None:
        submission_payload["inference_station_data_dir"] = str(
            inference_station_data_dir
        )
    if inference_project_root is not None:
        submission_payload["inference_project_root"] = str(inference_project_root)
    if batch_version:
        submission_payload["batch_version"] = batch_version
    serialized = json.dumps(
        submission_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _find_active_operator_job(
    output_root: Path,
    submission_hash: str,
    *,
    minimum_schema_version: int = 1,
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
        try:
            schema_version = int(handoff.get("schema_version", 0))
        except (TypeError, ValueError):
            continue
        if schema_version < minimum_schema_version:
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
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
