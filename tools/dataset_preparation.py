"""Phase 3C1 deterministic, fail-closed dataset preparation service.

The service prepares one immutable dataset for all READY_FOR_DATASET records.
It deliberately has no Qt, training, evaluation, deployment, or database
dependency.  Phase 1A remains the authority for snapshot conversion and
canonical duplicate selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from tools.export_review_dataset import (
    CanonicalLabelConflictError,
    DeduplicationAuditRecord,
    ExportedReviewItem,
    load_approved_annotation_selections,
    normalized_label_sha256,
    prepare_ready_review_item,
    select_canonical_ready_items,
)
from tools.processing_pipeline import (
    ProcessingPlan,
    ProcessingRecord,
    RoutingDecisionType,
)

DATASET_PREPARATION_SCHEMA_VERSION = 1
DATASET_PREPARATION_CODE_VERSION = "phase3c1-v1"
DEFAULT_SPLIT_SEED = 42
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)


class DatasetPreparationErrorCode(str, Enum):
    SOURCE_IMAGE_MISSING = "SOURCE_IMAGE_MISSING"
    SOURCE_IMAGE_STALE = "SOURCE_IMAGE_STALE"
    LABEL_MISSING = "LABEL_MISSING"
    LABEL_STALE = "LABEL_STALE"
    LABEL_PARSE_ERROR = "LABEL_PARSE_ERROR"
    CLASS_ID_INVALID = "CLASS_ID_INVALID"
    BBOX_INVALID = "BBOX_INVALID"
    CLASS_MAPPING_INVALID = "CLASS_MAPPING_INVALID"
    CANONICAL_SELECTION_STALE = "CANONICAL_SELECTION_STALE"
    ANNOTATION_CONFLICT = "ANNOTATION_CONFLICT"
    REANNOTATION_REQUIRED = "REANNOTATION_REQUIRED"
    UNSAFE_SOURCE_PATH = "UNSAFE_SOURCE_PATH"
    DATASET_COMMIT_FAILED = "DATASET_COMMIT_FAILED"
    DATASET_VERIFY_FAILED = "DATASET_VERIFY_FAILED"
    CANCELLED = "CANCELLED"


class DatasetPreparationStatus(str, Enum):
    DRY_RUN_VALIDATED = "DRY_RUN_VALIDATED"
    COMMITTED = "COMMITTED"
    REUSED = "REUSED"


class CancellationProbe(Protocol):
    def raise_if_cancelled(self) -> None: ...


class DatasetPreparationError(RuntimeError):
    def __init__(
        self,
        code: DatasetPreparationErrorCode,
        message: str,
        *,
        sample_id: str = "",
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.sample_id = sample_id
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class PreparedDatasetSample:
    sample_id: str
    source_index: int
    source_image: Path
    source_label: Path
    image_sha256: str
    label_sha256: str
    annotation_status: str
    source_type: str
    class_names: tuple[str, ...]
    split: str = ""


@dataclass(frozen=True)
class DatasetPreparationResult:
    status: DatasetPreparationStatus
    dataset_id: str
    dataset_hash: str
    artifact_path: Path | None
    preparation_report_path: Path
    accepted_count: int
    split_counts: Mapping[str, int]
    sample_splits: Mapping[str, str]
    deduplication_records: tuple[DeduplicationAuditRecord, ...]
    warnings: tuple[str, ...] = ()


def _noop_event(_name: str, _message: str, _metadata: Mapping[str, Any]) -> None:
    return None


class DatasetPreparationService:
    """Validate, hash, stage, and atomically commit one dataset batch."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        source_manifest: str | Path | None = None,
        clock: Callable[[], datetime] | None = None,
        split_seed: int = DEFAULT_SPLIT_SEED,
        split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
        commit_directory: Callable[[Path, Path], None] | None = None,
        source_snapshot: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        self.artifact_root = Path(artifact_root).resolve()
        self.source_manifest = (
            Path(source_manifest).resolve() if source_manifest is not None else None
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._split_seed = int(split_seed)
        self._split_ratios = tuple(float(value) for value in split_ratios)
        if len(self._split_ratios) != 3 or any(value < 0 for value in self._split_ratios):
            raise ValueError("split_ratios must contain three non-negative values")
        if abs(sum(self._split_ratios) - 1.0) > 1e-9:
            raise ValueError("split_ratios must sum to 1")
        self._commit_directory = commit_directory or os.replace
        self._source_snapshot = {
            str(sample_id): dict(values)
            for sample_id, values in (source_snapshot or {}).items()
        }

    def prepare(
        self,
        plan: ProcessingPlan,
        records: Sequence[ProcessingRecord],
        *,
        report_id: str,
        dry_run: bool,
        cancellation: CancellationProbe,
        emit: Callable[[str, str, Mapping[str, Any]], None] = _noop_event,
    ) -> DatasetPreparationResult:
        run_root = self._safe_child(self.artifact_root, report_id)
        preview_root = self._safe_child(run_root, "dataset_preview")
        final_root = self._safe_child(run_root, "dataset")
        scratch = self._safe_child(run_root, ".dataset-staging")
        if scratch.exists():
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.DATASET_COMMIT_FAILED,
                f"Dataset staging path already exists: {scratch}",
                retryable=True,
            )
        run_root.mkdir(parents=True, exist_ok=True)
        scratch.mkdir()
        committed = False
        emit("DATASET_PREPARATION_STARTED", "Dataset preparation started.", {
            "ready_count": len(records), "dry_run": dry_run
        })
        try:
            cancellation.raise_if_cancelled()
            samples, audits, warnings = self._collect_samples(
                records, scratch, cancellation, emit
            )
            assigned = self._assign_splits(samples)
            split_counts = {
                split: sum(item.split == split for item in assigned)
                for split in ("train", "val", "test")
            }
            emit("DATASET_SPLIT_CREATED", "Deterministic split created.", split_counts)
            stable_payload = self._stable_dataset_payload(plan, assigned, audits)
            dataset_hash = _canonical_sha256(stable_payload)
            dataset_id = f"dataset-{dataset_hash[:16]}"
            preview_payload = self._metadata_payload(
                plan, report_id, dataset_id, dataset_hash, assigned, audits,
                dry_run=dry_run,
            )
            if dry_run:
                shutil.rmtree(scratch, ignore_errors=True)
                preview_root.mkdir(parents=True, exist_ok=True)
                report_path = preview_root / "preparation_report.json"
                _write_json_atomic(report_path, {
                    **preview_payload,
                    "status": DatasetPreparationStatus.DRY_RUN_VALIDATED.value,
                    "message": "Dataset inputs validated; no dataset was created and training was not started.",
                    "split_counts": split_counts,
                })
                return DatasetPreparationResult(
                    status=DatasetPreparationStatus.DRY_RUN_VALIDATED,
                    dataset_id=dataset_id,
                    dataset_hash=dataset_hash,
                    artifact_path=None,
                    preparation_report_path=report_path,
                    accepted_count=len(assigned),
                    split_counts=split_counts,
                    sample_splits={item.sample_id: item.split for item in assigned},
                    deduplication_records=tuple(audits),
                    warnings=tuple(warnings),
                )

            cancellation.raise_if_cancelled()
            self._materialize_dataset(scratch, assigned, preview_payload, audits, cancellation)
            emit("DATASET_STAGED", "Dataset staged and ready for verification.", {
                "dataset_id": dataset_id, "accepted_count": len(assigned)
            })
            self._verify_dataset(scratch, assigned)
            emit("DATASET_VALIDATED", "Staged dataset verified.", {
                "dataset_id": dataset_id, "dataset_hash": dataset_hash
            })
            cancellation.raise_if_cancelled()
            reused = self._find_reusable_dataset(dataset_hash)
            if reused is not None:
                shutil.rmtree(scratch, ignore_errors=True)
                final_root.mkdir(parents=True, exist_ok=True)
                reference_path = final_root / "dataset_reference.json"
                _write_json_atomic(reference_path, {
                    "schema_version": 1,
                    "dataset_id": dataset_id,
                    "dataset_hash": dataset_hash,
                    "reused_artifact_path": str(reused),
                })
                report_path = final_root / "preparation_report.json"
                _write_json_atomic(report_path, {
                    **preview_payload,
                    "status": DatasetPreparationStatus.REUSED.value,
                    "artifact_path": str(reused),
                    "split_counts": split_counts,
                    "message": "Existing identical dataset reused; training was not started.",
                })
                return DatasetPreparationResult(
                    DatasetPreparationStatus.REUSED, dataset_id, dataset_hash,
                    reused, report_path, len(assigned), split_counts,
                    {item.sample_id: item.split for item in assigned}, tuple(audits),
                    tuple(warnings),
                )
            if final_root.exists():
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.DATASET_COMMIT_FAILED,
                    f"Final dataset path already exists: {final_root}",
                    retryable=False,
                )
            try:
                self._commit_directory(scratch, final_root)
                committed = True
            except OSError as exc:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.DATASET_COMMIT_FAILED,
                    f"Atomic dataset commit failed: {exc}",
                    retryable=True,
                ) from exc
            try:
                self._verify_dataset(final_root, assigned)
            except DatasetPreparationError:
                failed_root = self._safe_child(run_root, ".failed-dataset")
                try:
                    os.replace(final_root, failed_root)
                except OSError:
                    pass
                raise
            report_path = final_root / "preparation_report.json"
            _write_json_atomic(report_path, {
                **preview_payload,
                "status": DatasetPreparationStatus.COMMITTED.value,
                "artifact_path": str(final_root),
                "split_counts": split_counts,
                "message": "Dataset prepared; training was not started.",
            })
            emit("DATASET_COMMITTED", "Dataset committed atomically.", {
                "dataset_id": dataset_id, "dataset_hash": dataset_hash
            })
            return DatasetPreparationResult(
                DatasetPreparationStatus.COMMITTED, dataset_id, dataset_hash,
                final_root, report_path, len(assigned), split_counts,
                {item.sample_id: item.split for item in assigned}, tuple(audits),
                tuple(warnings),
            )
        except DatasetPreparationError:
            emit("DATASET_PREPARATION_FAILED", "Dataset preparation failed.", {})
            raise
        except CanonicalLabelConflictError as exc:
            emit("DATASET_PREPARATION_FAILED", "Annotation conflict blocked dataset preparation.", {
                "image_sha256": str(exc.conflict.get("image_sha256") or "")
            })
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.ANNOTATION_CONFLICT,
                str(exc),
            ) from exc
        except Exception as exc:
            if type(exc).__name__ == "ProcessingCancelledError":
                raise
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.DATASET_VERIFY_FAILED,
                f"Dataset preparation failed: {type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if not committed and scratch.exists():
                shutil.rmtree(scratch, ignore_errors=True)

    def _collect_samples(self, records, scratch, cancellation, emit):
        label_workspace = scratch / "canonical_labels"
        try:
            approved = (
                load_approved_annotation_selections(self.source_manifest)
                if self.source_manifest is not None and self.source_manifest.is_file()
                else {}
            )
        except ValueError as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.CANONICAL_SELECTION_STALE,
                str(exc),
            ) from exc
        candidates: list[tuple[ProcessingRecord, ExportedReviewItem]] = []
        item_to_record: dict[int, ProcessingRecord] = {}
        for record in sorted(records, key=lambda value: (value.source_index, value.sample_id)):
            cancellation.raise_if_cancelled()
            row = {str(key): str(value or "") for key, value in record.fields.items()}
            self._reject_reannotation(row, record.sample_id)
            annotation_status = row.get("annotation_status", "").strip()
            if annotation_status == "verified_annotation":
                item = self._human_item(record, row)
            else:
                if row.get("review_label", "").strip() == "confirmed_ng":
                    self._validate_snapshot_contract(row, record.sample_id)
                item, reason = prepare_ready_review_item(
                    row,
                    row_index=record.source_index + 1,
                    manifest_path=self.source_manifest or Path("review-manifest.csv"),
                    workspace=label_workspace / f"row-{record.source_index}",
                    copy_image=False,
                )
                if item is None:
                    code = (
                        DatasetPreparationErrorCode.SOURCE_IMAGE_MISSING
                        if "image" in reason or "canvas" in reason
                        else DatasetPreparationErrorCode.LABEL_MISSING
                    )
                    raise DatasetPreparationError(code, reason, sample_id=record.sample_id)
            self._validate_item(record, row, item)
            self._validate_annotation_revision_reference(
                record, row, item, approved
            )
            candidates.append((record, item))
            item_to_record[id(item)] = record
            emit("DATASET_SAMPLE_VALIDATED", "Dataset sample validated.", {
                "sample_id": record.sample_id, "image_sha256": item.image_sha256
            })
        audits: list[DeduplicationAuditRecord] = []
        try:
            indexed, _excluded = select_canonical_ready_items(
                [item for _record, item in candidates],
                audit_records=audits,
                approved_selections=approved,
            )
        except CanonicalLabelConflictError:
            raise
        except ValueError as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.CANONICAL_SELECTION_STALE,
                str(exc),
            ) from exc
        retained = {id(item) for item in indexed.values()}
        warnings = [
            f"Deduplicated sample {audit.excluded_sample}; kept {audit.kept_sample} ({audit.reason})."
            for audit in audits
        ]
        prepared: list[PreparedDatasetSample] = []
        for record, item in candidates:
            if id(item) not in retained:
                continue
            class_names = tuple(_json_string_list(record.fields.get("class_names_json")))
            prepared.append(PreparedDatasetSample(
                sample_id=record.sample_id,
                source_index=record.source_index,
                source_image=Path(item.source_image).resolve(),
                source_label=Path(item.output_label).resolve(),
                image_sha256=item.image_sha256,
                label_sha256=normalized_label_sha256(item.output_label),
                annotation_status=item.annotation_status,
                source_type=("human_annotation" if item.annotation_status == "verified_annotation"
                             else "human_empty" if item.annotation_status == "verified_empty"
                             else "ai_snapshot"),
                class_names=class_names,
            ))
        if not prepared:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_MISSING,
                "No canonical READY sample remained after validation.",
            )
        return tuple(prepared), audits, warnings

    @staticmethod
    def _validate_annotation_revision_reference(
        record: ProcessingRecord,
        row: Mapping[str, str],
        item: ExportedReviewItem,
        approved: Mapping[str, Mapping[str, Any]],
    ) -> None:
        revision_id = (row.get("annotation_revision_id") or "").strip()
        if not revision_id:
            return
        selection = approved.get(item.image_sha256)
        expected = (
            selection.get("label_sha256_by_sample", {}).get(record.sample_id)
            if selection
            else ""
        )
        actual = normalized_label_sha256(item.output_label)
        if (
            selection is None
            or selection.get("mode") != "new_annotation_revision"
            or expected != actual
        ):
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.CANONICAL_SELECTION_STALE,
                f"Annotation revision is revoked, superseded, or not canonical: {revision_id}",
                sample_id=record.sample_id,
            )

    def _human_item(self, record: ProcessingRecord, row: Mapping[str, str]) -> ExportedReviewItem:
        source = Path(row.get("source_image") or row.get("original_path") or "")
        label = Path(row.get("output_label") or "")
        if not label.is_file():
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_MISSING,
                f"Verified annotation label is missing: {label}",
                sample_id=record.sample_id,
            )
        image_sha = _sha256_file(self._validated_source_file(source, record.sample_id))
        product = row.get("product") or "unknown"
        area = row.get("area") or "unknown"
        return ExportedReviewItem(
            source_manifest=str(self.source_manifest or ""), review_label=row.get("review_label", ""),
            review_note=row.get("review_note", ""), source_image=str(source.resolve()),
            output_image="", output_label=str(label.resolve()), annotation_status="verified_annotation",
            sample_id=record.sample_id, image_sha256=image_sha, product=product, area=area,
            timestamp=row.get("timestamp", ""), status=row.get("status", ""),
            decision_reasons=row.get("decision_reasons", ""), model_version=row.get("model_version", ""),
            class_names_json=row.get("class_names_json", "[]"), class_map_json=row.get("class_map_json", "{}"),
            class_schema_hash=row.get("class_schema_hash", ""),
        )

    def _validate_item(self, record, row, item):
        image = self._validated_source_file(Path(item.source_image), record.sample_id)
        actual_image_sha = _sha256_file(image)
        captured = self._source_snapshot.get(record.sample_id, {})
        expected_image_sha = (
            row.get("image_sha256")
            or row.get("image_sha")
            or captured.get("image_sha256")
            or ""
        ).strip().lower()
        if expected_image_sha and expected_image_sha != actual_image_sha:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.SOURCE_IMAGE_STALE,
                f"Source image SHA changed for {record.sample_id}.", sample_id=record.sample_id,
            )
        label = self._validated_source_file(Path(item.output_label), record.sample_id, label=True)
        actual_label_sha = normalized_label_sha256(label)
        expected_label_sha = (
            row.get("label_sha256")
            or captured.get("label_sha256")
            or ""
        ).strip().lower()
        if expected_label_sha and expected_label_sha != actual_label_sha:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_STALE,
                f"Canonical label SHA changed for {record.sample_id}.", sample_id=record.sample_id,
            )
        class_names = _json_string_list(row.get("class_names_json"))
        if not class_names:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.CLASS_MAPPING_INVALID,
                f"Class mapping is missing for {record.sample_id}.", sample_id=record.sample_id,
            )
        self._validate_yolo_label(label, len(class_names), record.sample_id)
        if item.annotation_status == "verified_annotation" and not label.read_text(
            encoding="utf-8-sig"
        ).strip():
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_MISSING,
                f"Verified annotation is empty for {record.sample_id}.",
                sample_id=record.sample_id,
            )

    @staticmethod
    def _validate_snapshot_contract(row: Mapping[str, str], sample_id: str) -> None:
        try:
            detections = json.loads(row.get("detections_json") or "[]")
        except json.JSONDecodeError as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_PARSE_ERROR,
                f"Detection snapshot is invalid JSON for {sample_id}.", sample_id=sample_id,
            ) from exc
        if not isinstance(detections, list) or not detections:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_MISSING,
                f"Verified snapshot has no detections for {sample_id}.", sample_id=sample_id,
            )
        class_count = len(_json_string_list(row.get("class_names_json")))
        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.LABEL_PARSE_ERROR,
                    f"Detection {index} is not an object.", sample_id=sample_id,
                )
            try:
                class_id = int(detection["class_id"])
                x1, y1, x2, y2 = (float(value) for value in detection["bbox"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.LABEL_PARSE_ERROR,
                    f"Detection {index} is incomplete.", sample_id=sample_id,
                ) from exc
            if class_id < 0 or class_id >= class_count:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.CLASS_ID_INVALID,
                    f"Detection {index} class ID is invalid.", sample_id=sample_id,
                )
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.BBOX_INVALID,
                    f"Detection {index} bbox is invalid.", sample_id=sample_id,
                )

    @staticmethod
    def _validated_source_file(path: Path, sample_id: str, *, label: bool = False) -> Path:
        try:
            if path.is_symlink():
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.UNSAFE_SOURCE_PATH,
                    f"Symlink source is forbidden: {path}", sample_id=sample_id,
                )
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_MISSING if label else DatasetPreparationErrorCode.SOURCE_IMAGE_MISSING,
                f"Source file is missing: {path}", sample_id=sample_id,
            ) from exc
        if not resolved.is_file():
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_MISSING if label else DatasetPreparationErrorCode.SOURCE_IMAGE_MISSING,
                f"Source is not a regular file: {resolved}", sample_id=sample_id,
            )
        if not label:
            try:
                with Image.open(resolved) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as exc:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.SOURCE_IMAGE_MISSING,
                    f"Source image is unreadable: {resolved}", sample_id=sample_id,
                ) from exc
        return resolved

    @staticmethod
    def _validate_yolo_label(path: Path, class_count: int, sample_id: str) -> None:
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.LABEL_PARSE_ERROR,
                f"Label cannot be read: {path}", sample_id=sample_id,
            ) from exc
        for line_number, raw in enumerate(lines, start=1):
            if not raw.strip():
                continue
            parts = raw.split()
            if len(parts) != 5:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.LABEL_PARSE_ERROR,
                    f"YOLO label line {line_number} must have five fields.", sample_id=sample_id,
                )
            try:
                class_id = int(parts[0])
                x, y, width, height = (float(value) for value in parts[1:])
            except ValueError as exc:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.LABEL_PARSE_ERROR,
                    f"YOLO label line {line_number} is not numeric.", sample_id=sample_id,
                ) from exc
            if class_id < 0 or class_id >= class_count:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.CLASS_ID_INVALID,
                    f"Class ID {class_id} is outside [0, {class_count}).", sample_id=sample_id,
                )
            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < width <= 1
                and 0 < height <= 1
                and x - width / 2 >= 0
                and x + width / 2 <= 1
                and y - height / 2 >= 0
                and y + height / 2 <= 1
            ):
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.BBOX_INVALID,
                    f"YOLO bbox on line {line_number} is outside normalized bounds.", sample_id=sample_id,
                )

    @staticmethod
    def _reject_reannotation(row: Mapping[str, str], sample_id: str) -> None:
        status = (row.get("annotation_status") or "").strip().lower()
        resolution = (row.get("canonical_resolution") or row.get("resolution_mode") or "").strip().lower()
        if status in {"needs_reannotation", "revoked"} or resolution == "needs_reannotation":
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.REANNOTATION_REQUIRED,
                f"Sample requires reannotation: {sample_id}", sample_id=sample_id,
            )

    def _assign_splits(self, samples: Sequence[PreparedDatasetSample]):
        ordered = sorted(samples, key=lambda value: (value.image_sha256, value.sample_id))
        indices = list(range(len(ordered)))
        random.Random(self._split_seed).shuffle(indices)
        train_ratio, val_ratio, test_ratio = self._split_ratios
        if len(indices) < 3:
            counts = (len(indices), 0, 0)
        else:
            val_count = max(1 if val_ratio else 0, round(len(indices) * val_ratio))
            test_count = max(1 if test_ratio else 0, round(len(indices) * test_ratio))
            while val_count + test_count >= len(indices):
                if val_count >= test_count and val_count > int(bool(val_ratio)):
                    val_count -= 1
                elif test_count > int(bool(test_ratio)):
                    test_count -= 1
                else:
                    break
            counts = (len(indices) - val_count - test_count, val_count, test_count)
        split_by_index = {}
        offset = 0
        for split, count in zip(("train", "val", "test"), counts, strict=True):
            for index in indices[offset:offset + count]:
                split_by_index[index] = split
            offset += count
        return tuple(
            PreparedDatasetSample(**{**asdict(item), "split": split_by_index[index]})
            for index, item in enumerate(ordered)
        )

    def _stable_dataset_payload(self, plan, samples, audits):
        class_names = self._common_class_names(samples)
        return {
            "schema_version": DATASET_PREPARATION_SCHEMA_VERSION,
            "source_manifest_sha": plan.source_manifest_sha,
            "review_revision": plan.review_revision,
            "split_policy": {"name": "phase3c1_random42", "seed": self._split_seed,
                             "ratios": self._split_ratios},
            "class_names": class_names,
            "class_mapping_version": _canonical_sha256(class_names),
            "samples": [
                {"sample_id": item.sample_id, "image_sha256": item.image_sha256,
                 "label_sha256": item.label_sha256, "split": item.split,
                 "image_name": self._image_name(item), "label_name": self._label_name(item),
                 "source_type": item.source_type}
                for item in samples
            ],
            "excluded_samples": sorted(
                decision.sample_id
                for decision in plan.routing_decisions
                if decision.decision == RoutingDecisionType.EXCLUDED
            ),
            "canonical_deduplicated_samples": sorted(
                item.excluded_sample for item in audits
            ),
            "deduplication": [asdict(item) for item in audits],
            "deduplication_audit_reference": "deduplication_audit.json",
            "code_version": DATASET_PREPARATION_CODE_VERSION,
        }

    def _metadata_payload(self, plan, report_id, dataset_id, dataset_hash, samples, audits, *, dry_run):
        stable = self._stable_dataset_payload(plan, samples, audits)
        return {
            **stable,
            "dataset_id": dataset_id,
            "dataset_hash": dataset_hash,
            "plan_id": plan.plan_id,
            "report_id": report_id,
            "created_at": self._clock().astimezone(timezone.utc).isoformat(),
            "operator": plan.operator,
            "preparation_mode": "PREPARE_ONLY",
            "dry_run": dry_run,
            "sample_count": len(samples),
            "parent_dataset_id": "",
        }

    def _materialize_dataset(self, root, samples, metadata, audits, cancellation):
        class_names = self._common_class_names(samples)
        for item in samples:
            cancellation.raise_if_cancelled()
            image_dir = root / "images" / item.split
            label_dir = root / "labels" / item.split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item.source_image, image_dir / self._image_name(item))
            shutil.copy2(item.source_label, label_dir / self._label_name(item))
        shutil.rmtree(root / "canonical_labels", ignore_errors=True)
        _write_json_atomic(root / "source_sample_manifest.json", {
            "schema_version": 1,
            "samples": [
                {"sample_id": item.sample_id, "source_image_sha256": item.image_sha256,
                 "selected_annotation_sha256": item.label_sha256,
                 "source_type": item.source_type, "split": item.split}
                for item in samples
            ],
            "canonical_deduplicated_samples": [
                item.excluded_sample for item in audits
            ],
        })
        _write_json_atomic(root / "split_manifest.json", {
            "schema_version": 1, "seed": self._split_seed, "ratios": self._split_ratios,
            "assignments": {item.sample_id: item.split for item in samples},
        })
        _write_json_atomic(root / "deduplication_audit.json", {
            "schema_version": 1, "records": [asdict(item) for item in audits]
        })
        _write_json_atomic(root / "dataset_metadata.json", metadata)
        _write_text_atomic(root / "dataset.yaml", _dataset_yaml(class_names))
        checksums = self._checksums(root)
        _write_json_atomic(root / "checksums.json", {"schema_version": 1, "files": checksums})
        _write_json_atomic(root / "preparation_report.json", {
            **metadata, "status": "STAGED", "message": "Dataset staged; training was not started."
        })
        _fsync_tree(root)

    def _verify_dataset(self, root, samples):
        required = ("dataset.yaml", "source_sample_manifest.json", "split_manifest.json",
                    "dataset_metadata.json", "deduplication_audit.json", "checksums.json",
                    "preparation_report.json")
        if not all((root / name).is_file() for name in required):
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.DATASET_VERIFY_FAILED,
                f"Dataset metadata is incomplete: {root}",
            )
        for item in samples:
            image = root / "images" / item.split / self._image_name(item)
            label = root / "labels" / item.split / self._label_name(item)
            if _sha256_file(image) != item.image_sha256 or normalized_label_sha256(label) != item.label_sha256:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.DATASET_VERIFY_FAILED,
                    f"Committed dataset checksum mismatch for {item.sample_id}.",
                    sample_id=item.sample_id,
                )
        try:
            checksums = json.loads((root / "checksums.json").read_text(encoding="utf-8"))["files"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.DATASET_VERIFY_FAILED,
                f"Dataset checksums manifest is invalid: {root}",
            ) from exc
        for relative_path, expected in checksums.items():
            candidate = (root / str(relative_path)).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError as exc:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.DATASET_VERIFY_FAILED,
                    f"Checksum path escapes dataset root: {relative_path}",
                ) from exc
            if not candidate.is_file() or _sha256_file(candidate) != expected:
                raise DatasetPreparationError(
                    DatasetPreparationErrorCode.DATASET_VERIFY_FAILED,
                    f"Dataset checksum verification failed: {relative_path}",
                )

    def _find_reusable_dataset(self, dataset_hash: str) -> Path | None:
        if not self.artifact_root.is_dir():
            return None
        for metadata_path in sorted(self.artifact_root.glob("*/dataset/dataset_metadata.json")):
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if payload.get("dataset_hash") == dataset_hash:
                return metadata_path.parent.resolve()
        return None

    @staticmethod
    def _common_class_names(samples):
        contracts = {item.class_names for item in samples}
        if len(contracts) != 1:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.CLASS_MAPPING_INVALID,
                "READY samples contain different class mappings.",
            )
        return next(iter(contracts))

    @staticmethod
    def _image_name(item):
        return f"review_{item.image_sha256[:24]}{item.source_image.suffix.lower()}"

    @staticmethod
    def _label_name(item):
        return f"review_{item.image_sha256[:24]}.txt"

    @staticmethod
    def _checksums(root: Path):
        return {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name not in {"checksums.json", "preparation_report.json"}
        }

    @staticmethod
    def _safe_child(root: Path, name: str) -> Path:
        if not name or any(character in name for character in ("/", "\\", "..")):
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.DATASET_COMMIT_FAILED,
                f"Unsafe dataset artifact identifier: {name!r}",
            )
        child = (root / name).resolve()
        try:
            child.relative_to(root)
        except ValueError as exc:
            raise DatasetPreparationError(
                DatasetPreparationErrorCode.DATASET_COMMIT_FAILED,
                f"Dataset artifact path escapes root: {child}",
            ) from exc
        return child


def capture_dataset_source_snapshot(
    records: Sequence[ProcessingRecord],
) -> dict[str, dict[str, str]]:
    """Capture external source bytes when a plan is handed to the operator.

    Missing files are intentionally recorded as empty and are diagnosed by the
    preparation service. No source file is created or modified here.
    """
    captured: dict[str, dict[str, str]] = {}
    for record in records:
        fields = record.fields
        image_value = (
            fields.get("source_image")
            or fields.get("original_path")
            or fields.get("preprocessed_path")
            or ""
        )
        label_value = (
            fields.get("output_label")
            if str(fields.get("annotation_status") or "") == "verified_annotation"
            else ""
        )
        values: dict[str, str] = {}
        image = Path(str(image_value))
        if image.is_file() and not image.is_symlink():
            values["image_sha256"] = _sha256_file(image)
        label = Path(str(label_value))
        if label.is_file() and not label.is_symlink():
            values["label_sha256"] = normalized_label_sha256(label)
        captured[record.sample_id] = values
    return captured


def _json_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item).strip() for item in parsed if str(item).strip()] if isinstance(parsed, list) else []


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_yaml(class_names: Sequence[str]) -> str:
    names = "\n".join(f"  {index}: {json.dumps(name, ensure_ascii=False)}" for index, name in enumerate(class_names))
    return "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + names + "\n"


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            # Windows requires a writable descriptor for FlushFileBuffers,
            # which is what Python's fsync delegates to.
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())
