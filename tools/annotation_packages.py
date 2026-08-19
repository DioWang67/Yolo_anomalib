"""Immutable Phase 3C2 annotation work-package contracts and creation service."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.annotation_validation import AnnotationOperation, normalized_label_sha256
from tools.export_review_dataset import prepare_annotation_draft
from tools.processing_execution import CancellationToken, ProcessingCancelledError
from tools.processing_pipeline import ProcessingPlan, ProcessingRecord, RoutingDecision, RoutingDecisionType

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class AnnotationPackageStatus(str, Enum):
    WAITING_FOR_OPERATOR = "WAITING_FOR_OPERATOR"
    INCOMPLETE = "INCOMPLETE"
    COMPLETED = "COMPLETED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"
    CANCELLED_BEFORE_LAUNCH = "CANCELLED_BEFORE_LAUNCH"


class AnnotationPackageError(RuntimeError):
    def __init__(self, code: str, message: str, *, sample_id: str = "", retryable: bool = True) -> None:
        self.code = code
        self.sample_id = sample_id
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class AnnotationWorkItem:
    item_id: str
    sample_id: str
    source_index: int
    requested_operation: AnnotationOperation
    source_image: str
    source_image_sha256: str
    original_label: str
    working_label: str
    parent_label_sha256: str
    allow_empty: bool
    review_label: str
    primary_routing: str
    review_reason: str


@dataclass(frozen=True)
class AnnotationWorkPackage:
    package_id: str
    plan_id: str
    report_id: str
    created_at: datetime
    operator: str
    source_manifest_sha: str
    review_revision: str
    status: AnnotationPackageStatus
    root: Path
    class_mapping: tuple[str, ...]
    class_mapping_version: str
    items: tuple[AnnotationWorkItem, ...]

    @property
    def package_path(self) -> Path:
        return self.root / "package.json"


class AnnotationPackageService:
    """Create one all-or-nothing annotation package for a routed batch."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        copy_file: Callable[[Path, Path], Any] | None = None,
    ) -> None:
        self._artifact_root = Path(artifact_root).resolve()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_generator = id_generator or (lambda: str(uuid4()))
        self._copy_file = copy_file or shutil.copy2

    def create(
        self,
        plan: ProcessingPlan,
        records: Sequence[ProcessingRecord],
        decisions: Sequence[RoutingDecision],
        *,
        report_id: str,
        cancellation: CancellationToken,
    ) -> AnnotationWorkPackage:
        if not records or len(records) != len(decisions):
            raise AnnotationPackageError("ANNOTATION_PACKAGE_EMPTY", "Annotation package has no routed records.", retryable=False)
        decision_by_id = {item.sample_id: item for item in decisions}
        class_mapping = self._class_mapping(records)
        class_version = hashlib.sha256(
            json.dumps(class_mapping, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        package_id = self._safe_id(self._id_generator(), "package")
        parent = self._artifact_root / self._safe_id(report_id, "report") / "annotation"
        destination = parent / package_id
        parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise AnnotationPackageError("ANNOTATION_PACKAGE_EXISTS", f"Annotation package already exists: {destination}", retryable=False)
        cancellation.raise_if_cancelled()
        staging = Path(tempfile.mkdtemp(prefix=f".{package_id}.", dir=parent))
        try:
            for name in ("items", "source_images", "original_labels", "working_labels"):
                (staging / name).mkdir()
            items: list[AnnotationWorkItem] = []
            image_owners: dict[str, str] = {}
            for record in records:
                cancellation.raise_if_cancelled()
                decision = decision_by_id.get(record.sample_id)
                if decision is None:
                    raise AnnotationPackageError("ANNOTATION_ROUTE_MISSING", "Routed decision is missing.", sample_id=record.sample_id, retryable=False)
                item = self._stage_item(staging, record, decision)
                existing_owner = image_owners.get(item.source_image_sha256)
                if existing_owner is not None:
                    raise AnnotationPackageError(
                        "CANONICAL_CONFLICT",
                        "Identical image content appears more than once in an annotation package; explicit canonical selection is required "
                        f"for {existing_owner!r} and {item.sample_id!r}.",
                        sample_id=item.sample_id,
                        retryable=False,
                    )
                image_owners[item.source_image_sha256] = item.sample_id
                items.append(item)
                _write_json_atomic(staging / "items" / f"{item.item_id}.json", _item_dict(item))
            created_at = self._clock()
            if created_at.tzinfo is None:
                raise ValueError("Annotation package clock must return timezone-aware time")
            package = AnnotationWorkPackage(
                package_id=package_id,
                plan_id=plan.plan_id,
                report_id=report_id,
                created_at=created_at,
                operator=plan.operator,
                source_manifest_sha=plan.source_manifest_sha,
                review_revision=plan.review_revision,
                status=AnnotationPackageStatus.WAITING_FOR_OPERATOR,
                root=destination,
                class_mapping=class_mapping,
                class_mapping_version=class_version,
                items=tuple(items),
            )
            _write_json_atomic(staging / "class_mapping.json", {"version": class_version, "classes": list(class_mapping)})
            _write_json_atomic(staging / "instructions.json", {
                "schema_version": 1,
                "message": "Edit only files under working_labels, then run annotation resume validation.",
                "allowed_operations": [item.value for item in AnnotationOperation],
                "training_started": False,
            })
            _write_json_atomic(staging / "validation_report.json", {"status": "NOT_VALIDATED", "items": []})
            _write_json_atomic(staging / "completion.json", {"status": AnnotationPackageStatus.WAITING_FOR_OPERATOR.value, "revisions": [], "failures": []})
            _write_json_atomic(staging / "package.json", _package_dict(package, root=staging))
            _write_json_atomic(staging / "checksums.json", _immutable_checksums(staging))
            cancellation.raise_if_cancelled()
            _fsync_tree(staging)
            os.replace(staging, destination)
            _fsync_directory(parent)
            return package
        except ProcessingCancelledError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def _stage_item(self, staging: Path, record: ProcessingRecord, decision: RoutingDecision) -> AnnotationWorkItem:
        fields = dict(record.fields)
        try:
            source, parent_text = prepare_annotation_draft(fields)
        except (OSError, ValueError) as exc:
            raise AnnotationPackageError("LABEL_MISSING", f"Cannot prepare annotation evidence: {exc}", sample_id=record.sample_id) from exc
        resolved_source = source.resolve(strict=True)
        if source.is_symlink() or not resolved_source.is_file():
            raise AnnotationPackageError("IMAGE_STALE", "Source image is missing or is a symbolic link.", sample_id=record.sample_id)
        image_sha = _sha256_file(resolved_source)
        item_id = f"item-{hashlib.sha256(record.sample_id.encode('utf-8')).hexdigest()[:16]}"
        image_name = f"{item_id}{resolved_source.suffix.lower() or '.img'}"
        label_name = f"{item_id}.txt"
        self._copy_file(resolved_source, staging / "source_images" / image_name)
        _write_text_atomic(staging / "original_labels" / label_name, parent_text)
        _write_text_atomic(staging / "working_labels" / label_name, parent_text)
        operation = _operation_for(fields, decision.decision, bool(parent_text.strip()))
        return AnnotationWorkItem(
            item_id=item_id,
            sample_id=record.sample_id,
            source_index=record.source_index,
            requested_operation=operation,
            source_image=f"source_images/{image_name}",
            source_image_sha256=image_sha,
            original_label=f"original_labels/{label_name}",
            working_label=f"working_labels/{label_name}",
            parent_label_sha256=normalized_label_sha256(parent_text),
            allow_empty=str(fields.get("review_label") or "") in {"false_positive", "verified_empty"},
            review_label=str(fields.get("review_label") or ""),
            primary_routing=decision.decision.value,
            review_reason=str(fields.get("failure_category") or fields.get("review_note") or decision.reason),
        )

    @staticmethod
    def _class_mapping(records: Sequence[ProcessingRecord]) -> tuple[str, ...]:
        mappings: set[tuple[str, ...]] = set()
        for record in records:
            raw = str(record.fields.get("class_names_json") or "")
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise AnnotationPackageError("CLASS_ID_INVALID", f"Invalid class_names_json for {record.sample_id}", sample_id=record.sample_id) from exc
            if not isinstance(parsed, list) or not parsed or any(not str(value).strip() for value in parsed):
                raise AnnotationPackageError("CLASS_ID_INVALID", f"Missing class mapping for {record.sample_id}", sample_id=record.sample_id)
            mappings.add(tuple(str(value).strip() for value in parsed))
        if len(mappings) != 1:
            raise AnnotationPackageError("CLASS_MAPPING_CONFLICT", "All annotation items must use one exact class mapping.", retryable=False)
        return next(iter(mappings))

    @staticmethod
    def _safe_id(value: str, kind: str) -> str:
        cleaned = _SAFE_NAME.sub("-", str(value).strip()).strip(".-")
        if not cleaned or len(cleaned) > 128:
            raise AnnotationPackageError("WORKING_PATH_ESCAPE", f"Unsafe {kind} identifier.", retryable=False)
        return cleaned


def load_annotation_package(package_path: str | Path) -> AnnotationWorkPackage:
    path = Path(package_path).resolve()
    if path.name != "package.json" or not path.is_file():
        raise AnnotationPackageError("ANNOTATION_PACKAGE_STALE", f"Annotation package is unavailable: {path}")
    try:
        _verify_immutable_checksums(path.parent)
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = tuple(_item_from_dict(value) for value in raw["items"])
        return AnnotationWorkPackage(
            package_id=str(raw["package_id"]), plan_id=str(raw["plan_id"]), report_id=str(raw["report_id"]),
            created_at=datetime.fromisoformat(str(raw["created_at"])), operator=str(raw["operator"]),
            source_manifest_sha=str(raw["source_manifest_sha"]), review_revision=str(raw["review_revision"]),
            status=AnnotationPackageStatus(str(raw["status"])), root=path.parent,
            class_mapping=tuple(str(value) for value in raw["class_mapping"]),
            class_mapping_version=str(raw["class_mapping_version"]), items=items,
        )
    except AnnotationPackageError:
        raise
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        raise AnnotationPackageError("ANNOTATION_PACKAGE_STALE", f"Annotation package is unreadable: {path}") from exc


def resolve_package_member(package: AnnotationWorkPackage, relative_path: str) -> Path:
    candidate = (package.root / relative_path).resolve()
    try:
        candidate.relative_to(package.root.resolve())
    except ValueError as exc:
        raise AnnotationPackageError("WORKING_PATH_ESCAPE", f"Package path escapes root: {relative_path}", retryable=False) from exc
    if candidate.is_symlink():
        raise AnnotationPackageError("WORKING_PATH_ESCAPE", f"Symbolic links are not allowed: {relative_path}", retryable=False)
    return candidate


def _operation_for(fields: Mapping[str, Any], route: RoutingDecisionType, has_parent: bool) -> AnnotationOperation:
    label = str(fields.get("review_label") or "")
    if route == RoutingDecisionType.NEEDS_CLASS_FIX or label == "wrong_class":
        return AnnotationOperation.FIX_CLASS_ONLY
    if label == "false_positive":
        return AnnotationOperation.REVIEW_EMPTY_LABEL
    if label == "false_negative" or not has_parent:
        return AnnotationOperation.CREATE_ANNOTATION
    return AnnotationOperation.FIX_BOUNDING_BOX


def _item_dict(item: AnnotationWorkItem) -> dict[str, Any]:
    result = asdict(item)
    result["requested_operation"] = item.requested_operation.value
    return result


def _item_from_dict(raw: Mapping[str, Any]) -> AnnotationWorkItem:
    return AnnotationWorkItem(
        item_id=str(raw["item_id"]), sample_id=str(raw["sample_id"]), source_index=int(raw["source_index"]),
        requested_operation=AnnotationOperation(str(raw["requested_operation"])), source_image=str(raw["source_image"]),
        source_image_sha256=str(raw["source_image_sha256"]), original_label=str(raw["original_label"]),
        working_label=str(raw["working_label"]), parent_label_sha256=str(raw["parent_label_sha256"]),
        allow_empty=bool(raw["allow_empty"]), review_label=str(raw["review_label"]),
        primary_routing=str(raw.get("primary_routing") or ""),
        review_reason=str(raw.get("review_reason") or ""),
    )


def _package_dict(package: AnnotationWorkPackage, *, root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1, "package_id": package.package_id, "plan_id": package.plan_id,
        "report_id": package.report_id, "created_at": package.created_at.isoformat(), "operator": package.operator,
        "source_manifest_sha": package.source_manifest_sha, "review_revision": package.review_revision,
        "status": package.status.value, "class_mapping": list(package.class_mapping),
        "item_count": len(package.items),
        "sample_ids": [item.sample_id for item in package.items],
        "tool_type": "external_or_manual",
        "resumed_at": None,
        "completed_at": None,
        "class_mapping_version": package.class_mapping_version,
        "items": [_item_dict(item) for item in package.items],
    }


def _immutable_checksums(root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for directory in ("items", "source_images", "original_labels"):
        for path in sorted((root / directory).iterdir()):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = _sha256_file(path)
    for name in ("package.json", "class_mapping.json", "instructions.json"):
        files[name] = _sha256_file(root / name)
    return {"algorithm": "sha256", "files": files}


def _verify_immutable_checksums(root: Path) -> None:
    checksum_path = root / "checksums.json"
    try:
        payload = json.loads(checksum_path.read_text(encoding="utf-8"))
        files = payload["files"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise AnnotationPackageError("ANNOTATION_PACKAGE_STALE", "Package checksums are missing or invalid.") from exc
    if not isinstance(files, dict):
        raise AnnotationPackageError("ANNOTATION_PACKAGE_STALE", "Package checksum entries are invalid.")
    for relative, expected in files.items():
        # Evidence gets more specific IMAGE_STALE/PARENT_LABEL_STALE diagnostics
        # during resume; metadata is rejected here as package-level staleness.
        if str(relative).startswith(("source_images/", "original_labels/")):
            continue
        candidate = (root / str(relative)).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as exc:
            raise AnnotationPackageError("WORKING_PATH_ESCAPE", f"Checksum path escapes package: {relative}", retryable=False) from exc
        if candidate.is_symlink() or not candidate.is_file() or _sha256_file(candidate) != str(expected):
            raise AnnotationPackageError("ANNOTATION_PACKAGE_STALE", f"Immutable package artifact changed: {relative}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            # Windows rejects fsync on a descriptor opened read-only.
            with path.open("r+b") as handle:
                os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
