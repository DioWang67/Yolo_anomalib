"""Append-only annotation revision and revocation persistence for Phase 3C2."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.annotation_packages import AnnotationWorkItem, AnnotationWorkPackage, _write_json_atomic, _write_text_atomic
from tools.annotation_validation import (
    AnnotationDiff,
    AnnotationOperation,
    AnnotationValidationResult,
    normalized_label_sha256,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class AnnotationRevisionError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = True) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True)
class AnnotationRevision:
    revision_id: str
    package_id: str
    sample_id: str
    operator: str
    revision_reason: str
    created_at: datetime
    image_sha256: str
    parent_label_sha256: str
    new_label_sha256: str
    class_mapping_version: str
    requested_operation: AnnotationOperation
    actual_operation: AnnotationOperation
    diff: AnnotationDiff
    validation_warnings: tuple[str, ...]
    supersedes_revision_id: str
    source_tool: str
    source_tool_version: str
    root: Path

    @property
    def label_path(self) -> Path:
        return self.root / "label.txt"

    @property
    def metadata_path(self) -> Path:
        return self.root / "revision.json"


class AnnotationRevisionStore:
    def __init__(
        self,
        *,
        root: str | Path,
        source_manifest: str | Path,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.source_manifest = Path(source_manifest).resolve()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_generator = id_generator or (lambda: str(uuid4()))

    def commit(
        self,
        package: AnnotationWorkPackage,
        item: AnnotationWorkItem,
        *,
        label_text: str,
        validation: AnnotationValidationResult,
        operator: str,
        revision_reason: str,
        source_tool: str = "manual",
        source_tool_version: str = "",
        supersedes_revision_id: str = "",
    ) -> AnnotationRevision:
        if not revision_reason.strip():
            raise AnnotationRevisionError("REVISION_REASON_REQUIRED", "Revision reason is required.", retryable=False)
        self._ensure_no_canonical_conflict(
            image_sha256=item.source_image_sha256,
            new_label_sha256=validation.label_sha256,
            supersedes_revision_id=supersedes_revision_id,
        )
        revision_id = self._safe_id(self._id_generator())
        sample_dir = self.root / self._safe_sample(item.sample_id)
        destination = sample_dir / revision_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise AnnotationRevisionError("CANONICAL_CONFLICT", f"Revision already exists: {revision_id}", retryable=False)
        staging = Path(tempfile.mkdtemp(prefix=f".{revision_id}.", dir=sample_dir))
        created_at = self._clock()
        revision = AnnotationRevision(
            revision_id=revision_id, package_id=package.package_id, sample_id=item.sample_id,
            operator=operator.strip() or package.operator, revision_reason=revision_reason.strip(),
            created_at=created_at, image_sha256=item.source_image_sha256,
            parent_label_sha256=item.parent_label_sha256, new_label_sha256=validation.label_sha256,
            class_mapping_version=package.class_mapping_version,
            requested_operation=item.requested_operation, actual_operation=validation.actual_operation,
            diff=validation.diff, validation_warnings=validation.warnings,
            supersedes_revision_id=supersedes_revision_id, source_tool=source_tool,
            source_tool_version=source_tool_version, root=destination,
        )
        try:
            _write_text_atomic(staging / "label.txt", label_text)
            payload = _revision_dict(revision)
            payload["label_file"] = "label.txt"
            _write_json_atomic(staging / "revision.json", payload)
            _write_json_atomic(staging / "checksums.json", {
                "algorithm": "sha256", "label.txt": _sha256_file(staging / "label.txt"),
                "revision.json": _sha256_file(staging / "revision.json"),
            })
            os.replace(staging, destination)
            if normalized_label_sha256(
                (destination / "label.txt").read_text(encoding="utf-8")
            ) != validation.label_sha256:
                raise AnnotationRevisionError("REVISION_VERIFY_FAILED", f"Revision verification failed: {revision_id}")
            self._write_canonical_reference(revision)
            return revision
        except AnnotationRevisionError:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
        except (OSError, ValueError) as exc:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise AnnotationRevisionError("REVISION_COMMIT_FAILED", f"Revision commit failed: {exc}") from exc

    def revoke(self, revision: AnnotationRevision, *, operator: str, reason: str) -> Path:
        if not reason.strip():
            raise AnnotationRevisionError("REVISION_REASON_REQUIRED", "Revocation reason is required.", retryable=False)
        revocations = revision.root / "revocations"
        revocations.mkdir(parents=True, exist_ok=True)
        event_id = self._safe_id(self._id_generator())
        path = revocations / f"{event_id}.json"
        _write_json_atomic(path, {
            "schema_version": 1, "event": "annotation_revision_revoked",
            "event_id": event_id, "revision_id": revision.revision_id,
            "sample_id": revision.sample_id, "operator": operator.strip() or "unknown",
            "reason": reason.strip(), "created_at": self._clock().isoformat(),
        })
        reference_root = self.source_manifest.parent / ".review_repairs" / "annotation_resolutions" / "revocations"
        reference_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(reference_root / f"{event_id}.json", {
            "event": "annotation_resolution_revoked", "resolution_id": revision.revision_id,
            "reason": reason.strip(), "operator": operator.strip() or "unknown",
            "revoked_at": self._clock().isoformat(),
        })
        return path

    @staticmethod
    def is_revoked(revision: AnnotationRevision) -> bool:
        root = revision.root / "revocations"
        return root.is_dir() and any(root.glob("*.json"))

    def is_revision_id_revoked(self, revision_id: str) -> bool:
        if not revision_id:
            return False
        return any(
            path.parent.name == "revocations"
            for path in self.root.glob(f"*/{revision_id}/revocations/*.json")
        )

    def _write_canonical_reference(self, revision: AnnotationRevision) -> None:
        reference_root = self.source_manifest.parent / ".review_repairs" / "annotation_resolutions"
        reference_root.mkdir(parents=True, exist_ok=True)
        path = reference_root / f"{revision.revision_id}.json"
        _write_json_atomic(path, {
            "schema_version": 2,
            "resolution_id": revision.revision_id,
            "source": "phase3c2_annotation_revision",
            "image_sha256": revision.image_sha256,
            "sample_ids": [revision.sample_id],
            "revision_reason": revision.revision_reason,
            "resolution": {
                "mode": "new_annotation_revision",
                "selected_sample_id": revision.sample_id,
                "new_annotation_path": str(revision.label_path),
                "new_annotation_sha256": revision.new_label_sha256,
                "revision_id": revision.revision_id,
                "supersedes_revision_id": revision.supersedes_revision_id or None,
            },
            "old_annotations_preserved": True,
            "created_at": revision.created_at.isoformat(),
        })

    def _ensure_no_canonical_conflict(
        self,
        *,
        image_sha256: str,
        new_label_sha256: str,
        supersedes_revision_id: str,
    ) -> None:
        reference_root = self.source_manifest.parent / ".review_repairs" / "annotation_resolutions"
        if not reference_root.is_dir():
            return
        revoked = {
            str(payload.get("resolution_id") or "")
            for path in (reference_root / "revocations").glob("*.json")
            if isinstance((payload := _read_json(path)), dict)
        }
        for path in reference_root.glob("*.json"):
            payload = _read_json(path)
            if not isinstance(payload, dict) or str(payload.get("image_sha256") or "") != image_sha256:
                continue
            resolution = payload.get("resolution")
            resolution_id = str(payload.get("resolution_id") or payload.get("proposal_id") or "")
            if not isinstance(resolution, dict) or resolution_id in revoked:
                continue
            if supersedes_revision_id and resolution_id == supersedes_revision_id:
                continue
            existing_sha = str(
                resolution.get("new_annotation_sha256")
                or (payload.get("old_label_sha256s") or [""])[0]
            )
            if existing_sha:
                raise AnnotationRevisionError(
                    "CANONICAL_CONFLICT",
                    f"An active annotation resolution already exists for image SHA {image_sha256}; explicit supersession is required.",
                    retryable=False,
                )

    @staticmethod
    def _safe_id(value: str) -> str:
        value = str(value).strip()
        if not _SAFE_ID.fullmatch(value):
            raise AnnotationRevisionError("WORKING_PATH_ESCAPE", "Unsafe revision ID.", retryable=False)
        return value

    @staticmethod
    def _safe_sample(value: str) -> str:
        return f"sample-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def _revision_dict(revision: AnnotationRevision) -> dict[str, Any]:
    return {
        "schema_version": 1, "revision_id": revision.revision_id,
        "package_id": revision.package_id, "sample_id": revision.sample_id,
        "operator": revision.operator, "revision_reason": revision.revision_reason,
        "created_at": revision.created_at.isoformat(), "image_sha256": revision.image_sha256,
        "parent_label_sha256": revision.parent_label_sha256,
        "new_label_sha256": revision.new_label_sha256,
        "class_mapping_version": revision.class_mapping_version,
        "requested_operation": revision.requested_operation.value,
        "actual_operation": revision.actual_operation.value,
        "diff": asdict(revision.diff), "validation_warnings": list(revision.validation_warnings),
        "supersedes_revision_id": revision.supersedes_revision_id or None,
        "revoked": False, "source_tool": revision.source_tool,
        "source_tool_version": revision.source_tool_version,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
