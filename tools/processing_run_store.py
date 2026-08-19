"""Filesystem persistence for immutable processing plans, reports, and events."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from tools.processing_events import JsonlEventSink
from tools.processing_pipeline import ProcessingPlan, record_sha256
from tools.processing_reports import (
    ArtifactReference,
    ProcessingReport,
    processing_report_to_dict,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
logger = logging.getLogger(__name__)


class ProcessingPersistenceError(OSError):
    """Raised when an auditable run artifact cannot be persisted safely."""


@dataclass(frozen=True)
class StoredDocument:
    path: Path
    relative_path: str
    sha256: str

    def as_artifact(self, kind: str) -> ArtifactReference:
        return ArtifactReference(self.relative_path, self.sha256, kind)


class ProcessingRunStore:
    """Persist run artifacts under one traversal-safe `.processing_runs` root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.plans_dir = self.root / "plans"
        self.reports_dir = self.root / "reports"
        self.events_dir = self.root / "events"
        self.artifacts_dir = self.root / "artifacts"
        for directory in (
            self.plans_dir,
            self.reports_dir,
            self.events_dir,
            self.artifacts_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def persist_plan(self, plan: ProcessingPlan) -> StoredDocument:
        destination = self._document_path(self.plans_dir, plan.plan_id, ".json")
        return self._atomic_json(destination, processing_plan_to_dict(plan))

    def persist_report(self, report: ProcessingReport) -> StoredDocument:
        destination = self._document_path(
            self.reports_dir,
            report.report_id,
            ".json",
        )
        return self._atomic_json(destination, processing_report_to_dict(report))

    def prepare_event_log(
        self,
        report_id: str,
        *,
        fsync: bool = True,
    ) -> tuple[Path, JsonlEventSink]:
        path = self._document_path(self.events_dir, report_id, ".jsonl")
        try:
            with path.open("x", encoding="utf-8"):
                pass
        except FileExistsError as exc:
            raise ProcessingPersistenceError(
                f"Event log already exists for report_id={report_id}"
            ) from exc
        except OSError as exc:
            raise ProcessingPersistenceError(
                f"Could not create event log {path}: {exc}"
            ) from exc
        return path, JsonlEventSink(path, fsync=fsync)

    def describe_file(self, path: str | Path, *, kind: str) -> ArtifactReference:
        resolved = Path(path).resolve()
        self._ensure_within_root(resolved)
        if not resolved.is_file():
            raise ProcessingPersistenceError(f"Run artifact does not exist: {resolved}")
        return ArtifactReference(
            relative_path=resolved.relative_to(self.root).as_posix(),
            sha256=_sha256_file(resolved),
            kind=kind,
        )

    def update_latest(
        self,
        report: ProcessingReport,
        report_document: StoredDocument,
    ) -> StoredDocument:
        payload = {
            "schema_version": 1,
            "report_id": report.report_id,
            "plan_id": report.plan_id,
            "status": report.status.value,
            "finished_at": report.finished_at.isoformat(),
            "report_path": f"reports/{report.report_id}.json",
            "report_sha256": report_document.sha256,
        }
        return self._atomic_json(self.root / "latest.json", payload, replace=True)

    def report_path(self, report_id: str) -> Path:
        return self._document_path(self.reports_dir, report_id, ".json")

    def plan_path(self, plan_id: str) -> Path:
        return self._document_path(self.plans_dir, plan_id, ".json")

    def event_path(self, report_id: str) -> Path:
        return self._document_path(self.events_dir, report_id, ".jsonl")

    def _document_path(self, directory: Path, identifier: str, suffix: str) -> Path:
        if not _SAFE_ID.fullmatch(str(identifier)):
            raise ProcessingPersistenceError(
                f"Unsafe processing artifact identifier: {identifier!r}"
            )
        path = (directory / f"{identifier}{suffix}").resolve()
        self._ensure_within_root(path)
        return path

    def _atomic_json(
        self,
        destination: Path,
        payload: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> StoredDocument:
        encoded = (
            json.dumps(
                _json_value(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if destination.exists() and not replace:
            existing_sha = _sha256_file(destination)
            if existing_sha == digest:
                return self._stored_document(destination, existing_sha)
            raise ProcessingPersistenceError(
                f"Artifact ID collision would overwrite {destination}"
            )
        descriptor = -1
        temporary_path: Path | None = None
        try:
            descriptor, raw_temporary_path = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            temporary_path = Path(raw_temporary_path)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, destination)
            temporary_path = None
        except OSError as exc:
            raise ProcessingPersistenceError(
                f"Atomic write failed for {destination}: {exc}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.warning(
                        "Could not remove failed processing temp file path=%s error=%s",
                        temporary_path,
                        cleanup_error,
                    )
        return self._stored_document(destination, digest)

    def _stored_document(self, path: Path, sha256: str) -> StoredDocument:
        self._ensure_within_root(path)
        return StoredDocument(
            path=path,
            relative_path=path.relative_to(self.root).as_posix(),
            sha256=sha256,
        )

    def _ensure_within_root(self, path: Path) -> None:
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ProcessingPersistenceError(
                f"Processing artifact path escapes run root: {path}"
            ) from exc


def processing_plan_to_dict(plan: ProcessingPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "plan_id": plan.plan_id,
        "created_at": plan.created_at.isoformat(),
        "operator": plan.operator,
        "execution_mode": plan.execution_mode.value,
        "source_manifest_sha": plan.source_manifest_sha,
        "review_revision": plan.review_revision,
        "sample_count": plan.sample_count,
        "records": [
            {
                "source_index": record.source_index,
                "sample_id": record.sample_id,
                "record_sha": record_sha256(record.fields),
                "fields": _json_value(record.fields),
            }
            for record in plan.records
        ],
        "routing_decisions": [
            {
                "sample_id": decision.sample_id,
                "source_index": decision.source_index,
                "decision": decision.decision.value,
                "reason": decision.reason,
                "violation_codes": list(decision.violation_codes),
                "semantics": decision.semantics.to_dict(),
                "additional_decisions": [
                    item.value for item in decision.additional_decisions
                ],
            }
            for decision in plan.routing_decisions
        ],
        "blocking_items": [
            {
                "sample_id": item.sample_id,
                "source_index": item.source_index,
                "reason": item.reason,
                "violation_code": item.violation_code,
            }
            for item in plan.blocking_items
        ],
        "warnings": [
            {
                "sample_id": warning.sample_id,
                "code": warning.code,
                "message": warning.message,
            }
            for warning in plan.warnings
        ],
        "statistics": {
            "ready_count": plan.statistics.ready_count,
            "annotation_count": plan.statistics.annotation_count,
            "color_count": plan.statistics.color_count,
            "blocking_count": plan.statistics.blocking_count,
            "excluded_count": plan.statistics.excluded_count,
            "manual_review_count": plan.statistics.manual_review_count,
        },
        "retry_source_plan_id": plan.retry_source_plan_id or None,
        "retry_source_report_id": plan.retry_source_report_id or None,
        "attempt": plan.attempt,
        "routing_code_version": plan.routing_code_version,
        "annotation_source_package_id": plan.annotation_source_package_id or None,
        "annotation_revision_ids": list(plan.annotation_revision_ids),
        "color_source_plan_id": plan.color_source_plan_id or None,
        "color_source_report_id": plan.color_source_report_id or None,
        "color_source_package_id": plan.color_source_package_id or None,
        "color_revision_ids": list(plan.color_revision_ids),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(item) for item in value]
    return str(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
