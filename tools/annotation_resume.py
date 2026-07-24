"""Validate annotation work, commit immutable revisions, and re-plan safely."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.annotation_packages import (
    AnnotationPackageError,
    AnnotationPackageStatus,
    AnnotationWorkPackage,
    _write_json_atomic,
    load_annotation_package,
    resolve_package_member,
)
from tools.annotation_revisions import AnnotationRevision, AnnotationRevisionError, AnnotationRevisionStore
from tools.annotation_validation import AnnotationErrorCode, validate_annotation_revision
from tools.processing_events import (
    InMemoryEventSink,
    ProcessingEvent,
    ProcessingEventLevel,
    ProcessingEventPublisher,
    ProcessingEventType,
)
from tools.processing_execution import CancellationToken
from tools.processing_pipeline import ProcessingPlan, ProcessingPlanner, record_sha256
from tools.processing_plan_validation import ProcessingPlanValidator, ProcessingValidationContext
from tools.review_workflow import map_review_action_to_legacy_fields


@dataclass(frozen=True)
class AnnotationItemFailure:
    sample_id: str
    error_codes: tuple[str, ...]
    messages: tuple[str, ...]
    retryable: bool = True


@dataclass(frozen=True)
class AnnotationCompletionReport:
    completion_id: str
    package_id: str
    source_plan_id: str
    created_at: datetime
    status: AnnotationPackageStatus
    successful_revisions: tuple[AnnotationRevision, ...]
    failures: tuple[AnnotationItemFailure, ...]
    record_update_proposals: tuple[Mapping[str, Any], ...]
    follow_up_plan: ProcessingPlan | None
    report_path: Path
    event_path: Path
    events: tuple[ProcessingEvent, ...]


class AnnotationResumeService:
    """Resume is a separate command; it never blocks the processing engine."""

    def __init__(
        self,
        *,
        revision_store: AnnotationRevisionStore,
        planner: ProcessingPlanner | None = None,
        validator: ProcessingPlanValidator | None = None,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self._revision_store = revision_store
        self._planner = planner or ProcessingPlanner()
        self._validator = validator or ProcessingPlanValidator()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._id_generator = id_generator or (lambda: str(uuid4()))

    def resume(
        self,
        package_path: str | Path,
        original_plan: ProcessingPlan,
        *,
        current_entries: Sequence[tuple[int, Mapping[str, Any]]],
        current_manifest_sha: str,
        revision_reasons: Mapping[str, str],
        escalation_reasons: Mapping[str, str] | None = None,
        operator: str = "",
        source_tool: str = "manual",
        source_tool_version: str = "",
        build_follow_up: bool = True,
        cancellation: CancellationToken | None = None,
    ) -> AnnotationCompletionReport:
        package = load_annotation_package(package_path)
        completion_id = self._safe_id(self._id_generator())
        event_path = package.root / f"resume-{completion_id}.events.json"
        sink = InMemoryEventSink()
        publisher = ProcessingEventPublisher(
            plan_id=original_plan.plan_id, report_id=completion_id, sink=sink,
            clock=self._clock, id_generator=self._id_generator,
        )
        publisher.emit(ProcessingEventType.ANNOTATION_VALIDATION_STARTED,
                       "Annotation package validation started.", stage="annotation_resume",
                       metadata={"package_id": package.package_id})
        token = cancellation or CancellationToken(clock=self._clock)
        failures: list[AnnotationItemFailure] = []
        revisions: list[AnnotationRevision] = []
        prior = _read_completion(package.root / "completion.json")
        prior_successful = {
            str(value.get("sample_id") or ""): dict(value)
            for value in prior.get("successful_items", [])
            if isinstance(value, Mapping) and value.get("sample_id")
        }
        prior_successful_ids = set(prior_successful) | {
            str(value) for value in prior.get("successful_sample_ids", [])
        }
        prior_revision_ids = [str(value) for value in prior.get("revision_ids", [])]
        proposals: list[Mapping[str, Any]] = [
            dict(value) for value in prior.get("record_update_proposals", [])
            if isinstance(value, Mapping)
        ]
        if package.plan_id != original_plan.plan_id or package.source_manifest_sha != current_manifest_sha:
            failures.extend(
                AnnotationItemFailure(item.sample_id, (AnnotationErrorCode.ANNOTATION_PACKAGE_STALE.value,),
                                      ("Plan or source manifest changed after package creation.",))
                for item in package.items
            )
        else:
            current_by_id = {str(row.get("sample_id") or row.get("inspection_id") or row.get("config_snapshot_path") or row.get("source_path") or f"row:{index}"): row for index, row in current_entries}
            for item in package.items:
                if token.is_cancelled:
                    failures.append(AnnotationItemFailure(
                        item.sample_id, ("ANNOTATION_RESUME_CANCELLED",),
                        ("Resume was cancelled before this revision was committed.",),
                    ))
                    continue
                previous = prior_successful.get(item.sample_id)
                if previous is not None:
                    try:
                        current_working_sha = _normalized_file_sha(
                            resolve_package_member(package, item.working_label)
                        )
                    except (AnnotationPackageError, OSError, UnicodeError):
                        current_working_sha = ""
                    if (
                        current_working_sha == str(previous.get("label_sha256") or "")
                        and not self._revision_store.is_revision_id_revoked(
                            str(previous.get("revision_id") or "")
                        )
                    ):
                        continue
                failure, revision, proposal = self._resume_item(
                    package, item, current_by_id.get(item.sample_id),
                    revision_reason=str(revision_reasons.get(item.sample_id) or ""),
                    escalation_reason=str((escalation_reasons or {}).get(item.sample_id) or ""),
                    operator=operator or package.operator, source_tool=source_tool,
                    source_tool_version=source_tool_version, publisher=publisher,
                    supersedes_revision_id=(
                        str(previous.get("revision_id") or "") if previous else ""
                    ),
                    cancellation=token,
                )
                if failure is not None:
                    failures.append(failure)
                if revision is not None:
                    revisions.append(revision)
                if proposal is not None:
                    proposals.append(proposal)
        successful_ids = prior_successful_ids | {value.sample_id for value in revisions}
        successful_items = {
            **prior_successful,
            **{
                value.sample_id: {
                    "sample_id": value.sample_id,
                    "revision_id": value.revision_id,
                    "label_sha256": value.new_label_sha256,
                    "before_sha": value.parent_label_sha256,
                    "requested_operation": value.requested_operation.value,
                    "actual_operation": value.actual_operation.value,
                }
                for value in revisions
            },
        }
        status = _completion_status(len(package.items), len(successful_ids), len(failures))
        follow_up = None
        if build_follow_up and proposals:
            try:
                self._verify_canonical_revisions(revisions)
                follow_up = build_follow_up_plan_from_annotation(
                    original_plan, completion_id, package, tuple(revisions), tuple(proposals),
                    current_entries=current_entries, current_manifest_sha=current_manifest_sha,
                    planner=self._planner, validator=self._validator, created_at=self._clock(),
                    all_revision_ids=tuple(
                        prior_revision_ids + [value.revision_id for value in revisions]
                    ),
                )
            except (AnnotationPackageError, ValueError) as exc:
                failures.append(AnnotationItemFailure(
                    "batch", (AnnotationErrorCode.CANONICAL_CONFLICT.value,),
                    (str(exc),), retryable=False,
                ))
                status = AnnotationPackageStatus.PARTIAL_FAILURE if successful_ids else AnnotationPackageStatus.FAILED
        completion_path = package.root / f"completion-{completion_id}.json"
        payload = {
            "schema_version": 1, "completion_id": completion_id,
            "package_id": package.package_id, "source_plan_id": original_plan.plan_id,
            "created_at": self._clock().isoformat(), "status": status.value,
            "revision_ids": prior_revision_ids + [revision.revision_id for revision in revisions],
            "successful_sample_ids": sorted(successful_ids),
            "successful_items": list(successful_items.values()),
            "failures": [failure.__dict__ for failure in failures],
            "record_update_proposals": [dict(value) for value in proposals],
            "follow_up_plan_id": follow_up.plan_id if follow_up else None,
            "item_results": [
                {
                    "sample_id": item["sample_id"],
                    "status": "SUCCESS",
                    "action": "ANNOTATION_REVISION_CREATED",
                    "revision_id": item["revision_id"],
                    "before_sha": item.get("before_sha", ""),
                    "after_sha": item["label_sha256"],
                    "requested_operation": item.get("requested_operation", ""),
                    "actual_operation": item.get("actual_operation", ""),
                    "follow_up_required": "REPLAN",
                }
                for item in successful_items.values()
            ] + [
                {
                    "sample_id": failure.sample_id,
                    "status": "FAILED",
                    "error_codes": list(failure.error_codes),
                    "retryable": failure.retryable,
                }
                for failure in failures
            ],
            "training_started": False,
        }
        _write_json_atomic(completion_path, payload)
        _write_json_atomic(package.root / "validation_report.json", payload)
        _write_json_atomic(package.root / "completion.json", payload)
        event_type = (
            ProcessingEventType.ANNOTATION_PACKAGE_COMPLETED
            if status == AnnotationPackageStatus.COMPLETED
            else ProcessingEventType.ANNOTATION_PACKAGE_INCOMPLETE
        )
        publisher.emit(event_type, f"Annotation resume finished with status {status.value}.",
                       stage="annotation_resume", level=(ProcessingEventLevel.INFO if not failures else ProcessingEventLevel.WARNING),
                       metadata={"package_id": package.package_id, "revision_count": len(revisions), "failure_count": len(failures)})
        events = sink.events
        _write_json_atomic(event_path, {"schema_version": 1, "events": [event.to_dict() for event in events]})
        return AnnotationCompletionReport(
            completion_id=completion_id, package_id=package.package_id,
            source_plan_id=original_plan.plan_id, created_at=self._clock(), status=status,
            successful_revisions=tuple(revisions), failures=tuple(failures),
            record_update_proposals=tuple(proposals), follow_up_plan=follow_up,
            report_path=completion_path, event_path=event_path, events=events,
        )

    def _resume_item(
        self, package, item, current_row, *, revision_reason, escalation_reason,
        operator, source_tool, source_tool_version, publisher,
        supersedes_revision_id, cancellation,
    ):
        errors: list[str] = []
        messages: list[str] = []
        if current_row is None:
            errors.append(AnnotationErrorCode.ANNOTATION_PACKAGE_STALE.value)
            messages.append("Sample no longer exists in the current manifest context.")
        try:
            source_path = resolve_package_member(package, item.source_image)
            original_path = resolve_package_member(package, item.original_label)
            working_path = resolve_package_member(package, item.working_label)
            if not source_path.is_file() or _sha256_file(source_path) != item.source_image_sha256:
                errors.append(AnnotationErrorCode.IMAGE_STALE.value)
                messages.append("Packaged source image changed or is missing.")
            if not original_path.is_file() or _normalized_file_sha(original_path) != item.parent_label_sha256:
                errors.append(AnnotationErrorCode.PARENT_LABEL_STALE.value)
                messages.append("Parent label changed or is missing.")
            if not working_path.is_file():
                errors.append(AnnotationErrorCode.LABEL_MISSING.value)
                messages.append("Working label is missing.")
                working_text = ""
            else:
                working_text = working_path.read_text(encoding="utf-8")
        except (AnnotationPackageError, OSError, UnicodeError) as exc:
            errors.append(getattr(exc, "code", AnnotationErrorCode.LABEL_PARSE_ERROR.value))
            messages.append(str(exc))
            working_text = ""
            original_path = None
        if not errors and original_path is not None:
            validation = validate_annotation_revision(
                parent_label_text=original_path.read_text(encoding="utf-8"),
                working_label_text=working_text, class_count=len(package.class_mapping),
                requested_operation=item.requested_operation, allow_empty=item.allow_empty,
                revision_reason=revision_reason, escalation_reason=escalation_reason,
            )
            errors.extend(code.value for code in validation.errors)
            messages.extend(validation.messages)
        else:
            validation = None
        if errors:
            publisher.emit(ProcessingEventType.ANNOTATION_ITEM_REJECTED,
                           "Annotation item rejected.", stage="annotation_resume",
                           level=ProcessingEventLevel.WARNING, sample_id=item.sample_id,
                           metadata={"error_codes": errors})
            return AnnotationItemFailure(item.sample_id, tuple(dict.fromkeys(errors)), tuple(messages)), None, None
        publisher.emit(ProcessingEventType.ANNOTATION_ITEM_VALIDATED,
                       "Annotation item validated.", stage="annotation_resume", sample_id=item.sample_id,
                       metadata={
                           "package_id": package.package_id,
                           "requested_operation": item.requested_operation.value,
                           "before_sha": item.parent_label_sha256,
                           "after_sha": validation.label_sha256,
                       })
        if cancellation.is_cancelled:
            return AnnotationItemFailure(
                item.sample_id, ("ANNOTATION_RESUME_CANCELLED",),
                ("Resume was cancelled before revision commit.",),
            ), None, None
        try:
            revision = self._revision_store.commit(
                package, item, label_text=working_text, validation=validation,
                operator=operator, revision_reason=revision_reason,
                source_tool=source_tool, source_tool_version=source_tool_version,
                supersedes_revision_id=supersedes_revision_id,
            )
        except AnnotationRevisionError as exc:
            publisher.emit(ProcessingEventType.ANNOTATION_ITEM_REJECTED,
                           "Annotation revision commit failed.", stage="annotation_resume",
                           level=ProcessingEventLevel.ERROR, sample_id=item.sample_id,
                           metadata={"error_codes": [exc.code]})
            return AnnotationItemFailure(item.sample_id, (exc.code,), (str(exc),), exc.retryable), None, None
        publisher.emit(ProcessingEventType.ANNOTATION_REVISION_CREATED,
                       "Append-only annotation revision created.", stage="annotation_resume",
                       sample_id=item.sample_id, metadata={
                           "package_id": package.package_id,
                           "revision_id": revision.revision_id,
                           "requested_operation": revision.requested_operation.value,
                           "actual_operation": revision.actual_operation.value,
                           "before_sha": revision.parent_label_sha256,
                           "after_sha": revision.new_label_sha256,
                           "artifact": revision.metadata_path.relative_to(
                               self._revision_store.root
                           ).as_posix(),
                       })
        proposal = _record_update_proposal(current_row or {}, revision)
        return None, revision, proposal

    def _verify_canonical_revisions(
        self, revisions: Sequence[AnnotationRevision]
    ) -> None:
        if not revisions:
            return
        from tools.export_review_dataset import load_approved_annotation_selections

        selections = load_approved_annotation_selections(
            self._revision_store.source_manifest
        )
        for revision in revisions:
            selection = selections.get(revision.image_sha256)
            expected = (
                selection.get("label_sha256_by_sample", {}).get(revision.sample_id)
                if selection
                else ""
            )
            if expected != revision.new_label_sha256:
                raise AnnotationPackageError(
                    AnnotationErrorCode.CANONICAL_CONFLICT.value,
                    f"Committed revision is not the active canonical reference: {revision.revision_id}",
                    sample_id=revision.sample_id,
                    retryable=False,
                )

    @staticmethod
    def _safe_id(value: str) -> str:
        value = str(value).strip()
        if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value):
            raise ValueError("Unsafe annotation completion ID")
        return value


def build_follow_up_plan_from_annotation(
    original_plan: ProcessingPlan,
    completion_report_id: str,
    package: AnnotationWorkPackage,
    revisions: tuple[AnnotationRevision, ...],
    proposals: tuple[Mapping[str, Any], ...],
    *,
    current_entries: Sequence[tuple[int, Mapping[str, Any]]],
    current_manifest_sha: str,
    planner: ProcessingPlanner,
    validator: ProcessingPlanValidator,
    created_at: datetime,
    all_revision_ids: tuple[str, ...] = (),
) -> ProcessingPlan:
    """Overlay proposals and call the existing Planner; never force routing."""
    current_by_id = {
        str(row.get("sample_id") or row.get("inspection_id") or row.get("config_snapshot_path") or row.get("source_path") or f"row:{index}"): row
        for index, row in current_entries
    }
    hashes = {
        record.sample_id: record_sha256({
            key: current_by_id.get(record.sample_id, {}).get(key, "")
            for key in record.fields
        })
        for record in original_plan.records
        if record.sample_id in current_by_id
    }
    current_context = ProcessingValidationContext(
        current_manifest_sha=current_manifest_sha, current_record_hashes=hashes,
        artifact_root=package.root.parent,
    )
    if not validator.validate(original_plan, current_context).valid:
        raise AnnotationPackageError("ANNOTATION_PACKAGE_STALE", "Original processing plan is stale; follow-up plan was not created.")
    proposal_by_id = {str(value["sample_id"]): value for value in proposals}
    overlaid: list[tuple[int, Mapping[str, Any]]] = []
    for source_index, row in current_entries:
        sample_id = str(row.get("sample_id") or row.get("inspection_id") or row.get("config_snapshot_path") or row.get("source_path") or f"row:{source_index}")
        proposal = proposal_by_id.get(sample_id)
        overlaid.append((source_index, {**row, **(dict(proposal["legacy_fields"]) if proposal else {})}))
    planned = planner.create_plan(
        overlaid, operator=original_plan.operator, execution_mode=original_plan.execution_mode,
        source_manifest_sha=current_manifest_sha,
        review_revision=f"annotation:{completion_report_id}", created_at=created_at,
    )
    return replace(
        planned, retry_source_plan_id=original_plan.plan_id,
        retry_source_report_id=completion_report_id, attempt=original_plan.attempt + 1,
        annotation_source_package_id=package.package_id,
        annotation_revision_ids=(
            all_revision_ids
            or tuple(revision.revision_id for revision in revisions)
        ),
    )


def _record_update_proposal(row: Mapping[str, Any], revision: AnnotationRevision) -> Mapping[str, Any]:
    label = str(row.get("review_label") or "confirmed_ng")
    outcome = str(row.get("review_outcome") or "fail")
    failure = str(row.get("failure_category") or label)
    legacy = map_review_action_to_legacy_fields(
        label, review_outcome=outcome, failure_category=failure,
        skip_reason="", training_selected=True,
    )
    legacy.update({
        "annotation_status": "verified_annotation",
        "output_label": str(revision.label_path),
        "annotation_revision_id": revision.revision_id,
        "revision_reason": revision.revision_reason,
    })
    return {
        "sample_id": revision.sample_id, "revision_id": revision.revision_id,
        "record_sha_before": record_sha256(row), "legacy_fields": legacy,
        "apply_mode": "external_overlay_only", "manifest_modified": False,
    }


def _completion_status(total: int, successes: int, failures: int) -> AnnotationPackageStatus:
    if successes == total and not failures:
        return AnnotationPackageStatus.COMPLETED
    if successes:
        return AnnotationPackageStatus.PARTIAL_FAILURE
    return AnnotationPackageStatus.INCOMPLETE if failures else AnnotationPackageStatus.FAILED


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_file_sha(path: Path) -> str:
    from tools.annotation_validation import normalized_label_sha256
    return normalized_label_sha256(path.read_text(encoding="utf-8"))


def _read_completion(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
