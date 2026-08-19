"""Immutable Phase 3B processing report contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

from tools.processing_pipeline import ExecutionMode, RoutingDecisionType
from tools.processing_plan_validation import (
    PlanValidationIssue,
    PlanValidationResult,
)

PROCESSING_REPORT_SCHEMA_VERSION = 1


class ProcessingReportStatus(str, Enum):
    CREATED = "CREATED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class SampleProcessingStatus(str, Enum):
    DEFERRED = "DEFERRED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ProcessingStepStatus(str, Enum):
    NOT_EXECUTED = "NOT_EXECUTED"
    SUCCESS = "SUCCESS"
    DEFERRED = "DEFERRED"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class ArtifactReference:
    relative_path: str
    sha256: str = ""
    kind: str = "artifact"


@dataclass(frozen=True)
class StepResult:
    status: ProcessingStepStatus
    action: str
    artifacts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error_code: str = ""
    message: str = ""
    retryable: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))
        if self.status == ProcessingStepStatus.FAILED and not self.error_code:
            raise ValueError("A failed StepResult requires error_code")
        if self.status != ProcessingStepStatus.FAILED and self.error_code:
            raise ValueError("error_code is only valid for a failed StepResult")


@dataclass(frozen=True)
class SampleProcessingResult:
    sample_id: str
    primary_routing: RoutingDecisionType
    additional_routing: tuple[RoutingDecisionType, ...]
    status: SampleProcessingStatus
    action: str
    step_id: str
    started_at: datetime
    finished_at: datetime
    artifacts: tuple[ArtifactReference, ...] = ()
    warnings: tuple[str, ...] = ()
    error_code: str = ""
    message: str = ""
    retryable: bool = False
    attempt: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("SampleProcessingResult.sample_id must not be empty")
        if not _is_aware_utc(self.started_at) or not _is_aware_utc(self.finished_at):
            raise ValueError("Sample timestamps must be timezone-aware UTC")
        if self.finished_at < self.started_at:
            raise ValueError("Sample finished_at cannot precede started_at")
        if self.status == SampleProcessingStatus.FAILED and not self.error_code:
            raise ValueError("A failed sample requires error_code")
        object.__setattr__(self, "additional_routing", tuple(self.additional_routing))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "metadata", _freeze_json_mapping(self.metadata))


@dataclass(frozen=True)
class ProcessingReportSummary:
    total: int
    executable: int
    deferred: int
    succeeded: int
    failed: int
    blocked: int
    excluded: int
    cancelled: int
    warnings: int


@dataclass(frozen=True)
class CancellationSummary:
    requested: bool
    reason: str = ""
    requested_at: datetime | None = None
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        for value in (self.requested_at, self.completed_at):
            if value is not None and not _is_aware_utc(value):
                raise ValueError("Cancellation timestamps must be timezone-aware UTC")


@dataclass(frozen=True)
class ProcessingReport:
    report_id: str
    plan_id: str
    source_manifest_sha: str
    review_revision: str
    execution_mode: ExecutionMode
    started_at: datetime
    finished_at: datetime
    status: ProcessingReportStatus
    operator: str
    dry_run: bool
    validation_result: PlanValidationResult
    summary: ProcessingReportSummary
    sample_results: tuple[SampleProcessingResult, ...]
    events_reference: ArtifactReference | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    cancellation: CancellationSummary | None = None
    retry_source_report_id: str = ""
    schema_version: int = PROCESSING_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.report_id.strip() or not self.plan_id.strip():
            raise ValueError("report_id and plan_id must not be empty")
        if not _is_aware_utc(self.started_at) or not _is_aware_utc(self.finished_at):
            raise ValueError("Report timestamps must be timezone-aware UTC")
        if self.finished_at < self.started_at:
            raise ValueError("Report finished_at cannot precede started_at")
        object.__setattr__(self, "sample_results", tuple(self.sample_results))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "errors", tuple(self.errors))
        expected = build_report_summary(self.sample_results, self.warnings)
        if self.summary != expected:
            raise ValueError("ProcessingReport.summary does not match sample_results")

    def to_dict(self) -> dict[str, Any]:
        return processing_report_to_dict(self)


def build_report_summary(
    results: tuple[SampleProcessingResult, ...],
    warnings: tuple[str, ...] = (),
) -> ProcessingReportSummary:
    statuses = [result.status for result in results]
    excluded = sum(
        result.status == SampleProcessingStatus.SKIPPED
        and result.primary_routing == RoutingDecisionType.EXCLUDED
        for result in results
    )
    executable = sum(
        result.primary_routing
        not in {
            RoutingDecisionType.BLOCKED,
            RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
            RoutingDecisionType.EXCLUDED,
        }
        for result in results
    )
    return ProcessingReportSummary(
        total=len(results),
        executable=executable,
        deferred=statuses.count(SampleProcessingStatus.DEFERRED),
        succeeded=statuses.count(SampleProcessingStatus.SUCCESS),
        failed=statuses.count(SampleProcessingStatus.FAILED),
        blocked=statuses.count(SampleProcessingStatus.BLOCKED),
        excluded=excluded,
        cancelled=statuses.count(SampleProcessingStatus.CANCELLED),
        warnings=len(warnings) + sum(len(result.warnings) for result in results),
    )


def processing_report_to_dict(report: ProcessingReport) -> dict[str, Any]:
    return {
        "schema_version": report.schema_version,
        "report_id": report.report_id,
        "plan_id": report.plan_id,
        "source_manifest_sha": report.source_manifest_sha,
        "review_revision": report.review_revision,
        "execution_mode": report.execution_mode.value,
        "started_at": report.started_at.isoformat(),
        "finished_at": report.finished_at.isoformat(),
        "status": report.status.value,
        "operator": report.operator,
        "dry_run": report.dry_run,
        "validation_result": validation_result_to_dict(report.validation_result),
        "summary": _summary_to_dict(report.summary),
        "sample_results": [_sample_result_to_dict(item) for item in report.sample_results],
        "events_reference": _artifact_to_dict(report.events_reference),
        "artifacts": [_artifact_to_dict(item) for item in report.artifacts],
        "warnings": list(report.warnings),
        "errors": list(report.errors),
        "cancellation": _cancellation_to_dict(report.cancellation),
        "retry_source_report_id": report.retry_source_report_id or None,
    }


def validation_result_to_dict(result: PlanValidationResult) -> dict[str, Any]:
    return {
        "valid": result.valid,
        "blocking_issues": [_validation_issue_to_dict(item) for item in result.blocking_issues],
        "warnings": [_validation_issue_to_dict(item) for item in result.warnings],
        "checked_at": result.checked_at.isoformat(),
        "current_manifest_sha": result.current_manifest_sha,
        "expected_manifest_sha": result.expected_manifest_sha,
        "executable_sample_ids": list(result.executable_sample_ids),
        "skipped_sample_ids": list(result.skipped_sample_ids),
        "issues": [_validation_issue_to_dict(item) for item in result.issues],
    }


def _validation_issue_to_dict(issue: PlanValidationIssue) -> dict[str, Any]:
    return {
        "code": issue.code,
        "severity": issue.severity.value,
        "sample_id": issue.sample_id or None,
        "message": issue.message,
        "details": _thaw(issue.details),
        "retryable": issue.retryable,
    }


def _sample_result_to_dict(result: SampleProcessingResult) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "primary_routing": result.primary_routing.value,
        "additional_routing": [item.value for item in result.additional_routing],
        "status": result.status.value,
        "action": result.action,
        "step_id": result.step_id or None,
        "started_at": result.started_at.isoformat(),
        "finished_at": result.finished_at.isoformat(),
        "artifacts": [_artifact_to_dict(item) for item in result.artifacts],
        "warnings": list(result.warnings),
        "error_code": result.error_code or None,
        "message": result.message,
        "retryable": result.retryable,
        "attempt": result.attempt,
        "metadata": _thaw(result.metadata),
    }


def _summary_to_dict(summary: ProcessingReportSummary) -> dict[str, int]:
    return {
        "total": summary.total,
        "executable": summary.executable,
        "deferred": summary.deferred,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "blocked": summary.blocked,
        "excluded": summary.excluded,
        "cancelled": summary.cancelled,
        "warnings": summary.warnings,
    }


def _artifact_to_dict(artifact: ArtifactReference | None) -> dict[str, str] | None:
    if artifact is None:
        return None
    return {
        "relative_path": artifact.relative_path,
        "sha256": artifact.sha256,
        "kind": artifact.kind,
    }


def _cancellation_to_dict(
    cancellation: CancellationSummary | None,
) -> dict[str, Any] | None:
    if cancellation is None:
        return None
    return {
        "requested": cancellation.requested,
        "reason": cancellation.reason,
        "requested_at": (
            cancellation.requested_at.isoformat()
            if cancellation.requested_at is not None
            else None
        ),
        "completed_at": (
            cancellation.completed_at.isoformat()
            if cancellation.completed_at is not None
            else None
        ),
    }


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = _normalize_json(value)
    json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return _freeze(normalized)


def _normalize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _normalize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"Report metadata is not JSON serializable: {type(value).__name__}")


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _is_aware_utc(value: datetime) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timezone.utc.utcoffset(value)
    )
