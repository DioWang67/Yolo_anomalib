"""Pure validation of immutable Phase 3 processing plans."""

from __future__ import annotations

import csv
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from tools.processing_pipeline import (
    PROCESSING_PLAN_SCHEMA_VERSION,
    PROCESSING_ROUTING_CODE_VERSION,
    ExecutionMode,
    ProcessingPlan,
    RoutingDecisionType,
    record_sha256,
    sha256_file,
)
from tools.review_workflow import record_identity

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXECUTABLE_ROUTING = frozenset(
    {
        RoutingDecisionType.READY_FOR_DATASET,
        RoutingDecisionType.NEEDS_ANNOTATION,
        RoutingDecisionType.NEEDS_CLASS_FIX,
        RoutingDecisionType.NEEDS_COLOR_CALIBRATION,
    }
)
_NON_EXECUTABLE_ROUTING = frozenset(
    {
        RoutingDecisionType.BLOCKED,
        RoutingDecisionType.EXCLUDED,
        RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
    }
)


class PlanValidationSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    BLOCKING = "BLOCKING"


@dataclass(frozen=True)
class PlanValidationIssue:
    code: str
    severity: PlanValidationSeverity
    message: str
    sample_id: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


@dataclass(frozen=True)
class ProcessingValidationContext:
    current_manifest_sha: str
    current_record_hashes: Mapping[str, str]
    artifact_root: Path
    deployment_confirmation_allowed: bool = False
    expected_schema_version: int = PROCESSING_PLAN_SCHEMA_VERSION
    expected_routing_code_version: str = PROCESSING_ROUTING_CODE_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "current_record_hashes",
            MappingProxyType(dict(self.current_record_hashes)),
        )
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())


@dataclass(frozen=True)
class PlanValidationResult:
    valid: bool
    blocking_issues: tuple[PlanValidationIssue, ...]
    warnings: tuple[PlanValidationIssue, ...]
    checked_at: datetime
    current_manifest_sha: str
    expected_manifest_sha: str
    executable_sample_ids: tuple[str, ...]
    skipped_sample_ids: tuple[str, ...]
    issues: tuple[PlanValidationIssue, ...]


class ProcessingPlanValidator:
    """Validate plan integrity and freshness without mutating the plan."""

    def __init__(self, *, clock=None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def validate(
        self,
        plan: ProcessingPlan,
        current_context: ProcessingValidationContext,
    ) -> PlanValidationResult:
        issues: list[PlanValidationIssue] = []
        self._validate_header(plan, current_context, issues)
        self._validate_shape(plan, issues)
        self._validate_routing(plan, issues)
        self._validate_blocking_items(plan, issues)
        self._validate_statistics(plan, issues)
        self._validate_freshness(plan, current_context, issues)
        self._validate_artifact_paths(plan, current_context, issues)
        self._validate_immutability(plan, issues)

        executable: list[str] = []
        skipped: list[str] = []
        for decision in sorted(
            plan.routing_decisions,
            key=lambda item: (item.source_index, item.sample_id),
        ):
            if decision.decision in _EXECUTABLE_ROUTING:
                executable.append(decision.sample_id)
            else:
                skipped.append(decision.sample_id)
        blocking = tuple(
            issue
            for issue in issues
            if issue.severity
            in {PlanValidationSeverity.ERROR, PlanValidationSeverity.BLOCKING}
        )
        warnings = tuple(
            issue
            for issue in issues
            if issue.severity
            in {PlanValidationSeverity.INFO, PlanValidationSeverity.WARNING}
        )
        return PlanValidationResult(
            valid=not blocking,
            blocking_issues=blocking,
            warnings=warnings,
            checked_at=_require_utc(self._clock()),
            current_manifest_sha=current_context.current_manifest_sha,
            expected_manifest_sha=plan.source_manifest_sha,
            executable_sample_ids=tuple(executable),
            skipped_sample_ids=tuple(skipped),
            issues=tuple(issues),
        )

    @staticmethod
    def _validate_header(
        plan: ProcessingPlan,
        context: ProcessingValidationContext,
        issues: list[PlanValidationIssue],
    ) -> None:
        if plan.schema_version != context.expected_schema_version:
            _add(issues, "unsupported_plan_schema", "Unsupported processing plan schema version.")
        if plan.routing_code_version != context.expected_routing_code_version:
            _add(issues, "stale_routing_code", "The plan was created by an unsupported routing code version.", retryable=True)
        if not _SAFE_ID.fullmatch(plan.plan_id):
            _add(issues, "invalid_plan_id", "plan_id must be a safe non-empty identifier.")
        if not str(plan.operator).strip():
            _add(issues, "invalid_operator", "operator must not be empty.")
        if not _is_aware_utc(plan.created_at):
            _add(issues, "invalid_created_at", "created_at must be timezone-aware UTC.")
        if not _SHA256.fullmatch(str(plan.source_manifest_sha)):
            _add(issues, "invalid_source_manifest_sha", "source_manifest_sha must be a lowercase SHA-256 value.")
        if not str(plan.review_revision).strip():
            _add(issues, "invalid_review_revision", "review_revision must not be empty.")
        if not isinstance(plan.execution_mode, ExecutionMode):
            _add(issues, "unsupported_execution_mode", "execution_mode is not supported.")
        if plan.execution_mode == ExecutionMode.AUTO_DEPLOY_AFTER_GATE and not context.deployment_confirmation_allowed:
            _add(issues, "deployment_confirmation_required", "AUTO_DEPLOY_AFTER_GATE requires an explicit deployment confirmation policy.")
        if plan.attempt < 1:
            _add(issues, "invalid_attempt", "attempt must be at least 1.")
        if plan.attempt > 1 and (not plan.retry_source_plan_id or not plan.retry_source_report_id):
            _add(issues, "retry_reference_missing", "Retry plans must reference their source plan and report.")

    @staticmethod
    def _validate_shape(plan: ProcessingPlan, issues: list[PlanValidationIssue]) -> None:
        if not plan.records:
            _add(issues, "empty_plan", "A processing plan must contain at least one record.")
        if plan.sample_count != len(plan.records):
            _add(issues, "sample_count_mismatch", "sample_count does not match records.")
        sample_ids = [record.sample_id for record in plan.records]
        if any(not str(sample_id).strip() for sample_id in sample_ids):
            _add(issues, "empty_sample_id", "Every record must have a non-empty sample ID.")
        duplicate_ids = sorted(
            item for item, count in Counter(sample_ids).items() if count > 1
        )
        for sample_id in duplicate_ids:
            _add(issues, "duplicate_sample_id", "Sample IDs must be unique within a plan.", sample_id=sample_id)
        source_indices = [record.source_index for record in plan.records]
        for source_index in sorted(
            item for item, count in Counter(source_indices).items() if count > 1
        ):
            _add(issues, "duplicate_source_index", f"source_index {source_index} occurs more than once.")

    @staticmethod
    def _validate_routing(plan: ProcessingPlan, issues: list[PlanValidationIssue]) -> None:
        records = {record.sample_id: record for record in plan.records}
        decisions_by_sample: dict[str, list[Any]] = {}
        for decision in plan.routing_decisions:
            decisions_by_sample.setdefault(decision.sample_id, []).append(decision)
            record = records.get(decision.sample_id)
            if record is None:
                _add(issues, "routing_without_record", "Routing decision does not reference a plan record.", sample_id=decision.sample_id)
                continue
            if decision.source_index != record.source_index:
                _add(issues, "routing_source_index_mismatch", "Routing source_index does not match its record.", sample_id=decision.sample_id)
            if not isinstance(decision.decision, RoutingDecisionType):
                _add(issues, "unknown_primary_routing", "Primary routing is not supported.", sample_id=decision.sample_id)
            if len(set(decision.additional_decisions)) != len(decision.additional_decisions):
                _add(issues, "duplicate_additional_routing", "additional_decisions contains duplicates.", sample_id=decision.sample_id)
            for additional in decision.additional_decisions:
                if (
                    not isinstance(additional, RoutingDecisionType)
                    or additional != RoutingDecisionType.NEEDS_COLOR_CALIBRATION
                    or decision.decision
                    not in {
                        RoutingDecisionType.NEEDS_ANNOTATION,
                        RoutingDecisionType.NEEDS_CLASS_FIX,
                    }
                ):
                    _add(issues, "unknown_additional_routing", "Phase 3A additional routing must be color calibration after annotation/class correction.", sample_id=decision.sample_id)
                if additional == decision.decision:
                    _add(issues, "redundant_additional_routing", "Additional routing duplicates the primary routing.", sample_id=decision.sample_id)
            if decision.decision in _NON_EXECUTABLE_ROUTING and decision.additional_decisions:
                _add(issues, "non_executable_with_additional_routing", "A non-executable sample cannot contain executable additional routing.", sample_id=decision.sample_id)
        for sample_id in records:
            count = len(decisions_by_sample.get(sample_id, ()))
            if count == 0:
                _add(issues, "missing_routing_decision", "Record has no primary routing decision.", sample_id=sample_id)
            elif count > 1:
                _add(issues, "multiple_primary_routing", "Record has more than one primary routing decision.", sample_id=sample_id)

    @staticmethod
    def _validate_blocking_items(plan: ProcessingPlan, issues: list[PlanValidationIssue]) -> None:
        records = {record.sample_id for record in plan.records}
        blocking_samples = {item.sample_id for item in plan.blocking_items}
        for item in plan.blocking_items:
            if item.sample_id not in records:
                _add(issues, "blocking_item_without_record", "Blocking item does not reference a plan record.", sample_id=item.sample_id)
            if not item.violation_code.strip():
                _add(issues, "blocking_item_without_code", "Blocking item must contain a violation code.", sample_id=item.sample_id)
        for decision in plan.routing_decisions:
            should_block = decision.decision in {
                RoutingDecisionType.BLOCKED,
                RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
            }
            if should_block:
                _add(
                    issues,
                    "plan_contains_blocking_sample",
                    "BLOCKED and MANUAL_REVIEW_REQUIRED samples prevent execution.",
                    sample_id=decision.sample_id,
                    retryable=True,
                )
            if should_block and decision.sample_id not in blocking_samples:
                _add(issues, "blocking_decision_without_item", "Blocking routing must have a blocking item.", sample_id=decision.sample_id)
            if not should_block and decision.sample_id in blocking_samples:
                _add(issues, "executable_sample_has_blocking_item", "Executable or excluded routing cannot have a blocking item.", sample_id=decision.sample_id)

    @staticmethod
    def _validate_statistics(plan: ProcessingPlan, issues: list[PlanValidationIssue]) -> None:
        expected = _statistics(plan)
        actual = plan.statistics
        for statistic_name, value in expected.items():
            if getattr(actual, statistic_name) != value:
                _add(
                    issues,
                    "statistics_mismatch",
                    f"statistics.{statistic_name} is {getattr(actual, statistic_name)}, expected {value}.",
                    details={
                        "field": statistic_name,
                        "expected": value,
                        "actual": getattr(actual, statistic_name),
                    },
                )

    @staticmethod
    def _validate_freshness(
        plan: ProcessingPlan,
        context: ProcessingValidationContext,
        issues: list[PlanValidationIssue],
    ) -> None:
        if context.current_manifest_sha != plan.source_manifest_sha:
            _add(issues, "stale_manifest", "The source manifest changed after the plan was created.", retryable=True)
        if not context.current_record_hashes:
            _add(
                issues,
                "record_freshness_not_checked",
                "Current record hashes were not supplied.",
                severity=PlanValidationSeverity.WARNING,
            )
            return
        for record in plan.records:
            current_hash = context.current_record_hashes.get(record.sample_id)
            if current_hash is None:
                _add(issues, "stale_record_missing", "The source record no longer exists.", sample_id=record.sample_id, retryable=True)
            elif current_hash != record_sha256(record.fields):
                _add(issues, "stale_record", "The source record changed after the plan was created.", sample_id=record.sample_id, retryable=True)

    @staticmethod
    def _validate_artifact_paths(
        plan: ProcessingPlan,
        context: ProcessingValidationContext,
        issues: list[PlanValidationIssue],
    ) -> None:
        root = context.artifact_root
        for record in plan.records:
            for key in ("processing_artifact_path", "processing_working_directory"):
                raw = str(record.fields.get(key) or "").strip()
                if not raw:
                    continue
                candidate = Path(raw)
                resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
                if not _is_relative_to(resolved, root):
                    _add(issues, "artifact_path_outside_root", f"{key} points outside the allowed artifact root.", sample_id=record.sample_id)

    @staticmethod
    def _validate_immutability(plan: ProcessingPlan, issues: list[PlanValidationIssue]) -> None:
        tuple_fields = (
            plan.records,
            plan.routing_decisions,
            plan.blocking_items,
            plan.warnings,
            plan.annotation_revision_ids,
        )
        if not all(isinstance(value, tuple) for value in tuple_fields) or any(
            not isinstance(record.fields, MappingProxyType) for record in plan.records
        ):
            _add(issues, "mutable_plan_snapshot", "ProcessingPlan must contain immutable tuple and mapping snapshots.")


def build_validation_context_from_manifest(
    manifest_path: str | Path,
    plan: ProcessingPlan,
    *,
    artifact_root: str | Path,
    deployment_confirmation_allowed: bool = False,
) -> ProcessingValidationContext:
    """Read current CSV state for a freshness check without modifying it."""
    path = Path(manifest_path)
    planned_fields = {record.sample_id: tuple(record.fields) for record in plan.records}
    hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            row = dict(raw_row)
            sample_id = record_identity(row)
            keys = planned_fields.get(sample_id)
            if keys is None:
                continue
            hashes[sample_id] = record_sha256({key: row.get(key, "") for key in keys})
    return ProcessingValidationContext(
        current_manifest_sha=sha256_file(path),
        current_record_hashes=hashes,
        artifact_root=Path(artifact_root),
        deployment_confirmation_allowed=deployment_confirmation_allowed,
    )


def _statistics(plan: ProcessingPlan) -> dict[str, int]:
    counts = dict.fromkeys(RoutingDecisionType, 0)
    for decision in plan.routing_decisions:
        if isinstance(decision.decision, RoutingDecisionType):
            counts[decision.decision] += 1
        for additional in decision.additional_decisions:
            if isinstance(additional, RoutingDecisionType):
                counts[additional] += 1
    manual = counts[RoutingDecisionType.MANUAL_REVIEW_REQUIRED]
    return {
        "ready_count": counts[RoutingDecisionType.READY_FOR_DATASET],
        "annotation_count": counts[RoutingDecisionType.NEEDS_ANNOTATION] + counts[RoutingDecisionType.NEEDS_CLASS_FIX],
        "color_count": counts[RoutingDecisionType.NEEDS_COLOR_CALIBRATION],
        "blocking_count": counts[RoutingDecisionType.BLOCKED] + manual,
        "excluded_count": counts[RoutingDecisionType.EXCLUDED],
        "manual_review_count": manual,
    }


def _add(
    issues: list[PlanValidationIssue],
    code: str,
    message: str,
    *,
    severity: PlanValidationSeverity = PlanValidationSeverity.BLOCKING,
    sample_id: str = "",
    details: Mapping[str, Any] | None = None,
    retryable: bool = False,
) -> None:
    issues.append(
        PlanValidationIssue(
            code=code,
            severity=severity,
            sample_id=sample_id,
            message=message,
            details=details or {},
            retryable=retryable,
        )
    )


def _is_aware_utc(value: datetime) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() == timezone.utc.utcoffset(value)
    )


def _require_utc(value: datetime) -> datetime:
    if not _is_aware_utc(value):
        raise ValueError("Validation clock must return timezone-aware UTC")
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
