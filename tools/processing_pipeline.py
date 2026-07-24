"""Phase 3A batch planning domain built on the Phase 1B review contract.

This module deliberately has no Qt dependency.  It analyzes a complete review
snapshot and produces an immutable plan; execution is a separate interface and
is intentionally unavailable until a later phase.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from tools.review_workflow import (
    AnnotationValidity,
    RequiredAction,
    ReviewSemantics,
    WorkflowState,
    blocking_violations,
    derive_review_semantics,
    derive_workflow_state,
    record_identity,
    validate_record_consistency,
)

PROCESSING_PLAN_SCHEMA_VERSION = 1
PROCESSING_ROUTING_CODE_VERSION = "phase3a-v1"


class ExecutionMode(str, Enum):
    PREPARE_ONLY = "PREPARE_ONLY"
    PREPARE_AND_TRAIN = "PREPARE_AND_TRAIN"
    AUTO_DEPLOY_AFTER_GATE = "AUTO_DEPLOY_AFTER_GATE"


class RoutingDecisionType(str, Enum):
    READY_FOR_DATASET = "READY_FOR_DATASET"
    NEEDS_ANNOTATION = "NEEDS_ANNOTATION"
    NEEDS_CLASS_FIX = "NEEDS_CLASS_FIX"
    NEEDS_COLOR_CALIBRATION = "NEEDS_COLOR_CALIBRATION"
    BLOCKED = "BLOCKED"
    EXCLUDED = "EXCLUDED"
    MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"


@dataclass(frozen=True)
class ProcessingRecord:
    """One immutable input row and its stable manifest position."""

    source_index: int
    sample_id: str
    fields: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _freeze_mapping(self.fields))


@dataclass(frozen=True)
class RoutingDecision:
    sample_id: str
    source_index: int
    decision: RoutingDecisionType
    reason: str
    violation_codes: tuple[str, ...]
    semantics: ReviewSemantics
    additional_decisions: tuple[RoutingDecisionType, ...] = ()


@dataclass(frozen=True)
class BlockingItem:
    sample_id: str
    source_index: int
    reason: str
    violation_code: str


@dataclass(frozen=True)
class ProcessingWarning:
    sample_id: str
    code: str
    message: str


@dataclass(frozen=True)
class ProcessingStatistics:
    ready_count: int
    annotation_count: int
    color_count: int
    blocking_count: int
    excluded_count: int
    manual_review_count: int = 0


@dataclass(frozen=True)
class ProcessingPlan:
    plan_id: str
    created_at: datetime
    operator: str
    execution_mode: ExecutionMode
    source_manifest_sha: str
    review_revision: str
    sample_count: int
    records: tuple[ProcessingRecord, ...]
    routing_decisions: tuple[RoutingDecision, ...]
    blocking_items: tuple[BlockingItem, ...]
    warnings: tuple[ProcessingWarning, ...]
    statistics: ProcessingStatistics
    schema_version: int = PROCESSING_PLAN_SCHEMA_VERSION
    retry_source_plan_id: str = ""
    retry_source_report_id: str = ""
    attempt: int = 1
    routing_code_version: str = PROCESSING_ROUTING_CODE_VERSION
    annotation_source_package_id: str = ""
    annotation_revision_ids: tuple[str, ...] = ()
    color_source_plan_id: str = ""
    color_source_report_id: str = ""
    color_source_package_id: str = ""
    color_revision_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotation_revision_ids", tuple(self.annotation_revision_ids))
        object.__setattr__(self, "color_revision_ids", tuple(self.color_revision_ids))

    @property
    def can_execute(self) -> bool:
        actionable_count = (
            self.statistics.ready_count
            + self.statistics.annotation_count
            + self.statistics.color_count
        )
        return not self.blocking_items and actionable_count > 0


class ProcessingPlanner:
    """Analyze a whole snapshot using only Phase 1B semantics and validation."""

    def create_plan(
        self,
        entries: Sequence[tuple[int, Mapping[str, Any]]],
        *,
        operator: str,
        execution_mode: ExecutionMode = ExecutionMode.PREPARE_AND_TRAIN,
        source_manifest_sha: str = "",
        review_revision: str = "",
        created_at: datetime | None = None,
    ) -> ProcessingPlan:
        records = tuple(
            ProcessingRecord(
                source_index=int(source_index),
                sample_id=record_identity(row),
                fields=row,
            )
            for source_index, row in entries
        )
        effective_sha = source_manifest_sha or _snapshot_sha(entries)
        effective_revision = review_revision or f"sha256:{effective_sha[:12]}"
        decisions: list[RoutingDecision] = []
        blocking_items: list[BlockingItem] = []
        warnings: list[ProcessingWarning] = []

        for record in records:
            decision, record_blockers, record_warnings = self._route_record(record)
            decisions.append(decision)
            blocking_items.extend(record_blockers)
            warnings.extend(record_warnings)

        if execution_mode == ExecutionMode.AUTO_DEPLOY_AFTER_GATE:
            warnings.append(
                ProcessingWarning(
                    sample_id="batch",
                    code="manual_deployment_confirmation_required",
                    message=(
                        "AUTO_DEPLOY_AFTER_GATE is planning metadata only in Phase 3A; "
                        "deployment still requires explicit human confirmation."
                    ),
                )
            )

        statistics = build_processing_statistics(tuple(decisions))
        return ProcessingPlan(
            plan_id=str(uuid4()),
            created_at=created_at or datetime.now(timezone.utc),
            operator=operator.strip() or "unknown",
            execution_mode=execution_mode,
            source_manifest_sha=effective_sha,
            review_revision=effective_revision,
            sample_count=len(records),
            records=records,
            routing_decisions=tuple(decisions),
            blocking_items=tuple(blocking_items),
            warnings=tuple(warnings),
            statistics=statistics,
        )

    def _route_record(
        self,
        record: ProcessingRecord,
    ) -> tuple[RoutingDecision, tuple[BlockingItem, ...], tuple[ProcessingWarning, ...]]:
        fields = record.fields
        semantics = derive_review_semantics(fields)
        state = derive_workflow_state(fields)
        violations = validate_record_consistency(fields)
        blocking = blocking_violations(violations)
        warnings = tuple(
            ProcessingWarning(record.sample_id, issue.code, issue.message)
            for issue in violations
            if not issue.blocking
        )

        if blocking:
            items = tuple(
                BlockingItem(
                    sample_id=record.sample_id,
                    source_index=record.source_index,
                    reason=issue.message,
                    violation_code=issue.code,
                )
                for issue in blocking
            )
            return (
                RoutingDecision(
                    record.sample_id,
                    record.source_index,
                    RoutingDecisionType.BLOCKED,
                    "Phase 1B validation rejected this record.",
                    tuple(issue.code for issue in blocking),
                    semantics,
                ),
                items,
                warnings,
            )

        if state == WorkflowState.ERROR:
            item = BlockingItem(
                record.sample_id,
                record.source_index,
                "The record is in the ERROR workflow state.",
                "workflow_error",
            )
            return (
                RoutingDecision(
                    record.sample_id,
                    record.source_index,
                    RoutingDecisionType.BLOCKED,
                    item.reason,
                    (item.violation_code,),
                    semantics,
                ),
                (item,),
                warnings,
            )

        if semantics.required_action == RequiredAction.EXCLUDE:
            return self._decision(record, semantics, RoutingDecisionType.EXCLUDED, "Excluded by the reviewed evidence."), (), warnings

        if state in {WorkflowState.QUEUED, WorkflowState.PROCESSING, WorkflowState.COMPLETED}:
            return self._decision(record, semantics, RoutingDecisionType.EXCLUDED, f"Already in {state.value} lifecycle state."), (), warnings

        if semantics.required_action == RequiredAction.MANUAL_REVIEW:
            return self._manual_review(record, semantics, "An explicit current review is required.", warnings)

        if state in {WorkflowState.NEW, WorkflowState.SELECTED, WorkflowState.IN_REVIEW}:
            return self._manual_review(record, semantics, f"Workflow state {state.value} is not reviewed.", warnings)

        if not _compatibility_bool(fields.get("training_selected")):
            return self._decision(record, semantics, RoutingDecisionType.EXCLUDED, "Not selected for this processing plan."), (), warnings

        if semantics.required_action == RequiredAction.ANNOTATION:
            return self._decision(record, semantics, RoutingDecisionType.NEEDS_ANNOTATION, "Annotation must be created or corrected."), (), warnings

        if semantics.required_action == RequiredAction.CLASS_FIX:
            return self._decision(record, semantics, RoutingDecisionType.NEEDS_CLASS_FIX, "Annotation class must be corrected."), (), warnings

        if semantics.required_action == RequiredAction.COLOR_CALIBRATION:
            return self._decision(record, semantics, RoutingDecisionType.NEEDS_COLOR_CALIBRATION, "Color calibration is required."), (), warnings

        if semantics.annotation_validity == AnnotationValidity.VERIFIED:
            return self._decision(record, semantics, RoutingDecisionType.READY_FOR_DATASET, "Reviewed annotation is ready for dataset preparation."), (), warnings

        return self._manual_review(record, semantics, "Review semantics do not establish a usable annotation.", warnings)

    @staticmethod
    def _decision(
        record: ProcessingRecord,
        semantics: ReviewSemantics,
        decision: RoutingDecisionType,
        reason: str,
    ) -> RoutingDecision:
        additional_decisions = (
            (RoutingDecisionType.NEEDS_COLOR_CALIBRATION,)
            if semantics.legacy_action_route == "both"
            and decision
            in {
                RoutingDecisionType.NEEDS_ANNOTATION,
                RoutingDecisionType.NEEDS_CLASS_FIX,
            }
            else ()
        )
        return RoutingDecision(
            sample_id=record.sample_id,
            source_index=record.source_index,
            decision=decision,
            reason=reason,
            violation_codes=(),
            semantics=semantics,
            additional_decisions=additional_decisions,
        )

    def _manual_review(
        self,
        record: ProcessingRecord,
        semantics: ReviewSemantics,
        reason: str,
        warnings: tuple[ProcessingWarning, ...],
    ) -> tuple[RoutingDecision, tuple[BlockingItem, ...], tuple[ProcessingWarning, ...]]:
        decision = self._decision(
            record,
            semantics,
            RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
            reason,
        )
        item = BlockingItem(
            sample_id=record.sample_id,
            source_index=record.source_index,
            reason=reason,
            violation_code="manual_review_required",
        )
        return decision, (item,), warnings


@dataclass(frozen=True)
class ExecutionResult:
    plan_id: str
    accepted: bool
    message: str


class ProcessingExecutionUnavailableError(RuntimeError):
    """Raised when Phase 3A is asked to perform later-phase execution work."""


class ExecutionEngine(ABC):
    @abstractmethod
    def execute(self, plan: ProcessingPlan) -> ExecutionResult:
        """Execute one complete immutable plan."""


class Phase3AExecutionEngine(ExecutionEngine):
    """Safe placeholder: Phase 3A plans but never starts downstream work."""

    def execute(self, plan: ProcessingPlan) -> ExecutionResult:
        raise ProcessingExecutionUnavailableError(
            f"Processing plan {plan.plan_id} is ready, but execution is not implemented in Phase 3A."
        )


def sha256_file(path: str | Path) -> str:
    """Hash a manifest without placing filesystem code in a QWidget."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: Mapping[str, Any]) -> str:
    """Return the canonical SHA for one immutable record snapshot."""
    encoded = json.dumps(
        _json_value(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_processing_statistics(
    decisions: tuple[RoutingDecision, ...],
) -> ProcessingStatistics:
    """Derive batch statistics from existing Phase 3A routing decisions."""
    counts = dict.fromkeys(RoutingDecisionType, 0)
    for item in decisions:
        counts[item.decision] += 1
        for additional_decision in item.additional_decisions:
            counts[additional_decision] += 1
    manual = counts[RoutingDecisionType.MANUAL_REVIEW_REQUIRED]
    return ProcessingStatistics(
        ready_count=counts[RoutingDecisionType.READY_FOR_DATASET],
        annotation_count=(
            counts[RoutingDecisionType.NEEDS_ANNOTATION]
            + counts[RoutingDecisionType.NEEDS_CLASS_FIX]
        ),
        color_count=counts[RoutingDecisionType.NEEDS_COLOR_CALIBRATION],
        blocking_count=counts[RoutingDecisionType.BLOCKED] + manual,
        excluded_count=counts[RoutingDecisionType.EXCLUDED],
        manual_review_count=manual,
    )


def _snapshot_sha(entries: Sequence[tuple[int, Mapping[str, Any]]]) -> str:
    payload = [
        {"source_index": int(index), "record": _json_value(dict(row))}
        for index, row in entries
    ]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _compatibility_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
