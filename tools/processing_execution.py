"""Deterministic Phase 3B dry-run execution framework."""

from __future__ import annotations

import hashlib
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from tools.processing_events import (
    CompositeEventSink,
    InMemoryEventSink,
    ProcessingEvent,
    ProcessingEventLevel,
    ProcessingEventPublisher,
    ProcessingEventSink,
    ProcessingEventType,
    exception_metadata,
    redact_sensitive_text,
)
from tools.processing_pipeline import (
    ProcessingPlan,
    ProcessingRecord,
    RoutingDecision,
    RoutingDecisionType,
    build_processing_statistics,
)
from tools.processing_plan_validation import (
    PlanValidationIssue,
    PlanValidationResult,
    PlanValidationSeverity,
    ProcessingPlanValidator,
    ProcessingValidationContext,
)
from tools.processing_reports import (
    ArtifactReference,
    CancellationSummary,
    ProcessingReport,
    ProcessingReportStatus,
    ProcessingStepStatus,
    SampleProcessingResult,
    SampleProcessingStatus,
    StepResult,
    build_report_summary,
)
from tools.processing_run_store import (
    ProcessingPersistenceError,
    ProcessingRunStore,
    StoredDocument,
)

logger = logging.getLogger(__name__)


def _merge_sample_results(
    first: SampleProcessingResult,
    second: SampleProcessingResult,
) -> SampleProcessingResult:
    """Combine sequential primary/additional route results without losing lineage."""
    priority = {
        SampleProcessingStatus.FAILED: 6,
        SampleProcessingStatus.CANCELLED: 5,
        SampleProcessingStatus.BLOCKED: 4,
        SampleProcessingStatus.DEFERRED: 3,
        SampleProcessingStatus.SUCCESS: 2,
        SampleProcessingStatus.SKIPPED: 1,
    }
    winner = first if priority[first.status] >= priority[second.status] else second
    metadata = dict(first.metadata)
    metadata.update(dict(second.metadata))
    metadata["chained_step_ids"] = [first.step_id, second.step_id]
    return SampleProcessingResult(
        sample_id=first.sample_id,
        primary_routing=first.primary_routing,
        additional_routing=first.additional_routing,
        status=winner.status,
        action=";".join(value for value in (first.action, second.action) if value),
        step_id=";".join(value for value in (first.step_id, second.step_id) if value),
        started_at=min(first.started_at, second.started_at),
        finished_at=max(first.finished_at, second.finished_at),
        artifacts=(*first.artifacts, *second.artifacts),
        warnings=(*first.warnings, *second.warnings),
        error_code=winner.error_code,
        message=winner.message,
        retryable=winner.retryable,
        attempt=max(first.attempt, second.attempt),
        metadata=metadata,
    )


class ProcessingCancelledError(RuntimeError):
    """Raised cooperatively at a safe sample or step boundary."""


class FatalProcessingStepError(RuntimeError):
    """Explicitly stop the run after preserving a failed sample result."""


class ProcessingExecutionPersistenceError(RuntimeError):
    """A run could not preserve its final auditable report."""

    def __init__(
        self,
        message: str,
        *,
        report: ProcessingReport | None = None,
        partial_artifacts: Sequence[Path] = (),
    ) -> None:
        self.report = report
        self.partial_artifacts = tuple(partial_artifacts)
        super().__init__(message)


class RetryPlanError(ValueError):
    """Base class for retry-plan construction failures."""


class NoRetryableSamplesError(RetryPlanError):
    pass


class RetryPlanValidationError(RetryPlanError):
    def __init__(self, validation_result: PlanValidationResult) -> None:
        self.validation_result = validation_result
        codes = ", ".join(item.code for item in validation_result.blocking_issues)
        super().__init__(f"Retry plan validation failed: {codes}")


class CancellationToken:
    """Thread-safe cooperative cancellation without terminating worker threads."""

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._event = threading.Event()
        self._clock = clock or _utc_now
        self._lock = threading.Lock()
        self._reason = ""
        self._requested_at: datetime | None = None

    def request_cancel(self, reason: str = "operator_requested") -> bool:
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = str(reason).strip() or "operator_requested"
            self._requested_at = _require_utc(self._clock())
            self._event.set()
            return True

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason

    @property
    def requested_at(self) -> datetime | None:
        with self._lock:
            return self._requested_at

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise ProcessingCancelledError(self.reason or "cancelled")


@dataclass(frozen=True)
class ExecutionContext:
    plan: ProcessingPlan
    report_id: str
    record: ProcessingRecord
    decision: RoutingDecision
    event_sink: ProcessingEventPublisher
    cancellation_token: CancellationToken
    artifact_root: Path
    working_directory: Path
    dry_run: bool
    shared_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_root", Path(self.artifact_root).resolve())
        object.__setattr__(
            self,
            "working_directory",
            Path(self.working_directory).resolve(),
        )
        object.__setattr__(self, "shared_state", MappingProxyType(dict(self.shared_state)))


@dataclass(frozen=True)
class BatchExecutionContext:
    plan: ProcessingPlan
    report_id: str
    records: tuple[ProcessingRecord, ...]
    decisions: tuple[RoutingDecision, ...]
    event_sink: ProcessingEventPublisher
    cancellation_token: CancellationToken
    artifact_root: Path
    working_directory: Path
    dry_run: bool


@dataclass(frozen=True)
class BatchStepResult:
    sample_results: Mapping[str, StepResult]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_results",
            MappingProxyType(dict(self.sample_results)),
        )


class BatchProcessingStep(ABC):
    """Execute one batch artifact and map it back to per-sample reports."""

    step_id: str
    supported_routing: frozenset[RoutingDecisionType]

    @abstractmethod
    def execute_batch(self, context: BatchExecutionContext) -> BatchStepResult:
        """Return one result for every record in the batch."""

    @abstractmethod
    def describe(self) -> str:
        """Describe the batch operation without UI terminology."""


class ProcessingStep(ABC):
    step_id: str
    supported_routing: frozenset[RoutingDecisionType]

    def validate(self, context: ExecutionContext) -> tuple[str, ...]:
        if not context.dry_run:
            return ("Phase 3B steps are dry-run only.",)
        return ()

    @abstractmethod
    def execute(self, context: ExecutionContext) -> StepResult:
        """Return a result without changing review, model, or dataset state."""

    def rollback(self, context: ExecutionContext) -> None:
        """No rollback is needed because Phase 3B steps cannot mutate artifacts."""
        _ = context
        return None

    def is_retryable(self, error: BaseException) -> bool:
        return not isinstance(error, FatalProcessingStepError)

    @abstractmethod
    def describe(self) -> str:
        """Describe the deferred action in operator-neutral language."""


class NoOpReadyStep(ProcessingStep):
    step_id = "noop_ready"
    supported_routing = frozenset({RoutingDecisionType.READY_FOR_DATASET})

    def execute(self, context: ExecutionContext) -> StepResult:
        context.cancellation_token.raise_if_cancelled()
        return StepResult(
            status=ProcessingStepStatus.DEFERRED,
            action="would_prepare_dataset",
            message="Dataset preparation is deferred; Phase 3B does not create a dataset.",
        )

    def describe(self) -> str:
        return "Validate a dataset-ready sample without creating dataset artifacts."


class NoOpAnnotationStep(ProcessingStep):
    step_id = "noop_annotation"
    supported_routing = frozenset(
        {
            RoutingDecisionType.NEEDS_ANNOTATION,
            RoutingDecisionType.NEEDS_CLASS_FIX,
        }
    )

    def execute(self, context: ExecutionContext) -> StepResult:
        context.cancellation_token.raise_if_cancelled()
        return StepResult(
            status=ProcessingStepStatus.DEFERRED,
            action="would_request_annotation",
            message="Annotation work is deferred; Phase 3B does not modify labels.",
        )

    def describe(self) -> str:
        return "Validate an annotation request without opening or modifying labels."


class NoOpColorStep(ProcessingStep):
    step_id = "noop_color"
    supported_routing = frozenset(
        {RoutingDecisionType.NEEDS_COLOR_CALIBRATION}
    )

    def execute(self, context: ExecutionContext) -> StepResult:
        context.cancellation_token.raise_if_cancelled()
        return StepResult(
            status=ProcessingStepStatus.DEFERRED,
            action="would_request_color_calibration",
            message="Color calibration is deferred; Phase 3B does not change thresholds.",
        )

    def describe(self) -> str:
        return "Validate a color-calibration request without changing configuration."


class BlockedStep(ProcessingStep):
    step_id = "blocked"
    supported_routing = frozenset(
        {
            RoutingDecisionType.BLOCKED,
            RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
        }
    )

    def execute(self, context: ExecutionContext) -> StepResult:
        return StepResult(
            status=ProcessingStepStatus.BLOCKED,
            action="not_executable",
            message="The sample is blocked and no execution step was run.",
        )

    def describe(self) -> str:
        return "Represent a non-executable blocked sample."


class ExcludedStep(ProcessingStep):
    step_id = "excluded"
    supported_routing = frozenset({RoutingDecisionType.EXCLUDED})

    def execute(self, context: ExecutionContext) -> StepResult:
        return StepResult(
            status=ProcessingStepStatus.SKIPPED,
            action="excluded",
            message="The reviewed sample is excluded from processing.",
        )

    def describe(self) -> str:
        return "Represent a reviewed sample excluded from processing."


class ProcessingStepRegistry:
    def __init__(
        self,
        steps: Sequence[ProcessingStep],
        *,
        batch_steps: Sequence[BatchProcessingStep] = (),
    ) -> None:
        routes: dict[RoutingDecisionType, ProcessingStep] = {}
        for step in steps:
            if not step.step_id.strip():
                raise ValueError("Processing step_id must not be empty")
            for route in step.supported_routing:
                if route in routes:
                    raise ValueError(f"Duplicate processing step route: {route.value}")
                routes[route] = step
        batch_routes: dict[RoutingDecisionType, BatchProcessingStep] = {}
        for step in batch_steps:
            if not step.step_id.strip():
                raise ValueError("Batch processing step_id must not be empty")
            for route in step.supported_routing:
                if route in routes or route in batch_routes:
                    raise ValueError(f"Duplicate processing step route: {route.value}")
                batch_routes[route] = step
        self._routes = MappingProxyType(routes)
        self._batch_routes = MappingProxyType(batch_routes)

    def step_for(self, routing: RoutingDecisionType) -> ProcessingStep:
        try:
            return self._routes[routing]
        except KeyError as exc:
            raise LookupError(f"No processing step registered for {routing.value}") from exc

    def batch_step_for(self, routing: RoutingDecisionType) -> BatchProcessingStep | None:
        return self._batch_routes.get(routing)

    @classmethod
    def default(cls) -> ProcessingStepRegistry:
        return cls(
            (
                NoOpReadyStep(),
                NoOpAnnotationStep(),
                NoOpColorStep(),
                BlockedStep(),
                ExcludedStep(),
            )
        )


@dataclass(frozen=True)
class ProcessingExecutionOutcome:
    report: ProcessingReport
    plan_document: StoredDocument
    report_document: StoredDocument
    event_path: Path
    events: tuple[ProcessingEvent, ...]


class ProcessingExecutionEngine:
    """Validate, dry-run deterministic no-op steps, and persist an audit report."""

    def __init__(
        self,
        *,
        validator: ProcessingPlanValidator,
        context_provider: Callable[[ProcessingPlan], ProcessingValidationContext],
        store: ProcessingRunStore,
        step_registry: ProcessingStepRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        event_sink_factory: Callable[
            [InMemoryEventSink, ProcessingEventSink], ProcessingEventSink
        ]
        | None = None,
        dry_run: bool = True,
    ) -> None:
        self._validator = validator
        self._context_provider = context_provider
        self._store = store
        self._step_registry = step_registry or ProcessingStepRegistry.default()
        self._clock = clock or _utc_now
        self._id_generator = id_generator or (lambda: str(uuid4()))
        self._event_sink_factory = event_sink_factory or (
            lambda memory, persistent: CompositeEventSink((memory, persistent))
        )
        self._dry_run = bool(dry_run)

    def execute(
        self,
        plan: ProcessingPlan,
        *,
        cancellation_token: CancellationToken | None = None,
    ) -> ProcessingExecutionOutcome:
        started_at = _require_utc(self._clock())
        report_id = self._id_generator()
        token = cancellation_token or CancellationToken(clock=self._clock)
        partial_paths: list[Path] = []
        try:
            plan_document = self._store.persist_plan(plan)
            partial_paths.append(plan_document.path)
            event_path, persistent_sink = self._store.prepare_event_log(report_id)
            partial_paths.append(event_path)
        except ProcessingPersistenceError as exc:
            raise ProcessingExecutionPersistenceError(
                f"Could not initialize processing run: {exc}",
                partial_artifacts=partial_paths,
            ) from exc

        memory_sink = InMemoryEventSink()
        sink = self._event_sink_factory(memory_sink, persistent_sink)
        publisher = ProcessingEventPublisher(
            plan_id=plan.plan_id,
            report_id=report_id,
            sink=sink,
            clock=self._clock,
            id_generator=self._id_generator,
        )
        publisher.emit(
            ProcessingEventType.PLAN_VALIDATION_STARTED,
            "Processing plan validation started.",
            stage="validation",
        )
        publisher.emit(
            ProcessingEventType.ARTIFACT_CREATED,
            "Immutable processing plan snapshot persisted.",
            stage="persistence",
            metadata={"path": plan_document.relative_path, "sha256": plan_document.sha256},
        )
        validation = self._validate(plan, publisher)
        publisher.emit(
            ProcessingEventType.PLAN_VALIDATION_COMPLETED,
            "Processing plan validation completed.",
            stage="validation",
            level=(
                ProcessingEventLevel.INFO
                if validation.valid
                else ProcessingEventLevel.ERROR
            ),
            metadata={
                "valid": validation.valid,
                "blocking_issue_count": len(validation.blocking_issues),
                "warning_count": len(validation.warnings),
            },
        )

        if not validation.valid:
            results = self._validation_failure_results(plan)
            publisher.emit(
                ProcessingEventType.EXECUTION_COMPLETED,
                "Execution was not started because plan validation failed.",
                stage="execution",
                level=ProcessingEventLevel.ERROR,
                metadata={"status": ProcessingReportStatus.VALIDATION_FAILED.value},
            )
            report = self._finalize_report(
                plan=plan,
                report_id=report_id,
                started_at=started_at,
                status=ProcessingReportStatus.VALIDATION_FAILED,
                validation=validation,
                results=results,
                publisher=publisher,
                plan_document=plan_document,
                event_path=event_path,
                errors=tuple(
                    f"[{item.code}] {item.message}"
                    for item in validation.blocking_issues
                ),
            )
            return self._persist_outcome(
                report,
                plan_document,
                event_path,
                memory_sink,
                partial_paths,
            )

        publisher.emit(
            ProcessingEventType.EXECUTION_STARTED,
            "Deterministic processing execution started.",
            stage="execution",
            metadata={"dry_run": self._dry_run, "sample_count": plan.sample_count},
        )
        results, fatal_error = self._execute_samples(plan, report_id, publisher, token)
        cancellation = self._cancellation_summary(token)
        status = _derive_report_status(results, publisher.warnings, token, fatal_error)
        publisher.emit(
            (
                ProcessingEventType.EXECUTION_CANCELLED
                if token.is_cancelled
                else ProcessingEventType.EXECUTION_COMPLETED
            ),
            (
                "Processing execution cancelled cooperatively."
                if token.is_cancelled
                else "Processing execution completed."
            ),
            stage="execution",
            level=(
                ProcessingEventLevel.ERROR
                if status
                in {
                    ProcessingReportStatus.FAILED,
                    ProcessingReportStatus.INTERRUPTED,
                }
                else ProcessingEventLevel.INFO
            ),
            metadata={"status": status.value},
        )
        errors = (fatal_error,) if fatal_error else ()
        report = self._finalize_report(
            plan=plan,
            report_id=report_id,
            started_at=started_at,
            status=status,
            validation=validation,
            results=results,
            publisher=publisher,
            plan_document=plan_document,
            event_path=event_path,
            errors=errors,
            cancellation=cancellation,
        )
        return self._persist_outcome(
            report,
            plan_document,
            event_path,
            memory_sink,
            partial_paths,
        )

    def _validate(
        self,
        plan: ProcessingPlan,
        publisher: ProcessingEventPublisher,
    ) -> PlanValidationResult:
        try:
            context = self._context_provider(plan)
            return self._validator.validate(plan, context)
        except (OSError, ValueError) as exc:
            publisher.emit(
                ProcessingEventType.ERROR,
                "Could not read the current validation context.",
                stage="validation",
                level=ProcessingEventLevel.ERROR,
                metadata=exception_metadata(exc),
            )
            return _context_failure_validation(plan, self._clock(), exc)
        except Exception as exc:  # boundary: context adapters cannot escape the audit report
            publisher.emit(
                ProcessingEventType.ERROR,
                "Unexpected validation context failure.",
                stage="validation",
                level=ProcessingEventLevel.ERROR,
                metadata=exception_metadata(exc),
            )
            logger.exception(
                "Unexpected processing validation context failure plan_id=%s",
                plan.plan_id,
            )
            return _context_failure_validation(plan, self._clock(), exc)

    def _execute_samples(
        self,
        plan: ProcessingPlan,
        report_id: str,
        publisher: ProcessingEventPublisher,
        token: CancellationToken,
    ) -> tuple[tuple[SampleProcessingResult, ...], str]:
        results: list[SampleProcessingResult] = []
        fatal_error = ""
        cancel_event_emitted = False
        records = {record.sample_id: record for record in plan.records}
        decisions = sorted(
            plan.routing_decisions,
            key=lambda item: (item.source_index, item.sample_id),
        )
        batch_results, batch_fatal = self._execute_batch_steps(
            plan, report_id, publisher, token, records, decisions
        )
        if batch_fatal:
            fatal_error = batch_fatal
        for decision in decisions:
            record = records[decision.sample_id]
            if decision.sample_id in batch_results:
                batch_result = batch_results[decision.sample_id]
                remaining_routings = tuple(
                    routing
                    for routing in decision.additional_decisions
                    if self._step_registry.batch_step_for(routing) is None
                )
                if remaining_routings and batch_result.status not in {
                    SampleProcessingStatus.FAILED,
                    SampleProcessingStatus.CANCELLED,
                    SampleProcessingStatus.BLOCKED,
                }:
                    additional_result, sample_fatal = self._execute_sample(
                        plan,
                        report_id,
                        record,
                        decision,
                        publisher,
                        token,
                        routings_override=remaining_routings,
                    )
                    batch_result = _merge_sample_results(batch_result, additional_result)
                    if sample_fatal:
                        fatal_error = additional_result.message or additional_result.error_code
                results.append(batch_result)
                continue
            if fatal_error:
                result = self._not_started_cancelled(
                    plan,
                    decision,
                    "fatal_previous_sample",
                )
                results.append(result)
                self._emit_not_started_sample(publisher, result)
                continue
            if token.is_cancelled:
                if not cancel_event_emitted:
                    publisher.emit(
                        ProcessingEventType.EXECUTION_CANCEL_REQUESTED,
                        "Cooperative cancellation requested.",
                        stage="execution",
                        metadata={"reason": token.reason},
                    )
                    cancel_event_emitted = True
                if decision.decision == RoutingDecisionType.EXCLUDED:
                    results.append(self._excluded_result(plan, report_id, record, decision, publisher, token))
                else:
                    result = self._not_started_cancelled(plan, decision, token.reason)
                    results.append(result)
                    self._emit_not_started_sample(publisher, result)
                continue
            if decision.decision == RoutingDecisionType.EXCLUDED:
                results.append(self._excluded_result(plan, report_id, record, decision, publisher, token))
                continue
            sample_result, sample_fatal = self._execute_sample(
                plan,
                report_id,
                record,
                decision,
                publisher,
                token,
            )
            results.append(sample_result)
            if token.is_cancelled and not cancel_event_emitted:
                publisher.emit(
                    ProcessingEventType.EXECUTION_CANCEL_REQUESTED,
                    "Cooperative cancellation requested.",
                    stage="execution",
                    metadata={"reason": token.reason},
                )
                cancel_event_emitted = True
            if sample_fatal:
                fatal_error = sample_result.message or sample_result.error_code
        return tuple(results), fatal_error

    def _execute_batch_steps(
        self,
        plan: ProcessingPlan,
        report_id: str,
        publisher: ProcessingEventPublisher,
        token: CancellationToken,
        records_by_id: Mapping[str, ProcessingRecord],
        decisions: Sequence[RoutingDecision],
    ) -> tuple[dict[str, SampleProcessingResult], str]:
        grouped: dict[BatchProcessingStep, list[RoutingDecision]] = {}
        for decision in decisions:
            step = self._step_registry.batch_step_for(decision.decision)
            if step is not None:
                grouped.setdefault(step, []).append(decision)
        results: dict[str, SampleProcessingResult] = {}
        for step, grouped_decisions in grouped.items():
            started = _require_utc(self._clock())
            grouped_records = tuple(records_by_id[item.sample_id] for item in grouped_decisions)
            context = BatchExecutionContext(
                plan=plan,
                report_id=report_id,
                records=grouped_records,
                decisions=tuple(grouped_decisions),
                event_sink=publisher,
                cancellation_token=token,
                artifact_root=self._store.artifacts_dir,
                working_directory=self._store.artifacts_dir / report_id,
                dry_run=self._dry_run,
            )
            try:
                token.raise_if_cancelled()
                publisher.emit(
                    ProcessingEventType.STEP_STARTED,
                    step.describe(),
                    stage="batch_step",
                    step_id=step.step_id,
                    metadata={"sample_count": len(grouped_records)},
                )
                batch = step.execute_batch(context)
                expected = {record.sample_id for record in grouped_records}
                if set(batch.sample_results) != expected:
                    raise FatalProcessingStepError(
                        "Batch step did not return exactly one result per READY sample."
                    )
            except ProcessingCancelledError as exc:
                for decision in grouped_decisions:
                    results[decision.sample_id] = self._not_started_cancelled(plan, decision, str(exc))
                return results, ""
            except Exception as exc:
                error_code = (
                    str(exc.code.value)
                    if getattr(exc, "code", None) is not None
                    else "unexpected_batch_step_error"
                )
                retryable = bool(getattr(exc, "retryable", False))
                message = redact_sensitive_text(str(exc))
                publisher.emit(
                    ProcessingEventType.STEP_FAILED,
                    "Batch processing step failed closed.",
                    stage="batch_step",
                    level=ProcessingEventLevel.ERROR,
                    step_id=step.step_id,
                    metadata={"error_code": error_code, "retryable": retryable},
                )
                finished = _require_utc(self._clock())
                for decision in grouped_decisions:
                    results[decision.sample_id] = SampleProcessingResult(
                        sample_id=decision.sample_id,
                        primary_routing=decision.decision,
                        additional_routing=decision.additional_decisions,
                        status=SampleProcessingStatus.FAILED,
                        action=f"{step.step_id}_failed",
                        step_id=step.step_id,
                        started_at=started,
                        finished_at=finished,
                        error_code=error_code,
                        message=message,
                        retryable=retryable,
                        attempt=plan.attempt,
                        metadata={"not_started": False, "batch_failed": True},
                    )
                continue
            finished = _require_utc(self._clock())
            for decision in grouped_decisions:
                step_result = batch.sample_results[decision.sample_id]
                artifacts: list[ArtifactReference] = []
                for path in step_result.artifacts:
                    artifacts.append(self._store.describe_file(path, kind=f"step:{step.step_id}"))
                status = {
                    ProcessingStepStatus.SUCCESS: SampleProcessingStatus.SUCCESS,
                    ProcessingStepStatus.FAILED: SampleProcessingStatus.FAILED,
                    ProcessingStepStatus.CANCELLED: SampleProcessingStatus.CANCELLED,
                    ProcessingStepStatus.SKIPPED: SampleProcessingStatus.SKIPPED,
                    ProcessingStepStatus.BLOCKED: SampleProcessingStatus.BLOCKED,
                }.get(step_result.status, SampleProcessingStatus.DEFERRED)
                results[decision.sample_id] = SampleProcessingResult(
                    sample_id=decision.sample_id,
                    primary_routing=decision.decision,
                    additional_routing=decision.additional_decisions,
                    status=status,
                    action=step_result.action,
                    step_id=step.step_id,
                    started_at=started,
                    finished_at=finished,
                    artifacts=tuple(artifacts),
                    warnings=step_result.warnings,
                    error_code=step_result.error_code,
                    message=step_result.message,
                    retryable=step_result.retryable,
                    attempt=plan.attempt,
                    metadata=step_result.metadata,
                )
            publisher.emit(
                ProcessingEventType.STEP_COMPLETED,
                "Batch processing step completed.",
                stage="batch_step",
                step_id=step.step_id,
                metadata={"sample_count": len(grouped_records)},
            )
        return results, ""

    def _execute_sample(
        self,
        plan: ProcessingPlan,
        report_id: str,
        record: ProcessingRecord,
        decision: RoutingDecision,
        publisher: ProcessingEventPublisher,
        token: CancellationToken,
        routings_override: Sequence[RoutingDecisionType] | None = None,
    ) -> tuple[SampleProcessingResult, bool]:
        started = _require_utc(self._clock())
        publisher.emit(
            ProcessingEventType.SAMPLE_STARTED,
            "Sample dry-run started.",
            stage="sample",
            sample_id=record.sample_id,
        )
        actions: list[str] = []
        step_ids: list[str] = []
        warnings: list[str] = []
        artifacts: list[ArtifactReference] = []
        routings = tuple(routings_override) if routings_override is not None else (
            decision.decision,
            *decision.additional_decisions,
        )
        fatal = False
        error_code = ""
        message = ""
        retryable = False
        status = SampleProcessingStatus.DEFERRED
        result_metadata: dict[str, Any] = {"not_started": False}
        for routing in routings:
            try:
                step = self._step_registry.step_for(routing)
            except LookupError as exc:
                status = SampleProcessingStatus.FAILED
                error_code = "step_not_registered"
                message = str(exc)
                publisher.emit(
                    ProcessingEventType.STEP_FAILED,
                    "No processing step is registered for this routing.",
                    stage="step",
                    level=ProcessingEventLevel.ERROR,
                    sample_id=record.sample_id,
                    metadata=exception_metadata(exc),
                )
                break
            step_ids.append(step.step_id)
            context = ExecutionContext(
                plan=plan,
                report_id=report_id,
                record=record,
                decision=decision,
                event_sink=publisher,
                cancellation_token=token,
                artifact_root=self._store.artifacts_dir,
                working_directory=self._store.artifacts_dir
                / report_id
                / f"sample-{record.source_index}",
                dry_run=self._dry_run,
            )
            try:
                token.raise_if_cancelled()
                warnings.extend(step.validate(context))
                publisher.emit(
                    ProcessingEventType.STEP_STARTED,
                    step.describe(),
                    stage="step",
                    sample_id=record.sample_id,
                    step_id=step.step_id,
                    metadata={"routing": routing.value},
                )
                step_result = step.execute(context)
                result_metadata.update(dict(step_result.metadata))
                token.raise_if_cancelled()
                actions.append(step_result.action)
                warnings.extend(step_result.warnings)
                message = step_result.message
                for artifact_path in step_result.artifacts:
                    artifact = self._store.describe_file(
                        artifact_path,
                        kind=f"step:{step.step_id}",
                    )
                    artifacts.append(artifact)
                    publisher.emit(
                        ProcessingEventType.ARTIFACT_CREATED,
                        "Processing step artifact recorded.",
                        stage="persistence",
                        sample_id=record.sample_id,
                        step_id=step.step_id,
                        metadata={
                            "path": artifact.relative_path,
                            "sha256": artifact.sha256,
                        },
                    )
                if step_result.status == ProcessingStepStatus.FAILED:
                    status = SampleProcessingStatus.FAILED
                    error_code = step_result.error_code
                    retryable = step_result.retryable
                    publisher.emit(
                        ProcessingEventType.STEP_FAILED,
                        step_result.message or "Processing step failed.",
                        stage="step",
                        level=ProcessingEventLevel.ERROR,
                        sample_id=record.sample_id,
                        step_id=step.step_id,
                        metadata={"error_code": error_code, "retryable": retryable},
                    )
                    break
                if step_result.status == ProcessingStepStatus.SUCCESS:
                    status = SampleProcessingStatus.SUCCESS
                publisher.emit(
                    ProcessingEventType.STEP_COMPLETED,
                    step_result.message or "Processing step deferred.",
                    stage="step",
                    sample_id=record.sample_id,
                    step_id=step.step_id,
                    metadata={"status": step_result.status.value},
                )
            except ProcessingCancelledError as exc:
                status = SampleProcessingStatus.CANCELLED
                message = str(exc)
                error_code = "cancelled"
                break
            except FatalProcessingStepError as exc:
                status = SampleProcessingStatus.FAILED
                fatal = True
                error_code = "fatal_step_error"
                message = redact_sensitive_text(str(exc))
                publisher.emit(
                    ProcessingEventType.STEP_FAILED,
                    "Fatal processing step error.",
                    stage="step",
                    level=ProcessingEventLevel.ERROR,
                    sample_id=record.sample_id,
                    step_id=step.step_id,
                    metadata=exception_metadata(exc),
                )
                break
            except ProcessingPersistenceError as exc:
                status = SampleProcessingStatus.FAILED
                error_code = "invalid_step_artifact"
                message = redact_sensitive_text(str(exc))
                retryable = False
                publisher.emit(
                    ProcessingEventType.STEP_FAILED,
                    "Processing step artifact could not be recorded safely.",
                    stage="persistence",
                    level=ProcessingEventLevel.ERROR,
                    sample_id=record.sample_id,
                    step_id=step.step_id,
                    metadata=exception_metadata(exc),
                )
                break
            except Exception as exc:  # execution boundary converts one bad step into a result
                status = SampleProcessingStatus.FAILED
                error_code = "unexpected_step_error"
                message = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
                retryable = step.is_retryable(exc)
                publisher.emit(
                    ProcessingEventType.STEP_FAILED,
                    "Unexpected processing step error.",
                    stage="step",
                    level=ProcessingEventLevel.ERROR,
                    sample_id=record.sample_id,
                    step_id=step.step_id,
                    metadata={**exception_metadata(exc), "retryable": retryable},
                )
                break

        finished = _require_utc(self._clock())
        event_type = {
            SampleProcessingStatus.FAILED: ProcessingEventType.SAMPLE_FAILED,
            SampleProcessingStatus.CANCELLED: ProcessingEventType.SAMPLE_SKIPPED,
        }.get(status, ProcessingEventType.SAMPLE_COMPLETED)
        publisher.emit(
            event_type,
            message or "Sample dry-run completed with deferred work.",
            stage="sample",
            level=(
                ProcessingEventLevel.ERROR
                if status == SampleProcessingStatus.FAILED
                else ProcessingEventLevel.INFO
            ),
            sample_id=record.sample_id,
            metadata={"status": status.value},
        )
        return (
            SampleProcessingResult(
                sample_id=record.sample_id,
                primary_routing=decision.decision,
                additional_routing=decision.additional_decisions,
                status=status,
                action=" + ".join(actions) or "not_executed",
                step_id=" + ".join(step_ids),
                started_at=started,
                finished_at=finished,
                artifacts=tuple(artifacts),
                warnings=tuple(warnings),
                error_code=error_code,
                message=message,
                retryable=retryable,
                attempt=plan.attempt,
                metadata=result_metadata,
            ),
            fatal,
        )

    def _excluded_result(
        self,
        plan: ProcessingPlan,
        report_id: str,
        record: ProcessingRecord,
        decision: RoutingDecision,
        publisher: ProcessingEventPublisher,
        token: CancellationToken,
    ) -> SampleProcessingResult:
        timestamp = _require_utc(self._clock())
        step = self._step_registry.step_for(RoutingDecisionType.EXCLUDED)
        context = ExecutionContext(
            plan=plan,
            report_id=report_id,
            record=record,
            decision=decision,
            event_sink=publisher,
            cancellation_token=token,
            artifact_root=self._store.artifacts_dir,
            working_directory=self._store.artifacts_dir / report_id,
            dry_run=self._dry_run,
        )
        step_result = step.execute(context)
        publisher.emit(
            ProcessingEventType.SAMPLE_SKIPPED,
            step_result.message,
            stage="sample",
            sample_id=record.sample_id,
            step_id=step.step_id,
            metadata={"status": SampleProcessingStatus.SKIPPED.value},
        )
        return SampleProcessingResult(
            sample_id=record.sample_id,
            primary_routing=decision.decision,
            additional_routing=(),
            status=SampleProcessingStatus.SKIPPED,
            action=step_result.action,
            step_id=step.step_id,
            started_at=timestamp,
            finished_at=timestamp,
            message=step_result.message,
            attempt=plan.attempt,
            metadata={"not_started": True},
        )

    def _not_started_cancelled(
        self,
        plan: ProcessingPlan,
        decision: RoutingDecision,
        reason: str,
    ) -> SampleProcessingResult:
        timestamp = _require_utc(self._clock())
        return SampleProcessingResult(
            sample_id=decision.sample_id,
            primary_routing=decision.decision,
            additional_routing=decision.additional_decisions,
            status=SampleProcessingStatus.CANCELLED,
            action="not_started",
            step_id="",
            started_at=timestamp,
            finished_at=timestamp,
            error_code="cancelled",
            message=reason or "cancelled",
            retryable=True,
            attempt=plan.attempt,
            metadata={"not_started": True},
        )

    @staticmethod
    def _emit_not_started_sample(
        publisher: ProcessingEventPublisher,
        result: SampleProcessingResult,
    ) -> None:
        publisher.emit(
            ProcessingEventType.SAMPLE_SKIPPED,
            "Sample was not started.",
            stage="sample",
            sample_id=result.sample_id,
            metadata={
                "status": result.status.value,
                "reason": result.message,
            },
        )

    def _validation_failure_results(
        self,
        plan: ProcessingPlan,
    ) -> tuple[SampleProcessingResult, ...]:
        results: list[SampleProcessingResult] = []
        for decision in sorted(
            plan.routing_decisions,
            key=lambda item: (item.source_index, item.sample_id),
        ):
            timestamp = _require_utc(self._clock())
            excluded = decision.decision == RoutingDecisionType.EXCLUDED
            results.append(
                SampleProcessingResult(
                    sample_id=decision.sample_id,
                    primary_routing=decision.decision,
                    additional_routing=decision.additional_decisions,
                    status=(
                        SampleProcessingStatus.SKIPPED
                        if excluded
                        else SampleProcessingStatus.BLOCKED
                    ),
                    action="excluded" if excluded else "validation_failed",
                    step_id="",
                    started_at=timestamp,
                    finished_at=timestamp,
                    message=(
                        "Excluded by the processing plan."
                        if excluded
                        else "No step ran because plan validation failed."
                    ),
                    attempt=plan.attempt,
                    metadata={"not_started": True},
                )
            )
        return tuple(results)

    def _cancellation_summary(
        self,
        token: CancellationToken,
    ) -> CancellationSummary | None:
        if not token.is_cancelled:
            return None
        return CancellationSummary(
            requested=True,
            reason=token.reason,
            requested_at=token.requested_at,
            completed_at=_require_utc(self._clock()),
        )

    def _finalize_report(
        self,
        *,
        plan: ProcessingPlan,
        report_id: str,
        started_at: datetime,
        status: ProcessingReportStatus,
        validation: PlanValidationResult,
        results: tuple[SampleProcessingResult, ...],
        publisher: ProcessingEventPublisher,
        plan_document: StoredDocument,
        event_path: Path,
        errors: tuple[str, ...] = (),
        cancellation: CancellationSummary | None = None,
    ) -> ProcessingReport:
        try:
            events_reference = self._store.describe_file(event_path, kind="events")
        except ProcessingPersistenceError as exc:
            raise ProcessingExecutionPersistenceError(
                f"Could not hash the processing event log: {exc}",
                partial_artifacts=(plan_document.path, event_path),
            ) from exc
        warnings = tuple(
            [f"[{warning.code}] {warning.message}" for warning in plan.warnings]
            + [f"[{issue.code}] {issue.message}" for issue in validation.warnings]
            + list(publisher.warnings)
        )
        return ProcessingReport(
            report_id=report_id,
            plan_id=plan.plan_id,
            source_manifest_sha=plan.source_manifest_sha,
            review_revision=plan.review_revision,
            execution_mode=plan.execution_mode,
            started_at=started_at,
            finished_at=_require_utc(self._clock()),
            status=status,
            operator=plan.operator,
            dry_run=self._dry_run,
            validation_result=validation,
            summary=build_report_summary(results, warnings),
            sample_results=results,
            events_reference=events_reference,
            artifacts=_unique_artifacts(
                (
                    plan_document.as_artifact("plan"),
                    *(
                        artifact
                        for result in results
                        for artifact in result.artifacts
                    ),
                )
            ),
            warnings=warnings,
            errors=errors,
            cancellation=cancellation,
            retry_source_report_id=plan.retry_source_report_id,
        )

    def _persist_outcome(
        self,
        report: ProcessingReport,
        plan_document: StoredDocument,
        event_path: Path,
        memory_sink: InMemoryEventSink,
        partial_paths: list[Path],
    ) -> ProcessingExecutionOutcome:
        try:
            report_document = self._store.persist_report(report)
            partial_paths.append(report_document.path)
        except ProcessingPersistenceError as exc:
            failed_report = replace(
                report,
                status=ProcessingReportStatus.FAILED,
                errors=(*report.errors, f"report_persistence_failed: {exc}"),
            )
            raise ProcessingExecutionPersistenceError(
                f"Final processing report could not be persisted: {exc}",
                report=failed_report,
                partial_artifacts=partial_paths,
            ) from exc
        try:
            self._store.update_latest(report, report_document)
        except ProcessingPersistenceError as exc:
            logger.warning(
                "Processing report persisted but latest index update failed report_id=%s error=%s",
                report.report_id,
                exc,
            )
        return ProcessingExecutionOutcome(
            report=report,
            plan_document=plan_document,
            report_document=report_document,
            event_path=event_path,
            events=memory_sink.events,
        )


def build_retry_plan(
    report: ProcessingReport,
    original_plan: ProcessingPlan,
    current_context: ProcessingValidationContext,
    *,
    validator: ProcessingPlanValidator,
    clock: Callable[[], datetime] | None = None,
    include_deferred: bool = False,
) -> ProcessingPlan:
    """Build an idempotent subset retry without rerunning Phase 3A routing."""
    if report.plan_id != original_plan.plan_id:
        raise RetryPlanError("Report does not belong to the supplied original plan")
    eligible_ids: set[str] = set()
    for result in report.sample_results:
        if result.status == SampleProcessingStatus.FAILED and result.retryable:
            eligible_ids.add(result.sample_id)
        elif (
            result.status == SampleProcessingStatus.CANCELLED
            and bool(result.metadata.get("not_started"))
        ):
            eligible_ids.add(result.sample_id)
        elif include_deferred and result.status == SampleProcessingStatus.DEFERRED:
            eligible_ids.add(result.sample_id)
    if not eligible_ids:
        raise NoRetryableSamplesError("The report contains no retryable samples")
    decisions = tuple(
        decision
        for decision in original_plan.routing_decisions
        if decision.sample_id in eligible_ids
        and decision.decision
        not in {
            RoutingDecisionType.BLOCKED,
            RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
            RoutingDecisionType.EXCLUDED,
        }
    )
    selected_ids = {decision.sample_id for decision in decisions}
    records = tuple(
        record for record in original_plan.records if record.sample_id in selected_ids
    )
    if not records:
        raise NoRetryableSamplesError(
            "Retryable results do not map to executable original-plan records"
        )
    next_attempt = original_plan.attempt + 1
    retry_id = _retry_plan_id(
        original_plan.plan_id,
        report.report_id,
        next_attempt,
        tuple(sorted(selected_ids)),
    )
    retry_plan = replace(
        original_plan,
        plan_id=retry_id,
        created_at=_require_utc((clock or _utc_now)()),
        sample_count=len(records),
        records=records,
        routing_decisions=decisions,
        blocking_items=(),
        statistics=build_processing_statistics(decisions),
        retry_source_plan_id=original_plan.plan_id,
        retry_source_report_id=report.report_id,
        attempt=next_attempt,
    )
    validation = validator.validate(retry_plan, current_context)
    if not validation.valid:
        raise RetryPlanValidationError(validation)
    return retry_plan


def _derive_report_status(
    results: tuple[SampleProcessingResult, ...],
    publisher_warnings: tuple[str, ...],
    token: CancellationToken,
    fatal_error: str,
) -> ProcessingReportStatus:
    if fatal_error:
        return ProcessingReportStatus.INTERRUPTED
    if token.is_cancelled and any(
        result.status == SampleProcessingStatus.CANCELLED for result in results
    ):
        return ProcessingReportStatus.CANCELLED
    failed = sum(result.status == SampleProcessingStatus.FAILED for result in results)
    executable = sum(
        result.primary_routing
        not in {
            RoutingDecisionType.BLOCKED,
            RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
            RoutingDecisionType.EXCLUDED,
        }
        for result in results
    )
    if failed:
        return (
            ProcessingReportStatus.FAILED
            if failed >= executable
            else ProcessingReportStatus.PARTIAL_FAILURE
        )
    if publisher_warnings or any(
        result.status == SampleProcessingStatus.DEFERRED or result.warnings
        for result in results
    ):
        return ProcessingReportStatus.COMPLETED_WITH_WARNINGS
    return ProcessingReportStatus.COMPLETED


def _context_failure_validation(
    plan: ProcessingPlan,
    checked_at: datetime,
    error: BaseException,
) -> PlanValidationResult:
    issue = PlanValidationIssue(
        code="validation_context_error",
        severity=PlanValidationSeverity.BLOCKING,
        message=f"Could not load current validation context: {type(error).__name__}: {error}",
        retryable=True,
    )
    return PlanValidationResult(
        valid=False,
        blocking_issues=(issue,),
        warnings=(),
        checked_at=_require_utc(checked_at),
        current_manifest_sha="",
        expected_manifest_sha=plan.source_manifest_sha,
        executable_sample_ids=(),
        skipped_sample_ids=tuple(record.sample_id for record in plan.records),
        issues=(issue,),
    )


def _retry_plan_id(
    plan_id: str,
    report_id: str,
    attempt: int,
    sample_ids: tuple[str, ...],
) -> str:
    encoded = "\0".join((plan_id, report_id, str(attempt), *sample_ids)).encode(
        "utf-8"
    )
    return f"retry-{hashlib.sha256(encoded).hexdigest()[:24]}-a{attempt}"


def _unique_artifacts(
    artifacts: Sequence[ArtifactReference],
) -> tuple[ArtifactReference, ...]:
    indexed: dict[tuple[str, str], ArtifactReference] = {}
    for artifact in artifacts:
        indexed.setdefault((artifact.relative_path, artifact.kind), artifact)
    return tuple(indexed.values())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_utc(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() != timezone.utc.utcoffset(value)
    ):
        raise ValueError("Execution clock must return timezone-aware UTC")
    return value
