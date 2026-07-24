"""Phase 3B execution adapter for the Phase 3C1 dataset service."""

from __future__ import annotations

from tools.dataset_preparation import (
    DatasetPreparationError,
    DatasetPreparationService,
    DatasetPreparationStatus,
)
from tools.processing_events import ProcessingEventLevel, ProcessingEventType
from tools.processing_execution import (
    BatchExecutionContext,
    BatchProcessingStep,
    BatchStepResult,
)
from tools.processing_pipeline import RoutingDecisionType
from tools.processing_reports import ProcessingStepStatus, StepResult


class DatasetPreparationStep(BatchProcessingStep):
    """Prepare all READY records as one fail-closed dataset artifact."""

    step_id = "dataset_preparation"
    supported_routing = frozenset({RoutingDecisionType.READY_FOR_DATASET})

    def __init__(
        self,
        service: DatasetPreparationService,
        *,
        dry_run_override: bool | None = None,
    ) -> None:
        self._service = service
        self._dry_run_override = dry_run_override

    def execute_batch(self, context: BatchExecutionContext) -> BatchStepResult:
        def emit(name, message, metadata):
            context.event_sink.emit(
                ProcessingEventType(name),
                message,
                stage="dataset_preparation",
                step_id=self.step_id,
                metadata=metadata,
            )

        try:
            prepared = self._service.prepare(
                context.plan,
                context.records,
                report_id=context.report_id,
                dry_run=(
                    context.dry_run
                    if self._dry_run_override is None
                    else self._dry_run_override
                ),
                cancellation=context.cancellation_token,
                emit=emit,
            )
        except DatasetPreparationError as exc:
            context.event_sink.emit(
                ProcessingEventType.DATASET_SAMPLE_REJECTED,
                "Dataset sample rejected.",
                stage="dataset_preparation",
                level=ProcessingEventLevel.ERROR,
                sample_id=exc.sample_id,
                step_id=self.step_id,
                metadata={"error_code": exc.code.value, "retryable": exc.retryable},
            )
            raise

        dry_run = prepared.status == DatasetPreparationStatus.DRY_RUN_VALIDATED
        status = (
            ProcessingStepStatus.DEFERRED
            if dry_run
            else ProcessingStepStatus.SUCCESS
        )
        action = "dataset_dry_run_validated" if dry_run else "DATASET_PREPARED"
        message = (
            "Dataset inputs validated; no dataset was created and training was not started."
            if dry_run
            else "Dataset prepared; training was not started."
        )
        artifacts = (str(prepared.preparation_report_path),)
        warnings = list(prepared.warnings)
        if not dry_run and context.cancellation_token.is_cancelled:
            warnings.append(
                "Cancellation arrived after atomic dataset commit; the completed dataset was retained."
            )
        shared = {
            "dataset_id": prepared.dataset_id,
            "dataset_hash": prepared.dataset_hash,
            "dataset_path": str(prepared.artifact_path or ""),
            "accepted_count": prepared.accepted_count,
            "split_counts": dict(prepared.split_counts),
            "dry_run": dry_run,
            "training_started": False,
        }
        return BatchStepResult(
            {
                record.sample_id: StepResult(
                    status=status,
                    action=action,
                    artifacts=artifacts,
                    warnings=tuple(warnings),
                    message=message,
                    metadata={
                        **shared,
                        "split": prepared.sample_splits.get(record.sample_id, "deduplicated"),
                    },
                )
                for record in context.records
            }
        )

    def describe(self) -> str:
        return "Validate and prepare one immutable dataset for all READY samples."
