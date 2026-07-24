"""Batch execution adapter that creates a Phase 3C2 annotation package."""

from __future__ import annotations

from tools.annotation_packages import AnnotationPackageService
from tools.processing_events import ProcessingEventType
from tools.processing_execution import BatchExecutionContext, BatchProcessingStep, BatchStepResult
from tools.processing_pipeline import RoutingDecisionType
from tools.processing_reports import ProcessingStepStatus, StepResult


class AnnotationPreparationStep(BatchProcessingStep):
    step_id = "annotation_preparation"
    supported_routing = frozenset({
        RoutingDecisionType.NEEDS_ANNOTATION,
        RoutingDecisionType.NEEDS_CLASS_FIX,
    })

    def __init__(self, service: AnnotationPackageService) -> None:
        self._service = service

    def execute_batch(self, context: BatchExecutionContext) -> BatchStepResult:
        context.event_sink.emit(
            ProcessingEventType.ANNOTATION_PACKAGE_STARTED,
            "Annotation work-package creation started.",
            stage="annotation_package", step_id=self.step_id,
            metadata={"sample_count": len(context.records)},
        )
        package = self._service.create(
            context.plan, context.records, context.decisions,
            report_id=context.report_id, cancellation=context.cancellation_token,
        )
        context.event_sink.emit(
            ProcessingEventType.ANNOTATION_PACKAGE_CREATED,
            "Immutable annotation work package created; operator work is deferred.",
            stage="annotation_package", step_id=self.step_id,
            metadata={"package_id": package.package_id, "sample_count": len(package.items)},
        )
        artifact = str(package.package_path)
        return BatchStepResult({
            record.sample_id: StepResult(
                status=ProcessingStepStatus.DEFERRED,
                action="ANNOTATION_PACKAGE_CREATED",
                artifacts=(artifact,),
                message="Annotation package created; resume after the working labels are reviewed.",
                retryable=True,
                metadata={
                    "package_id": package.package_id,
                    "package_path": artifact,
                    "package_status": package.status.value,
                    "requires_resume": True,
                    "training_started": False,
                },
            )
            for record in context.records
        })

    def describe(self) -> str:
        return "Create one immutable annotation work package for all routed samples."
