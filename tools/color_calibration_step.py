"""Execution-framework adapter for Phase 3C3 color package preparation."""

from __future__ import annotations

import threading

from tools.color_calibration_packages import ColorCalibrationPackage, ColorCalibrationPackageService
from tools.color_calibration_service import ColorCalibrationError, ColorProposalStatus
from tools.processing_events import ProcessingEventLevel, ProcessingEventType
from tools.processing_execution import ExecutionContext, ProcessingStep
from tools.processing_pipeline import RoutingDecisionType
from tools.processing_reports import ProcessingStepStatus, StepResult


class ColorCalibrationPreparationStep(ProcessingStep):
    step_id = "color_calibration_preparation"
    supported_routing = frozenset({RoutingDecisionType.NEEDS_COLOR_CALIBRATION})

    def __init__(self, service: ColorCalibrationPackageService) -> None:
        self.service = service
        self._lock = threading.Lock()
        self._packages: dict[str, ColorCalibrationPackage] = {}

    def validate(self, context: ExecutionContext) -> tuple[str, ...]:
        return ()

    def execute(self, context: ExecutionContext) -> StepResult:
        context.cancellation_token.raise_if_cancelled()
        context.event_sink.emit(
            ProcessingEventType.COLOR_PACKAGE_STARTED,
            "Color calibration package preparation started.",
            stage="color_calibration",
            sample_id=context.record.sample_id,
            step_id=self.step_id,
        )
        try:
            with self._lock:
                package = self._packages.get(context.report_id)
                if package is None:
                    package = self.service.create(
                        context.plan, context.plan.records, context.plan.routing_decisions,
                        report_id=context.report_id,
                        cancellation=context.cancellation_token,
                    )
                    self._packages[context.report_id] = package
            context.event_sink.emit(
                ProcessingEventType.COLOR_PACKAGE_CREATED,
                "Color calibration proposal, preview, and gate artifacts were created.",
                stage="color_calibration",
                sample_id=context.record.sample_id,
                step_id=self.step_id,
                metadata={"package_id": package.package_id, "scope_count": len(package.scopes), "artifact_relative_path": str(package.package_path.relative_to(context.artifact_root))},
            )
            for proposal, gate in zip(package.proposals, package.gates, strict=True):
                proposal_event = (
                    ProcessingEventType.COLOR_PROPOSAL_INSUFFICIENT_DATA
                    if proposal.status == ColorProposalStatus.INSUFFICIENT_DATA
                    else ProcessingEventType.COLOR_PROPOSAL_CREATED
                )
                context.event_sink.emit(
                    proposal_event,
                    "Color threshold proposal created.",
                    stage="color_calibration",
                    sample_id=context.record.sample_id,
                    step_id=self.step_id,
                    metadata={
                        "package_id": package.package_id,
                        "scope": proposal.scope.key,
                        "proposal_status": proposal.status.value,
                        "proposal_sha256": proposal.proposal_sha256,
                    },
                )
                context.event_sink.emit(
                    ProcessingEventType.COLOR_PREVIEW_CREATED,
                    "Color before/after preview created.",
                    stage="color_calibration",
                    sample_id=context.record.sample_id,
                    step_id=self.step_id,
                    metadata={
                        "package_id": package.package_id,
                        "scope": proposal.scope.key,
                    },
                )
                event_type = (
                    ProcessingEventType.COLOR_GATE_PASSED
                    if gate.passed
                    else ProcessingEventType.COLOR_GATE_FAILED
                )
                context.event_sink.emit(
                    event_type,
                    "Color calibration gate evaluated.",
                    stage="color_calibration",
                    sample_id=context.record.sample_id,
                    step_id=self.step_id,
                    level=ProcessingEventLevel.INFO if gate.passed else ProcessingEventLevel.WARNING,
                    metadata={"package_id": package.package_id, "scope": proposal.scope.key, "proposal_sha256": proposal.proposal_sha256, "blocking_issues": list(gate.blocking_issues)},
                )
            return StepResult(
                status=ProcessingStepStatus.DEFERRED,
                action="COLOR_CALIBRATION_PROPOSED",
                artifacts=(str(package.package_path), str(package.root / "validation_report.json")),
                warnings=tuple(
                    f"{proposal.scope.key}: {', '.join(gate.blocking_issues)}"
                    for proposal, gate in zip(
                        package.proposals, package.gates, strict=True
                    )
                    if not gate.passed
                ),
                message="Color calibration package is waiting for explicit per-scope approval.",
                metadata={
                    "requires_approval": True,
                    "package_id": package.package_id,
                    "package_path": str(package.package_path),
                    "scope_count": len(package.scopes),
                    "gate_passed_count": sum(gate.passed for gate in package.gates),
                    "pending_approval_count": len(package.scopes),
                },
            )
        except ColorCalibrationError as exc:
            context.event_sink.emit(
                ProcessingEventType.COLOR_PACKAGE_FAILED,
                "Color calibration package preparation failed.",
                stage="color_calibration",
                sample_id=context.record.sample_id,
                step_id=self.step_id,
                level=ProcessingEventLevel.ERROR,
                metadata={"error_code": exc.code},
            )
            return StepResult(
                status=ProcessingStepStatus.FAILED,
                action="COLOR_PACKAGE_FAILED",
                error_code=exc.code,
                message=str(exc),
                retryable=exc.retryable,
            )

    def describe(self) -> str:
        return "Prepare one immutable color-calibration proposal package for the full plan."
