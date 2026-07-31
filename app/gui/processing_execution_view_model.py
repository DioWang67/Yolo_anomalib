"""Presentation adapter for the Phase 3B dry-run execution framework."""

from __future__ import annotations

from dataclasses import dataclass

from app.gui.annotation_package_view_model import AnnotationPackageViewModel
from app.gui.color_calibration_view_model import ColorCalibrationViewModel
from app.gui.dataset_preparation_view_model import DatasetPreparationViewModel
from app.gui.processing_summary_view_model import (
    ProcessingSummaryViewModel,
    StartProcessingViewModel,
    SummaryMetricViewModel,
)
from tools.processing_execution import (
    NoRetryableSamplesError,
    ProcessingExecutionEngine,
    ProcessingExecutionPersistenceError,
    RetryPlanError,
    build_retry_plan,
)
from tools.processing_pipeline import ProcessingPlan
from tools.processing_plan_validation import (
    ProcessingPlanValidator,
    ProcessingValidationContext,
)
from tools.processing_reports import (
    ProcessingReportStatus,
    SampleProcessingStatus,
)
from tools.processing_run_store import ProcessingPersistenceError, ProcessingRunStore


@dataclass(frozen=True)
class RetryPlanViewModel:
    created: bool
    title: str
    message: str
    plan_path: str = ""


class ProcessingExecutionViewModel(ProcessingSummaryViewModel):
    """Run Phase 3B and expose only operator-facing text and counts."""

    block_before_start = False

    def __init__(
        self,
        plan: ProcessingPlan,
        *,
        engine: ProcessingExecutionEngine,
        validator: ProcessingPlanValidator,
        context_provider,
        store: ProcessingRunStore,
        language: str = "zh_TW",
        annotation_launcher=None,
        annotation_resume=None,
        color_decider=None,
        color_resume=None,
        color_rollback=None,
        color_history=None,
    ) -> None:
        super().__init__(plan, language=language)
        self._execution_engine = engine
        self._validator = validator
        self._context_provider = context_provider
        self._store = store
        self._outcome = None
        self.dataset_preparation = DatasetPreparationViewModel()
        self.annotation_package = AnnotationPackageViewModel()
        self.color_calibration = ColorCalibrationViewModel()
        self._annotation_launcher = annotation_launcher
        self._annotation_resume = annotation_resume
        self._color_decider = color_decider
        self._color_resume = color_resume
        self._color_rollback = color_rollback
        self._color_history = color_history
        self.execution_details = (
            f"Mode: {'Dry-run' if getattr(engine, '_dry_run', True) else 'Actual'} | "
            f"Artifact root: {store.artifacts_dir}"
        )
        self.requires_dataset_commit_confirmation = not getattr(engine, "_dry_run", True)

    def start_processing(self) -> StartProcessingViewModel:
        try:
            outcome = self._execution_engine.execute(self._plan)
        except ProcessingExecutionPersistenceError as exc:
            partial = "\n".join(str(path) for path in exc.partial_artifacts)
            suffix = f"\n\nPartial artifacts:\n{partial}" if partial else ""
            return StartProcessingViewModel(
                accepted=False,
                close_dialog=False,
                title=self._text("處理報告保存失敗", "Processing report persistence failed"),
                message=f"{exc}{suffix}",
                report_status=(exc.report.status.value if exc.report else "FAILED"),
            )
        self._outcome = outcome
        report = outcome.report
        self.dataset_preparation = DatasetPreparationViewModel.from_report(
            report, self._store.root
        )
        self.annotation_package = AnnotationPackageViewModel.from_report(
            report,
            launcher=self._annotation_launcher,
            resume=self._annotation_resume,
        )
        self.color_calibration = ColorCalibrationViewModel.from_report(
            report,
            decider=self._color_decider,
            resume=self._color_resume,
            rollback=self._color_rollback,
            history=self._color_history,
        )
        failed_statuses = {
            ProcessingReportStatus.VALIDATION_FAILED,
            ProcessingReportStatus.FAILED,
            ProcessingReportStatus.INTERRUPTED,
            ProcessingReportStatus.CANCELLED,
        }
        accepted = report.status not in failed_statuses
        message = self._text(
            "Dry-run 已完成；所有實際工作仍為 deferred。",
            "Dry-run completed; all downstream work remains deferred.",
        )
        if self.dataset_preparation.visible:
            message = self.dataset_preparation.message
        if self.annotation_package.visible:
            message = self._text(
                "補標工作包已建立；完成 working labels 後按『繼續補標結果』。",
                "Annotation package created. Edit working labels, then resume the package.",
            )
        if self.color_calibration.visible:
            message = self._text(
                "Color calibration proposal 已建立，需逐 scope 核准後才會啟用。",
                "Color calibration proposal created; each scope requires explicit approval before activation.",
            )
        if report.status == ProcessingReportStatus.VALIDATION_FAILED:
            message = self._text(
                "計畫驗證失敗，未執行任何 processing step；報告已保存。",
                "Plan validation failed. No processing step ran; the report was persisted.",
            )
        elif report.status == ProcessingReportStatus.CANCELLED:
            message = self._text(
                "Dry-run 已在安全邊界取消；報告已保存。",
                "Dry-run was cancelled at a safe boundary; the report was persisted.",
            )
        return StartProcessingViewModel(
            accepted=accepted,
            close_dialog=False,
            title=self._text("Processing Report", "Processing Report"),
            message=message,
            report_path=str(outcome.report_document.path),
            event_path=str(outcome.event_path),
            report_status=report.status.value,
            report_metrics=self._report_metrics(report.summary),
            event_lines=tuple(
                f"{event.sequence:04d}  {event.event_type.value}  {event.message}"
                for event in outcome.events
            ),
            can_build_retry=self._has_retryable_result(),
        )

    def build_retry_plan(self) -> RetryPlanViewModel:
        if self._outcome is None:
            return RetryPlanViewModel(
                False,
                self._text("無法建立 Retry Plan", "Retry plan unavailable"),
                self._text("請先執行 dry-run。", "Run the dry-run first."),
            )
        try:
            context: ProcessingValidationContext = self._context_provider(self._plan)
            retry_plan = build_retry_plan(
                self._outcome.report,
                self._plan,
                context,
                validator=self._validator,
            )
            document = self._store.persist_plan(retry_plan)
        except (NoRetryableSamplesError, RetryPlanError) as exc:
            return RetryPlanViewModel(
                False,
                self._text("無法建立 Retry Plan", "Retry plan unavailable"),
                str(exc),
            )
        except ProcessingPersistenceError as exc:
            return RetryPlanViewModel(
                False,
                self._text("Retry Plan 保存失敗", "Retry plan persistence failed"),
                str(exc),
            )
        return RetryPlanViewModel(
            True,
            self._text("Retry Plan 已建立", "Retry plan created"),
            self._text(
                "Retry Plan 僅保存，尚未執行。",
                "The retry plan was persisted but not executed.",
            ),
            str(document.path),
        )

    def _has_retryable_result(self) -> bool:
        if self._outcome is None:
            return False
        return any(
            (
                result.status == SampleProcessingStatus.FAILED
                and result.retryable
            )
            or (
                result.status == SampleProcessingStatus.CANCELLED
                and bool(result.metadata.get("not_started"))
            )
            for result in self._outcome.report.sample_results
        )

    def _report_metrics(self, summary) -> tuple[SummaryMetricViewModel, ...]:
        return (
            SummaryMetricViewModel("deferred", self._text("Deferred", "Deferred"), summary.deferred),
            SummaryMetricViewModel("blocked", self._text("Blocked", "Blocked"), summary.blocked),
            SummaryMetricViewModel("excluded", self._text("Excluded", "Excluded"), summary.excluded),
            SummaryMetricViewModel("failed", self._text("Failed", "Failed"), summary.failed),
            SummaryMetricViewModel("cancelled", self._text("Cancelled", "Cancelled"), summary.cancelled),
        )
