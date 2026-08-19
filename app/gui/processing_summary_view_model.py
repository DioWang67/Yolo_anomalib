"""Presentation-only model for the Phase 3A processing summary."""

from __future__ import annotations

from dataclasses import dataclass

from tools.processing_pipeline import (
    ExecutionEngine,
    ExecutionResult,
    Phase3AExecutionEngine,
    ProcessingExecutionUnavailableError,
    ProcessingPlan,
    RoutingDecisionType,
)


@dataclass(frozen=True)
class SummaryMetricViewModel:
    key: str
    label: str
    value: int


@dataclass(frozen=True)
class BlockingItemViewModel:
    sample_id: str
    reason: str
    violation_code: str


@dataclass(frozen=True)
class StartProcessingViewModel:
    accepted: bool
    title: str
    message: str
    close_dialog: bool = True
    report_path: str = ""
    event_path: str = ""
    report_status: str = ""
    report_metrics: tuple[SummaryMetricViewModel, ...] = ()
    event_lines: tuple[str, ...] = ()
    can_build_retry: bool = False


class ProcessingSummaryViewModel:
    """Convert a domain plan into stable UI data and one start command."""

    def __init__(
        self,
        plan: ProcessingPlan,
        *,
        language: str = "zh_TW",
        engine: ExecutionEngine | None = None,
    ) -> None:
        self._plan = plan
        self._engine = engine or Phase3AExecutionEngine()
        self._is_zh = str(language).lower().startswith("zh")
        statistics = plan.statistics
        self.plan_id = plan.plan_id
        self.sample_count = plan.sample_count
        self.execution_mode = plan.execution_mode.value
        self.annotation_fix_count = sum(
            decision.decision == RoutingDecisionType.NEEDS_ANNOTATION
            for decision in plan.routing_decisions
        )
        self.class_fix_count = sum(
            decision.decision == RoutingDecisionType.NEEDS_CLASS_FIX
            for decision in plan.routing_decisions
        )
        self.metrics = (
            SummaryMetricViewModel("ready", self._text("可準備", "Ready"), statistics.ready_count),
            SummaryMetricViewModel(
                "annotation",
                self._text("需要標註", "Need Annotation"),
                statistics.annotation_count,
            ),
            SummaryMetricViewModel(
                "color",
                self._text("需要顏色校正", "Need Color Calibration"),
                statistics.color_count,
            ),
            SummaryMetricViewModel(
                "blocked",
                self._text("阻擋", "Blocked"),
                statistics.blocking_count,
            ),
            SummaryMetricViewModel(
                "excluded",
                self._text("排除", "Excluded"),
                statistics.excluded_count,
            ),
        )
        self.blocking_items = tuple(
            BlockingItemViewModel(item.sample_id, item.reason, item.violation_code)
            for item in plan.blocking_items
        )
        self.warnings = tuple(
            f"[{warning.code}] {warning.message}"
            + (f" ({warning.sample_id})" if warning.sample_id != "batch" else "")
            for warning in plan.warnings
        )

    @property
    def has_blocking_items(self) -> bool:
        return bool(self.blocking_items)

    def start_processing(self) -> StartProcessingViewModel:
        """Pass the complete plan to the engine without exposing it to Qt widgets."""
        if self.has_blocking_items:
            return StartProcessingViewModel(
                accepted=False,
                title=self._text("無法開始處理", "Processing blocked"),
                message=self._text(
                    "請先處理 Blocking Summary 中的項目。",
                    "Resolve the items in Blocking Summary before starting.",
                ),
            )
        try:
            result = self._engine.execute(self._plan)
        except ProcessingExecutionUnavailableError as exc:
            return StartProcessingViewModel(
                accepted=False,
                title=self._text("處理計畫已建立", "Processing plan created"),
                message=str(exc),
            )
        return self._result_view_model(result)

    def _result_view_model(self, result: ExecutionResult) -> StartProcessingViewModel:
        return StartProcessingViewModel(
            accepted=result.accepted,
            title=self._text(
                "處理已接受" if result.accepted else "無法開始處理",
                "Processing accepted" if result.accepted else "Processing not started",
            ),
            message=result.message,
        )

    def _text(self, zh: str, en: str) -> str:
        return zh if self._is_zh else en
