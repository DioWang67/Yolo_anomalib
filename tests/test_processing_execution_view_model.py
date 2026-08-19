from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from app.gui.processing_batch_dialog import (
    PROCESSING_EXECUTION_FRAMEWORK_ENV,
    PROCESSING_PIPELINE_ENV,
    ProcessingBatchDialog,
    processing_execution_framework_enabled,
    processing_pipeline_enabled,
)
from app.gui.processing_execution_view_model import ProcessingExecutionViewModel
from app.gui.processing_summary_view_model import ProcessingSummaryViewModel
from tools.processing_execution import (
    BlockedStep,
    ExcludedStep,
    NoOpAnnotationStep,
    NoOpColorStep,
    ProcessingExecutionEngine,
    ProcessingExecutionPersistenceError,
    ProcessingStep,
    ProcessingStepRegistry,
)
from tools.processing_pipeline import (
    ProcessingPlanner,
    RoutingDecisionType,
    record_sha256,
)
from tools.processing_plan_validation import (
    ProcessingPlanValidator,
    ProcessingValidationContext,
)
from tools.processing_reports import ProcessingStepStatus, StepResult
from tools.processing_run_store import ProcessingRunStore

NOW = datetime(2026, 7, 21, 1, 2, 3, tzinfo=timezone.utc)


class Ids:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return f"id-{self.value}"


def _record(sample_id="sample-1", **updates):
    record = {
        "sample_id": sample_id,
        "review_selected": "1",
        "review_outcome": "fail",
        "review_label": "confirmed_ng",
        "failure_category": "threshold_not_met",
        "skip_reason": "",
        "product_verdict": "ng",
        "detection_verdict": "correct",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }
    record.update(updates)
    return record


def _plan(*records):
    plan = ProcessingPlanner().create_plan(
        list(enumerate(records)), operator="operator-a", created_at=NOW
    )
    return replace(plan, plan_id="plan-1")


def _context(plan):
    return ProcessingValidationContext(
        current_manifest_sha=plan.source_manifest_sha,
        current_record_hashes={
            record.sample_id: record_sha256(record.fields) for record in plan.records
        },
        artifact_root=".processing_runs/artifacts",
    )


def _view_model(tmp_path, plan, *, registry=None):
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    validator = ProcessingPlanValidator(clock=lambda: NOW)

    def provider(current):
        return _context(current)

    engine = ProcessingExecutionEngine(
        validator=validator,
        context_provider=provider,
        store=store,
        step_registry=registry,
        clock=lambda: NOW,
        id_generator=Ids(),
    )
    return ProcessingExecutionViewModel(
        plan,
        engine=engine,
        validator=validator,
        context_provider=provider,
        store=store,
        language="en",
    )


def test_execution_view_model_exposes_report_summary_events_and_paths(tmp_path):
    view_model = _view_model(tmp_path, _plan(_record()))

    result = view_model.start_processing()

    assert result.accepted is True
    assert result.close_dialog is False
    assert result.report_status == "COMPLETED_WITH_WARNINGS"
    assert result.report_path.endswith(".json")
    assert result.event_path.endswith(".jsonl")
    assert {metric.key: metric.value for metric in result.report_metrics} == {
        "deferred": 1,
        "blocked": 0,
        "excluded": 0,
        "failed": 0,
        "cancelled": 0,
    }
    assert any("PLAN_VALIDATION_STARTED" in line for line in result.event_lines)
    assert any("EXECUTION_COMPLETED" in line for line in result.event_lines)


def test_validation_failed_report_does_not_show_fake_execution(tmp_path):
    plan = _plan(_record(review_selected="0"))
    view_model = _view_model(tmp_path, plan)

    result = view_model.start_processing()

    assert view_model.block_before_start is False
    assert result.accepted is False
    assert result.report_status == "VALIDATION_FAILED"
    assert not any("STEP_STARTED" in line for line in result.event_lines)
    assert result.report_path


def test_deferred_dry_run_does_not_offer_automatic_retry(tmp_path):
    result = _view_model(tmp_path, _plan(_record())).start_processing()

    assert result.can_build_retry is False


class RetryableFailureStep(ProcessingStep):
    step_id = "retryable_failure"
    supported_routing = frozenset({RoutingDecisionType.READY_FOR_DATASET})

    def execute(self, context):
        return StepResult(
            ProcessingStepStatus.FAILED,
            "failed",
            error_code="temporary",
            retryable=True,
        )

    def describe(self):
        return "Controlled retryable failure."


def _failure_registry():
    return ProcessingStepRegistry(
        (
            RetryableFailureStep(),
            NoOpAnnotationStep(),
            NoOpColorStep(),
            BlockedStep(),
            ExcludedStep(),
        )
    )


def test_view_model_builds_but_does_not_execute_retry_plan(tmp_path):
    view_model = _view_model(
        tmp_path,
        _plan(_record()),
        registry=_failure_registry(),
    )

    result = view_model.start_processing()
    retry = view_model.build_retry_plan()

    assert result.can_build_retry is True
    assert retry.created is True
    assert retry.plan_path.endswith(".json")
    assert "not executed" in retry.message


def test_retry_plan_requires_a_previous_run(tmp_path):
    view_model = _view_model(tmp_path, _plan(_record()))

    result = view_model.build_retry_plan()

    assert result.created is False
    assert "Run the dry-run first" in result.message


def test_execution_persistence_error_is_operator_visible(tmp_path):
    plan = _plan(_record())
    store = ProcessingRunStore(tmp_path / ".processing_runs")
    engine = MagicMock()
    engine.execute.side_effect = ProcessingExecutionPersistenceError("write failed")
    view_model = ProcessingExecutionViewModel(
        plan,
        engine=engine,
        validator=ProcessingPlanValidator(clock=lambda: NOW),
        context_provider=lambda current: _context(current),
        store=store,
        language="en",
    )

    result = view_model.start_processing()

    assert result.accepted is False
    assert result.report_status == "FAILED"
    assert "write failed" in result.message


def test_dialog_renders_execution_result_without_closing(qtbot, tmp_path, monkeypatch):
    view_model = _view_model(tmp_path, _plan(_record()))
    dialog = ProcessingBatchDialog(view_model, language="en")
    qtbot.addWidget(dialog)
    information = MagicMock()
    monkeypatch.setattr(
        "app.gui.processing_batch_dialog.QMessageBox.information",
        information,
    )

    qtbot.mouseClick(dialog.start_button, Qt.LeftButton)

    assert dialog.execution_panel.isHidden() is False
    assert "COMPLETED_WITH_WARNINGS" in dialog.report_status_label.text()
    assert "Deferred 1" in dialog.report_summary_label.text()
    assert "PLAN_VALIDATION_STARTED" in dialog.event_log_view.toPlainText()
    assert dialog.result() == 0
    information.assert_called_once()


def test_dialog_enables_retry_only_for_retryable_result(qtbot, tmp_path, monkeypatch):
    view_model = _view_model(
        tmp_path,
        _plan(_record()),
        registry=_failure_registry(),
    )
    dialog = ProcessingBatchDialog(view_model, language="en")
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "app.gui.processing_batch_dialog.QMessageBox.warning",
        MagicMock(),
    )

    qtbot.mouseClick(dialog.start_button, Qt.LeftButton)

    assert dialog.build_retry_button.isEnabled() is True


def test_copy_report_path_uses_clipboard(qtbot, tmp_path, monkeypatch):
    view_model = _view_model(tmp_path, _plan(_record()))
    dialog = ProcessingBatchDialog(view_model, language="en")
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "app.gui.processing_batch_dialog.QMessageBox.information",
        MagicMock(),
    )
    dialog._start_processing()

    dialog._copy_report_path()

    assert dialog.report_path_label.text()
    assert dialog.report_path_label.text() == QApplication.clipboard().text()


def test_execution_framework_flag_is_independent_from_pipeline_flag():
    assert processing_pipeline_enabled({PROCESSING_PIPELINE_ENV: "1"}) is True
    assert processing_execution_framework_enabled({}) is False
    assert processing_execution_framework_enabled(
        {PROCESSING_EXECUTION_FRAMEWORK_ENV: "1"}
    ) is True
    assert processing_execution_framework_enabled(
        {PROCESSING_PIPELINE_ENV: "1"}
    ) is False


def test_review_dialog_wires_phase3b_view_model_only_when_execution_flag_is_on(
    tmp_path,
    monkeypatch,
):
    from app.gui import review_cases_dialog as module

    manifest = tmp_path / "review.csv"
    manifest.write_text("sample_id\nsample-1\n", encoding="utf-8")
    host = MagicMock()
    host.store.rows = [_record()]
    host.manifest_path = manifest
    host.language = "en"
    captured = {}

    class DialogCapture:
        def __init__(self, view_model, **_kwargs):
            captured["view_model"] = view_model

        def exec_(self):
            return 0

    monkeypatch.setattr(module, "ProcessingBatchDialog", DialogCapture)
    monkeypatch.setattr(
        module,
        "processing_execution_framework_enabled",
        lambda: True,
    )

    module.ReviewCasesDialog._open_processing_pipeline(host)

    assert isinstance(captured["view_model"], ProcessingExecutionViewModel)


def test_review_dialog_keeps_phase3a_stub_when_execution_flag_is_off(
    tmp_path,
    monkeypatch,
):
    from app.gui import review_cases_dialog as module

    manifest = tmp_path / "review.csv"
    manifest.write_text("sample_id\nsample-1\n", encoding="utf-8")
    host = MagicMock()
    host.store.rows = [_record()]
    host.manifest_path = manifest
    host.language = "en"
    captured = {}

    class DialogCapture:
        def __init__(self, view_model, **_kwargs):
            captured["view_model"] = view_model

        def exec_(self):
            return 0

    monkeypatch.setattr(module, "ProcessingBatchDialog", DialogCapture)
    monkeypatch.setattr(
        module,
        "processing_execution_framework_enabled",
        lambda: False,
    )

    module.ReviewCasesDialog._open_processing_pipeline(host)

    assert type(captured["view_model"]) is ProcessingSummaryViewModel
