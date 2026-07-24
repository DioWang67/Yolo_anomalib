from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea, QTableWidget

from app.gui.processing_batch_dialog import (
    PROCESSING_PIPELINE_ENV,
    ProcessingBatchDialog,
    processing_pipeline_enabled,
)
from app.gui.processing_summary_view_model import ProcessingSummaryViewModel
from tools.processing_pipeline import (
    ExecutionEngine,
    ExecutionResult,
    ProcessingPlanner,
)


def _ready(sample_id: str = "ready") -> dict[str, str]:
    return {
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


def _blocked() -> dict[str, str]:
    return {**_ready("blocked-sample"), "review_selected": "0"}


def _plan(*records: dict[str, str]):
    return ProcessingPlanner().create_plan(
        list(enumerate(records)),
        operator="operator-a",
    )


class RecordingEngine(ExecutionEngine):
    def __init__(self) -> None:
        self.plan = None

    def execute(self, plan):
        self.plan = plan
        return ExecutionResult(plan.plan_id, True, "accepted")


def test_view_model_displays_summary_and_passes_the_complete_plan_to_engine():
    plan = _plan(_ready("one"), _ready("two"))
    engine = RecordingEngine()
    view_model = ProcessingSummaryViewModel(
        plan,
        language="en",
        engine=engine,
    )

    result = view_model.start_processing()

    assert [(metric.label, metric.value) for metric in view_model.metrics] == [
        ("Ready", 2),
        ("Need Annotation", 0),
        ("Need Color Calibration", 0),
        ("Blocked", 0),
        ("Excluded", 0),
    ]
    assert engine.plan is plan
    assert result.accepted is True


def test_view_model_exposes_blocking_summary_without_exposing_domain_decisions():
    view_model = ProcessingSummaryViewModel(_plan(_blocked()), language="en")

    assert view_model.has_blocking_items is True
    assert view_model.blocking_items[0].sample_id == "blocked-sample"
    assert view_model.blocking_items[0].violation_code == "review_without_selection"
    assert view_model.start_processing().accepted is False


def test_processing_dialog_renders_view_model_and_advanced_actions(qtbot):
    dialog = ProcessingBatchDialog(
        ProcessingSummaryViewModel(_plan(_ready()), language="en"),
        language="en",
    )
    qtbot.addWidget(dialog)

    assert dialog.metric_value_labels["ready"].text() == "1"
    assert dialog.metric_value_labels["blocked"].text() == "0"
    assert dialog.advanced_panel.isHidden() is True

    qtbot.mouseClick(dialog.advanced_button, Qt.LeftButton)

    assert dialog.advanced_panel.isHidden() is False
    assert dialog.offline_button.text() == "Export Offline Package"
    assert dialog.history_button.text() == "History"
    assert dialog.diagnostics_button.text() == "Diagnostics"

    body_scroll = dialog.findChild(QScrollArea, "ProcessingBodyScroll")
    assert body_scroll is not None
    assert body_scroll.widgetResizable() is True
    assert body_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_processing_dialog_shows_annotation_package_actions_only_when_available(qtbot):
    view_model = ProcessingSummaryViewModel(_plan(_ready()), language="en")
    view_model.annotation_package = SimpleNamespace(
        visible=True,
        package_id="package-1",
        package_root="C:/annotation/package-1",
        status="WAITING_FOR_OPERATOR",
        requires_resume=True,
        completion_path="",
    )
    dialog = ProcessingBatchDialog(view_model, language="en")
    qtbot.addWidget(dialog)
    dialog._render_annotation_result()

    assert dialog.open_annotation_package_button.isHidden() is False
    assert dialog.launch_annotation_tool_button.isHidden() is False
    assert dialog.resume_annotation_button.isHidden() is False
    assert dialog.open_annotation_completion_button.isHidden() is True
    assert "training has not started" in dialog.report_summary_label.text()


def test_blocking_summary_table_contains_required_columns(qtbot):
    dialog = ProcessingBatchDialog(
        ProcessingSummaryViewModel(_plan(_blocked()), language="en"),
        language="en",
    )
    qtbot.addWidget(dialog)

    summary = dialog._build_blocking_summary_dialog()
    qtbot.addWidget(summary)
    table = summary.findChild(QTableWidget, "BlockingSummaryTable")

    assert table is not None
    assert table.rowCount() == 1
    assert table.horizontalHeaderItem(0).text() == "Sample ID"
    assert table.horizontalHeaderItem(1).text() == "Reason"
    assert table.horizontalHeaderItem(2).text() == "Violation Code"
    assert table.item(0, 0).text() == "blocked-sample"
    assert table.item(0, 2).text() == "review_without_selection"


def test_processing_pipeline_feature_flag_can_fall_back_to_legacy():
    assert processing_pipeline_enabled({}) is False
    assert processing_pipeline_enabled({PROCESSING_PIPELINE_ENV: "0"}) is False
    assert processing_pipeline_enabled({PROCESSING_PIPELINE_ENV: "1"}) is True
    assert processing_pipeline_enabled({PROCESSING_PIPELINE_ENV: "true"}) is True


def test_review_dialog_uses_processing_path_when_feature_flag_is_enabled(monkeypatch):
    from app.gui import review_cases_dialog as module

    host = MagicMock()
    monkeypatch.setattr(module, "processing_pipeline_enabled", lambda: True)

    module.ReviewCasesDialog._open_selected_training_queue(host)

    host._open_processing_pipeline.assert_called_once_with()


def test_review_dialog_falls_back_to_legacy_path_when_feature_flag_is_disabled(
    monkeypatch,
):
    from app.gui import review_cases_dialog as module

    host = MagicMock()
    host.store.rows = []
    host._text.side_effect = lambda _zh, en: en
    monkeypatch.setattr(module, "processing_pipeline_enabled", lambda: False)
    information = MagicMock()
    monkeypatch.setattr(module.QMessageBox, "information", information)

    module.ReviewCasesDialog._open_selected_training_queue(host)

    host._open_processing_pipeline.assert_not_called()
    information.assert_called_once()


def test_processing_dialog_has_no_review_routing_dependency():
    import app.gui.processing_batch_dialog as module

    source = inspect.getsource(module)
    assert "review_routing" not in source
    assert "review_workflow" not in source
    assert "RoutingDecision" not in source
