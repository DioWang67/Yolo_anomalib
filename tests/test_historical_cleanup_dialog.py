from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from PyQt5.QtCore import Qt

from app.gui.historical_cleanup_dialog import HistoricalCleanupDialog
from app.gui.historical_cleanup_view_model import HistoricalCleanupViewModel
from app.gui.processing_batch_dialog import ProcessingBatchDialog
from app.gui.processing_summary_view_model import ProcessingSummaryViewModel
from tools.historical_cleanup import (
    CleanupAnalysis,
    CleanupGroup,
    CleanupRecord,
    HistoricalCleanupSession,
)
from tools.processing_pipeline import ProcessingPlanner


def _analysis(tmp_path, count: int = 55) -> CleanupAnalysis:
    manifest = tmp_path / "review.csv"
    manifest.write_text("sample_id\n", encoding="utf-8")
    records = tuple(
        CleanupRecord(
            record_id=f"record-{index}",
            sample_id=f"sample-{index}",
            source_index=index,
            root_cause="legacy_review_outcome_missing",
            contributing_causes=("legacy_review_outcome_missing",),
            violation_codes=(),
            reason="Manual review required",
            confidence="NONE",
            confidence_reason="No human outcome exists.",
            suggested_fix="Open the existing Review UI.",
            potential_risk="Do not invent training truth.",
            blocking_removed_estimate=0,
            apply_ready=False,
            phase1c_proposal_id="",
            proposed_field_changes={},
            original_fields={"review_selected": "0"},
            derived_semantics={"required_action": "manual_review"},
            open_targets={
                "sample": "",
                "image": "",
                "annotation": "",
                "review": "",
                "conflict_report": "",
            },
        )
        for index in range(count)
    )
    group = CleanupGroup(
        group_id="manual-group",
        root_cause="legacy_review_outcome_missing",
        title="legacy_review_outcome_missing (NONE)",
        record_ids=tuple(item.record_id for item in records),
        record_count=count,
        confidence="NONE",
        confidence_reason="No human outcome exists.",
        suggested_fix="Open the existing Review UI.",
        potential_risk="Do not invent training truth.",
        blocking_removed_estimate=0,
        batch_approvable=False,
    )
    return CleanupAnalysis(
        analysis_id="analysis-1",
        created_at="2026-07-21T00:00:00+00:00",
        manifest_path=str(manifest.resolve()),
        manifest_sha256="a" * 64,
        sample_count=count,
        audit_summary={"blocking_inconsistency_count": 0, "error_count": 0},
        planner_statistics={
            "ready_count": 0,
            "annotation_count": 0,
            "color_count": 0,
            "blocking_count": count,
            "excluded_count": 0,
            "manual_review_count": count,
        },
        root_cause_counts={"legacy_review_outcome_missing": count},
        groups=(group,),
        records=records,
        warnings=(),
        phase1c_plan={"schema_version": 1, "plan_id": "unused", "proposals": []},
    )


def test_dialog_uses_three_columns_and_pages_large_groups(qtbot, tmp_path):
    analysis = _analysis(tmp_path)
    session = HistoricalCleanupSession(
        analysis,
        session_path=tmp_path / "session" / "session.json",
    )
    view_model = HistoricalCleanupViewModel(
        analysis,
        session=session,
        page_size=50,
    )
    dialog = HistoricalCleanupDialog(view_model, language="en")
    qtbot.addWidget(dialog)

    assert dialog.group_list.count() == 1
    assert dialog.record_table.rowCount() == 50
    assert "Page 1/2" in dialog.page_label.text()
    assert "Suggested Fix:" in dialog.detail_view.toPlainText()

    qtbot.mouseClick(dialog.next_page_button, Qt.LeftButton)

    assert dialog.record_table.rowCount() == 5
    assert "Page 2/2" in dialog.page_label.text()


def test_processing_summary_exposes_cleanup_callback_only_for_blocking(qtbot):
    blocked = {
        "sample_id": "blocked",
        "review_selected": "0",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "failure_category": "",
        "skip_reason": "",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "0",
    }
    plan = ProcessingPlanner().create_plan([(0, blocked)], operator="tester")
    callback = MagicMock()
    dialog = ProcessingBatchDialog(
        ProcessingSummaryViewModel(plan, language="en"),
        language="en",
        cleanup_launcher=callback,
    )
    qtbot.addWidget(dialog)

    assert dialog.historical_cleanup_button.isHidden() is False
    qtbot.mouseClick(dialog.historical_cleanup_button, Qt.LeftButton)
    callback.assert_called_once_with()


def test_cleanup_dialog_has_no_workflow_or_routing_business_dependency():
    import app.gui.historical_cleanup_dialog as module

    source = inspect.getsource(module)
    assert "derive_review_semantics" not in source
    assert "validate_record_consistency" not in source
    assert "RoutingDecision" not in source


def test_review_dialog_launches_cleanup_from_complete_manifest(monkeypatch, tmp_path):
    from app.gui import review_cases_dialog as module

    host = MagicMock()
    host.manifest_path = tmp_path / "review.csv"
    host.language = "en"
    analyzer = MagicMock()
    analysis = MagicMock()
    analyzer.analyze.return_value = analysis
    view_model = MagicMock()
    dialog = MagicMock()
    monkeypatch.setattr(module, "HistoricalCleanupAnalyzer", lambda: analyzer)
    monkeypatch.setattr(
        module,
        "HistoricalCleanupViewModel",
        lambda value, page_size: view_model,
    )
    monkeypatch.setattr(
        module,
        "HistoricalCleanupDialog",
        lambda value, language, parent: dialog,
    )

    module.ReviewCasesDialog._open_historical_cleanup(host)

    analyzer.analyze.assert_called_once()
    assert analyzer.analyze.call_args.args == (host.manifest_path,)
    assert analyzer.analyze.call_args.kwargs["operator"]
    dialog.exec_.assert_called_once_with()
