import csv
import inspect
import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtWidgets import QDialog, QLabel, QMessageBox, QPushButton

from app.gui.review_cases_dialog import (
    LEGACY_REVIEW_LAYOUT_ENV,
    LEGACY_SELECTED_PAGE_ENV,
    ReviewCasesDialog,
    ReviewManifestStore,
    _assert_rows_match_target,
    _environment_flag_enabled,
    _find_saved_case_index,
    _is_color_only_submission,
    _is_confirmation_only_submission,
    _is_failure_review_candidate,
    _row_detection_state,
    _row_has_ordered_class_contract,
    _select_target,
    _target_manifest_path,
    _visible_action_values,
    _with_pass_sampling,
)
from app.gui.review_selection_gallery import (
    ROW_INDEX_ROLE,
    UNCLASSIFIED_REASON_FILTER,
    ReviewSelectionGallery,
    record_reason_keys,
)
from app.gui.training_batch_dialog import TrainingBatchDialog
from tools.retraining_workspaces import create_retraining_workspace
from tools.review_classification import (
    COLOR_ISSUE,
    MISSED_DETECTION,
    OTHER_FAILURE,
    THRESHOLD_NOT_MET,
    ReviewFailureClassification,
    threshold_source_keys,
)
from tools.review_routing import ReviewDecision
from tools.review_workflow import ReviewWorkflowValidationError


def test_training_target_guard_rejects_mixed_products() -> None:
    with pytest.raises(ValueError, match="cannot mix"):
        _assert_rows_match_target(
            [
                {"product": "Cable1", "area": "A"},
                {"product": "PCBA1", "area": "TOP"},
            ],
            product="Cable1",
            area="A",
        )


def test_training_target_guard_rejects_active_target_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _assert_rows_match_target(
            [{"product": "Cable1", "area": "B"}],
            product="Cable1",
            area="A",
        )


def test_embedded_workspace_close_returns_to_inspection_without_shutdown(
    tmp_path, qtbot
):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    requested = []
    dialog.back_to_inspection_requested.connect(lambda: requested.append(True))
    dialog.image_service.shutdown = MagicMock()

    dialog.accept()

    assert requested == [True]
    dialog.image_service.shutdown.assert_not_called()


def test_photo_selection_keeps_only_short_heading_and_hover_help(tmp_path, qtbot):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
        embedded=True,
    )
    qtbot.addWidget(dialog)

    heading = dialog.findChild(QLabel, "reviewSelectionHeading")
    help_badge = dialog.findChild(QLabel, "reviewSelectionHeadingHelp")
    target_scope = dialog.findChild(QLabel, "reviewTargetScope")

    assert heading is not None
    assert heading.text() == "第 1 階段：選照片"
    assert help_badge is not None
    assert help_badge.text() == "ⓘ 提示"
    assert "勾選會立即保存" in help_badge.toolTip()
    assert target_scope is not None
    assert target_scope.text() == "Cable1／A"
    assert "不會混入其他機種" in target_scope.toolTip()


def test_embedded_workspace_opens_progress_as_same_window_page(tmp_path, qtbot):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    return_page = dialog.workflow_stack.currentWidget()

    dialog._open_update_progress()

    progress_page = dialog.workflow_stack.currentWidget()
    assert progress_page is not return_page
    assert progress_page.windowType() == Qt.Widget
    progress_page.accept()
    assert dialog.workflow_stack.currentWidget() is return_page


def test_embedded_workspace_opens_training_queue_as_same_window_page(
    tmp_path, qtbot, monkeypatch
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    dialog.store.set_review(0, "verified_empty")
    return_page = dialog.workflow_stack.currentWidget()
    monkeypatch.setattr(
        TrainingBatchDialog,
        "exec_",
        lambda _self: (_ for _ in ()).throw(AssertionError("must not be modal")),
    )

    dialog._open_selected_training_queue()

    queue_page = dialog.workflow_stack.currentWidget()
    assert isinstance(queue_page, TrainingBatchDialog)
    queue_page.reject()
    assert dialog.workflow_stack.currentWidget() is return_page


def test_batch_folder_keeps_its_selected_manifest_after_handoff(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    workspace = create_retraining_workspace(
        tmp_path / "training-data",
        product="Cable1",
        area="A",
        batch_version="Cable1_A_v0.0.1",
    )
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=workspace.manifest_path,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
        batch_version=workspace.batch_version,
        batch_workspace_dir=workspace.root,
        embedded=True,
    )
    qtbot.addWidget(dialog)
    dialog.store.set_review(0, "verified_empty")

    dialog._remove_submitted_rows_from_queue({0})

    assert dialog.store.rows[0]["training_selected"] == "1"
    assert dialog._handed_off_indices == {0}
    assert workspace.batch_version in dialog.windowTitle()


def test_submitted_training_starts_orchestrator_in_background(
    tmp_path, qtbot, monkeypatch
):
    launcher = tmp_path / "open_operator_training.bat"
    launcher.write_text("@echo off\n", encoding="utf-8")
    handoff = tmp_path / "handoff.json"
    handoff.write_text("{}", encoding="utf-8")
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    launches = []

    class DetachedProcess:
        @staticmethod
        def startDetached(program, arguments, working_directory):
            launches.append((program, arguments, working_directory))
            return True, 1234

    monkeypatch.setattr(
        "app.gui.review_cases_dialog.QProcess",
        DetachedProcess,
    )

    assert dialog._start_training_center(handoff) is True
    assert launches == [
        (
            "cmd.exe",
            ["/c", str(launcher), str(handoff), "--background"],
            str(tmp_path),
        )
    ]


def test_saved_case_lookup_requires_exact_persisted_path(tmp_path):
    saved = tmp_path / "Result" / "saved.png"
    saved.parent.mkdir()
    saved.write_bytes(b"saved")
    copied = tmp_path / "outside" / "saved.png"
    copied.parent.mkdir()
    copied.write_bytes(b"saved")
    rows = [
        {
            "original_path": str(saved),
            "preprocessed_path": "",
            "annotated_path": "",
        }
    ]

    assert _find_saved_case_index(rows, saved) == 0
    assert _find_saved_case_index(rows, copied) is None


@pytest.mark.parametrize(
    ("class_names_json", "expected"),
    [
        ('["Black","Green"]', True),
        ('["Black","Black"]', False),
        ('["Black",""]', False),
        ("[]", False),
        ("not-json", False),
    ],
)
def test_saved_missed_case_requires_ordered_class_contract(class_names_json, expected):
    assert _row_has_ordered_class_contract({"class_names_json": class_names_json}) is expected


def _write_manifest(path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["config_snapshot_path", "review_label", "review_note"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "config_snapshot_path": "result/one.json",
                "review_label": "",
                "review_note": "",
            }
        )


def _write_failure_cases(result_root, *, count=2):
    metadata = (
        result_root
        / "20260720"
        / "Cable1"
        / "A"
        / "FAIL"
        / "metadata"
        / "yolo"
    )
    metadata.mkdir(parents=True)
    for index in range(count):
        (metadata / f"phase2a_{index}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": datetime.now().replace(microsecond=index).isoformat(),
                    "product": "Cable1",
                    "area": "A",
                    "status": "FAIL",
                    "detector": "yolo",
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )


def test_review_manifest_store_saves_each_button_decision(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_review(0, "confirmed_ng")

    reloaded = ReviewManifestStore(manifest)
    assert reloaded.rows[0]["review_label"] == "confirmed_ng"
    assert reloaded.rows[0]["review_selected"] == "1"
    assert reloaded.reviewed_count() == 1


def test_known_main_screen_target_skips_discovery_scan(monkeypatch, tmp_path):
    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("known target must not rescan Result")

    monkeypatch.setattr(
        "app.gui.review_cases_dialog.collect_review_cases",
        unexpected_scan,
    )

    selected = _select_target(
        tmp_path / "Result",
        product="Cable1",
        area="A",
        language="en",
        parent=None,
    )

    assert selected == ("Cable1", "A", True)


def test_manifest_store_only_syncs_missing_or_changed_database_rows(tmp_path):
    snapshot = tmp_path / "case.json"
    snapshot.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "config_snapshot_path",
                "review_outcome",
                "review_label",
                "failure_category",
                "failure_source",
                "failure_note",
                "review_note",
                "skip_reason",
                "action_route",
                "training_selected",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "config_snapshot_path": str(snapshot),
                "review_outcome": "pass",
                "review_label": "false_positive",
                "failure_category": "",
                "failure_source": "yolo",
                "failure_note": "verified",
                "review_note": "",
                "skip_reason": "",
                "action_route": "yolo",
                "training_selected": "0",
            }
        )
    store = ReviewManifestStore(manifest)
    expected = store._database_review_state(store.rows[0])
    repository = MagicMock()
    repository.load_manifest_sync_state.return_value = {
        str(snapshot.resolve()): expected
    }
    store.repository = repository

    store._sync_existing_rows()

    repository.upsert_snapshot_file.assert_not_called()
    repository.sync_review_row.assert_not_called()

    repository.load_manifest_sync_state.return_value = {
        str(snapshot.resolve()): {**expected, "review_label": "confirmed_ng"}
    }
    store._sync_existing_rows()
    repository.upsert_snapshot_file.assert_not_called()
    repository.sync_review_row.assert_called_once_with(
        store.rows[0], append_event=False
    )


def test_manifest_store_indexes_new_snapshot_before_review_sync(tmp_path):
    snapshot = tmp_path / "case.json"
    snapshot.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["config_snapshot_path", "review_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "config_snapshot_path": str(snapshot),
                "review_label": "",
            }
        )
    store = ReviewManifestStore(manifest)
    repository = MagicMock()
    repository.load_manifest_sync_state.return_value = {}
    store.repository = repository

    store._sync_existing_rows()

    repository.upsert_snapshot_file.assert_called_once_with(snapshot)
    repository.sync_review_row.assert_not_called()


def test_legacy_inconsistent_manifest_loads_with_diagnostic_without_rewrite(tmp_path):
    manifest = tmp_path / "review.csv"
    manifest.write_text(
        "config_snapshot_path,review_selected,review_label,training_selected\n"
        "case.json,0,confirmed_ng,0\n",
        encoding="utf-8",
    )
    original = manifest.read_bytes()

    store = ReviewManifestStore(manifest)

    assert manifest.read_bytes() == original
    diagnostics = next(iter(store.workflow_diagnostics.values()))
    assert "review_without_selection" in {
        violation.code for violation in diagnostics
    }


def test_new_inconsistent_review_is_rejected_before_manifest_write(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    original = manifest.read_bytes()
    store = ReviewManifestStore(manifest)
    inconsistent = ReviewDecision(
        review_label="confirmed_ng",
        product_verdict="ok",
        detection_verdict="correct",
        color_verdict="not_applicable",
        action_route="yolo",
    )

    with pytest.raises(
        ReviewWorkflowValidationError,
        match="product_ok_confirmed_ng",
    ):
        store.set_decision(0, inconsistent)

    assert manifest.read_bytes() == original


def test_submitted_review_cannot_be_silently_changed(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)
    store.set_review(0, "confirmed_ng")
    store.mark_submitted_indices({0})
    submitted = manifest.read_bytes()

    with pytest.raises(
        ReviewWorkflowValidationError,
        match="submitted_review_modified_without_revision",
    ):
        store.set_review(0, "wrong_box")

    assert manifest.read_bytes() == submitted


def test_submitted_review_accepts_explicit_transient_revision_reason(
    tmp_path, caplog
):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)
    store.set_review(0, "confirmed_ng")
    store.mark_submitted_indices({0})
    decision = ReviewDecision.from_legacy_label("wrong_box")
    updates = {
        **decision.to_columns(),
        "review_selected": "1",
        "training_selected": "1",
    }

    with caplog.at_level("INFO"):
        store.apply_review_updates(
            0,
            updates,
            revision_reason="operator requested a new annotation",
        )

    assert store.rows[0]["review_label"] == "wrong_box"
    assert "revision_reason" not in store.rows[0]
    assert "revision_reason" not in manifest.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "Submitted review revision saved" in caplog.text
    assert "operator requested a new annotation" in caplog.text


def test_training_selection_boundary_rejects_skipped_record(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)
    store.set_triage(
        0,
        outcome="skip",
        decision=ReviewDecision(
            review_label="confirmed_ng",
            product_verdict="ng",
            detection_verdict="correct",
            color_verdict="not_applicable",
            action_route="none",
        ),
        skip_reason="confirmed_failure",
    )
    skipped = manifest.read_bytes()

    with pytest.raises(
        ReviewWorkflowValidationError,
        match="skip_selected_for_training",
    ):
        store.set_training_selection({0}, {0})

    assert manifest.read_bytes() == skipped


def test_review_manifest_store_routes_color_truth_without_yolo(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_color_review(0, "actually_ok")

    row = ReviewManifestStore(manifest).rows[0]
    assert row["review_label"] == "color_false_reject"
    assert row["product_verdict"] == "ok"
    assert row["detection_verdict"] == "correct"
    assert row["color_verdict"] == "actually_ok"
    assert row["action_route"] == "color"
    assert row["training_selected"] == "1"


def test_review_manifest_store_routes_mixed_color_and_box_failure(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_color_review(0, "confirmed_ng", detection_verdict="wrong_box")

    row = ReviewManifestStore(manifest).rows[0]
    assert row["review_label"] == "color_confirmed_ng"
    assert row["detection_verdict"] == "wrong_box"
    assert row["action_route"] == "both"


def test_review_manifest_store_rejects_unknown_action(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    with pytest.raises(ValueError, match="Unsupported review label"):
        store.set_review(0, "auto_accept_everything")


def test_review_manifest_store_persists_training_exclusion(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_training_selection({0}, set())

    assert ReviewManifestStore(manifest).rows[0]["training_selected"] == "0"


@pytest.mark.parametrize("review_label", ["confirmed_ok", "image_quality_issue"])
def test_review_hold_decisions_are_never_selected_for_training(tmp_path, review_label):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_review(0, review_label)

    row = ReviewManifestStore(manifest).rows[0]
    assert row["review_label"] == review_label
    assert row["training_selected"] == "0"


def test_review_scope_selection_persists_across_store_reload(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_review_selection({0}, {0})

    assert ReviewManifestStore(manifest).rows[0]["review_selected"] == "1"
    ReviewManifestStore(manifest).set_review_selection({0}, set())
    assert ReviewManifestStore(manifest).rows[0]["review_selected"] == "0"


def test_parallel_store_decision_preserves_saved_review_selection(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    selection_window = ReviewManifestStore(manifest)
    classification_window = ReviewManifestStore(manifest)

    selection_window.set_review_selection({0}, {0})
    classification_window.set_review(0, "confirmed_ng")

    saved = ReviewManifestStore(manifest).rows[0]
    assert saved["review_selected"] == "1"
    assert saved["review_label"] == "confirmed_ng"


def test_failure_classification_is_persisted_without_changing_review_route(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)
    store.set_review(0, "false_negative")

    store.set_failure_classification(
        0,
        ReviewFailureClassification(
            category=THRESHOLD_NOT_MET,
            source="yolo",
        ),
    )

    saved = ReviewManifestStore(manifest).rows[0]
    assert saved["failure_category"] == "threshold_not_met"
    assert saved["failure_source"] == "yolo"
    assert saved["review_label"] == "false_negative"
    assert saved["action_route"] == "yolo"


def test_fail_triage_saves_custom_reason_and_selects_training_set(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_triage(
        0,
        outcome="fail",
        decision=ReviewDecision.from_legacy_label("false_negative"),
        classification=ReviewFailureClassification(
            category=MISSED_DETECTION,
            source="yolo",
            note="左側端子完全沒有框到",
        ),
    )

    saved = ReviewManifestStore(manifest).rows[0]
    assert saved["review_outcome"] == "fail"
    assert saved["failure_category"] == MISSED_DETECTION
    assert saved["failure_note"] == "左側端子完全沒有框到"
    assert saved["training_selected"] == "1"


def test_pass_false_reject_can_be_selected_for_training(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_triage(
        0,
        outcome="pass",
        decision=ReviewDecision.from_legacy_label("false_positive"),
        add_to_training_set=True,
    )

    saved = ReviewManifestStore(manifest).rows[0]
    assert saved["review_outcome"] == "pass"
    assert saved["review_label"] == "false_positive"
    assert saved["training_selected"] == "1"


def test_review_gallery_uses_obvious_checked_badge_and_color(qtbot):
    gallery = ReviewSelectionGallery(language="zh_TW")
    qtbot.addWidget(gallery)
    gallery.set_entries([(4, {"status": "FAIL", "timestamp": "2026-07-20"})], set())
    item = gallery.thumbnail_list.item(0)

    assert item.text().startswith("○ 未選取")
    assert item.background().color().name() == "#fff7ed"

    item.setCheckState(Qt.Checked)

    assert item.text().startswith("✓ 已選取")
    assert item.background().color().name() == "#b7ebc6"
    assert item.font().bold() is True


def test_review_gallery_can_refresh_synchronously_after_selection_change(qtbot):
    gallery = ReviewSelectionGallery(language="en")
    qtbot.addWidget(gallery)
    entries = [(4, {"status": "FAIL", "timestamp": "2026-07-20"})]
    gallery.set_entries(entries, set())
    emitted = []

    def refresh_gallery(selected_indices):
        emitted.append(selected_indices)
        gallery.set_entries(entries, selected_indices)

    gallery.selection_changed.connect(refresh_gallery)

    gallery.thumbnail_list.item(0).setCheckState(Qt.Checked)

    assert emitted == [{4}]
    assert gallery.selected_indices() == {4}


def test_reviewed_gallery_item_can_be_unchecked(qtbot):
    gallery = ReviewSelectionGallery(language="en")
    qtbot.addWidget(gallery)
    reviewed = {
        "status": "FAIL",
        "review_selected": "1",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }
    gallery.set_entries([(4, reviewed)], {4})
    emitted = []
    gallery.selection_changed.connect(emitted.append)
    item = gallery.thumbnail_list.item(0)

    assert bool(item.flags() & Qt.ItemIsUserCheckable)

    item.setCheckState(Qt.Unchecked)

    assert item.checkState() == Qt.Unchecked
    assert gallery.selected_indices() == set()
    assert emitted == [set()]


def test_historical_review_without_selection_can_be_selected_for_resolution(qtbot):
    gallery = ReviewSelectionGallery(language="en")
    qtbot.addWidget(gallery)
    historical_review = {
        "status": "FAIL",
        "review_selected": "0",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "0",
    }

    gallery.set_entries([(4, historical_review)], set())
    item = gallery.thumbnail_list.item(0)

    assert bool(item.flags() & Qt.ItemIsUserCheckable)


def test_review_gallery_filters_by_selection_and_derived_workflow(qtbot):
    gallery = ReviewSelectionGallery(language="en")
    qtbot.addWidget(gallery)
    reviewed = {
        "status": "FAIL",
        "review_selected": "1",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }
    gallery.set_entries(
        [(0, {"status": "FAIL"}), (1, reviewed), (2, {"status": "FAIL"})],
        {1},
    )

    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("selected"))
    assert gallery.entry_indices() == {1}
    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("unselected"))
    assert gallery.entry_indices() == {0, 2}
    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("reviewed"))
    assert gallery.entry_indices() == {1}
    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("unreviewed"))
    assert gallery.entry_indices() == {0, 2}


def test_review_gallery_reason_filter_combines_with_existing_filters(qtbot):
    gallery = ReviewSelectionGallery(language="zh_TW")
    qtbot.addWidget(gallery)
    entries = [
        (0, {"status": "FAIL", "decision_reasons": "MISSING"}),
        (
            1,
            {
                "status": "FAIL",
                "decision_reasons": "MISSING|COLOR_MISMATCH",
            },
        ),
        (2, {"status": "FAIL", "failure_category": "wrong_box"}),
        (3, {"status": "FAIL"}),
        (4, {"status": "FAIL", "skip_reason": "equipment_lighting"}),
    ]
    gallery.set_entries(entries, {1, 2})

    missing_index = gallery.reason_filter_combo.findData("system:MISSING")
    assert missing_index >= 0
    assert "缺件 (2)" == gallery.reason_filter_combo.itemText(missing_index)
    gallery.reason_filter_combo.setCurrentIndex(missing_index)
    assert gallery.entry_indices() == {0, 1}
    assert gallery.selected_indices() == {1}

    gallery.filter_combo.setCurrentIndex(
        gallery.filter_combo.findData("selected")
    )
    assert gallery.entry_indices() == {1}
    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("all"))
    gallery.reason_filter_combo.setCurrentIndex(
        gallery.reason_filter_combo.findData("failure:wrong_box")
    )
    assert gallery.entry_indices() == {2}

    gallery.reason_filter_combo.setCurrentIndex(
        gallery.reason_filter_combo.findData(UNCLASSIFIED_REASON_FILTER)
    )
    assert gallery.entry_indices() == {3}
    assert gallery.source_entry_indices() == {0, 1, 2, 3, 4}

    gallery.reason_filter_combo.setCurrentIndex(
        gallery.reason_filter_combo.findData("system:MISSING")
    )
    gallery.set_entries(entries, {1, 2})
    assert gallery.reason_filter_combo.currentData() == "system:MISSING"
    assert gallery.entry_indices() == {0, 1}
    assert gallery.selected_indices() == {1}


def test_reason_parser_accepts_pipe_json_and_future_codes() -> None:
    assert record_reason_keys(
        {
            "failure_category": "threshold_not_met",
            "decision_reasons": '["MISSING", "FUTURE_DEFECT"]',
            "skip_reason": "image_quality",
        }
    ) == (
        "failure:threshold_not_met",
        "system:MISSING",
        "system:FUTURE_DEFECT",
        "skip:image_quality",
    )
    assert record_reason_keys(
        {"decision_reasons": "['MISSING', 'COLOR_MISMATCH']"}
    ) == ("system:MISSING", "system:COLOR_MISMATCH")


def test_review_gallery_selects_and_clears_only_current_filter_results(qtbot):
    gallery = ReviewSelectionGallery(language="en")
    qtbot.addWidget(gallery)
    gallery.set_entries(
        [(0, {"status": "FAIL"}), (1, {"status": "FAIL"}), (2, {"status": "FAIL"})],
        {1},
    )
    emitted = []
    gallery.selection_changed.connect(emitted.append)
    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("unselected"))

    qtbot.mouseClick(gallery.select_all_button, Qt.LeftButton)

    assert emitted[-1] == {0, 2}
    gallery.filter_combo.setCurrentIndex(gallery.filter_combo.findData("selected"))
    assert gallery.entry_indices() == {0, 1, 2}
    qtbot.mouseClick(gallery.clear_button, Qt.LeftButton)
    assert emitted[-1] == set()


def test_review_gallery_lazy_pages_preserve_unrendered_selections(qtbot):
    gallery = ReviewSelectionGallery(language="en")
    qtbot.addWidget(gallery)
    entries = [
        (index, {"status": "FAIL", "timestamp": f"2026-07-20T00:{index:02d}"})
        for index in range(250)
    ]

    gallery.set_entries(entries, {205})

    assert gallery.thumbnail_list.count() == 100
    assert gallery.entry_indices() == set(range(250))
    assert gallery.selected_indices() == {205}
    assert gallery.load_more_button.isHidden() is False

    qtbot.mouseClick(gallery.select_all_button, Qt.LeftButton)
    assert gallery.selected_indices() == set(range(250))

    qtbot.mouseClick(gallery.load_more_button, Qt.LeftButton)
    assert gallery.thumbnail_list.count() == 200


def test_reviewed_selection_can_be_removed_without_erasing_review_result(
    tmp_path, qtbot
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    manifest = tmp_path / "review.csv"
    kwargs = {
        "result_root": result_root,
        "manifest_path": manifest,
        "training_data_dir": tmp_path / "training-data",
        "language": "en",
        "start_in_overview": True,
    }
    dialog = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(dialog)
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("all"))
    dialog._apply_time_filter()
    item = dialog.review_gallery.thumbnail_list.item(0)
    row_index = int(item.data(ROW_INDEX_ROLE))
    item.setCheckState(Qt.Checked)
    dialog.store.set_triage(
        row_index,
        outcome="pass",
        decision=ReviewDecision.from_legacy_label("false_positive"),
        add_to_training_set=False,
    )
    dialog._refresh_review_gallery([row_index])

    dialog.review_gallery.thumbnail_list.item(0).setCheckState(Qt.Unchecked)

    saved = ReviewManifestStore(manifest).rows[row_index]
    assert dialog._review_scope_indices == set()
    assert saved["review_selected"] == "1"
    assert saved["review_outcome"] == "pass"
    assert saved["review_label"] == "false_positive"
    assert saved["training_selected"] == "0"

    reopened = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(reopened)
    reopened.time_range_combo.setCurrentIndex(reopened.time_range_combo.findData("all"))
    reopened._apply_time_filter()

    assert reopened._review_scope_indices == set()
    restored = ReviewManifestStore(manifest).rows[row_index]
    assert restored["review_outcome"] == "pass"
    assert restored["review_label"] == "false_positive"

    reopened.review_gallery.thumbnail_list.item(0).setCheckState(Qt.Checked)

    assert reopened._review_scope_indices == {row_index}
    reselected = ReviewManifestStore(manifest).rows[row_index]
    assert reselected["review_outcome"] == "pass"
    assert reselected["review_label"] == "false_positive"


def test_selecting_pending_row_does_not_rewrite_unselected_historical_review(
    tmp_path, qtbot, monkeypatch
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=2)
    manifest = tmp_path / "review.csv"
    kwargs = {
        "result_root": result_root,
        "manifest_path": manifest,
        "training_data_dir": tmp_path / "training-data",
        "language": "en",
        "start_in_overview": True,
    }
    first = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(first)
    first.store.set_triage(
        0,
        outcome="pass",
        decision=ReviewDecision.from_legacy_label("false_positive"),
        add_to_training_set=False,
    )
    legacy_rows = [dict(row) for row in first.store.rows]
    legacy_rows[0]["review_selected"] = "0"
    first.store._write_rows_atomic(legacy_rows)

    errors = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda *_args: errors.append(_args[-1]),
    )
    reopened = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(reopened)
    reopened.time_range_combo.setCurrentIndex(reopened.time_range_combo.findData("all"))
    reopened._apply_time_filter()
    pending_item = next(
        reopened.review_gallery.thumbnail_list.item(item_index)
        for item_index in range(reopened.review_gallery.thumbnail_list.count())
        if int(
            reopened.review_gallery.thumbnail_list.item(item_index).data(
                ROW_INDEX_ROLE
            )
        )
        == 1
    )

    pending_item.setCheckState(Qt.Checked)

    saved = ReviewManifestStore(manifest).rows
    assert errors == []
    assert saved[0]["review_selected"] == "0"
    assert saved[0]["review_outcome"] == "pass"
    assert saved[0]["review_label"] == "false_positive"
    assert saved[1]["review_selected"] == "1"


def test_dialog_reject_stops_image_service_and_releases_caches(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=2)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
        start_in_overview=True,
    )
    qtbot.addWidget(dialog)

    dialog.reject()

    assert dialog.image_service.active is False
    assert dialog.image_service.pending_count == 0
    assert dialog.image_service.thumbnail_cache.entry_count == 0
    assert dialog.image_service.full_cache.entry_count == 0


def test_other_failure_requires_custom_reason(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    with pytest.raises(ValueError, match="custom failure note"):
        store.set_triage(
            0,
            outcome="fail",
            decision=ReviewDecision.from_legacy_label("false_negative"),
            classification=ReviewFailureClassification(
                category=OTHER_FAILURE,
                source="yolo",
            ),
        )


def test_skip_confirmed_failure_is_saved_but_never_selected_for_training(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_triage(
        0,
        outcome="skip",
        decision=ReviewDecision(
            review_label="confirmed_ng",
            product_verdict="ng",
            detection_verdict="correct",
            color_verdict="not_applicable",
            action_route="none",
        ),
        skip_reason="confirmed_failure",
    )

    saved = ReviewManifestStore(manifest).rows[0]
    assert saved["review_outcome"] == "skip"
    assert saved["skip_reason"] == "confirmed_failure"
    assert saved["training_selected"] == "0"
    assert saved["action_route"] == "none"


def test_threshold_sources_keep_builtins_and_extend_from_future_detector():
    assert threshold_source_keys({"detector": "anomalib"}) == (
        "yolo",
        "color",
        "anomalib",
    )


def test_target_manifest_path_keeps_product_decisions_separate(tmp_path):
    path = _target_manifest_path(tmp_path / "review.csv", product="Cable 1", area="Top/A")

    assert path.name == "review_Cable_1_Top_A.csv"


def test_pass_sampling_keeps_all_failures_and_one_in_every_hundred_passes():
    cases = [SimpleNamespace(status="PASS", product="Cable1", area="A") for _ in range(201)]
    failure = SimpleNamespace(status="FAIL", product="Cable1", area="A")
    cases.insert(50, failure)

    selected = _with_pass_sampling(cases)

    assert selected.count(failure) == 1
    assert sum(case.status == "PASS" for case in selected) == 3


@pytest.mark.parametrize(
    ("status", "expected"),
    [("FAIL", True), ("DETECTION_FAIL", True), ("PASS", False), ("OK", False)],
)
def test_failure_overview_excludes_success_rows(status, expected):
    assert _is_failure_review_candidate({"status": status}) is expected


def test_review_overview_selects_failures_before_classification(
    tmp_path, qtbot, monkeypatch
):
    metadata = (
        tmp_path
        / "Result"
        / "20260717"
        / "Cable1"
        / "A"
        / "FAIL"
        / "metadata"
        / "yolo"
    )
    metadata.mkdir(parents=True)
    timestamps = [
        datetime.now().replace(microsecond=100).isoformat(),
        datetime.now().replace(microsecond=200).isoformat(),
    ]
    for index, timestamp in enumerate(timestamps):
        (metadata / f"failure_{index}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "product": "Cable1",
                    "area": "A",
                    "status": "FAIL",
                    "detector": "yolo",
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )
    pass_metadata = (
        tmp_path
        / "Result"
        / "20260717"
        / "Cable1"
        / "A"
        / "PASS"
        / "metadata"
        / "yolo"
    )
    pass_metadata.mkdir(parents=True)
    (pass_metadata / "pass_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "product": "Cable1",
                "area": "A",
                "status": "PASS",
                "detector": "yolo",
                "detections": [],
            }
        ),
        encoding="utf-8",
    )

    manifest = tmp_path / "review.csv"
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        start_in_overview=True,
    )
    qtbot.addWidget(dialog)
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("all"))
    dialog._apply_time_filter()

    assert dialog.workflow_stack.currentWidget() is dialog.review_selection_page
    assert dialog.review_gallery.thumbnail_list.count() == 2
    assert dialog.start_review_button.isEnabled() is False
    assert "稍後補訓" in dialog.save_review_selection_button.text()

    selected_item = dialog.review_gallery.thumbnail_list.item(1)
    selected_index = int(selected_item.data(ROW_INDEX_ROLE))
    selected_item.setCheckState(Qt.Checked)

    assert dialog.start_review_button.isEnabled() is True
    assert "已保存 1 張" in dialog.review_selection_saved_label.text()
    reopened = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        start_in_overview=True,
    )
    qtbot.addWidget(reopened)
    assert reopened.workflow_stack.currentWidget() is reopened.review_selection_page
    assert reopened._review_scope_indices == {selected_index}
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("custom"))
    dialog.start_time_edit.setDateTime(QDateTime.fromString("2000-01-01T00:00:00", Qt.ISODate))
    dialog.end_time_edit.setDateTime(QDateTime.fromString("2000-01-02T00:00:00", Qt.ISODate))
    dialog._apply_time_filter()
    assert dialog.review_gallery.thumbnail_list.count() == 0
    assert "另保留 1 張勾選" in dialog.review_selection_summary.text()

    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("all"))
    dialog._apply_time_filter()
    restored_item = next(
        dialog.review_gallery.thumbnail_list.item(index)
        for index in range(dialog.review_gallery.thumbnail_list.count())
        if int(dialog.review_gallery.thumbnail_list.item(index).data(ROW_INDEX_ROLE))
        == selected_index
    )
    assert restored_item.checkState() == Qt.Checked
    qtbot.mouseClick(dialog.start_review_button, Qt.LeftButton)

    assert dialog.workflow_stack.currentWidget() is dialog.classification_page
    assert dialog.visible_indices == [selected_index]
    dialog.fail_reason_combo.setCurrentIndex(
        dialog.fail_reason_combo.findData(THRESHOLD_NOT_MET)
    )
    qtbot.mouseClick(dialog.save_fail_button, Qt.LeftButton)

    rows = ReviewManifestStore(manifest).rows
    assert rows[selected_index]["review_label"] == "false_negative"
    assert rows[selected_index]["failure_category"] == "threshold_not_met"
    assert rows[selected_index]["failure_source"] == "yolo"
    assert sum(bool(row["review_label"]) for row in rows) == 1


def test_phase2a_legacy_flag_restores_selected_only_page_and_scope(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    kwargs = {
        "result_root": result_root,
        "manifest_path": tmp_path / "review.csv",
        "training_data_dir": tmp_path / "training-data",
        "language": "en",
        "start_in_overview": True,
    }
    direct_dialog = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(direct_dialog)
    direct_dialog.time_range_combo.setCurrentIndex(
        direct_dialog.time_range_combo.findData("all")
    )
    direct_dialog._apply_time_filter()
    qtbot.mouseClick(direct_dialog.review_gallery.select_all_button, Qt.LeftButton)
    expected_scope = set(direct_dialog._review_scope_indices)

    legacy_dialog = ReviewCasesDialog(
        **kwargs,
        use_legacy_selected_page=True,
    )
    qtbot.addWidget(legacy_dialog)

    assert legacy_dialog.workflow_stack.currentWidget() is legacy_dialog.selected_review_page
    assert legacy_dialog.selected_review_gallery.entry_indices() == expected_scope
    qtbot.mouseClick(legacy_dialog.classify_selected_button, Qt.LeftButton)
    assert set(legacy_dialog.visible_indices) == expected_scope


def test_phase2a_legacy_page_environment_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv(LEGACY_SELECTED_PAGE_ENV, raising=False)
    assert _environment_flag_enabled(LEGACY_SELECTED_PAGE_ENV) is False

    monkeypatch.setenv(LEGACY_SELECTED_PAGE_ENV, "true")
    assert _environment_flag_enabled(LEGACY_SELECTED_PAGE_ENV) is True


def test_phase2b_legacy_layout_environment_flag_is_opt_in(monkeypatch):
    monkeypatch.delenv(LEGACY_REVIEW_LAYOUT_ENV, raising=False)
    assert _environment_flag_enabled(LEGACY_REVIEW_LAYOUT_ENV) is False

    monkeypatch.setenv(LEGACY_REVIEW_LAYOUT_ENV, "on")
    assert _environment_flag_enabled(LEGACY_REVIEW_LAYOUT_ENV) is True


def test_phase2b_opens_first_pending_and_handles_all_completed(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    manifest = tmp_path / "review.csv"
    first = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(first)
    first.store.set_review(0, "false_negative")

    resumed = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(resumed)
    assert resumed.current_index == 1

    resumed.store.set_review(1, "false_negative")
    completed = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(completed)
    assert completed.current_index == 0
    assert "Reviewed 2" in completed.progress_label.text()


def test_phase2b_thumbnail_navigation_and_status_refresh(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    dialog.review_thumbnail_panel.list_widget.setCurrentRow(1)
    assert dialog.current_index == 1

    qtbot.mouseClick(dialog.pass_review_button, Qt.LeftButton)

    dialog.review_thumbnail_panel.filter_combo.setCurrentIndex(
        dialog.review_thumbnail_panel.filter_combo.findData("completed")
    )
    assert dialog.review_thumbnail_panel.displayed_indices() == {1}


def test_phase2b_primary_actions_fit_without_horizontal_scroll(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    dialog.resize(1000, 700)
    dialog.show()
    qtbot.wait(10)

    assert dialog.review_detail_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.skip_review_button.isVisible() is True
    assert dialog.review_details_panel.isHidden() is True

    qtbot.mouseClick(dialog.review_details_toggle, Qt.LeftButton)

    assert dialog.review_details_panel.isHidden() is False
    assert "▲" in dialog.review_details_toggle.text()


def test_phase2b_save_failure_keeps_position_input_and_offers_retry(
    tmp_path, qtbot, monkeypatch
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    start_index = dialog.current_index
    dialog.fail_reason_combo.setCurrentIndex(
        dialog.fail_reason_combo.findData(MISSED_DETECTION)
    )
    dialog.custom_failure_note.setText("keep this input")
    original_apply = dialog.store.apply_review_updates
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("disk full")
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(dialog.store, "apply_review_updates", fail_once)

    qtbot.mouseClick(dialog.save_fail_button, Qt.LeftButton)

    assert dialog.current_index == start_index
    assert dialog.custom_failure_note.text() == "keep this input"
    assert "OSError" in dialog.feedback_label.text()
    assert "phase2a_" in dialog.feedback_label.text()
    assert not dialog.retry_save_button.isHidden()

    qtbot.mouseClick(dialog.retry_save_button, Qt.LeftButton)
    saved = dialog.store.rows[start_index]
    assert saved["review_outcome"] == "fail"
    assert saved["failure_note"] == "keep this input"
    assert attempts == 2


def test_phase2b_validation_error_shows_sample_and_code(
    tmp_path, qtbot, monkeypatch
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    dialog.store.mark_submitted_indices({0})
    monkeypatch.setattr(
        "app.gui.review_cases_dialog.QInputDialog.getText",
        lambda *_args, **_kwargs: ("", False),
    )

    qtbot.mouseClick(dialog.pass_review_button, Qt.LeftButton)

    assert "submitted_review_modified_without_revision" in dialog.feedback_label.text()
    assert "phase2a_0_config_snapshot.json" in dialog.feedback_label.text()
    assert dialog.store.rows[0]["review_label"] == ""


def test_phase2b_submitted_review_prompts_for_reason_and_saves(
    tmp_path, qtbot, monkeypatch, caplog
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    dialog.store.mark_submitted_indices({0})
    monkeypatch.setattr(
        "app.gui.review_cases_dialog.QInputDialog.getText",
        lambda *_args, **_kwargs: ("second operator review", True),
    )

    with caplog.at_level("INFO"):
        qtbot.mouseClick(dialog.pass_review_button, Qt.LeftButton)

    assert dialog.store.rows[0]["review_outcome"] == "pass"
    assert dialog.store.rows[0]["review_label"] == "false_positive"
    assert "Submitted review revision saved" in caplog.text


def test_phase2b_shortcuts_focus_guard_navigation_and_overlay(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(10)
    start_index = dialog.current_index
    dialog.pass_review_button.setFocus()

    qtbot.keyClick(dialog.pass_review_button, Qt.Key_F)
    assert not dialog.fail_details_panel.isHidden()
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_Escape)
    assert dialog.fail_details_panel.isHidden()
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_S)
    assert not dialog.skip_details_panel.isHidden()
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_Escape)
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_Right)
    assert dialog.current_index != start_index
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_Left)
    assert dialog.current_index == start_index
    original_mode = dialog.image_viewer.mode
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_Space)
    assert dialog.image_viewer.mode != original_mode

    dialog.custom_failure_note.setFocus()
    qtbot.keyClick(dialog.custom_failure_note, Qt.Key_P)
    assert dialog.store.rows[start_index]["review_label"] == ""
    assert dialog.custom_failure_note.text().endswith("p")
    dialog.pass_review_button.setFocus()
    qtbot.keyClick(dialog.pass_review_button, Qt.Key_P)
    assert dialog.store.rows[start_index]["review_label"] == "false_positive"


def test_phase2b_enter_confirms_fail_and_skip_reason_is_legacy_compatible(
    tmp_path, qtbot
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    dialog.show()
    first_index = dialog.current_index
    dialog.fail_reason_combo.setCurrentIndex(
        dialog.fail_reason_combo.findData(MISSED_DETECTION)
    )
    dialog._dispatch_review_shortcut("fail")
    assert not dialog.fail_details_panel.isHidden()
    dialog.question_label.setFocusPolicy(Qt.StrongFocus)
    dialog.question_label.setFocus()
    qtbot.keyClick(dialog.question_label, Qt.Key_Enter)
    assert dialog.store.rows[first_index]["review_outcome"] == "fail"

    second_index = dialog.current_index
    dialog.skip_reason_combo.setCurrentIndex(
        dialog.skip_reason_combo.findData("equipment_lighting")
    )
    qtbot.mouseClick(dialog.skip_review_button, Qt.LeftButton)
    qtbot.mouseClick(dialog.save_skip_button, Qt.LeftButton)
    saved = dialog.store.rows[second_index]
    assert saved["review_outcome"] == "skip"
    assert saved["skip_reason"] == "image_quality_issue"
    assert saved["review_note"].startswith("[skip:equipment_lighting]")
    assert saved["training_selected"] == "0"


def test_phase2b_new_and_legacy_layout_save_identical_pass_result(tmp_path, qtbot):
    results = []
    for name, legacy in (("new", False), ("legacy", True)):
        result_root = tmp_path / name / "Result"
        _write_failure_cases(result_root, count=1)
        dialog = ReviewCasesDialog(
            result_root=result_root,
            manifest_path=tmp_path / name / "review.csv",
            training_data_dir=tmp_path / name / "training-data",
            language="en",
            use_legacy_review_layout=legacy,
        )
        qtbot.addWidget(dialog)
        qtbot.mouseClick(dialog.pass_review_button, Qt.LeftButton)
        row = ReviewManifestStore(tmp_path / name / "review.csv").rows[0]
        results.append(
            {
                key: row[key]
                for key in (
                    "review_outcome",
                    "review_label",
                    "product_verdict",
                    "detection_verdict",
                    "color_verdict",
                    "action_route",
                    "training_selected",
                )
            }
        )
    assert results[0] == results[1]


def test_phase2b_workspace_does_not_choose_action_route_in_qwidget():
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            ReviewCasesDialog._build_review_workspace_page,
            ReviewCasesDialog._save_pass_triage,
            ReviewCasesDialog._save_fail_triage,
            ReviewCasesDialog._save_skip_triage,
        )
    )

    assert "action_route" not in source


def test_phase2b_primary_actions_remain_operable_at_small_window(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
    )
    qtbot.addWidget(dialog)
    dialog.resize(900, 650)
    dialog.show()
    qtbot.wait(20)

    assert all(
        button.isVisible() and button.height() >= 40
        for button in (
            dialog.pass_review_button,
            dialog.fail_review_button,
            dialog.skip_review_button,
        )
    )


def test_phase2a_empty_selection_warns_without_leaving_overview(
    tmp_path, qtbot, monkeypatch
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
        start_in_overview=True,
    )
    qtbot.addWidget(dialog)
    warnings = []
    monkeypatch.setattr(
        "app.gui.review_cases_dialog.QMessageBox.warning",
        lambda *_args: warnings.append(_args[-1]),
    )

    dialog._start_selected_review()

    assert warnings
    assert "Select at least one" in warnings[-1]
    assert dialog.workflow_stack.currentWidget() is dialog.review_selection_page


def test_phase2a_inconsistent_selection_is_blocked_by_domain_validation(
    tmp_path, qtbot, monkeypatch
):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="en",
        start_in_overview=True,
    )
    qtbot.addWidget(dialog)
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("all"))
    dialog._apply_time_filter()
    dialog.review_gallery.thumbnail_list.item(0).setCheckState(Qt.Checked)
    selected_index = next(iter(dialog._review_scope_indices))
    dialog.store.rows[selected_index].update(
        {
            "review_outcome": "skip",
            "skip_reason": "image_quality_issue",
            "training_selected": "1",
        }
    )
    errors = []
    monkeypatch.setattr(
        "app.gui.review_cases_dialog.QMessageBox.critical",
        lambda *_args: errors.append(_args[-1]),
    )

    dialog._start_selected_review()

    assert errors
    assert "skip_selected_for_training" in errors[-1]
    assert dialog.workflow_stack.currentWidget() is dialog.review_selection_page


def test_phase2a_return_and_reenter_preserves_completed_review(tmp_path, qtbot):
    result_root = tmp_path / "Result"
    _write_failure_cases(result_root, count=1)
    manifest = tmp_path / "review.csv"
    dialog = ReviewCasesDialog(
        result_root=result_root,
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="en",
        start_in_overview=True,
    )
    qtbot.addWidget(dialog)
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("all"))
    dialog._apply_time_filter()
    dialog.review_gallery.thumbnail_list.item(0).setCheckState(Qt.Checked)
    selected_index = next(iter(dialog._review_scope_indices))
    dialog._start_selected_review()
    dialog.store.set_review(selected_index, "false_negative")

    dialog._show_review_overview()
    dialog._start_selected_review()

    assert dialog.workflow_stack.currentWidget() is dialog.classification_page
    assert dialog.visible_indices == [selected_index]
    assert ReviewManifestStore(manifest).rows[selected_index]["review_label"] == "false_negative"


def test_review_actions_are_reduced_to_context_relevant_choices():
    assert _visible_action_values("FAIL", True) == {
        "confirmed_ng",
        "false_positive",
        "needs_annotation",
        "image_quality_issue",
    }
    assert _visible_action_values("FAIL", False) == {
        "verified_empty",
        "false_negative",
        "image_quality_issue",
    }
    assert _visible_action_values("PASS", True) == {
        "confirmed_ok",
        "false_negative",
        "image_quality_issue",
    }
    assert _visible_action_values("FAIL", True, color_failure=True) == {
        "color_confirmed_ng",
        "color_false_reject",
        "color_needs_annotation",
        "image_quality_issue",
    }


def test_legacy_saved_crops_are_treated_as_detected_boxes():
    row = {
        "detections_json": "[]",
        "detected_box_count": "6",
        "detection_evidence_source": "saved_crops",
    }

    assert _row_detection_state(row) == "present"
    assert _visible_action_values("DETECTION_FAIL", True) == {
        "confirmed_ng",
        "false_positive",
        "needs_annotation",
        "image_quality_issue",
    }


def test_missing_legacy_box_evidence_is_not_called_a_confirmed_miss():
    row = {
        "detections_json": "[]",
        "detected_box_count": "",
        "detection_evidence_source": "unknown",
    }

    assert _row_detection_state(row) == "unknown"
    assert _visible_action_values(
        "DETECTION_FAIL", False, detection_unknown=True
    ) == {
        "confirmed_ng",
        "false_positive",
        "needs_annotation",
        "verified_empty",
        "false_negative",
        "image_quality_issue",
    }


def test_confirmation_only_submission_is_saved_as_replay_feedback():
    rows = [
        {"review_label": "confirmed_ng"},
        {"review_label": "false_negative"},
    ]

    assert _is_confirmation_only_submission(rows, {0}) is True
    assert _is_confirmation_only_submission(rows, {0, 1}) is False


def test_color_only_submission_is_not_launched_as_yolo_training():
    rows = [
        {"review_label": "color_false_reject", "action_route": "color"},
        {"review_label": "color_confirmed_ng", "action_route": "both"},
    ]

    assert _is_color_only_submission(rows, {0}) is True
    assert _is_color_only_submission(rows, {0, 1}) is False


def test_review_dialog_records_decision_with_one_button_click(tmp_path, qtbot, monkeypatch):
    metadata = tmp_path / "Result" / "20260714" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "yolo_Cable1_A_120000_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "decision": {"reasons": ["MISSING"]},
                "detections": [
                    {
                        "class_id": 0,
                        "confidence": 0.9,
                        "bbox": [1, 2, 10, 20],
                        "image_width": 100,
                        "image_height": 100,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "review.csv"
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        use_legacy_review_layout=True,
    )
    qtbot.addWidget(dialog)
    confirm_button = dialog.review_buttons["confirmed_ng"]

    assert confirm_button.text() == "確認 NG（AI 判定正確）"
    qtbot.mouseClick(confirm_button, Qt.LeftButton)

    reloaded = ReviewManifestStore(manifest)
    assert reloaded.rows[0]["review_label"] == "confirmed_ng"
    assert "已記錄" in dialog.feedback_label.text()
    assert dialog.export_button.isEnabled() is True

    monkeypatch.setattr(
        "app.gui.review_cases_dialog.QInputDialog.getItem",
        lambda *_args, **_kwargs: ("框的位置／數量錯誤", True),
    )
    qtbot.mouseClick(dialog.review_buttons["needs_annotation"], Qt.LeftButton)

    assert ReviewManifestStore(manifest).rows[0]["review_label"] == "wrong_box"


def test_pass_on_ai_failure_is_added_to_training_queue(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260720" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "false_reject_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "product": "Cable1",
                "area": "A",
                "status": "DETECTION_FAIL",
                "detector": "yolo",
                "detections": [{"class_id": 0, "confidence": 0.9, "bbox": [1, 2, 10, 20]}],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "review.csv"
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.pass_review_button, Qt.LeftButton)

    saved = ReviewManifestStore(manifest).rows[0]
    assert saved["review_outcome"] == "pass"
    assert saved["review_label"] == "false_positive"
    assert saved["training_selected"] == "1"
    assert dialog.export_button.isEnabled() is True


def test_review_dialog_shows_color_failure_score_and_routes_false_reject(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260716" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "color_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "fail_reasons": ["COLOR_MISMATCH"],
                "detections": [
                    {"class_id": 0, "class": "Red", "bbox": [1, 2, 10, 20]}
                ],
                "color_result": {
                    "is_ok": False,
                    "items": [
                        {
                            "index": 0,
                            "class_name": "Red",
                            "best_color": "Red",
                            "diff": 0.55,
                            "threshold": 0.4,
                            "is_ok": False,
                        }
                    ],
                },
                "config": {"color_checker_type": "stats"},
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "review.csv"
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)

    assert "顏色檢查未過" in dialog.question_label.text()
    assert "diff 0.550 > 門檻 0.400" in dialog.details_label.text()
    qtbot.mouseClick(dialog.fail_review_button, Qt.LeftButton)
    dialog.fail_reason_combo.setCurrentIndex(
        dialog.fail_reason_combo.findData(COLOR_ISSUE)
    )
    qtbot.mouseClick(dialog.save_fail_button, Qt.LeftButton)

    row = ReviewManifestStore(manifest).rows[0]
    assert row["review_outcome"] == "fail"
    assert row["failure_category"] == COLOR_ISSUE
    assert row["review_label"] == "color_false_reject"
    assert row["color_verdict"] == "actually_ok"
    assert row["action_route"] == "color"


def test_review_dialog_requires_annotation_when_no_box_exists(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260714" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "missed_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "review.csv"
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=manifest,
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        use_legacy_review_layout=True,
    )
    qtbot.addWidget(dialog)
    confirm_button = dialog.review_buttons["confirmed_ng"]

    assert confirm_button.isEnabled() is False
    assert "未偵測到任何框" in dialog.details_label.text()

    qtbot.mouseClick(dialog.review_buttons["false_negative"], Qt.LeftButton)
    assert ReviewManifestStore(manifest).rows[0]["review_label"] == "false_negative"


def test_review_dialog_routes_legacy_crop_evidence_as_boxed_ng(tmp_path, qtbot):
    base = tmp_path / "Result" / "20260623" / "Cable1" / "A" / "DETECTION_FAIL"
    metadata_dir = base / "metadata" / "yolo"
    cropped_dir = base / "cropped" / "yolo"
    metadata_dir.mkdir(parents=True)
    cropped_dir.mkdir(parents=True)
    stem = "yolo_Cable1_A_122008"
    (metadata_dir / f"{stem}_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-06-23T12:20:08",
                "product": "Cable1",
                "area": "A",
                "status": "DETECTION_FAIL",
                "detector": "yolo",
            }
        ),
        encoding="utf-8",
    )
    for index, class_name in enumerate(("Red", "Green", "Black")):
        (cropped_dir / f"{stem}_{class_name}_{index}.png").write_bytes(b"crop")

    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        use_legacy_review_layout=True,
    )
    qtbot.addWidget(dialog)
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("all"))
    dialog._apply_time_filter()

    assert dialog.store.rows[0]["detected_box_count"] == "3"
    assert dialog.store.rows[0]["detection_evidence_source"] == "saved_crops"
    assert "系統沒有畫框" not in dialog.question_label.text()
    assert "未偵測到任何框" not in dialog.details_label.text()
    enabled_actions = {
        value
        for value, button in dialog.review_buttons.items()
        if button.isEnabled()
    }
    assert enabled_actions == {
        "confirmed_ng",
        "false_positive",
        "needs_annotation",
        "image_quality_issue",
    }


def test_review_dialog_filters_and_exports_selected_time_range(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260714" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    for hour in (10, 14):
        (metadata / f"case_{hour}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": f"2026-07-14T{hour:02d}:00:00",
                    "product": "Cable1",
                    "area": "A",
                    "status": "FAIL",
                    "detector": "yolo",
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    dialog.time_range_combo.setCurrentIndex(dialog.time_range_combo.findData("custom"))
    dialog.start_time_edit.setDateTime(QDateTime.fromString("2026-07-14T09:00:00", Qt.ISODate))
    dialog.end_time_edit.setDateTime(QDateTime.fromString("2026-07-14T11:00:00", Qt.ISODate))

    dialog._apply_time_filter()
    selected_manifest = dialog._write_selected_manifest()

    assert len(dialog.visible_indices) == 1
    with selected_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["timestamp"] for row in rows] == ["2026-07-14T10:00:00"]


def test_review_dialog_restores_last_custom_time_range(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260714" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-14T10:00:00",
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "detections": [],
            }
        ),
        encoding="utf-8",
    )
    kwargs = {
        "result_root": tmp_path / "Result",
        "manifest_path": tmp_path / "review.csv",
        "training_data_dir": tmp_path / "training-data",
        "language": "zh_TW",
        "product": "Cable1",
        "area": "A",
    }
    first = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(first)
    first.time_range_combo.setCurrentIndex(first.time_range_combo.findData("custom"))
    first.start_time_edit.setDateTime(QDateTime.fromString("2026-07-14T09:00:00", Qt.ISODate))
    first.end_time_edit.setDateTime(QDateTime.fromString("2026-07-14T11:00:00", Qt.ISODate))
    first._apply_time_filter()

    reopened = ReviewCasesDialog(**kwargs)
    qtbot.addWidget(reopened)

    assert reopened.time_range_combo.currentData() == "custom"
    assert reopened.start_time_edit.dateTime().toString(Qt.ISODate).startswith("2026-07-14T09:00:00")
    assert reopened.end_time_edit.dateTime().toString(Qt.ISODate).startswith("2026-07-14T11:00:00")


def test_submitted_batch_keeps_review_dialog_context_open(tmp_path, qtbot):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
    )
    qtbot.addWidget(dialog)

    dialog._show_submission_active("本次可訓練：1 張", reused_existing=False)

    assert dialog.result() == QDialog.Rejected
    assert dialog.export_button.isEnabled() is False
    assert dialog.export_button.text() == "本批次已送出"
    assert dialog.progress_button.isHidden() is False


def test_color_submission_shows_same_window_result_and_clear_semantics(
    tmp_path, qtbot
):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    summary = "顏色校正回饋：2 項"

    dialog._show_submission_active(
        summary,
        reused_existing=False,
        color_feedback=True,
    )
    dialog._show_color_submission_result(summary)

    result_page = dialog.workflow_stack.currentWidget()
    assert result_page.objectName() == "ColorSubmissionResultPage"
    assert dialog.export_button.text() == "顏色資料已送出"
    result_text = result_page.findChild(QLabel, "ColorSubmissionSummary").text()
    assert "加入顏色校正樣本庫" in result_text
    assert "不會啟動 YOLO 模型訓練" in result_text
    assert "不會立即改變門檻" in result_text
    experimental_button = result_page.findChild(
        QPushButton, "ExperimentalColorVersionButton"
    )
    assert experimental_button.isHidden() is True

    back_button = next(
        button
        for button in result_page.findChildren(QPushButton)
        if button.text() == "返回資料複核"
    )
    qtbot.mouseClick(back_button, Qt.LeftButton)
    assert dialog.workflow_stack.currentWidget() is dialog.classification_page


def test_color_submission_offers_ok_only_version_when_scope_is_eligible(
    tmp_path, qtbot, monkeypatch
):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        dialog,
        "_eligible_experimental_color_scopes",
        lambda _report: (SimpleNamespace(),),
    )

    dialog._show_color_submission_result(
        "顏色校正回饋：40 項",
        report=SimpleNamespace(color_manifest_paths=("feedback.csv",)),
    )

    result_page = dialog.workflow_stack.currentWidget()
    experimental_button = result_page.findChild(
        QPushButton, "ExperimentalColorVersionButton"
    )
    rollback_button = result_page.findChild(
        QPushButton, "ExperimentalColorRollbackButton"
    )
    assert experimental_button.isHidden() is False
    assert experimental_button.text() == "建立 OK-only 實驗版本"
    assert rollback_button.isHidden() is True


def test_successful_color_submission_routes_to_visible_result_page(
    tmp_path, qtbot, monkeypatch
):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        embedded=True,
    )
    qtbot.addWidget(dialog)
    dialog.store.rows = [
        {
            "product": "Cable1",
            "area": "A",
            "review_label": "color_false_reject",
            "action_route": "color",
            "training_selected": "1",
        }
    ]
    report = SimpleNamespace(
        handoff_path=tmp_path / "color-feedback.json",
        ready_count=0,
        pending_count=0,
        skipped_count=0,
        targets=(("Cable1", "A"),),
        total_ready_count=0,
        total_pending_count=0,
        job_id="",
        status_path=None,
        reused_existing=False,
        color_feedback_count=1,
    )
    monkeypatch.setattr(
        "app.gui.review_cases_dialog.export_operator_handoff",
        lambda *_args, **_kwargs: report,
    )
    dialog._record_submission_audit = MagicMock(return_value=True)
    dialog._remove_submitted_rows_from_queue = MagicMock()
    dialog._show_color_submission_result = MagicMock()
    dialog._color_feedback_progress_message = MagicMock(
        return_value="red: 1/30｜OK 1/5｜NG 0/5｜繼續累積"
    )

    dialog._submit_selected_indices({0})

    dialog._record_submission_audit.assert_called_once()
    dialog._color_feedback_progress_message.assert_called_once_with(report)
    dialog._remove_submitted_rows_from_queue.assert_called_once_with({0})
    dialog._show_color_submission_result.assert_called_once()
    assert dialog.export_button.text() == "顏色資料已送出"


def test_review_dialog_always_exposes_submission_history(tmp_path, qtbot):
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)

    assert dialog.submission_history_button.text() == "補訓批次紀錄"
    assert dialog.submission_history_button.isEnabled() is True


def test_selected_manifest_excludes_unchecked_batch_rows(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260715" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    now = datetime.now()
    for index in range(2):
        (metadata / f"case_{index}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": now.replace(microsecond=index).isoformat(),
                    "product": "Cable1",
                    "area": "A",
                    "status": "FAIL",
                    "detector": "yolo",
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    selected_index = dialog.visible_indices[1]

    selected_manifest = dialog._write_selected_manifest({selected_index})

    with selected_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["config_snapshot_path"].endswith("case_1_config_snapshot.json")


def test_selected_manifest_can_submit_rows_outside_visible_date_range(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260715" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    for index, timestamp in enumerate(("2026-07-14T10:00:00", "2026-07-15T10:00:00")):
        (metadata / f"case_{index}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "product": "Cable1",
                    "area": "A",
                    "status": "FAIL",
                    "detector": "yolo",
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    outside_visible_index = 0
    dialog.visible_indices = [1]

    selected_manifest = dialog._write_selected_manifest(
        {outside_visible_index},
        scope_indices=[outside_visible_index],
    )

    with selected_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["timestamp"] for row in rows] == ["2026-07-14T10:00:00"]


def test_selected_queue_count_is_independent_of_visible_date_range(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260715" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    for index, timestamp in enumerate(("2026-07-14T10:00:00", "2026-07-15T10:00:00")):
        (metadata / f"case_{index}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "product": "Cable1",
                    "area": "A",
                    "status": "FAIL",
                    "detector": "yolo",
                    "detections": [],
                }
            ),
            encoding="utf-8",
        )
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    dialog.store.set_review(0, "verified_empty")
    dialog.store.set_review(1, "false_negative")
    dialog.visible_indices = [1]

    dialog._update_submit_button()

    assert dialog.batch_preview_button.text() == "待送清單（2）"

    dialog._remove_submitted_rows_from_queue({0})

    assert dialog.batch_preview_button.text() == "待送清單（1）"


def test_selected_queue_excludes_legacy_rows_already_handed_off(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260715" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    preprocessed = tmp_path / "Result" / "saved_case.png"
    preprocessed.write_bytes(b"saved-evidence")
    (metadata / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-15T10:00:00",
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "detections": [],
                "artifacts": {"preprocessed_path": str(preprocessed)},
            }
        ),
        encoding="utf-8",
    )
    training_data = tmp_path / "training-data"
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=training_data,
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    dialog.store.set_review(0, "verified_empty")
    ready_manifest = training_data / "Cable1" / "A" / "metadata" / "review_dataset_manifest.csv"
    ready_manifest.parent.mkdir(parents=True)
    with ready_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_image", "config_snapshot_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_image": dialog.store.rows[0]["preprocessed_path"],
                "config_snapshot_path": "",
            }
        )

    dialog._reconcile_handed_off_selection()
    dialog.visible_indices = [0]
    dialog._show_current()
    dialog._update_submit_button()

    assert dialog.store.rows[0]["training_selected"] == "0"
    assert dialog.reprocess_button.isHidden() is False

    qtbot.mouseClick(dialog.reprocess_button, Qt.LeftButton)

    assert dialog.store.rows[0]["training_selected"] == "1"
    assert dialog.batch_preview_button.text() == "待送清單（1）"


def test_handed_off_reconciliation_does_not_rewrite_unchanged_exclusion(
    tmp_path, monkeypatch
):
    sample = tmp_path / "sample.json"
    sample.write_text("{}", encoding="utf-8")
    store = MagicMock()
    store.rows = [
        {
            "product": "Cable1",
            "area": "A",
            "config_snapshot_path": str(sample),
            "original_path": "",
            "preprocessed_path": "",
            "annotated_path": "",
            "training_selected": "0",
        }
    ]
    host = SimpleNamespace(training_data_dir=tmp_path / "training", store=store)
    monkeypatch.setattr(
        "app.gui.review_cases_dialog._load_handed_off_artifacts",
        lambda _root, _rows: {str(sample.resolve()).replace("\\", "/").casefold()},
    )

    ReviewCasesDialog._reconcile_handed_off_selection(host)

    assert host._handed_off_indices == {0}
    store.mark_submitted_indices.assert_called_once_with({0})
    store.set_training_selection.assert_not_called()


def test_cancelled_retraining_settings_keeps_queue_and_does_not_export(
    tmp_path, qtbot, monkeypatch
):
    metadata = (
        tmp_path
        / "Result"
        / "20260715"
        / "Cable1"
        / "A"
        / "FAIL"
        / "metadata"
        / "yolo"
    )
    metadata.mkdir(parents=True)
    preprocessed = tmp_path / "Result" / "saved_case.png"
    preprocessed.write_bytes(b"saved-evidence")
    (metadata / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-15T10:00:00",
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "detections": [],
                "artifacts": {"preprocessed_path": str(preprocessed)},
            }
        ),
        encoding="utf-8",
    )
    dialog = ReviewCasesDialog(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
    )
    qtbot.addWidget(dialog)
    dialog.store.set_review(0, "verified_empty")
    exported = False

    def fail_if_exported(*_args, **_kwargs):
        nonlocal exported
        exported = True
        raise AssertionError("export must not run after settings cancellation")

    monkeypatch.setattr(
        "app.gui.review_cases_dialog.RetrainingSettingsDialog.exec_",
        lambda _self: QDialog.Rejected,
    )
    monkeypatch.setattr(
        "app.gui.review_cases_dialog.export_operator_handoff", fail_if_exported
    )

    dialog._submit_selected_indices({0})

    assert exported is False
    assert dialog.store.rows[0]["training_selected"] == "1"
