import csv
import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QDateTime, Qt
from PyQt5.QtWidgets import QDialog

from app.gui.review_cases_dialog import (
    ReviewCasesDialog,
    ReviewManifestStore,
    _find_saved_case_index,
    _is_confirmation_only_submission,
    _row_has_ordered_class_contract,
    _target_manifest_path,
    _visible_action_values,
    _with_pass_sampling,
)


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


def test_review_manifest_store_saves_each_button_decision(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_review(0, "confirmed_ng")

    reloaded = ReviewManifestStore(manifest)
    assert reloaded.rows[0]["review_label"] == "confirmed_ng"
    assert reloaded.reviewed_count() == 1


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


@pytest.mark.parametrize("review_label", ["image_quality_issue"])
def test_review_hold_decisions_are_never_selected_for_training(tmp_path, review_label):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest)
    store = ReviewManifestStore(manifest)

    store.set_review(0, review_label)

    row = ReviewManifestStore(manifest).rows[0]
    assert row["review_label"] == review_label
    assert row["training_selected"] == "0"


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


def test_review_actions_are_reduced_to_context_relevant_choices():
    assert _visible_action_values(True) == {
        "confirmed_ng",
        "false_positive",
        "wrong_class",
        "image_quality_issue",
    }
    assert _visible_action_values(False) == {
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


def test_review_dialog_records_decision_with_one_button_click(tmp_path, qtbot):
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
    )
    qtbot.addWidget(dialog)
    confirm_button = dialog.review_buttons["confirmed_ng"]

    qtbot.mouseClick(confirm_button, Qt.LeftButton)

    reloaded = ReviewManifestStore(manifest)
    assert reloaded.rows[0]["review_label"] == "confirmed_ng"
    assert "已記錄" in dialog.feedback_label.text()
    assert dialog.export_button.isEnabled() is True


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
    )
    qtbot.addWidget(dialog)
    confirm_button = dialog.review_buttons["confirmed_ng"]

    assert confirm_button.isEnabled() is False
    assert "未偵測到任何框" in dialog.details_label.text()

    qtbot.mouseClick(dialog.review_buttons["false_negative"], Qt.LeftButton)
    assert ReviewManifestStore(manifest).rows[0]["review_label"] == "false_negative"


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

    assert dialog.batch_preview_button.text() == "已選擇清單（2）"

    dialog._remove_submitted_rows_from_queue({0})

    assert dialog.batch_preview_button.text() == "已選擇清單（1）"


def test_selected_queue_excludes_legacy_rows_already_handed_off(tmp_path, qtbot):
    metadata = tmp_path / "Result" / "20260715" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata.mkdir(parents=True)
    (metadata / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-07-15T10:00:00",
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "detector": "yolo",
                "detections": [],
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
    dialog._update_submit_button()

    assert dialog.store.rows[0]["training_selected"] == "0"
    assert dialog.batch_preview_button.text() == "已選擇清單（0）"
