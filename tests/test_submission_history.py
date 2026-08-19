from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QMessageBox

from app.gui.submission_history_dialog import SubmissionHistoryDialog
from tools.submission_history import (
    DuplicateTrainingBatchVersionError,
    ensure_training_batch_version_available,
    load_submission_entries,
    load_submission_history,
    record_submission_history,
    suggest_next_training_batch_version,
)


def _write_review_manifest(path: Path, *, label: str = "confirmed_ng") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product",
                "area",
                "timestamp",
                "review_label",
                "annotated_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "timestamp": "2026-07-16T10:00:00",
                "review_label": label,
                "annotated_path": "preview.jpg",
            }
        )


def test_submission_history_is_idempotent_and_keeps_manifest_snapshot(tmp_path):
    manifest = tmp_path / "selected.csv"
    _write_review_manifest(manifest)
    data_root = tmp_path / "data"

    first = record_submission_history(
        data_root,
        manifest,
        action="direct",
        product="Cable1",
        area="A",
        case_count=1,
        ready_count=1,
        job_id="job-1",
    )
    second = record_submission_history(
        data_root,
        manifest,
        action="direct",
        product="Cable1",
        area="A",
        case_count=1,
        ready_count=1,
        job_id="job-1",
    )

    assert first.submission_id == second.submission_id
    assert len(load_submission_history(data_root)) == 1
    assert first.manifest_path is not None
    assert first.manifest_path.read_bytes() == manifest.read_bytes()
    assert load_submission_entries(first)[0][1]["review_label"] == "confirmed_ng"


def test_submission_history_accepts_portable_package_action(tmp_path):
    manifest = tmp_path / "selected.csv"
    _write_review_manifest(manifest)

    record = record_submission_history(
        tmp_path / "data",
        manifest,
        action="portable",
        product="Cable1",
        area="A",
        case_count=1,
    )

    assert record.action == "portable"


def test_versioned_submission_batches_keep_independent_samples_and_settings(
    tmp_path,
):
    data_root = tmp_path / "data"
    first_manifest = tmp_path / "selected-v1.csv"
    second_manifest = tmp_path / "selected-v2.csv"
    _write_review_manifest(first_manifest, label="confirmed_ng")
    _write_review_manifest(second_manifest, label="verified_empty")
    options = {
        "epochs": 60,
        "augmentations_per_image": 5,
        "batch": 4,
        "imgsz": 960,
        "position_training_mode": "yolo_only",
        "position_activation": "preserve",
    }

    first = record_submission_history(
        data_root,
        first_manifest,
        action="direct",
        product="Cable1",
        area="A",
        case_count=1,
        ready_count=1,
        job_id="job-v1",
        batch_version="Cable1_A_v0.0.1",
        training_options=options,
    )
    second = record_submission_history(
        data_root,
        second_manifest,
        action="direct",
        product="Cable1",
        area="A",
        case_count=1,
        ready_count=1,
        job_id="job-v2",
        batch_version="Cable1_A_v0.0.2",
        training_options={**options, "epochs": 80},
    )

    assert first.submission_id != second.submission_id
    assert dict(first.training_options)["epochs"] == 60
    assert dict(second.training_options)["epochs"] == 80
    assert load_submission_entries(first)[0][1]["review_label"] == "confirmed_ng"
    assert load_submission_entries(second)[0][1]["review_label"] == "verified_empty"
    assert (
        suggest_next_training_batch_version(
            data_root,
            product="Cable1",
            area="A",
        )
        == "Cable1_A_v0.0.3"
    )


def test_training_batch_version_cannot_be_reused_for_different_samples(tmp_path):
    data_root = tmp_path / "data"
    first_manifest = tmp_path / "selected-v1.csv"
    replacement_manifest = tmp_path / "replacement-v1.csv"
    _write_review_manifest(first_manifest, label="confirmed_ng")
    _write_review_manifest(replacement_manifest, label="verified_empty")
    record_submission_history(
        data_root,
        first_manifest,
        action="direct",
        product="Cable1",
        area="A",
        case_count=1,
        job_id="job-v1",
        batch_version="Cable1_A_v0.0.1",
    )

    with pytest.raises(DuplicateTrainingBatchVersionError):
        ensure_training_batch_version_available(
            data_root,
            product="Cable1",
            area="A",
            batch_version="Cable1_A_v0.0.1",
        )
    with pytest.raises(DuplicateTrainingBatchVersionError):
        record_submission_history(
            data_root,
            replacement_manifest,
            action="direct",
            product="Cable1",
            area="A",
            case_count=1,
            job_id="job-replacement",
            batch_version="Cable1_A_v0.0.1",
        )


def test_submission_history_reconstructs_legacy_job_images(tmp_path):
    data_root = tmp_path / "data"
    job_dir = data_root / ".operator_handoff" / "jobs" / "job-old"
    target_root = job_dir / "dataset" / "Cable1" / "A"
    metadata_dir = target_root / "metadata"
    image_dir = target_root / "raw" / "images"
    metadata_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    image = image_dir / "review_sample-1.jpg"
    image.write_bytes(b"image")
    (job_dir / "handoff.json").write_text(
        json.dumps(
            {
                "job_id": "job-old",
                "created_at": "2026-07-16T10:00:00+00:00",
                "targets": [
                    {
                        "product": "Cable1",
                        "area": "A",
                        "ready_count": 1,
                        "sample_ids": ["sample-1"],
                        "pending_sample_ids": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with (metadata_dir / "review_dataset_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "review_label",
                "timestamp",
                "source_image",
                "output_image",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample-1",
                "review_label": "confirmed_ng",
                "timestamp": "2026-07-15T09:00:00",
                "source_image": "source.jpg",
                "output_image": str(tmp_path / "missing" / image.name),
            }
        )

    record = load_submission_history(data_root)[0]
    entries = load_submission_entries(record)

    assert record.source_type == "legacy_job"
    assert record.case_count == 1
    assert entries[0][1]["annotated_path"] == str(image)


def test_submission_history_dialog_lists_batches(tmp_path, qtbot):
    manifest = tmp_path / "selected.csv"
    _write_review_manifest(manifest)
    data_root = tmp_path / "data"
    record_submission_history(
        data_root,
        manifest,
        action="annotation",
        product="Cable1",
        area="A",
        case_count=1,
        pending_count=1,
        job_id="job-annotation",
        batch_version="Cable1_A_v0.0.1",
        training_options={"epochs": 60, "batch": 4, "imgsz": 960},
    )

    dialog = SubmissionHistoryDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 1).text() == "Cable1_A_v0.0.1"
    assert dialog.table.item(0, 2).text() == "補標後訓練"
    assert dialog.table.item(0, 5).text() == "1"
    assert "Epochs 60" in dialog.details_label.text()
    assert dialog.open_button.isEnabled() is True


def test_submission_history_dialog_can_clear_record_without_deleting_manifest(
    tmp_path, qtbot, monkeypatch
):
    manifest = tmp_path / "selected.csv"
    _write_review_manifest(manifest)
    data_root = tmp_path / "data"
    record = record_submission_history(
        data_root,
        manifest,
        action="direct",
        product="Cable1",
        area="A",
        case_count=1,
        ready_count=1,
        job_id="job-direct",
    )
    monkeypatch.setattr(
        "app.gui.submission_history_dialog.QMessageBox.question",
        lambda *_args, **_kwargs: QMessageBox.Yes,
    )
    dialog = SubmissionHistoryDialog(data_root=data_root, language="zh_TW")
    qtbot.addWidget(dialog)

    dialog._clear_selected_record()

    assert dialog.table.rowCount() == 0
    assert load_submission_history(data_root) == []
    assert record.manifest_path is not None
    assert record.manifest_path.is_file()
