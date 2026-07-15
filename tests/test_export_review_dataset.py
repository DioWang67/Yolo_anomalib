import csv
import hashlib
import json
import socket
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from tools.export_review_dataset import (
    ExportedReviewItem,
    _read_image_size,
    _snapshot_yolo_label_lines,
    export_operator_handoff,
    export_review_dataset,
)


def test_operator_handoff_rejects_pending_case_without_class_contract(tmp_path):
    processed = tmp_path / "processed.jpg"
    processed.write_bytes(b"image")
    manifest = tmp_path / "review.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product",
                "area",
                "config_snapshot_path",
                "preprocessed_path",
                "detections_json",
                "class_names_json",
                "review_label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "config_snapshot_path": "case.json",
                "preprocessed_path": str(processed),
                "detections_json": "[]",
                "class_names_json": "[]",
                "review_label": "false_negative",
            }
        )
    output = tmp_path / "training-data"

    with pytest.raises(ValueError, match="完整類別順序"):
        export_operator_handoff(manifest, output)

    assert not (output / "Cable1" / "A" / "review_pending").exists()


def test_snapshot_labels_use_stored_image_dimensions_over_camera_metadata(tmp_path):
    image_path = tmp_path / "review.jpg"
    Image.new("RGB", (640, 640)).save(image_path)
    detections = json.dumps(
        [
            {
                "class_id": 3,
                "confidence": 0.9,
                "bbox": [320, 160, 384, 224],
                "image_width": 3072,
                "image_height": 2048,
            }
        ]
    )

    lines = _snapshot_yolo_label_lines(
        detections,
        image_size=_read_image_size(image_path),
    )

    assert lines == ["3 0.55000000 0.30000000 0.10000000 0.10000000"]


def test_snapshot_labels_use_verified_class_and_remove_overlapping_candidate():
    detections = json.dumps(
        [
            {
                "class_id": 2,
                "verified_class": "Green",
                "confidence": 0.8,
                "bbox": [100, 100, 130, 150],
                "image_width": 640,
                "image_height": 640,
            },
            {
                "class_id": 1,
                "verified_class": "Green",
                "confidence": 0.7,
                "bbox": [101, 100, 131, 150],
                "image_width": 640,
                "image_height": 640,
            },
        ]
    )

    lines = _snapshot_yolo_label_lines(
        detections,
        image_size=(640, 640),
        class_names=["Black", "Green", "Orange"],
    )

    assert lines == ["1 0.17968750 0.19531250 0.04687500 0.07812500"]


def test_operator_handoff_recovers_checksum_valid_legacy_contract(tmp_path):
    output = tmp_path / "training-data"
    target_manifest = (
        output
        / "Cable1"
        / "A"
        / "metadata"
        / "review_dataset_manifest.csv"
    )
    target_manifest.parent.mkdir(parents=True)
    class_names = ["Black", "Green"]
    serialized_names = json.dumps(
        class_names, ensure_ascii=False, separators=(",", ":")
    )
    class_hash = hashlib.sha256(serialized_names.encode("utf-8")).hexdigest()
    existing = ExportedReviewItem(
        source_manifest="existing.csv",
        review_label="confirmed_ng",
        review_note="",
        source_image="existing.jpg",
        output_image="existing.jpg",
        output_label="existing.txt",
        annotation_status="verified_snapshot",
        sample_id="existing",
        image_sha256="a" * 64,
        product="Cable1",
        area="A",
        timestamp="",
        status="FAIL",
        decision_reasons="",
        model_version="",
        class_names_json=serialized_names,
        class_map_json='{"0":"Black","1":"Green"}',
        class_schema_hash=class_hash,
    )
    with target_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=ExportedReviewItem.__dataclass_fields__.keys()
        )
        writer.writeheader()
        writer.writerow(asdict(existing))

    processed = tmp_path / "legacy.jpg"
    processed.write_bytes(b"legacy")
    manifest = tmp_path / "legacy_review.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product",
                "area",
                "weights",
                "config_snapshot_path",
                "preprocessed_path",
                "detections_json",
                "class_names_json",
                "class_map_json",
                "review_label",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "weights": "models/Cable1/A/yolo/weights/best.onnx",
                "config_snapshot_path": "legacy.json",
                "preprocessed_path": str(processed),
                "detections_json": "[]",
                "class_names_json": "[]",
                "class_map_json": '{"0":"Black","1":"Green"}',
                "review_label": "false_negative",
            }
        )

    report = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )

    assert report.pending_count == 1
    handoff = json.loads(report.handoff_path.read_text(encoding="utf-8"))
    assert handoff["targets"][0]["class_names"] == class_names
    assert handoff["targets"][0]["class_schema_hash"] == class_hash


def test_export_review_dataset_copies_clean_original_into_target_root(tmp_path):
    original = tmp_path / "original.png"
    original.write_bytes(b"png")
    manifest = tmp_path / "review_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "product",
                "area",
                "status",
                "detector",
                "decision_reasons",
                "model_version",
                "weights",
                "inference_time",
                "config_snapshot_path",
                "original_path",
                "annotated_path",
                "failure_crop_paths",
                "review_label",
                "review_note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "PCBA",
                "area": "TOP",
                "status": "FAIL",
                "decision_reasons": "MISSING",
                "model_version": "1.2.3",
                "original_path": str(original),
                "review_label": "false_positive",
                "review_note": "operator confirmed OK",
            }
        )

    exported = export_review_dataset(manifest, tmp_path / "dataset")

    assert len(exported) == 1
    output_image = Path(exported[0].output_image)
    assert output_image.exists()
    assert output_image.parent == tmp_path / "dataset" / "PCBA" / "TOP" / "raw" / "images"
    assert exported[0].output_label == ""
    assert exported[0].annotation_status == "pending"
    assert len(exported[0].image_sha256) == 64
    assert "false_positive_PCBA_TOP_MISSING" in output_image.name
    assert (
        tmp_path
        / "dataset"
        / "PCBA"
        / "TOP"
        / "metadata"
        / "review_dataset_manifest.csv"
    ).exists()


def test_export_review_dataset_skips_unreviewed_rows(tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"png")
    manifest = tmp_path / "review_manifest.csv"
    manifest.write_text(
        "product,area,status,decision_reasons,failure_crop_paths,review_label,review_note\n"
        f"PCBA,TOP,FAIL,MISSING,{crop},,\n",
        encoding="utf-8",
    )

    exported = export_review_dataset(
        manifest, tmp_path / "dataset", source_kind="failure_crops"
    )

    assert exported == []


def test_export_review_dataset_skips_batch_exclusions(tmp_path):
    original = tmp_path / "original.png"
    original.write_bytes(b"png")
    manifest = tmp_path / "review_manifest.csv"
    manifest.write_text(
        "product,area,original_path,review_label,training_selected\n"
        f"PCBA,TOP,{original},confirmed_ng,0\n",
        encoding="utf-8",
    )

    assert export_review_dataset(manifest, tmp_path / "dataset") == []


def test_export_review_dataset_deduplicates_identical_images(tmp_path):
    original = tmp_path / "original.png"
    original.write_bytes(b"same-image")
    manifest = tmp_path / "review_manifest.csv"
    manifest.write_text(
        "product,area,decision_reasons,original_path,review_label,review_note\n"
        f"PCBA,B,MISSING,{original},false_negative,first\n"
        f"PCBA,B,MISSING,{original},false_negative,duplicate\n",
        encoding="utf-8",
    )

    exported = export_review_dataset(manifest, tmp_path / "dataset")

    assert len(exported) == 1


def test_export_review_dataset_deduplicates_within_each_target(tmp_path):
    """Identical pixels in two products must not silently drop one product."""
    original = tmp_path / "original.png"
    original.write_bytes(b"same-image")
    manifest = tmp_path / "review_manifest.csv"
    manifest.write_text(
        "product,area,decision_reasons,original_path,review_label,review_note\n"
        f"PCBA,A,MISSING,{original},false_negative,first\n"
        f"Cable,A,MISSING,{original},false_negative,second\n",
        encoding="utf-8",
    )

    exported = export_review_dataset(manifest, tmp_path / "dataset")

    assert len(exported) == 2
    assert {item.product for item in exported} == {"PCBA", "Cable"}


def test_operator_handoff_exports_verified_boxes_and_routes_missed_cases(tmp_path):
    processed = tmp_path / "processed.jpg"
    processed_missed = tmp_path / "processed_missed.jpg"
    processed_false = tmp_path / "processed_false.jpg"
    original = tmp_path / "original.jpg"
    processed.write_bytes(b"processed")
    processed_missed.write_bytes(b"processed-missed")
    processed_false.write_bytes(b"processed-false")
    original.write_bytes(b"original")
    detections = [
        {
            "class_id": 0,
            "confidence": 0.95,
            "bbox": [10, 20, 30, 60],
            "image_width": 100,
            "image_height": 100,
        },
        {
            "class_id": 1,
            "confidence": 0.40,
            "bbox": [10, 20, 30, 60],
            "image_width": 100,
            "image_height": 100,
        },
    ]
    manifest = tmp_path / "review.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product",
                "area",
                "status",
                "decision_reasons",
                "model_version",
                "config_snapshot_path",
                "original_path",
                "preprocessed_path",
                "detections_json",
                "class_names_json",
                "class_map_json",
                "review_label",
                "review_note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "config_snapshot_path": "one.json",
                "original_path": str(original),
                "preprocessed_path": str(processed),
                "detections_json": json.dumps(detections),
                "review_label": "confirmed_ng",
            }
        )
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "config_snapshot_path": "two.json",
                "original_path": str(original),
                "preprocessed_path": str(processed_missed),
                "detections_json": "[]",
                "class_names_json": json.dumps(["Black", "Green"]),
                "review_label": "false_negative",
            }
        )
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "config_snapshot_path": "three.json",
                "original_path": str(original),
                "preprocessed_path": str(processed_false),
                "detections_json": json.dumps(detections),
                "class_names_json": json.dumps(["Black", "Green"]),
                "review_label": "false_positive",
            }
        )

    output = tmp_path / "training-data"
    report = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )
    duplicate = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )

    assert report.ready_count == 1
    assert report.pending_count == 2
    assert report.targets == (("Cable1", "A"),)
    labels = list((output / "Cable1" / "A" / "raw" / "labels").glob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text(encoding="utf-8").startswith("0 0.20000000")
    pending_manifest = output / "Cable1" / "A" / "review_pending" / "manifest.csv"
    assert pending_manifest.exists()
    with pending_manifest.open("r", encoding="utf-8", newline="") as handle:
        pending_rows = list(csv.DictReader(handle))
    assert len(pending_rows) == 2
    assert {row["reason"] for row in pending_rows} == {
        "false_detection_requires_correction",
        "missed_detection_requires_box_annotation",
    }
    pending_by_reason = {row["reason"]: row for row in pending_rows}
    correction_row = pending_by_reason["false_detection_requires_correction"]
    assert len(correction_row["label_baseline_sha256"]) == 64
    assert pending_by_reason["missed_detection_requires_box_annotation"][
        "label_baseline_sha256"
    ] == "missing"
    assert {
        Path(row["output_image"]).read_bytes() for row in pending_rows
    } == {b"processed-missed", b"processed-false"}
    handoff = json.loads(report.handoff_path.read_text(encoding="utf-8"))
    assert handoff["schema_version"] == 3
    assert handoff["job_id"] == report.job_id
    assert report.handoff_path.name == "handoff.json"
    assert report.handoff_path.parent.name == report.job_id
    assert report.status_path == report.handoff_path.parent / "status.json"
    status = json.loads(report.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "waiting_annotation"
    assert status["pending_count"] == 2
    assert duplicate.reused_existing is True
    assert duplicate.handoff_path == report.handoff_path
    assert handoff["ready_count"] == 1
    with (
        output / "Cable1" / "A" / "metadata" / "review_dataset_manifest.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_operator_handoff_replaces_stale_non_terminal_job(tmp_path, monkeypatch):
    processed = tmp_path / "processed.jpg"
    processed.write_bytes(b"image")
    manifest = tmp_path / "review.csv"
    manifest.write_text(
        "product,area,config_snapshot_path,preprocessed_path,detections_json,"
        "class_names_json,review_label\n"
        f'Cable1,A,case.json,{processed},[],"[""Black""]",false_negative\n',
        encoding="utf-8",
    )
    output = tmp_path / "training-data"
    first = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )
    status = json.loads(first.status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": "waiting_annotation",
            "training_process_id": 42424242,
            "training_process_host": socket.gethostname(),
        }
    )
    first.status_path.write_text(json.dumps(status), encoding="utf-8")

    def report_dead_process(_process_id, _signal):
        raise OSError("process is not running")

    monkeypatch.setattr(
        "tools.export_review_dataset.os.kill", report_dead_process
    )

    replacement = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )

    assert replacement.reused_existing is False
    assert replacement.job_id != first.job_id


def test_operator_handoff_reclassification_revokes_previous_raw_sample(tmp_path):
    processed = tmp_path / "processed.jpg"
    processed.write_bytes(b"same-sample")
    manifest = tmp_path / "review.csv"
    fields = [
        "product",
        "area",
        "config_snapshot_path",
        "preprocessed_path",
        "detections_json",
        "class_names_json",
        "review_label",
    ]
    detection = json.dumps(
        [
            {
                "class_id": 0,
                "class": "Black",
                "confidence": 0.9,
                "bbox": [1, 1, 8, 8],
                "image_width": 10,
                "image_height": 10,
            }
        ]
    )

    def write_review(label: str) -> None:
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "product": "Cable1",
                    "area": "A",
                    "config_snapshot_path": "case.json",
                    "preprocessed_path": str(processed),
                    "detections_json": detection,
                    "class_names_json": json.dumps(["Black"]),
                    "review_label": label,
                }
            )

    output = tmp_path / "training-data"
    write_review("confirmed_ng")
    export_operator_handoff(manifest, output, inference_models_dir=tmp_path / "models")
    assert len(list((output / "Cable1" / "A" / "raw" / "images").glob("*"))) == 1

    write_review("false_positive")
    report = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )

    assert report.total_ready_count == 0
    assert list((output / "Cable1" / "A" / "raw" / "images").glob("*")) == []
    assert len(list(csv.DictReader(
        (output / "Cable1" / "A" / "review_pending" / "manifest.csv").open(
            "r", encoding="utf-8"
        )
    ))) == 1


def test_operator_handoff_batch_exclusion_revokes_previous_sample(tmp_path):
    processed = tmp_path / "processed.jpg"
    processed.write_bytes(b"same-sample")
    manifest = tmp_path / "review.csv"
    fields = [
        "product",
        "area",
        "config_snapshot_path",
        "preprocessed_path",
        "detections_json",
        "class_names_json",
        "review_label",
        "training_selected",
    ]
    detection = json.dumps(
        [
            {
                "class_id": 0,
                "class": "Black",
                "confidence": 0.9,
                "bbox": [1, 1, 8, 8],
                "image_width": 10,
                "image_height": 10,
            }
        ]
    )

    def write_review(selected: str) -> None:
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "product": "Cable1",
                    "area": "A",
                    "config_snapshot_path": "case.json",
                    "preprocessed_path": str(processed),
                    "detections_json": detection,
                    "class_names_json": json.dumps(["Black"]),
                    "review_label": "confirmed_ng",
                    "training_selected": selected,
                }
            )

    output = tmp_path / "training-data"
    write_review("1")
    export_operator_handoff(manifest, output)
    assert len(list((output / "Cable1" / "A" / "raw" / "images").glob("*"))) == 1

    write_review("0")
    report = export_operator_handoff(manifest, output)

    assert report.total_ready_count == 0
    assert list((output / "Cable1" / "A" / "raw" / "images").glob("*")) == []


def test_operator_handoff_holds_uncertain_and_bad_quality_images(tmp_path):
    uncertain = tmp_path / "uncertain.jpg"
    bad_quality = tmp_path / "overexposed.jpg"
    uncertain.write_bytes(b"uncertain")
    bad_quality.write_bytes(b"overexposed")
    manifest = tmp_path / "review.csv"
    manifest.write_text(
        "product,area,config_snapshot_path,preprocessed_path,review_label,training_selected\n"
        f"Cable1,A,one.json,{uncertain},uncertain,1\n"
        f"Cable1,A,two.json,{bad_quality},image_quality_issue,1\n",
        encoding="utf-8",
    )

    report = export_operator_handoff(manifest, tmp_path / "training-data")

    assert report.ready_count == 0
    assert report.pending_count == 0
    assert report.targets == ()


def test_operator_handoff_verified_empty_creates_explicit_negative_label(tmp_path):
    processed = tmp_path / "background.jpg"
    processed.write_bytes(b"background")
    manifest = tmp_path / "review.csv"
    manifest.write_text(
        "product,area,config_snapshot_path,preprocessed_path,detections_json,class_names_json,review_label\n"
        f'Cable1,A,case.json,{processed},[],"[""Black""]",verified_empty\n',
        encoding="utf-8",
    )

    output = tmp_path / "training-data"
    report = export_operator_handoff(
        manifest, output, inference_models_dir=tmp_path / "models"
    )

    assert report.ready_count == 1
    label = next((output / "Cable1" / "A" / "raw" / "labels").glob("*.txt"))
    assert label.read_text(encoding="utf-8") == ""
