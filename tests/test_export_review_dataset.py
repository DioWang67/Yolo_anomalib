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
    _index_ready_items_by_content,
    _read_image_size,
    _sample_id,
    _snapshot_yolo_label_lines,
    _write_export_manifest,
    export_operator_handoff,
    export_review_dataset,
)
from tools.review_repair import (
    apply_repair_plan,
    generate_repair_plan,
    rollback_repair,
    write_repair_plan,
)


def _save_test_image(
    path: Path,
    *,
    size: tuple[int, int] = (100, 100),
    color: tuple[int, int, int] = (20, 40, 60),
) -> None:
    Image.new("RGB", size, color).save(path)


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


def test_operator_handoff_blocks_inconsistent_review_before_artifact_write(tmp_path):
    manifest = tmp_path / "review.csv"
    manifest.write_text(
        "sample_id,product,area,review_selected,review_outcome,review_label,"
        "product_verdict,detection_verdict,color_verdict,action_route,"
        "training_selected\n"
        "contradiction,Cable1,A,1,fail,confirmed_ng,ok,correct,"
        "not_applicable,yolo,1\n",
        encoding="utf-8",
    )
    output = tmp_path / "training-data"

    with pytest.raises(ValueError, match="product_ok_confirmed_ng"):
        export_operator_handoff(manifest, output)

    assert not (output / ".operator_handoff" / "jobs").exists()


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


def test_snapshot_labels_reverse_letterbox_into_original_camera_frame():
    detections = json.dumps(
        [
            {
                "class_id": 0,
                "confidence": 0.95,
                "bbox": [160.0, 213.5, 320.0, 320.0],
                "image_width": 3072,
                "image_height": 2048,
            }
        ]
    )

    lines = _snapshot_yolo_label_lines(
        detections,
        image_size=(3072, 2048),
        detection_image_size=(640, 640),
    )

    assert lines == ["0 0.37500000 0.37500000 0.25000000 0.25000000"]


def test_snapshot_labels_drop_box_that_is_entirely_in_letterbox_padding():
    detections = json.dumps(
        [
            {
                "class_id": 0,
                "confidence": 0.95,
                "bbox": [100, 20, 200, 80],
                "image_width": 3072,
                "image_height": 2048,
            }
        ]
    )

    assert _snapshot_yolo_label_lines(
        detections,
        image_size=(3072, 2048),
        detection_image_size=(640, 640),
    ) == []


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


def test_legacy_duplicate_prefers_canonical_operator_annotation(tmp_path):
    image_sha256 = "a" * 64
    canonical_id = _sample_id("Cable1", "A", image_sha256)
    operator_label = tmp_path / "operator.txt"
    snapshot_label = tmp_path / "snapshot.txt"
    operator_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    snapshot_label.write_text("0 0.4 0.5 0.2 0.2\n", encoding="utf-8")

    def item(sample_id, annotation_status, output_label):
        return ExportedReviewItem(
            source_manifest="review.csv",
            review_label="false_positive",
            review_note="",
            source_image="source.jpg",
            output_image=str(tmp_path / f"review_{sample_id}.jpg"),
            output_label=str(output_label),
            annotation_status=annotation_status,
            sample_id=sample_id,
            image_sha256=image_sha256,
            product="Cable1",
            area="A",
            timestamp="",
            status="FAIL",
            decision_reasons="",
            model_version="",
            class_names_json="[]",
            class_map_json="{}",
            class_schema_hash="",
        )

    operator_item = item(canonical_id, "verified_annotation", operator_label)
    legacy_item = item("legacy-snapshot-id", "verified_snapshot", snapshot_label)

    audit_records = []
    indexed, superseded = _index_ready_items_by_content(
        [operator_item, legacy_item], audit_records=audit_records
    )

    assert indexed == {canonical_id: operator_item}
    assert superseded == [legacy_item]
    assert len(audit_records) == 1
    assert audit_records[0].image_sha256 == image_sha256
    assert audit_records[0].kept_sample == canonical_id
    assert audit_records[0].excluded_sample == "legacy-snapshot-id"
    assert audit_records[0].reason == "human_annotation_over_ai_snapshot"
    assert audit_records[0].kept_source_type == "human_annotation"
    assert audit_records[0].excluded_source_type == "ai_snapshot"
    assert len(audit_records[0].kept_label_sha256) == 64
    assert len(audit_records[0].excluded_label_sha256) == 64


def test_two_human_annotations_with_different_labels_are_blocked(tmp_path):
    image_sha256 = "b" * 64
    canonical_id = _sample_id("Cable1", "A", image_sha256)
    first_label = tmp_path / "first.txt"
    second_label = tmp_path / "second.txt"
    first_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    second_label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def item(sample_id, label, status="verified_annotation"):
        return ExportedReviewItem(
            source_manifest="review.csv",
            review_label="wrong_class",
            review_note="",
            source_image="source.jpg",
            output_image=str(tmp_path / f"review_{sample_id}.jpg"),
            output_label=str(label),
            annotation_status=status,
            sample_id=sample_id,
            image_sha256=image_sha256,
            product="Cable1",
            area="A",
            timestamp="",
            status="FAIL",
            decision_reasons="",
            model_version="",
            class_names_json="[]",
            class_map_json="{}",
            class_schema_hash="",
        )

    with pytest.raises(ValueError, match="automatic canonical selection is forbidden"):
        _index_ready_items_by_content(
            [item(canonical_id, first_label), item("older-human", second_label)]
        )
    with pytest.raises(ValueError, match="automatic canonical selection is forbidden"):
        _index_ready_items_by_content(
            [item(canonical_id, first_label), item("legacy", second_label, "")]
        )


def test_identical_human_and_snapshot_labels_keep_human_with_audit(tmp_path):
    image_sha256 = "c" * 64
    label = tmp_path / "same.txt"
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def item(sample_id, status):
        return ExportedReviewItem(
            source_manifest="review.csv",
            review_label="confirmed_ng",
            review_note="",
            source_image="source.jpg",
            output_image=str(tmp_path / f"review_{sample_id}.jpg"),
            output_label=str(label),
            annotation_status=status,
            sample_id=sample_id,
            image_sha256=image_sha256,
            product="Cable1",
            area="A",
            timestamp="",
            status="FAIL",
            decision_reasons="",
            model_version="",
            class_names_json="[]",
            class_map_json="{}",
            class_schema_hash="",
        )

    human = item("human-record", "verified_annotation")
    snapshot = item(_sample_id("Cable1", "A", image_sha256), "verified_snapshot")
    audit_records = []

    indexed, _superseded = _index_ready_items_by_content(
        [snapshot, human], audit_records=audit_records
    )

    assert next(iter(indexed.values())) is human
    assert audit_records[0].reason == "identical_label_prefer_human_over_ai"


def test_operator_handoff_exports_verified_boxes_and_routes_missed_cases(tmp_path):
    processed = tmp_path / "processed.jpg"
    processed_missed = tmp_path / "processed_missed.jpg"
    processed_false = tmp_path / "processed_false.jpg"
    processed_wrong_box = tmp_path / "processed_wrong_box.jpg"
    original = tmp_path / "original.jpg"
    original_missed = tmp_path / "original_missed.jpg"
    original_false = tmp_path / "original_false.jpg"
    original_wrong_box = tmp_path / "original_wrong_box.jpg"
    _save_test_image(processed, color=(1, 1, 1))
    _save_test_image(processed_missed, color=(2, 2, 2))
    _save_test_image(processed_false, color=(3, 3, 3))
    _save_test_image(processed_wrong_box, color=(4, 4, 4))
    _save_test_image(original, color=(11, 11, 11))
    _save_test_image(original_missed, color=(12, 12, 12))
    _save_test_image(original_false, color=(13, 13, 13))
    _save_test_image(original_wrong_box, color=(14, 14, 14))
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
                "original_path": str(original_missed),
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
                "original_path": str(original_false),
                "preprocessed_path": str(processed_false),
                "detections_json": json.dumps(detections),
                "class_names_json": json.dumps(["Black", "Green"]),
                "review_label": "false_positive",
            }
        )
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "status": "FAIL",
                "config_snapshot_path": "four.json",
                "original_path": str(original_wrong_box),
                "preprocessed_path": str(processed_wrong_box),
                "detections_json": json.dumps(detections),
                "class_names_json": json.dumps(["Black", "Green"]),
                "review_label": "wrong_box",
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
    assert report.pending_count == 3
    assert report.targets == (("Cable1", "A"),)
    labels = list((output / "Cable1" / "A" / "raw" / "labels").glob("*.txt"))
    assert len(labels) == 1
    assert labels[0].read_text(encoding="utf-8").startswith("0 0.20000000")
    pending_manifest = output / "Cable1" / "A" / "review_pending" / "manifest.csv"
    assert pending_manifest.exists()
    with pending_manifest.open("r", encoding="utf-8", newline="") as handle:
        pending_rows = list(csv.DictReader(handle))
    assert len(pending_rows) == 3
    assert {row["reason"] for row in pending_rows} == {
        "box_geometry_requires_correction",
        "false_detection_requires_correction",
        "missed_detection_requires_box_annotation",
    }
    pending_by_reason = {row["reason"]: row for row in pending_rows}
    correction_row = pending_by_reason["false_detection_requires_correction"]
    assert len(correction_row["label_baseline_sha256"]) == 64
    assert len(pending_by_reason["box_geometry_requires_correction"]["label_baseline_sha256"]) == 64
    assert pending_by_reason["missed_detection_requires_box_annotation"][
        "label_baseline_sha256"
    ] == "missing"
    assert {row["source_image"] for row in pending_rows} == {
        str(original_missed),
        str(original_false),
        str(original_wrong_box),
    }
    handoff = json.loads(report.handoff_path.read_text(encoding="utf-8"))
    assert handoff["schema_version"] == 4
    assert handoff["training_options"] == {
        "epochs": 20,
        "augmentations_per_image": 20,
        "batch": 8,
        "imgsz": 640,
    }
    assert handoff["job_id"] == report.job_id
    assert report.handoff_path.name == "handoff.json"
    assert report.handoff_path.parent.name == report.job_id
    assert report.status_path == report.handoff_path.parent / "status.json"
    status = json.loads(report.status_path.read_text(encoding="utf-8"))
    assert status["state"] == "waiting_annotation"
    assert status["pending_count"] == 3
    assert duplicate.reused_existing is True
    assert duplicate.handoff_path == report.handoff_path
    assert handoff["ready_count"] == 1
    with (
        output / "Cable1" / "A" / "metadata" / "review_dataset_manifest.csv"
    ).open("r", encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_operator_handoff_blocks_conflicting_same_image_before_production_write(
    tmp_path,
):
    processed = tmp_path / "same-image.jpg"
    _save_test_image(processed, size=(100, 100))
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
    detections = [
        [
            {
                "class_id": 0,
                "confidence": 0.9,
                "bbox": [10, 10, 30, 30],
                "image_width": 100,
                "image_height": 100,
            }
        ],
        [
            {
                "class_id": 1,
                "confidence": 0.9,
                "bbox": [50, 50, 80, 80],
                "image_width": 100,
                "image_height": 100,
            }
        ],
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, snapshot in enumerate(detections, start=1):
            writer.writerow(
                {
                    "product": "Cable1",
                    "area": "A",
                    "config_snapshot_path": f"case-{index}.json",
                    "preprocessed_path": str(processed),
                    "detections_json": json.dumps(snapshot),
                    "class_names_json": json.dumps(["Black", "Green"]),
                    "review_label": "confirmed_ng",
                }
            )

    output = tmp_path / "training-data"
    with pytest.raises(ValueError, match="Conflict report"):
        export_operator_handoff(manifest, output)

    assert not (output / "Cable1" / "A" / "raw").exists()
    conflict_reports = list(
        (output / ".operator_handoff" / "conflict_reports").glob("*.json")
    )
    assert len(conflict_reports) == 1
    report = json.loads(conflict_reports[0].read_text(encoding="utf-8"))
    assert report["mutation_performed"] is False
    assert report["conflicts"][0]["reason"] == (
        "conflicting_equally_authoritative_labels"
    )
    assert len(report["conflicts"][0]["image_sha256"]) == 64
    assert len(set(report["conflicts"][0]["label_sha256s"])) == 2
    assert report["conflicts"][0]["normalized_labels"] == [
        ["0 0.20000000 0.20000000 0.20000000 0.20000000"],
        ["1 0.65000000 0.65000000 0.30000000 0.30000000"],
    ]


def test_pending_reannotation_archives_conflicting_active_human_labels(tmp_path):
    source = tmp_path / "same-image.jpg"
    _save_test_image(source, size=(100, 100))
    image_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "training-data"
    target = output / "Cable1" / "A"
    ready_path = target / "metadata" / "review_dataset_manifest.csv"
    old_items = []
    old_label_texts = (
        "0 0.20 0.20 0.20 0.20\n",
        "1 0.65 0.65 0.30 0.30\n",
    )
    for index, label_text in enumerate(old_label_texts, start=1):
        output_image = target / "raw" / "images" / f"legacy-{index}.jpg"
        output_label = target / "raw" / "labels" / f"legacy-{index}.txt"
        output_image.parent.mkdir(parents=True, exist_ok=True)
        output_label.parent.mkdir(parents=True, exist_ok=True)
        output_image.write_bytes(source.read_bytes())
        output_label.write_text(label_text, encoding="utf-8")
        old_items.append(
            ExportedReviewItem(
                source_manifest="old-review.csv",
                review_label="false_positive",
                review_note="",
                source_image=str(source),
                output_image=str(output_image),
                output_label=str(output_label),
                annotation_status="verified_annotation",
                sample_id=f"legacy-{index}",
                image_sha256=image_sha,
                product="Cable1",
                area="A",
                timestamp="2026-07-13T14:32:39",
                status="DETECTION_FAIL",
                decision_reasons="UNEXPECTED_COMPONENT",
                model_version="",
                class_names_json=json.dumps(["Black", "Green"]),
                class_map_json=json.dumps({"0": "Black", "1": "Green"}),
                class_schema_hash="schema",
            )
        )
    _write_export_manifest(old_items, ready_path)
    manifest = tmp_path / "review.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "product",
                "area",
                "config_snapshot_path",
                "original_path",
                "preprocessed_path",
                "detections_json",
                "class_names_json",
                "review_label",
                "training_selected",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "Cable1",
                "area": "A",
                "config_snapshot_path": "current-review.json",
                "original_path": str(source),
                "preprocessed_path": str(source),
                "detections_json": "[]",
                "class_names_json": json.dumps(["Black", "Green"]),
                "review_label": "false_positive",
                "training_selected": "1",
            }
        )

    handoff = export_operator_handoff(manifest, output)

    assert handoff.pending_count == 1
    assert handoff.total_ready_count == 0
    assert not list((target / "raw" / "images").glob("*"))
    assert not list((target / "raw" / "labels").glob("*"))
    assert not list((output / ".operator_handoff" / "conflict_reports").glob("*.json"))
    archive_files = list(
        (target / ".operator_handoff" / "superseded_ready").glob(
            "*/archive.json"
        )
    )
    assert len(archive_files) == 1
    archive = json.loads(archive_files[0].read_text(encoding="utf-8"))
    assert archive["status"] == "committed"
    assert archive["old_annotations_preserved"] is True
    assert len(archive["records"]) == 2
    archived_labels = {
        (archive_files[0].parent / record["archived_files"]["output_label"])
        .read_text(encoding="utf-8")
        for record in archive["records"]
    }
    assert archived_labels == set(old_label_texts)


def test_approved_human_conflict_selection_becomes_auditable_canonical_rule(
    tmp_path,
):
    processed = tmp_path / "same-image.jpg"
    _save_test_image(processed, size=(100, 100))
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
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (class_id, bbox) in enumerate(
            ((0, [10, 10, 30, 30]), (1, [50, 50, 80, 80])),
            start=1,
        ):
            writer.writerow(
                {
                    "product": "Cable1",
                    "area": "A",
                    "config_snapshot_path": f"case-{index}.json",
                    "preprocessed_path": str(processed),
                    "detections_json": json.dumps(
                        [
                            {
                                "class_id": class_id,
                                "confidence": 0.9,
                                "bbox": bbox,
                                "image_width": 100,
                                "image_height": 100,
                            }
                        ]
                    ),
                    "class_names_json": json.dumps(["Black", "Green"]),
                    "review_label": "confirmed_ng",
                }
            )
    output = tmp_path / "training-data"
    with pytest.raises(ValueError, match="Conflict report"):
        export_operator_handoff(manifest, output)
    conflict_report = next(
        (output / ".operator_handoff" / "conflict_reports").glob("*.json")
    )
    plan = generate_repair_plan(manifest, conflict_report_paths=[conflict_report])
    annotation = next(
        proposal
        for proposal in plan["proposals"]
        if proposal["target_kind"] == "annotation_conflict"
    )
    selected_sample = annotation["conflicting_sample_ids"][0]
    for proposal in plan["proposals"]:
        proposal["reviewer"] = "reviewer"
        proposal["decision_reason"] = "Reviewed original production evidence"
        if proposal is annotation:
            proposal["approval_status"] = "approved"
            proposal["approved"] = True
            proposal["revision_reason"] = "Resolve submitted label conflict"
            proposal["resolution"] = {
                "mode": "selected_sample",
                "selected_sample_id": selected_sample,
                "new_annotation_path": "",
                "new_label_sha256": "",
            }
        else:
            proposal["approval_status"] = "rejected"
    plan_path = tmp_path / "repair-plan.json"
    write_repair_plan(plan, plan_path)
    apply_repair_plan(plan_path)

    result = export_operator_handoff(manifest, output)

    assert result.ready_count == 1
    ready_manifest = output / "Cable1" / "A" / "metadata" / "review_dataset_manifest.csv"
    with ready_manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["sample_id"] for row in rows] == [selected_sample.split("@", 1)[0]]
    assert Path(rows[0]["output_label"]).read_text(encoding="utf-8").startswith("0 ")
    audit_path = max(
        (output / ".operator_handoff" / "deduplication_audits").glob("*.json")
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["records"][0]["reason"] == "approved_human_canonical_selection"

    applied_report = (
        manifest.parent
        / ".review_repairs"
        / "reports"
        / f"{plan['plan_id']}.json"
    )
    rollback_repair(applied_report)
    with pytest.raises(ValueError, match="Conflict report"):
        export_operator_handoff(manifest, output)


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
    _save_test_image(processed, size=(10, 10))
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
    _save_test_image(processed, size=(10, 10))
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


def test_operator_handoff_blocks_selected_uncertain_and_bad_quality_images(tmp_path):
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

    output = tmp_path / "training-data"
    with pytest.raises(
        ValueError,
        match="manual_review_record_in_handoff.*excluded_record_in_handoff",
    ):
        export_operator_handoff(manifest, output)

    assert not (output / ".operator_handoff" / "jobs").exists()


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
        manifest,
        output,
        inference_models_dir=tmp_path / "models",
        training_options={
            "epochs": 60,
            "augmentations_per_image": 12,
            "batch": 4,
            "imgsz": 960,
        },
    )

    assert report.ready_count == 1
    handoff = json.loads(report.handoff_path.read_text(encoding="utf-8"))
    assert handoff["training_options"] == {
        "epochs": 60,
        "augmentations_per_image": 12,
        "batch": 4,
        "imgsz": 960,
    }
    label = next((output / "Cable1" / "A" / "raw" / "labels").glob("*.txt"))
    assert label.read_text(encoding="utf-8") == ""
