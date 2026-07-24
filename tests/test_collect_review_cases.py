import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.collect_review_cases import (
    ReviewManifestReadError,
    collect_review_cases,
    write_manifest,
)


def test_collect_review_cases_finds_fail_snapshot_artifacts(tmp_path):
    base = tmp_path / "Result" / "20260518" / "PCBA" / "TOP" / "FAIL"
    metadata_dir = base / "metadata" / "yolo"
    annotated_dir = base / "annotated" / "yolo"
    original_dir = base / "original" / "yolo"
    cropped_dir = base / "cropped" / "yolo"
    metadata_dir.mkdir(parents=True)
    annotated_dir.mkdir(parents=True)
    original_dir.mkdir(parents=True)
    cropped_dir.mkdir(parents=True)

    stem = "yolo_PCBA_TOP_123456"
    snapshot_path = metadata_dir / f"{stem}_config_snapshot.json"
    annotated_path = annotated_dir / f"{stem}.jpg"
    original_path = original_dir / f"{stem}.jpg"
    crop_path = cropped_dir / f"{stem}_NG_MISSING_R101_0.png"
    annotated_path.write_bytes(b"jpg")
    original_path.write_bytes(b"original")
    crop_path.write_bytes(b"png")
    snapshot_path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-18T12:34:56",
                "status": "FAIL",
                "detector": "yolo",
                "product": "PCBA",
                "area": "TOP",
                "decision": {"reasons": ["MISSING"]},
                "model_info": {
                    "model_version": "1.2.3",
                    "weights": "models/pcba.pt",
                },
                "detections": [
                    {
                        "class_id": 0,
                        "confidence": 0.95,
                        "bbox": [1, 2, 10, 20],
                        "image_width": 100,
                        "image_height": 100,
                    }
                ],
                "inference_time": 0.123,
            }
        ),
        encoding="utf-8",
    )

    cases = collect_review_cases(tmp_path / "Result")

    assert len(cases) == 1
    case = cases[0]
    assert case.status == "FAIL"
    assert case.decision_reasons == "MISSING"
    assert case.model_version == "1.2.3"
    assert case.inference_time == "0.123000"
    assert Path(case.annotated_path) == annotated_path
    assert Path(case.original_path) == original_path
    assert json.loads(case.detections_json)[0]["class_id"] == 0
    assert case.failure_crop_paths == str(crop_path)


def test_write_manifest_refuses_to_overwrite_unreadable_existing_reviews(tmp_path):
    manifest = tmp_path / "review_manifest.csv"
    original = b"\xff\xfe\x00broken-review-manifest"
    manifest.write_bytes(original)

    with pytest.raises(ReviewManifestReadError, match="original file was not changed"):
        write_manifest([], manifest)

    assert manifest.read_bytes() == original


def test_write_manifest_replace_failure_preserves_original(tmp_path, monkeypatch):
    manifest = tmp_path / "review_manifest.csv"
    original = b"config_snapshot_path,review_label\nold.json,confirmed_ng\n"
    manifest.write_bytes(original)

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("tools.collect_review_cases.os.replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_manifest([], manifest)

    assert manifest.read_bytes() == original
    assert not list(tmp_path.glob(".review_manifest.csv.*.tmp"))


def test_collect_review_cases_skips_pass_by_default(tmp_path):
    metadata_dir = tmp_path / "Result" / "20260518" / "PCBA" / "TOP" / "PASS" / "metadata" / "yolo"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "yolo_PCBA_TOP_123456_config_snapshot.json").write_text(
        json.dumps({"status": "PASS", "detector": "yolo"}),
        encoding="utf-8",
    )

    assert collect_review_cases(tmp_path / "Result") == []
    assert len(collect_review_cases(tmp_path / "Result", include_pass=True)) == 1


def test_collect_review_cases_persists_color_calibration_contract(tmp_path):
    metadata_dir = tmp_path / "Result" / "20260716" / "Cable1" / "A" / "FAIL" / "metadata" / "yolo"
    metadata_dir.mkdir(parents=True)
    color_result = {
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
    }
    (metadata_dir / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "detector": "yolo",
                "product": "Cable1",
                "area": "A",
                "decision": {"reasons": []},
                "fail_reasons": ["COLOR_MISMATCH"],
                "color_result": color_result,
                "config": {"color_checker_type": "stats"},
            }
        ),
        encoding="utf-8",
    )

    case = collect_review_cases(tmp_path / "Result")[0]

    assert case.decision_reasons == "COLOR_MISMATCH"
    assert json.loads(case.color_result_json) == color_result
    assert case.color_checker_type == "stats"
    assert case.color_failure_count == "1"


def test_collect_review_cases_recovers_legacy_box_count_from_saved_crops(tmp_path):
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
                "status": "DETECTION_FAIL",
                "detector": "yolo",
                "product": "Cable1",
                "area": "A",
            }
        ),
        encoding="utf-8",
    )
    (cropped_dir / f"{stem}_Red_0.png").write_bytes(b"crop")
    (cropped_dir / f"{stem}_Black_1.png").write_bytes(b"crop")
    (cropped_dir / f"{stem}_NG_MISSING_Black_0.png").write_bytes(b"failure")

    case = collect_review_cases(tmp_path / "Result")[0]

    assert json.loads(case.detections_json) == []
    assert case.detected_box_count == "2"
    assert case.detection_evidence_source == "saved_crops"


def test_collect_review_cases_applies_inclusive_time_range(tmp_path):
    metadata_dir = tmp_path / "Result" / "20260714" / "PCBA" / "TOP" / "FAIL" / "metadata" / "yolo"
    metadata_dir.mkdir(parents=True)
    for hour in (10, 12, 14):
        (metadata_dir / f"case_{hour}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "timestamp": f"2026-07-14T{hour:02d}:00:00",
                    "status": "FAIL",
                    "detector": "yolo",
                    "product": "PCBA",
                    "area": "TOP",
                }
            ),
            encoding="utf-8",
        )

    cases = collect_review_cases(
        tmp_path / "Result",
        start_time=datetime(2026, 7, 14, 10),
        end_time="2026-07-14T12:00:00",
    )

    assert [case.timestamp for case in cases] == [
        "2026-07-14T10:00:00",
        "2026-07-14T12:00:00",
    ]


def test_collect_review_cases_rejects_reversed_time_range(tmp_path):
    (tmp_path / "Result").mkdir()

    with pytest.raises(ValueError, match="start_time"):
        collect_review_cases(
            tmp_path / "Result",
            start_time="2026-07-15T00:00:00",
            end_time="2026-07-14T00:00:00",
        )


def test_target_filter_runs_before_expensive_artifact_discovery(
    tmp_path, monkeypatch
):
    result_root = tmp_path / "Result"
    for product, area in (("Cable1", "A"), ("PCBA1", "TOP")):
        metadata = (
            result_root
            / "20260721"
            / product
            / area
            / "FAIL"
            / "metadata"
            / "yolo"
        )
        metadata.mkdir(parents=True)
        (metadata / f"{product}_config_snapshot.json").write_text(
            json.dumps(
                {
                    "status": "FAIL",
                    "detector": "yolo",
                    "product": product,
                    "area": area,
                }
            ),
            encoding="utf-8",
        )
    discovered = []

    def record_discovery(snapshot_path, _detector, **_kwargs):
        discovered.append(snapshot_path.name)
        return []

    monkeypatch.setattr(
        "tools.collect_review_cases._find_detection_crop_paths",
        record_discovery,
    )
    monkeypatch.setattr(
        "tools.collect_review_cases._find_failure_crop_paths",
        record_discovery,
    )

    cases = collect_review_cases(
        result_root,
        include_pass=True,
        product="Cable1",
        area="A",
    )

    assert [(case.product, case.area) for case in cases] == [("Cable1", "A")]
    assert discovered == ["Cable1_config_snapshot.json"] * 2


def test_schema_v2_detection_and_crop_contracts_skip_legacy_globs(
    tmp_path, monkeypatch
):
    metadata = (
        tmp_path
        / "Result"
        / "20260721"
        / "Cable1"
        / "A"
        / "FAIL"
        / "metadata"
        / "yolo"
    )
    metadata.mkdir(parents=True)
    normal_crop = tmp_path / "case_Red_0.png"
    failure_crop = tmp_path / "case_NG_MISSING_0.png"
    (metadata / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "detector": "yolo",
                "product": "Cable1",
                "area": "A",
                "detections": [],
                "artifacts": {
                    "cropped_paths": [str(normal_crop), str(failure_crop)]
                },
            }
        ),
        encoding="utf-8",
    )

    def legacy_glob_is_forbidden(*_args, **_kwargs):
        raise AssertionError("schema-v2 evidence must not run legacy crop globs")

    monkeypatch.setattr(
        "tools.collect_review_cases._find_detection_crop_paths",
        legacy_glob_is_forbidden,
    )
    monkeypatch.setattr(
        "tools.collect_review_cases._find_failure_crop_paths",
        legacy_glob_is_forbidden,
    )

    case = collect_review_cases(tmp_path / "Result")[0]

    assert case.crop_paths == f"{normal_crop}|{failure_crop}"
    assert case.failure_crop_paths == str(failure_crop)


def test_collect_review_cases_adds_saved_image_dimensions(tmp_path):
    base = tmp_path / "Result" / "20260518" / "PCBA" / "TOP" / "FAIL"
    metadata_dir = base / "metadata" / "yolo"
    processed_dir = base / "preprocessed" / "yolo"
    metadata_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    processed_path = processed_dir / "case.png"
    success, encoded = cv2.imencode(".png", np.zeros((20, 40, 3), dtype=np.uint8))
    assert success
    encoded.tofile(processed_path)
    (metadata_dir / "case_config_snapshot.json").write_text(
        json.dumps(
            {
                "status": "FAIL",
                "detector": "yolo",
                "product": "PCBA",
                "area": "TOP",
                "artifacts": {"preprocessed_path": str(processed_path)},
                "detections": [
                    {"class_id": 0, "confidence": 0.9, "bbox": [1, 2, 10, 15]}
                ],
            }
        ),
        encoding="utf-8",
    )

    case = collect_review_cases(tmp_path / "Result")[0]
    detection = json.loads(case.detections_json)[0]

    assert detection["image_width"] == 40
    assert detection["image_height"] == 20


def test_write_manifest_outputs_csv_and_json(tmp_path):
    metadata_dir = tmp_path / "Result" / "20260518" / "PCBA" / "TOP" / "DETECTION_FAIL" / "metadata" / "yolo"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "yolo_PCBA_TOP_123456_config_snapshot.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-05-18T12:00:00",
                "status": "DETECTION_FAIL",
                "detector": "yolo",
                "product": "PCBA",
                "area": "TOP",
                "decision": {"reasons": ["POSITION_SHIFT"]},
            }
        ),
        encoding="utf-8",
    )
    cases = collect_review_cases(tmp_path / "Result")
    csv_path = tmp_path / "manifest.csv"
    json_path = tmp_path / "manifest.json"

    write_manifest(cases, csv_path, json_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["review_label"] == ""
    assert rows[0]["decision_reasons"] == "POSITION_SHIFT"
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["status"] == "DETECTION_FAIL"


def test_write_manifest_preserves_existing_operator_review(tmp_path):
    metadata_dir = (
        tmp_path
        / "Result"
        / "20260518"
        / "PCBA"
        / "TOP"
        / "FAIL"
        / "metadata"
        / "yolo"
    )
    metadata_dir.mkdir(parents=True)
    snapshot = metadata_dir / "yolo_PCBA_TOP_123456_config_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "status": "FAIL",
                "detector": "yolo",
                "product": "PCBA",
                "area": "TOP",
            }
        ),
        encoding="utf-8",
    )
    cases = collect_review_cases(tmp_path / "Result")
    csv_path = tmp_path / "manifest.csv"
    write_manifest(cases, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    row["review_label"] = "confirmed_ng"
    row["review_note"] = "operator checked"
    row["failure_category"] = "threshold_not_met"
    row["failure_source"] = "yolo"
    row["review_selected"] = "1"
    row["training_selected"] = "0"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)

    write_manifest(cases, csv_path)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        refreshed = next(csv.DictReader(handle))
    assert refreshed["review_label"] == "confirmed_ng"
    assert refreshed["review_note"] == "operator checked"
    assert refreshed["failure_category"] == "threshold_not_met"
    assert refreshed["failure_source"] == "yolo"
    assert refreshed["review_selected"] == "1"
    assert refreshed["training_selected"] == "0"
