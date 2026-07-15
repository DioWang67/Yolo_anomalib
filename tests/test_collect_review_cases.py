import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.collect_review_cases import collect_review_cases, write_manifest


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


def test_collect_review_cases_skips_pass_by_default(tmp_path):
    metadata_dir = tmp_path / "Result" / "20260518" / "PCBA" / "TOP" / "PASS" / "metadata" / "yolo"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "yolo_PCBA_TOP_123456_config_snapshot.json").write_text(
        json.dumps({"status": "PASS", "detector": "yolo"}),
        encoding="utf-8",
    )

    assert collect_review_cases(tmp_path / "Result") == []
    assert len(collect_review_cases(tmp_path / "Result", include_pass=True)) == 1


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
    assert refreshed["training_selected"] == "0"
