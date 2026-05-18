import csv
import json
from pathlib import Path

from tools.collect_review_cases import collect_review_cases, write_manifest


def test_collect_review_cases_finds_fail_snapshot_artifacts(tmp_path):
    base = tmp_path / "Result" / "20260518" / "PCBA" / "TOP" / "FAIL"
    metadata_dir = base / "metadata" / "yolo"
    annotated_dir = base / "annotated" / "yolo"
    cropped_dir = base / "cropped" / "yolo"
    metadata_dir.mkdir(parents=True)
    annotated_dir.mkdir(parents=True)
    cropped_dir.mkdir(parents=True)

    stem = "yolo_PCBA_TOP_123456"
    snapshot_path = metadata_dir / f"{stem}_config_snapshot.json"
    annotated_path = annotated_dir / f"{stem}.jpg"
    crop_path = cropped_dir / f"{stem}_NG_MISSING_R101_0.png"
    annotated_path.write_bytes(b"jpg")
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
