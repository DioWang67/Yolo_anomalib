import json

import pytest

from core.services.inspection_repository import InspectionRepository


def _snapshot(tmp_path):
    original = tmp_path / "original.jpg"
    annotated = tmp_path / "annotated.jpg"
    crop = tmp_path / "crop.png"
    for path in (original, annotated, crop):
        path.write_bytes(b"evidence")
    snapshot = tmp_path / "case_config_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "timestamp": "2026-07-20T10:30:00",
                "status": "FAIL",
                "detector": "yolo",
                "product": "Cable1",
                "area": "A",
                "equipment": {
                    "machine_id": "M-01",
                    "station": "A",
                    "work_order": "WO-7788",
                    "camera_id": "CAM-02",
                },
                "model_info": {
                    "model_version": "yolo11_20260715_v8",
                    "weights": "models/cable1.onnx",
                },
                "fail_reasons": ["MISSING"],
                "detections": [
                    {
                        "class_id": 3,
                        "class": "Red",
                        "confidence": 0.91,
                        "bbox": [1, 2, 10, 20],
                    }
                ],
                "artifacts": {
                    "original_path": str(original),
                    "annotated_path": str(annotated),
                    "cropped_paths": [str(crop)],
                    "mask_paths": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return snapshot


def test_repository_indexes_evidence_predictions_equipment_and_model(tmp_path):
    snapshot = _snapshot(tmp_path)
    repository = InspectionRepository(tmp_path / "inspection_records.sqlite3")

    inspection_id = repository.upsert_snapshot_file(snapshot)

    inspections = repository.query(
        "SELECT * FROM inspections WHERE inspection_id=?", (inspection_id,)
    )
    assert inspections[0]["product"] == "Cable1"
    assert inspections[0]["machine_id"] == "M-01"
    assert inspections[0]["work_order"] == "WO-7788"
    assert inspections[0]["camera_id"] == "CAM-02"
    assert inspections[0]["model_version"] == "yolo11_20260715_v8"
    predictions = repository.query(
        "SELECT * FROM ai_predictions WHERE inspection_id=?", (inspection_id,)
    )
    assert predictions[0]["class_name"] == "Red"
    assert predictions[0]["confidence"] == pytest.approx(0.91)
    assert predictions[0]["bbox_x2"] == pytest.approx(10)
    artifacts = repository.query(
        "SELECT artifact_type FROM inspection_artifacts WHERE inspection_id=? "
        "ORDER BY artifact_type",
        (inspection_id,),
    )
    assert [row["artifact_type"] for row in artifacts] == [
        "annotated",
        "crop",
        "original",
    ]


def test_repository_saves_review_and_training_set_state(tmp_path):
    snapshot = _snapshot(tmp_path)
    repository = InspectionRepository(tmp_path / "inspection_records.sqlite3")
    inspection_id = repository.upsert_snapshot_file(snapshot)

    repository.sync_review_row(
        {
            "config_snapshot_path": str(snapshot),
            "review_outcome": "fail",
            "review_label": "false_negative",
            "failure_category": "new_defect_type",
            "failure_source": "yolo",
            "failure_note": "端子出現新的裂痕型態",
            "skip_reason": "",
            "action_route": "yolo",
            "training_selected": "1",
        }
    )

    record = repository.query(
        "SELECT * FROM inspections WHERE inspection_id=?", (inspection_id,)
    )[0]
    assert record["review_outcome"] == "fail"
    assert record["failure_category"] == "new_defect_type"
    assert record["failure_note"] == "端子出現新的裂痕型態"
    assert record["training_set_state"] == "selected"
    events = repository.query(
        "SELECT * FROM review_events WHERE inspection_id=?", (inspection_id,)
    )
    assert len(events) == 1


def test_repository_rejects_mutating_report_queries(tmp_path):
    repository = InspectionRepository(tmp_path / "inspection_records.sqlite3")

    with pytest.raises(ValueError, match="Only SELECT"):
        repository.query("DELETE FROM inspections")


def test_repository_loads_manifest_sync_state_in_one_snapshot_view(tmp_path):
    snapshot = _snapshot(tmp_path)
    repository = InspectionRepository(tmp_path / "inspection_records.sqlite3")
    repository.upsert_snapshot_file(snapshot)
    repository.sync_review_row(
        {
            "config_snapshot_path": str(snapshot),
            "review_outcome": "pass",
            "review_label": "false_positive",
            "failure_category": "",
            "failure_source": "yolo",
            "failure_note": "verified",
            "skip_reason": "",
            "action_route": "yolo",
            "training_selected": "0",
        },
        append_event=False,
    )

    state = repository.load_manifest_sync_state()

    assert state[str(snapshot.resolve())] == {
        "review_outcome": "pass",
        "review_label": "false_positive",
        "failure_category": "",
        "failure_source": "yolo",
        "failure_note": "verified",
        "skip_reason": "",
        "action_route": "yolo",
        "training_selected": "0",
    }
