import csv
import json

import pytest
from PIL import Image

import tools.color_feedback as color_feedback_module
from tools.color_feedback import (
    COLOR_FEEDBACK_FIELDS,
    export_color_feedback,
    read_color_feedback_progress,
)
from tools.export_review_dataset import export_operator_handoff


def _color_row(image_path, *, verdict="actually_ok", route="color"):
    label = "color_false_reject" if verdict == "actually_ok" else "color_confirmed_ng"
    return {
        "timestamp": "2026-07-16T10:00:00",
        "product": "Cable1",
        "area": "A",
        "status": "FAIL",
        "detector": "yolo",
        "decision_reasons": "COLOR_MISMATCH",
        "model_version": "v1",
        "config_snapshot_path": "result.json",
        "original_path": str(image_path),
        "preprocessed_path": str(image_path),
        "review_label": label,
        "product_verdict": "ok" if verdict == "actually_ok" else "ng",
        "detection_verdict": "correct",
        "color_verdict": verdict,
        "action_route": route,
        "color_checker_type": "stats",
        "color_result_json": json.dumps(
            {
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
        ),
        "review_note": "box is correct",
        "training_selected": "1",
    }


def _write_manifest(path, rows):
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_color_feedback_is_isolated_from_yolo_training_data(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"image-content")
    manifest = tmp_path / "selected.csv"
    _write_manifest(manifest, [_color_row(source)])

    report = export_operator_handoff(manifest, tmp_path / "training-data")

    feedback_path = tmp_path / "training-data" / "Cable1" / "A" / "color_review" / "feedback.csv"
    with feedback_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert report.color_feedback_count == 1
    assert report.color_case_count == 1
    assert report.ready_count == 0
    assert report.pending_count == 0
    assert len(rows) == 1
    assert set(rows[0]) == set(COLOR_FEEDBACK_FIELDS)
    assert rows[0]["actual_is_ok"] == "1"
    assert rows[0]["threshold_key"] == "red"
    assert rows[0]["failure_kind"] == "threshold"
    assert float(rows[0]["score"]) == 0.45
    assert not (tmp_path / "training-data" / "Cable1" / "A" / "raw").exists()

    progress = read_color_feedback_progress(report.color_manifest_paths)
    assert len(progress) == 1
    assert progress[0].threshold_key == "red"
    assert progress[0].total_count == 1
    assert progress[0].ok_count == 1
    assert progress[0].ng_count == 0


def test_color_feedback_rolls_back_when_readback_verification_fails(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.png"
    source.write_bytes(b"image-content")
    destination = tmp_path / "training-data"
    monkeypatch.setattr(color_feedback_module, "_read_feedback", lambda _path: [])

    with pytest.raises(OSError, match="verification failed"):
        export_color_feedback(
            [_color_row(source)],
            source_manifest=tmp_path / "selected.csv",
            output_root=destination,
        )

    assert not (
        destination / "Cable1" / "A" / "color_review" / "feedback.csv"
    ).exists()


def test_re_review_replaces_item_truth_without_duplicating_feedback(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"image-content")
    output = tmp_path / "data"
    first = _color_row(source, verdict="actually_ok")
    second = _color_row(source, verdict="confirmed_ng")

    export_color_feedback([first], source_manifest="first.csv", output_root=output)
    export_color_feedback([second], source_manifest="second.csv", output_root=output)

    feedback_path = output / "Cable1" / "A" / "color_review" / "feedback.csv"
    with feedback_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["actual_is_ok"] == "0"
    assert rows[0]["review_label"] == "color_confirmed_ng"


def test_color_feedback_rejects_legacy_snapshot_without_item_scores(tmp_path):
    source = tmp_path / "source.png"
    source.write_bytes(b"image-content")
    row = _color_row(source)
    row["color_result_json"] = "{}"

    try:
        export_color_feedback([row], source_manifest="legacy.csv", output_root=tmp_path)
    except ValueError as exc:
        assert "no failed color item" in str(exc)
    else:
        raise AssertionError("legacy color feedback must fail closed")


def test_mixed_color_and_box_failure_routes_to_both_queues(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (20, 20), "red").save(source)
    row = _color_row(source, verdict="confirmed_ng", route="both")
    row.update(
        {
            "detection_verdict": "wrong_box",
            "class_names_json": '["Red"]',
            "class_map_json": '{"0":"Red"}',
            "detections_json": json.dumps(
                [
                    {
                        "class_id": 0,
                        "class": "Red",
                        "bbox": [1, 1, 10, 10],
                        "image_width": 20,
                        "image_height": 20,
                    }
                ]
            ),
        }
    )
    manifest = tmp_path / "selected.csv"
    _write_manifest(manifest, [row])

    report = export_operator_handoff(manifest, tmp_path / "training-data")

    assert report.color_feedback_count == 1
    assert report.pending_count == 1
    pending_path = tmp_path / "training-data" / "Cable1" / "A" / "review_pending" / "manifest.csv"
    with pending_path.open("r", encoding="utf-8", newline="") as handle:
        pending = next(csv.DictReader(handle))
    assert pending["reason"] == "box_geometry_requires_correction"
