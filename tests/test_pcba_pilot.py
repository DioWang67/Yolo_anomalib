import csv
import json

import yaml

from tools.pcba_pilot import (
    default_config_path,
    default_readiness_report_path,
    main,
)


def test_default_paths_use_pcba_conventions():
    assert default_config_path("PCBA1", "A").as_posix() == "models/PCBA1/A/yolo/config.yaml"
    assert default_readiness_report_path("A").as_posix() == "readiness_report_A.json"


def test_readiness_command_uses_short_area_argument(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "models" / "PCBA1" / "A" / "yolo" / "config.yaml"
    weights_path = tmp_path / "models" / "PCBA1" / "A" / "yolo" / "weights" / "best.onnx"
    weights_path.parent.mkdir(parents=True)
    weights_path.write_bytes(b"model")
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "models/PCBA1/A/yolo/weights/best.onnx",
                "current_product": "PCBA1",
                "current_area": "A",
                "output_dir": "Result",
                "expected_items": {"PCBA1": {"A": ["J5"]}},
                "position_config": {
                    "PCBA1": {
                        "A": {
                            "enabled": True,
                            "expected_boxes": {"J5": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(["readiness", "A"])

    assert exit_code == 0
    report = json.loads((tmp_path / "readiness_report_A.json").read_text(encoding="utf-8"))
    assert any(item["name"] == "weights_exists" and item["status"] == "PASS" for item in report)


def test_collect_command_writes_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["collect"])

    assert exit_code == 0
    assert (tmp_path / "review_manifest.csv").exists()
    assert (tmp_path / "review_manifest.json").exists()


def test_summary_command_uses_default_area_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "readiness_report_A.json").write_text(
        json.dumps([{"name": "weights_exists", "status": "PASS", "message": "ok"}]),
        encoding="utf-8",
    )
    with (tmp_path / "review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["status", "decision_reasons", "review_label"])
        writer.writeheader()
        writer.writerow(
            {
                "status": "FAIL",
                "decision_reasons": "MISSING",
                "review_label": "confirmed_ng",
            }
        )

    exit_code = main(["summary", "A"])

    assert exit_code == 0
    summary = json.loads((tmp_path / "pilot_acceptance_summary_A.json").read_text(encoding="utf-8"))
    assert summary["recommendation"] == "SUPERVISED_PILOT_READY"
    assert (tmp_path / "pilot_acceptance_summary_A.md").exists()
