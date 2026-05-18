import json

import yaml

from tools.production_readiness_check import (
    has_blocking_failures,
    run_readiness_checks,
    write_report,
)


def test_run_readiness_checks_passes_complete_pcba_config(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": str(weights),
                "current_product": "PCBA",
                "current_area": "TOP",
                "conf_thres": 0.4,
                "iou_thres": 0.45,
                "output_dir": "Result",
                "expected_items": {"PCBA": {"TOP": ["R101", "C205"]}},
                "position_config": {
                    "PCBA": {
                        "TOP": {
                            "enabled": True,
                            "expected_boxes": {
                                "R101": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                                "C205": {"x1": 5, "y1": 6, "x2": 7, "y2": 8},
                            },
                            "missing_slot_check": {"enabled": True},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    checks = run_readiness_checks(config_path)

    assert not has_blocking_failures(checks)


def test_run_readiness_checks_fails_missing_expected_boxes(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": str(weights),
                "current_product": "PCBA",
                "current_area": "TOP",
                "expected_items": {"PCBA": {"TOP": ["R101"]}},
                "position_config": {"PCBA": {"TOP": {"enabled": True, "expected_boxes": {}}}},
            }
        ),
        encoding="utf-8",
    )

    checks = run_readiness_checks(config_path)

    assert has_blocking_failures(checks)
    assert any(check.name == "expected_boxes" and check.status == "FAIL" for check in checks)


def test_run_readiness_checks_resolves_repo_relative_weights(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    config_dir = repo_root / "models" / "PCBA1" / "A" / "yolo"
    weights = config_dir / "weights" / "best.onnx"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"model")
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": "models/PCBA1/A/yolo/weights/best.onnx",
                "current_product": "PCBA1",
                "current_area": "A",
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
    monkeypatch.chdir(repo_root)

    checks = run_readiness_checks(config_path)

    assert any(check.name == "weights_exists" and check.status == "PASS" for check in checks)


def test_run_readiness_checks_warns_for_loose_position_tolerance(tmp_path):
    weights = tmp_path / "best.onnx"
    weights.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": str(weights),
                "current_product": "PCBA1",
                "current_area": "A",
                "expected_items": {"PCBA1": {"A": ["J5"]}},
                "position_config": {
                    "PCBA1": {
                        "A": {
                            "enabled": True,
                            "mode": "iou",
                            "tolerance": 1.06,
                            "expected_boxes": {"J5": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    checks = run_readiness_checks(config_path)

    assert any(check.name == "position_iou_tolerance" and check.status == "WARN" for check in checks)


def test_run_readiness_checks_fails_enabled_color_without_model(tmp_path):
    weights = tmp_path / "best.onnx"
    weights.write_bytes(b"model")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": str(weights),
                "current_product": "PCBA1",
                "current_area": "A",
                "enable_color_check": True,
                "color_fail_closed": False,
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

    checks = run_readiness_checks(config_path)

    assert has_blocking_failures(checks)
    assert any(check.name == "color_model_configured" and check.status == "FAIL" for check in checks)
    assert any(check.name == "color_fail_closed" and check.status == "FAIL" for check in checks)


def test_write_report_outputs_json(tmp_path):
    config_path = tmp_path / "missing.yaml"
    checks = run_readiness_checks(config_path, product="P", area="A")
    report_path = tmp_path / "readiness.json"

    write_report(checks, report_path)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data[0]["name"] == "config_exists"
