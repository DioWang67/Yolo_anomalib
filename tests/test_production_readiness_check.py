import json

import pytest
import yaml

from tools.production_readiness_check import (
    has_blocking_failures,
    main,
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
    assert any(
        check.name == "color_check_enabled" and check.status == "PASS" and check.message == "enabled=false"
        for check in checks
    )


@pytest.mark.parametrize(
    "setting",
    ("save_original", "save_annotated", "save_crops", "fail_on_unexpected"),
)
def test_run_readiness_checks_rejects_truthy_string_for_required_boolean(
    tmp_path,
    setting,
):
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
                "position_config": {
                    "PCBA": {
                        "TOP": {
                            "enabled": True,
                            "expected_boxes": {"R101": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                        }
                    }
                },
                setting: "false",
            }
        ),
        encoding="utf-8",
    )

    checks = run_readiness_checks(config_path)

    check = next(item for item in checks if item.name == setting)
    assert check.status == "FAIL"
    assert "must be a YAML boolean" in check.message


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
    assert any(
        check.name == "color_check_enabled" and check.status == "PASS" and check.message == "enabled=true"
        for check in checks
    )
    assert any(check.name == "color_model_configured" and check.status == "FAIL" for check in checks)
    assert any(check.name == "color_fail_closed" and check.status == "FAIL" for check in checks)


def test_run_readiness_checks_rejects_string_color_enable_flag(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text('enable_color_check: "false"\n', encoding="utf-8")

    checks = run_readiness_checks(config_path, product="PCBA", area="TOP")

    marker = next(check for check in checks if check.name == "color_check_enabled")
    assert marker.status == "FAIL"
    assert "must be a YAML boolean" in marker.message


def test_run_readiness_checks_warns_when_defect_coverage_is_missing(tmp_path):
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
                            "expected_boxes": {"J5": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    checks = run_readiness_checks(config_path)

    assert any(check.name == "defect_coverage_declared" and check.status == "WARN" for check in checks)


def test_run_readiness_checks_passes_declared_defect_coverage_with_limitations(tmp_path):
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
                "defect_coverage": {
                    "covered": ["missing_component", "position_shift"],
                    "not_covered": ["solder_quality"],
                },
                "position_config": {
                    "PCBA1": {
                        "A": {
                            "enabled": True,
                            "expected_boxes": {"J5": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}},
                            "alignment": {
                                "enabled": True,
                                "quality_gate": {"enabled": True, "max_shift_px": 10},
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    checks = run_readiness_checks(config_path)

    assert any(check.name == "defect_coverage_declared" and check.status == "PASS" for check in checks)
    assert any(check.name == "defect_coverage_limitations" and check.status == "WARN" for check in checks)
    assert any(check.name == "alignment_quality_gate" and check.status == "PASS" for check in checks)


def test_write_report_outputs_json(tmp_path):
    config_path = tmp_path / "missing.yaml"
    checks = run_readiness_checks(config_path, product="P", area="A")
    report_path = tmp_path / "readiness.json"

    write_report(checks, report_path)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data[0]["name"] == "config_exists"
    assert report_path.read_bytes().endswith(b"\n")
    assert list(tmp_path.glob(f".{report_path.name}.*.tmp")) == []


def test_write_report_cannot_overwrite_protected_source(tmp_path):
    config_path = tmp_path / "config.json"
    original = b'{"current_product": "PCBA"}\n'
    config_path.write_bytes(original)
    checks = run_readiness_checks(config_path, product="PCBA", area="A")

    with pytest.raises(ValueError, match="cannot overwrite source evidence"):
        write_report(
            checks,
            config_path,
            protected_sources=(config_path,),
        )

    assert config_path.read_bytes() == original


def test_main_cannot_overwrite_color_model_source(tmp_path):
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    model_dir.mkdir(parents=True)
    weights_path = model_dir / "best.onnx"
    weights_path.write_bytes(b"model")
    color_model_path = model_dir / "color_stats.json"
    original = b'{"classes": {"red": {"mean": [1, 2, 3]}}}\n'
    color_model_path.write_bytes(original)
    config_path = model_dir / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "weights": str(weights_path),
                "current_product": "Cable1",
                "current_area": "A",
                "enable_color_check": True,
                "color_model_path": str(color_model_path),
                "color_fail_closed": True,
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--output-json",
            str(color_model_path),
        ]
    )

    assert exit_code == 1
    assert color_model_path.read_bytes() == original


def test_global_config_loads_selected_model_config(tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "Cable1" / "A" / "yolo"
    weights_dir = model_dir / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / "best.onnx").write_bytes(b"model")
    (model_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "weights": "models/Cable1/A/yolo/weights/best.onnx",
                "current_product": "Cable1",
                "current_area": "A",
                "expected_items": {"Cable1": {"A": ["Red"]}},
                "position_config": {"Cable1": {"A": {"enabled": False}}},
            }
        ),
        encoding="utf-8",
    )
    global_config = tmp_path / "config.yaml"
    global_config.write_text("weights: missing.onnx\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    checks = run_readiness_checks(global_config, product="Cable1", area="A")

    by_name = {check.name: check for check in checks}
    assert by_name["model_config_loaded"].status == "PASS"
    assert by_name["weights_exists"].status == "PASS"
    assert by_name["expected_items"].status == "PASS"
    assert by_name["position_check_enabled"].status == "PASS"
