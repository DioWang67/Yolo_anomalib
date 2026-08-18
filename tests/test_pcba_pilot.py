import csv
import json
from pathlib import Path

import pytest
import yaml

from tools.pcba_pilot import (
    default_config_path,
    default_readiness_report_path,
    main,
)
from tools.pilot_acceptance_report import REQUIRED_READINESS_CHECK_NAMES


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
    result_root = tmp_path / "Result"
    result_root.mkdir()
    output_csv = tmp_path / "review_manifest.csv"
    output_json = tmp_path / "review_manifest.json"

    exit_code = main(
        [
            "collect",
            "--result-root",
            str(result_root),
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    assert output_csv.exists()
    assert output_json.exists()


def test_collect_command_forwards_optional_filters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()
    received = {}
    written = {}

    def capture_collect(root, **kwargs):
        received["root"] = root
        received.update(kwargs)
        return []

    monkeypatch.setattr("tools.pcba_pilot.collect_review_cases", capture_collect)

    def capture_write(cases, output_csv, output_json):
        written["cases"] = cases
        written["csv"] = output_csv
        written["json"] = output_json

    monkeypatch.setattr("tools.pcba_pilot.write_manifest", capture_write)

    exit_code = main(
        [
            "collect",
            "--result-root",
            str(result_root),
            "--product",
            "Cable1",
            "--area",
            "A",
            "--start-time",
            "2026-08-18T08:00:00",
            "--end-time",
            "2026-08-18T17:00:00",
            "--include-pass",
        ]
    )

    assert exit_code == 0
    assert received == {
        "root": result_root,
        "include_pass": True,
        "start_time": "2026-08-18T08:00:00",
        "end_time": "2026-08-18T17:00:00",
        "product": "Cable1",
        "area": "A",
        "strict_evidence": False,
    }
    assert written["cases"] == []
    assert written["csv"].stem == written["json"].stem
    assert written["csv"].name.startswith("review_manifest_Cable1_A_")
    assert written["csv"].name != "review_manifest.csv"
    assert written["json"].name != "review_manifest.json"


def test_collect_command_preserves_unfiltered_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()
    received = {}

    def capture_collect(root, **kwargs):
        received["root"] = root
        received.update(kwargs)
        return []

    monkeypatch.setattr("tools.pcba_pilot.collect_review_cases", capture_collect)

    exit_code = main(["collect", "--result-root", str(result_root)])

    assert exit_code == 0
    assert received == {
        "root": result_root,
        "include_pass": False,
        "start_time": None,
        "end_time": None,
        "product": None,
        "area": None,
        "strict_evidence": False,
    }


def test_collect_command_reuses_service_time_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()

    with pytest.raises(ValueError, match="start_time"):
        main(
            [
                "collect",
                "--result-root",
                str(result_root),
                "--start-time",
                "2026-08-18T17:00:00",
                "--end-time",
                "2026-08-18T08:00:00",
            ]
        )


def test_collect_command_derives_json_peer_from_explicit_scoped_csv(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()
    output_csv = tmp_path / "evidence" / "Cable1_A.csv"
    written = {}
    monkeypatch.setattr("tools.pcba_pilot.collect_review_cases", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        "tools.pcba_pilot.write_manifest",
        lambda _cases, csv_path, json_path: written.update({"csv": csv_path, "json": json_path}),
    )

    assert (
        main(
            [
                "collect",
                "--result-root",
                str(result_root),
                "--product",
                "Cable1",
                "--area",
                "A",
                "--output-csv",
                str(output_csv),
            ]
        )
        == 0
    )

    assert written == {
        "csv": output_csv.resolve(),
        "json": output_csv.with_suffix(".json").resolve(),
    }


def test_collect_command_rejects_output_inside_result_tree_before_collection(
    tmp_path,
    monkeypatch,
):
    result_root = tmp_path / "Result"
    result_root.mkdir()

    def unexpected_collect(*_args, **_kwargs):
        raise AssertionError("collection must not run for an unsafe destination")

    monkeypatch.setattr("tools.pcba_pilot.collect_review_cases", unexpected_collect)

    with pytest.raises(ValueError, match="outside the result root"):
        main(
            [
                "collect",
                "--result-root",
                str(result_root),
                "--output-csv",
                str(result_root / "inspection_records.csv"),
            ]
        )


def test_summary_command_uses_default_area_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_readiness(tmp_path / "readiness_report_A.json")
    with (tmp_path / "review_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["product", "area", "status", "decision_reasons", "review_label"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "PCBA1",
                "area": "A",
                "status": "FAIL",
                "decision_reasons": "MISSING",
                "review_label": "confirmed_ng",
            }
        )

    exit_code = main(
        [
            "summary",
            "A",
            "--review-manifest-csv",
            str(tmp_path / "review_manifest.csv"),
        ]
    )

    assert exit_code == 0
    summary = json.loads((tmp_path / "pilot_acceptance_summary_A.json").read_text(encoding="utf-8"))
    assert summary["recommendation"] == "READY_TO_START_SUPERVISED_PILOT"
    assert summary["operational_acceptance_status"] == "NOT_CAPTURED"
    assert summary["merge_eligible"] is False
    assert (tmp_path / "pilot_acceptance_summary_A.md").exists()


def test_summary_command_returns_failure_for_no_go(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_readiness(
        tmp_path / "readiness_report_A.json",
        overrides={"weights_exists": ("FAIL", "missing")},
    )
    manifest_path = tmp_path / "review_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["product", "area", "status", "decision_reasons", "review_label"],
        )
        writer.writeheader()

    exit_code = main(
        [
            "summary",
            "A",
            "--review-manifest-csv",
            str(manifest_path),
        ]
    )

    assert exit_code == 1


def test_pilot_command_propagates_blocking_summary_exit(monkeypatch):
    captured = {}

    monkeypatch.setattr("tools.pcba_pilot.run_readiness_command", lambda _args: 0)

    def capture_collect(args):
        captured.update(vars(args))
        return 0

    monkeypatch.setattr("tools.pcba_pilot.run_collect_command", capture_collect)
    monkeypatch.setattr("tools.pcba_pilot.run_summary_command", lambda _args: 1)

    assert (
        main(
            [
                "pilot",
                "A",
                "--product",
                "Cable1",
                "--start-time",
                "2026-08-18T08:00:00",
                "--end-time",
                "2026-08-18T17:00:00",
            ]
        )
        == 1
    )
    assert captured["product"] == "Cable1"
    assert captured["area"] == "A"
    assert captured["start_time"] == "2026-08-18T08:00:00"
    assert captured["end_time"] == "2026-08-18T17:00:00"
    assert captured["strict_evidence"] is True
    assert captured["output_csv"].name.startswith("review_manifest_Cable1_A_")


def test_pilot_rejects_reversed_time_window_before_any_stage(monkeypatch):
    def unexpected_stage(_args):
        raise AssertionError("no pilot stage may run for an invalid time window")

    monkeypatch.setattr("tools.pcba_pilot.run_readiness_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_collect_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_summary_command", unexpected_stage)

    with pytest.raises(ValueError, match="start_time must not be later"):
        main(
            [
                "pilot",
                "A",
                "--start-time",
                "2026-08-18T17:00:00",
                "--end-time",
                "2026-08-18T08:00:00",
            ]
        )


def test_pilot_rejects_cross_stage_output_collision_before_writing(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()

    def unexpected_stage(_args):
        raise AssertionError("no pilot stage may run before output validation")

    monkeypatch.setattr("tools.pcba_pilot.run_readiness_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_collect_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_summary_command", unexpected_stage)

    with pytest.raises(ValueError, match="destinations must be unique"):
        main(
            [
                "pilot",
                "A",
                "--product",
                "Cable1",
                "--result-root",
                str(result_root),
                "--readiness-json",
                "evidence.json",
                "--review-manifest-json",
                "evidence.json",
                "--start-time",
                "2026-08-18T08:00:00",
                "--end-time",
                "2026-08-18T17:00:00",
            ]
        )

    assert not (tmp_path / "evidence.json").exists()


def test_pilot_outputs_cannot_overwrite_active_color_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()
    weights_path = tmp_path / "best.onnx"
    weights_path.write_bytes(b"model")
    color_model_path = tmp_path / "color_stats.json"
    original = b'{"classes": {"red": {"mean": [1, 2, 3]}}}\n'
    color_model_path.write_bytes(original)
    config_path = tmp_path / "config.yaml"
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

    def unexpected_stage(_args):
        raise AssertionError("no pilot stage may run before source validation")

    monkeypatch.setattr("tools.pcba_pilot.run_readiness_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_collect_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_summary_command", unexpected_stage)

    with pytest.raises(ValueError, match="active readiness source"):
        main(
            [
                "pilot",
                "A",
                "--product",
                "Cable1",
                "--config",
                str(config_path),
                "--result-root",
                str(result_root),
                "--review-manifest-json",
                str(color_model_path),
                "--start-time",
                "2026-08-18T08:00:00",
                "--end-time",
                "2026-08-18T17:00:00",
            ]
        )

    assert color_model_path.read_bytes() == original


def test_pilot_defaults_are_unique_for_each_product_and_scope(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()
    captured: list[tuple[str, str, str, str, str]] = []
    current: dict[str, str] = {}

    def capture_readiness(args):
        current["readiness"] = Path(args.output_json).name
        return 0

    def capture_collect(args):
        current["manifest_csv"] = Path(args.output_csv).name
        current["manifest_json"] = Path(args.output_json).name
        return 0

    def capture_summary(args):
        captured.append(
            (
                current["readiness"],
                current["manifest_csv"],
                current["manifest_json"],
                Path(args.output_json).name,
                Path(args.output_md).name,
            )
        )
        return 0

    monkeypatch.setattr("tools.pcba_pilot.run_readiness_command", capture_readiness)
    monkeypatch.setattr("tools.pcba_pilot.run_collect_command", capture_collect)
    monkeypatch.setattr("tools.pcba_pilot.run_summary_command", capture_summary)

    for product in ("Cable1", "LED"):
        assert (
            main(
                [
                    "pilot",
                    "A",
                    "--product",
                    product,
                    "--result-root",
                    str(result_root),
                    "--start-time",
                    "2026-08-18T08:00:00",
                    "--end-time",
                    "2026-08-18T17:00:00",
                ]
            )
            == 0
        )

    assert captured[0] != captured[1]
    for outputs, product in zip(captured, ("Cable1", "LED"), strict=True):
        assert all(product in output for output in outputs)


def test_pilot_rejects_symlink_output_before_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result_root = tmp_path / "Result"
    result_root.mkdir()
    target = tmp_path / "existing_summary.json"
    target.write_text("verified evidence\n", encoding="utf-8")
    link = tmp_path / "summary_link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    def unexpected_stage(_args):
        raise AssertionError("no pilot stage may run before output validation")

    monkeypatch.setattr("tools.pcba_pilot.run_readiness_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_collect_command", unexpected_stage)
    monkeypatch.setattr("tools.pcba_pilot.run_summary_command", unexpected_stage)

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        main(
            [
                "pilot",
                "A",
                "--product",
                "Cable1",
                "--result-root",
                str(result_root),
                "--summary-json",
                str(link),
                "--start-time",
                "2026-08-18T08:00:00",
                "--end-time",
                "2026-08-18T17:00:00",
            ]
        )

    assert target.read_text(encoding="utf-8") == "verified evidence\n"


def _write_readiness(path, *, overrides=None):
    overrides = overrides or {}
    names = set(REQUIRED_READINESS_CHECK_NAMES)
    names.update(
        {
            "position_tolerance_percent",
            "alignment_shift_limits",
            "defect_coverage_limitations",
        }
    )
    checks = []
    for name in sorted(names):
        status, message = overrides.get(name, ("PASS", "ok"))
        if name == "product_area":
            message = "product=PCBA1, area=A"
        elif name == "color_check_enabled":
            message = "enabled=false"
        checks.append({"name": name, "status": status, "message": message})
    path.write_text(json.dumps(checks), encoding="utf-8")
