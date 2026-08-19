from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import yaml

from tools.audit_cross_class_duplicates import main


def _write_matching_snapshot(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "inspection_id": "inspection-1",
                "timestamp": "2026-08-18T08:00:00+00:00",
                "status": "PASS",
                "product": "Cable1",
                "area": "A",
                "model_info": {"model_version": "model-1"},
                "config": {"position_config": {"Cable1": {"A": {"enabled": False}}}},
                "raw_detections": [
                    {
                        "class": "red_component",
                        "verified_class": "green",
                        "confidence": 0.95,
                        "bbox": [0, 0, 100, 100],
                    },
                    {
                        "class": "blue_component",
                        "verified_class": "green",
                        "confidence": 0.90,
                        "bbox": [0, 0, 100, 100],
                    },
                ],
                "color_result": {
                    "items": [
                        {"index": 0, "is_ok": True, "best_color": "green"},
                        {"index": 1, "is_ok": True, "best_color": "green"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


def test_main_writes_versioned_report_and_keeps_json_stdout_compatible(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    snapshot = snapshot_root / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    output = tmp_path / "evidence" / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot_root),
            "--json",
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 0
    stdout_report = json.loads(capsys.readouterr().out)
    evidence_report = json.loads(output.read_text(encoding="utf-8"))
    assert "schema_version" not in stdout_report
    assert evidence_report == {"schema_version": 1, **stdout_report}
    assert evidence_report["scanned_count"] == 1
    assert evidence_report["matched_snapshot_count"] == 1
    assert evidence_report["proposed_suppression_count"] == 1
    assert len(evidence_report["source"]["snapshot_inventory_sha256"]) == 64
    assert evidence_report["records"][0]["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert output.read_bytes().endswith(b"\n")
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_atomic_output_failure_preserves_existing_report_and_cleans_temp_file(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    output = tmp_path / "duplicate_audit.json"
    output.write_text("previous verified report\n", encoding="utf-8")

    def deny_replace(_source: Path, _destination: Path) -> None:
        raise PermissionError("replacement denied")

    monkeypatch.setattr(
        "tools.audit_cross_class_duplicates.os.replace",
        deny_replace,
    )

    exit_code = main([os.fspath(snapshot), "--output-json", os.fspath(output)])

    assert exit_code == 1
    assert "Audit report could not be written: replacement denied" in (capsys.readouterr().err)
    assert output.read_text(encoding="utf-8") == "previous verified report\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_main_can_bind_audit_to_exact_model_config(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    snapshot_payload = json.loads(snapshot.read_text(encoding="utf-8"))
    snapshot_payload["config"]["position_config"]["Cable1"]["A"]["enabled"] = True
    snapshot.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config_bytes = (
        b"pipeline:\n"
        b"  - color_check\n"
        b"  - cross_class_duplicate_filter\n"
        b"steps:\n"
        b"  cross_class_duplicate_filter:\n"
        b"    enabled: true\n"
        b"    mode: suppress\n"
        b"    iou_threshold: 0.95\n"
        b"    center_distance_ratio_max: 0.10\n"
        b"    area_similarity_min: 0.80\n"
    )
    config.write_bytes(config_bytes)
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            "--config",
            os.fspath(config),
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy"]["mode"] == "suppress"
    assert report["policy"]["iou_threshold"] == 0.95
    assert report["policy_source"] == {
        "kind": "model_config",
        "path": os.fspath(config.resolve()),
        "sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def test_snapshot_parse_error_is_reported_and_exits_nonzero(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    broken = snapshot_root / "broken_config_snapshot.json"
    broken.write_text("{not-json", encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot_root),
            "--json",
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 1
    stdout_report = json.loads(capsys.readouterr().out)
    evidence_report = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_report["scanned_count"] == 1
    assert stdout_report["errors"] == evidence_report["errors"]
    assert evidence_report["schema_version"] == 1
    assert evidence_report["errors"][0]["path"] == os.fspath(broken)


def test_missing_path_fails_clearly_without_replacing_output(
    tmp_path: Path,
    capsys,
) -> None:
    missing = tmp_path / "missing"
    output = tmp_path / "duplicate_audit.json"
    output.write_text("previous verified report\n", encoding="utf-8")

    exit_code = main([os.fspath(missing), "--output-json", os.fspath(output)])

    assert exit_code == 1
    assert f"Path does not exist: {missing}" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "previous verified report\n"


def test_empty_snapshot_directory_is_fail_closed(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    output = tmp_path / "duplicate_audit.json"

    exit_code = main([os.fspath(snapshot_root), "--output-json", os.fspath(output)])

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scanned_count"] == 0
    assert report["errors"][0]["error"] == "no snapshot files found"


def test_explicit_empty_raw_detections_do_not_fall_back_to_effective_list(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["raw_detections"] = []
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["matched_snapshot_count"] == 0


def test_invalid_fallback_detections_shape_is_fail_closed(tmp_path: Path) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload.pop("raw_detections")
    payload["detections"] = {}
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["errors"][0]["error"] == ("snapshot detections must be a list of objects")


def test_output_cannot_overwrite_single_snapshot_input(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    original = snapshot.read_bytes()

    exit_code = main(
        [
            os.fspath(snapshot),
            "--output-json",
            os.fspath(tmp_path / "." / snapshot.name),
        ]
    )

    assert exit_code == 1
    assert "must not overwrite the snapshot input" in capsys.readouterr().err
    assert snapshot.read_bytes() == original


def test_output_cannot_overwrite_model_config(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    config = tmp_path / "config.yaml"
    config.write_text("invalid until destination validation\n", encoding="utf-8")
    original = config.read_bytes()

    exit_code = main(
        [
            os.fspath(snapshot),
            "--config",
            os.fspath(config),
            "--output-json",
            os.fspath(tmp_path / "nested" / ".." / config.name),
        ]
    )

    assert exit_code == 1
    assert "must not overwrite the model config" in capsys.readouterr().err
    assert config.read_bytes() == original


def test_output_must_be_outside_snapshot_scan_tree(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    snapshot = snapshot_root / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    original = snapshot.read_bytes()
    output = snapshot_root / "evidence_config_snapshot.json"

    exit_code = main([os.fspath(snapshot_root), "--output-json", os.fspath(output)])

    assert exit_code == 1
    assert "must be outside the snapshot scan directory" in (capsys.readouterr().err)
    assert not output.exists()
    assert snapshot.read_bytes() == original


def test_non_mapping_model_info_is_a_structured_snapshot_error(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["model_info"] = ["model-1"]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    exit_code = main([os.fspath(snapshot), "--output-json", os.fspath(output)])

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["records"] == []
    assert report["errors"] == [
        {
            "path": os.fspath(snapshot.resolve()),
            "error": "snapshot model_info must be a JSON mapping",
        }
    ]


def test_position_enabled_snapshot_is_blocked_without_suppression(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["config"]["position_config"]["Cable1"]["A"]["enabled"] = True
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["records"] == []
    assert report["proposed_suppression_count"] == 0
    assert report["errors"] == []
    assert report["blocked_snapshot_count"] == 1
    assert report["blocked_reason_counts"] == {"blocked_position_enabled": 1}
    assert report["blocked_records"][0]["reason"] == "blocked_position_enabled"
    assert report["blocked_records"][0]["position_check_state"] == "enabled"


def test_position_state_unknown_snapshot_is_blocked_without_suppression(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    del payload["config"]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["records"] == []
    assert report["proposed_suppression_count"] == 0
    assert report["errors"][0]["code"] == "blocked_position_state_unknown"
    assert report["errors"][0]["position_check_state"] == "unknown"


def test_position_disabled_snapshot_is_analyzed_with_guard_evidence(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["proposed_suppression_count"] == 1
    assert report["records"][0]["position_guard"] == {
        "required": True,
        "state": "disabled",
        "evidence": ("snapshot.config.position_config resolved product=Cable1, area=A, enabled=False"),
    }


def test_missing_position_mapping_matches_runtime_disabled_default(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["config"]["position_config"] = {}
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["errors"] == []
    assert report["blocked_snapshot_count"] == 0
    assert report["proposed_suppression_count"] == 1
    assert report["records"][0]["position_guard"]["state"] == "disabled"


@pytest.mark.parametrize(
    "pipeline",
    [
        ["cross_class_duplicate_filter", "save_results"],
        ["cross_class_duplicate_filter", "color_check", "save_results"],
        [
            "color_check",
            "cross_class_duplicate_filter",
            "cross_class_duplicate_filter",
            "save_results",
        ],
        ["color_check", "count_check", "cross_class_duplicate_filter"],
        ["color_check", "sequence_check", "cross_class_duplicate_filter"],
        ["color_check", "save_results", "cross_class_duplicate_filter"],
    ],
)
def test_model_config_rejects_runtime_invalid_duplicate_filter_order(
    tmp_path: Path,
    capsys,
    pipeline: list[str],
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pipeline": pipeline,
                "steps": {
                    "cross_class_duplicate_filter": {
                        "enabled": True,
                        "mode": "report_only",
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            "--config",
            os.fspath(config),
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 1
    assert "cross_class_duplicate_filter" in capsys.readouterr().err
    assert not output.exists()


def test_empty_detections_are_safe_noop_before_unknown_position_guard(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["raw_detections"] = []
    del payload["config"]
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["scanned_count"] == 1
    assert report["selected_snapshot_count"] == 1
    assert report["no_detection_snapshot_count"] == 1
    assert report["analyzed_snapshot_count"] == 0
    assert report["blocked_snapshot_count"] == 0
    assert report["errors"] == []


def test_exact_product_area_and_time_filters_skip_nonmatching_snapshots(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    fixtures = [
        ("selected", "Cable1", "A", "2026-08-18T10:00:00"),
        ("wrong_product", "PCBA1", "A", "2026-08-18T10:00:00"),
        ("wrong_area", "Cable1", "B", "2026-08-18T10:00:00"),
        ("wrong_time", "Cable1", "A", "2026-08-18T08:59:59"),
    ]
    for name, product, area, timestamp in fixtures:
        snapshot = snapshot_root / f"{name}_config_snapshot.json"
        _write_matching_snapshot(snapshot)
        payload = json.loads(snapshot.read_text(encoding="utf-8"))
        payload.update({"product": product, "area": area, "timestamp": timestamp})
        if name == "wrong_product":
            payload["raw_detections"] = "outside the selected pilot scope"
        snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot_root),
            "--product",
            "Cable1",
            "--area",
            "A",
            "--start-time",
            "2026-08-18T09:00:00",
            "--end-time",
            "2026-08-18T11:00:00",
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["filters"] == {
        "product": "Cable1",
        "area": "A",
        "start_time": "2026-08-18T09:00:00",
        "end_time": "2026-08-18T11:00:00",
    }
    assert report["scanned_count"] == 4
    assert report["selected_snapshot_count"] == 1
    assert report["filter_skipped_snapshot_count"] == 3
    assert report["filter_reason_counts"] == {
        "area": 1,
        "product": 1,
        "timestamp": 1,
    }
    assert report["matched_snapshot_count"] == 1
    assert report["errors"] == []


def test_output_report_requires_json_suffix(
    tmp_path: Path,
    capsys,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    output = tmp_path / "inspection.sqlite3"
    output.write_bytes(b"database bytes")

    exit_code = main([os.fspath(snapshot), "--output-json", os.fspath(output)])

    assert exit_code == 1
    assert "must use a .json suffix" in capsys.readouterr().err
    assert output.read_bytes() == b"database bytes"


def test_inventory_digest_commits_filtered_raw_sha_and_filter_outcome(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    cable = snapshot_root / "cable_config_snapshot.json"
    pcba = snapshot_root / "pcba_config_snapshot.json"
    _write_matching_snapshot(cable)
    _write_matching_snapshot(pcba)
    pcba_payload = json.loads(pcba.read_text(encoding="utf-8"))
    pcba_payload["product"] = "PCBA1"
    pcba.write_text(json.dumps(pcba_payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert (
        main(
            [
                os.fspath(snapshot_root),
                "--product",
                "Cable1",
                "--output-json",
                os.fspath(output),
            ]
        )
        == 0
    )
    first_digest = json.loads(output.read_text(encoding="utf-8"))["source"]["snapshot_inventory_sha256"]

    assert (
        main(
            [
                os.fspath(snapshot_root),
                "--product",
                "PCBA1",
                "--output-json",
                os.fspath(output),
            ]
        )
        == 0
    )
    second_digest = json.loads(output.read_text(encoding="utf-8"))["source"]["snapshot_inventory_sha256"]
    assert second_digest != first_digest

    cable_payload = json.loads(cable.read_text(encoding="utf-8"))
    cable_payload["outside_selected_scope_revision"] = 1
    cable.write_text(json.dumps(cable_payload), encoding="utf-8")
    assert (
        main(
            [
                os.fspath(snapshot_root),
                "--product",
                "PCBA1",
                "--output-json",
                os.fspath(output),
            ]
        )
        == 0
    )
    third_report = json.loads(output.read_text(encoding="utf-8"))
    assert third_report["source"]["snapshot_inventory_sha256"] != second_digest
    assert third_report["filter_reason_counts"] == {"product": 1}


@pytest.mark.parametrize("timestamp", [None, "not-an-iso-timestamp"])
def test_active_time_filter_rejects_missing_or_invalid_snapshot_timestamp(
    tmp_path: Path,
    timestamp: str | None,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    if timestamp is None:
        payload.pop("timestamp")
    else:
        payload["timestamp"] = timestamp
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            "--start-time",
            "2026-08-18T09:00:00",
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_snapshot_count"] == 0
    assert report["filter_skipped_snapshot_count"] == 0
    assert report["errors"][0]["code"] == "invalid_snapshot_timestamp"


def test_inference_error_status_is_runtime_equivalent_noop(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["status"] = "inference_error"
    payload["raw_detections"] = "must not be parsed"
    payload["color_result"] = "must not be parsed"
    payload["model_info"] = "must not be parsed"
    payload.pop("config")
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_snapshot_count"] == 1
    assert report["runtime_skipped_snapshot_count"] == 1
    assert report["runtime_skip_reason_counts"] == {"inference_error_status": 1}
    assert report["runtime_skipped_records"][0]["status"] == "INFERENCE_ERROR"
    assert report["no_detection_snapshot_count"] == 0
    assert report["analyzed_snapshot_count"] == 0
    assert report["blocked_snapshot_count"] == 0
    assert report["proposed_suppression_count"] == 0
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("field", "filter_flag", "invalid_value", "error_code"),
    [
        ("product", "--product", None, "invalid_snapshot_product"),
        ("product", "--product", "   ", "invalid_snapshot_product"),
        ("product", "--product", 123, "invalid_snapshot_product"),
        ("area", "--area", None, "invalid_snapshot_area"),
        ("area", "--area", "   ", "invalid_snapshot_area"),
        ("area", "--area", ["A"], "invalid_snapshot_area"),
    ],
)
def test_active_scope_filter_rejects_missing_empty_or_non_string_field(
    tmp_path: Path,
    field: str,
    filter_flag: str,
    invalid_value,
    error_code: str,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    if invalid_value is None:
        payload.pop(field)
    else:
        payload[field] = invalid_value
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            filter_flag,
            "Cable1" if field == "product" else "A",
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_snapshot_count"] == 0
    assert report["filter_skipped_snapshot_count"] == 0
    assert report["errors"][0]["code"] == error_code


def test_scope_filters_exclude_non_target_legacy_bad_timestamp_before_time_parse(
    tmp_path: Path,
) -> None:
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    wrong_product = snapshot_root / "wrong_product_config_snapshot.json"
    wrong_area = snapshot_root / "wrong_area_config_snapshot.json"
    _write_matching_snapshot(wrong_product)
    _write_matching_snapshot(wrong_area)

    product_payload = json.loads(wrong_product.read_text(encoding="utf-8"))
    product_payload["product"] = "PCBA1"
    product_payload["timestamp"] = "legacy-invalid-timestamp"
    wrong_product.write_text(json.dumps(product_payload), encoding="utf-8")

    area_payload = json.loads(wrong_area.read_text(encoding="utf-8"))
    area_payload["area"] = "B"
    area_payload.pop("timestamp")
    wrong_area.write_text(json.dumps(area_payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot_root),
            "--product",
            "Cable1",
            "--area",
            "A",
            "--start-time",
            "2026-08-18T09:00:00",
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["selected_snapshot_count"] == 0
    assert report["filter_skipped_snapshot_count"] == 2
    assert report["filter_reason_counts"] == {"area": 1, "product": 1}
    assert report["errors"] == [
        {
            "path": os.fspath(snapshot_root.resolve()),
            "code": "no_snapshots_in_selected_scope",
            "error": "no snapshots matched the selected audit scope",
        }
    ]


@pytest.mark.parametrize("color_result", [{}, {"items": []}])
def test_missing_or_empty_color_items_are_runtime_equivalent_blocked_snapshots(
    tmp_path: Path,
    color_result: dict,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["color_result"] = color_result
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "duplicate_audit.json"

    assert main([os.fspath(snapshot), "--output-json", os.fspath(output)]) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["blocked_snapshot_count"] == 1
    assert report["blocked_reason_counts"] == {
        "blocked_color_result_unavailable": 1
    }
    assert report["blocked_records"][0]["reason"] == (
        "blocked_color_result_unavailable"
    )
    assert report["analyzed_snapshot_count"] == 0
    assert report["proposed_suppression_count"] == 0
    assert report["errors"] == []


def test_missing_color_items_block_even_when_color_pass_policy_is_disabled(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["color_result"] = {}
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pipeline": [
                    "color_check",
                    "cross_class_duplicate_filter",
                    "save_results",
                ],
                "steps": {
                    "cross_class_duplicate_filter": {
                        "require_color_check_pass": False,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            "--config",
            os.fspath(config),
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy"]["require_color_check_pass"] is False
    assert report["blocked_reason_counts"] == {
        "blocked_color_result_unavailable": 1
    }
    assert report["proposed_suppression_count"] == 0


def test_model_config_empty_duplicate_options_use_runtime_enabled_default(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pipeline": [
                    "color_check",
                    "cross_class_duplicate_filter",
                    "save_results",
                ],
                "steps": {"cross_class_duplicate_filter": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            "--config",
            os.fspath(config),
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy"]["mode"] == "report_only"
    assert report["matched_snapshot_count"] == 1
    assert report["proposed_suppression_count"] == 1
    assert report["errors"] == []


@pytest.mark.parametrize(
    "steps",
    [pytest.param(None, id="omitted"), pytest.param({}, id="empty")],
)
def test_model_config_missing_duplicate_options_use_runtime_enabled_default(
    tmp_path: Path,
    steps: dict | None,
) -> None:
    snapshot = tmp_path / "sample_config_snapshot.json"
    _write_matching_snapshot(snapshot)
    config_payload = {
        "pipeline": [
            "color_check",
            "cross_class_duplicate_filter",
            "save_results",
        ]
    }
    if steps is not None:
        config_payload["steps"] = steps
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(config_payload, sort_keys=False),
        encoding="utf-8",
    )
    output = tmp_path / "duplicate_audit.json"

    exit_code = main(
        [
            os.fspath(snapshot),
            "--config",
            os.fspath(config),
            "--output-json",
            os.fspath(output),
        ]
    )

    assert exit_code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["policy"]["mode"] == "report_only"
    assert report["matched_snapshot_count"] == 1
    assert report["proposed_suppression_count"] == 1
    assert report["errors"] == []
