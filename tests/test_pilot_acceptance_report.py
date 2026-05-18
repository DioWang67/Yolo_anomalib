import csv
import json

import pytest

from tools.pilot_acceptance_report import build_acceptance_summary, write_summary


def test_build_acceptance_summary_recommends_supervised_pilot_with_warnings(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps(
            [
                {"name": "weights_exists", "status": "PASS", "message": "ok"},
                {"name": "position_tolerance_percent", "status": "WARN", "message": "wide"},
            ]
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            {
                "status": "FAIL",
                "decision_reasons": "MISSING|POSITION_SHIFT",
                "review_label": "confirmed_ng",
            },
            {
                "status": "FAIL",
                "decision_reasons": "WRONG_COMPONENT",
                "review_label": "false_positive",
            },
        ],
    )

    summary = build_acceptance_summary(
        product="PCBA1",
        area="A",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )

    assert summary.readiness_fail_count == 0
    assert summary.readiness_warn_count == 1
    assert summary.reviewed_case_count == 2
    assert summary.review_label_counts == {"confirmed_ng": 1, "false_positive": 1}
    assert summary.decision_reason_counts["MISSING"] == 1
    assert summary.recommendation == "SUPERVISED_PILOT_WITH_ACCEPTED_WARNINGS"


def test_build_acceptance_summary_blocks_on_false_negative(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps([{"name": "weights_exists", "status": "PASS", "message": "ok"}]),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": "PASS", "decision_reasons": "", "review_label": "false_negative"}],
    )

    summary = build_acceptance_summary(
        product="PCBA1",
        area="B",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )

    assert summary.recommendation == "NO_GO_INVESTIGATE_FALSE_NEGATIVES"


def test_build_acceptance_summary_requires_operator_review(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps([{"name": "weights_exists", "status": "PASS", "message": "ok"}]),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": "FAIL", "decision_reasons": "MISSING", "review_label": ""}],
    )

    summary = build_acceptance_summary(
        product="PCBA1",
        area="B",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )

    assert summary.unreviewed_case_count == 1
    assert summary.recommendation == "HOLD_COMPLETE_OPERATOR_REVIEW"


def test_write_summary_outputs_json_and_markdown(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(
        json.dumps([{"name": "weights_exists", "status": "PASS", "message": "ok"}]),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": "FAIL", "decision_reasons": "MISSING", "review_label": "confirmed_ng"}],
    )
    summary = build_acceptance_summary(
        product="PCBA1",
        area="A",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )
    output_json = tmp_path / "summary.json"
    output_md = tmp_path / "summary.md"

    write_summary(summary, output_json=output_json, output_md=output_md)

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["product"] == "PCBA1"
    assert "Recommendation" in output_md.read_text(encoding="utf-8")


def test_build_acceptance_summary_rejects_invalid_readiness_format(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(manifest_path, [])

    with pytest.raises(ValueError):
        build_acceptance_summary(
            product="PCBA1",
            area="A",
            readiness_json=readiness_path,
            review_manifest_csv=manifest_path,
        )


def _write_manifest(path, rows):
    fieldnames = ["status", "decision_reasons", "review_label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
