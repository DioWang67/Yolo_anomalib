import csv
import json

import pytest

from tools.pilot_acceptance_report import (
    REQUIRED_READINESS_CHECK_NAMES,
    build_acceptance_summary,
    write_summary,
)


def test_build_acceptance_summary_holds_until_warnings_are_accepted_elsewhere(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(
        readiness_path,
        product="PCBA1",
        area="A",
        overrides={"position_tolerance_percent": ("WARN", "wide")},
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
    assert summary.recommendation == "HOLD_DOCUMENT_READINESS_WARNING_ACCEPTANCE"
    assert summary.evidence_kind == "pre_pilot_screening"
    assert summary.operational_acceptance_status == "NOT_CAPTURED"
    assert summary.merge_eligible is False


def test_build_acceptance_summary_blocks_on_false_negative(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="B")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": "PASS", "decision_reasons": "", "review_label": "false_negative"}],
        area="B",
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
    _write_readiness(readiness_path, product="PCBA1", area="B")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": "FAIL", "decision_reasons": "MISSING", "review_label": ""}],
        area="B",
    )

    summary = build_acceptance_summary(
        product="PCBA1",
        area="B",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )

    assert summary.unreviewed_case_count == 1
    assert summary.recommendation == "HOLD_COMPLETE_OPERATOR_REVIEW"


@pytest.mark.parametrize(
    ("status", "label", "recommendation", "field"),
    (
        ("FAIL", "made_up_label", "NO_GO_FIX_UNKNOWN_REVIEW_LABELS", "unknown_review_label_count"),
        ("PASS", "confirmed_ng", "NO_GO_FIX_INCONSISTENT_REVIEW_LABELS", "inconsistent_review_count"),
        ("FAIL", "uncertain", "HOLD_RESOLVE_UNCERTAIN_OPERATOR_REVIEWS", "uncertain_case_count"),
    ),
)
def test_build_acceptance_summary_fails_closed_on_unusable_review_labels(
    tmp_path,
    status,
    label,
    recommendation,
    field,
):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="B")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": status, "decision_reasons": "", "review_label": label}],
        area="B",
    )

    summary = build_acceptance_summary(
        product="PCBA1",
        area="B",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )

    assert summary.recommendation == recommendation
    assert getattr(summary, field) == 1
    assert summary.reviewed_case_count == 0


def test_build_acceptance_summary_rejects_invalid_machine_status(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="B")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(
        manifest_path,
        [{"status": "UNKNOWN", "decision_reasons": "", "review_label": "confirmed_ok"}],
        area="B",
    )

    with pytest.raises(ValueError, match="invalid status"):
        build_acceptance_summary(
            product="PCBA1",
            area="B",
            readiness_json=readiness_path,
            review_manifest_csv=manifest_path,
        )


def test_write_summary_outputs_json_and_markdown(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="A")
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
    assert data["merge_eligible"] is False
    assert len(data["readiness_sha256"]) == 64
    assert len(data["review_manifest_sha256"]) == 64
    markdown = output_md.read_text(encoding="utf-8")
    assert "Recommendation" in markdown
    assert "does not prove completed operational acceptance" in markdown


def test_write_summary_rejects_source_and_cross_output_collisions(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="A")
    original_readiness = readiness_path.read_text(encoding="utf-8")
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

    with pytest.raises(ValueError, match="cannot overwrite source evidence"):
        write_summary(summary, output_json=readiness_path)
    assert readiness_path.read_text(encoding="utf-8") == original_readiness

    shared_output = tmp_path / "shared-output"
    with pytest.raises(ValueError, match="must be different"):
        write_summary(
            summary,
            output_json=shared_output,
            output_md=shared_output,
        )
    assert not shared_output.exists()


def test_write_summary_rejects_symbolic_link_destination(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="A")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(manifest_path, [])
    summary = build_acceptance_summary(
        product="PCBA1",
        area="A",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )
    target = tmp_path / "existing-summary.json"
    original_target = b"verified summary\n"
    target.write_bytes(original_target)
    link = tmp_path / "summary-link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with pytest.raises(ValueError, match="cannot be a symbolic link"):
        write_summary(summary, output_json=link)
    assert target.read_bytes() == original_target


def test_write_summary_rejects_source_changed_after_parsing(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="A")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(manifest_path, [])
    summary = build_acceptance_summary(
        product="PCBA1",
        area="A",
        readiness_json=readiness_path,
        review_manifest_csv=manifest_path,
    )
    manifest_path.write_text("changed\n", encoding="utf-8")
    output = tmp_path / "summary.json"

    with pytest.raises(ValueError, match="review manifest changed"):
        write_summary(summary, output_json=output)
    assert not output.exists()


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


@pytest.mark.parametrize(
    "readiness",
    (
        [],
        ["not-an-object"],
        [{"name": "weights", "status": "UNKNOWN", "message": "bad"}],
        [
            {"name": "weights", "status": "PASS", "message": "ok"},
            {"name": "weights", "status": "PASS", "message": "duplicate"},
        ],
    ),
)
def test_build_acceptance_summary_rejects_incomplete_readiness_evidence(
    tmp_path,
    readiness,
):
    readiness_path = tmp_path / "readiness.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(manifest_path, [])

    with pytest.raises(ValueError, match="readiness"):
        build_acceptance_summary(
            product="PCBA1",
            area="A",
            readiness_json=readiness_path,
            review_manifest_csv=manifest_path,
        )


def test_build_acceptance_summary_rejects_cross_scope_evidence(tmp_path):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="OtherProduct", area="A")
    manifest_path = tmp_path / "review_manifest.csv"
    _write_manifest(manifest_path, [])

    with pytest.raises(ValueError, match="does not match requested scope"):
        build_acceptance_summary(
            product="PCBA1",
            area="A",
            readiness_json=readiness_path,
            review_manifest_csv=manifest_path,
        )


@pytest.mark.parametrize(
    "csv_text",
    (
        "product,area,status,decision_reasons,review_label,review_label\nPCBA1,A,PASS,,confirmed_ok,confirmed_ok\n",
        "product,area,status,decision_reasons,review_label\nPCBA1,A,PASS,,confirmed_ok,extra\n",
    ),
)
def test_build_acceptance_summary_rejects_ambiguous_csv_rows(tmp_path, csv_text):
    readiness_path = tmp_path / "readiness.json"
    _write_readiness(readiness_path, product="PCBA1", area="A")
    manifest_path = tmp_path / "review_manifest.csv"
    manifest_path.write_text(csv_text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate CSV columns|unexpected extra columns"):
        build_acceptance_summary(
            product="PCBA1",
            area="A",
            readiness_json=readiness_path,
            review_manifest_csv=manifest_path,
        )

    _write_readiness(readiness_path, product="PCBA1", area="A")
    _write_manifest(
        manifest_path,
        [{"status": "PASS", "decision_reasons": "", "review_label": "confirmed_ok"}],
        product="OtherProduct",
    )
    with pytest.raises(ValueError, match="scope mismatch"):
        build_acceptance_summary(
            product="PCBA1",
            area="A",
            readiness_json=readiness_path,
            review_manifest_csv=manifest_path,
        )


def _write_readiness(path, *, product, area, overrides=None):
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
            message = f"product={product}, area={area}"
        elif name == "color_check_enabled":
            message = "enabled=false"
        checks.append({"name": name, "status": status, "message": message})
    path.write_text(json.dumps(checks), encoding="utf-8")


def _write_manifest(path, rows, *, product="PCBA1", area="A"):
    fieldnames = ["product", "area", "status", "decision_reasons", "review_label"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({"product": product, "area": area, **row})
