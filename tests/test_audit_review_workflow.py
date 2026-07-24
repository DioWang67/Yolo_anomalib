import csv
import json

from tools.audit_review_workflow import audit_review_workflow, main

FIELDS = [
    "sample_id",
    "review_selected",
    "review_outcome",
    "review_label",
    "failure_category",
    "skip_reason",
    "product_verdict",
    "detection_verdict",
    "color_verdict",
    "action_route",
    "training_selected",
]


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _valid_false_positive(sample_id="valid"):
    return {
        "sample_id": sample_id,
        "review_selected": "1",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "failure_category": "",
        "skip_reason": "",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
    }


def test_dry_run_reports_states_violations_and_original_fields(tmp_path):
    manifest = tmp_path / "review.csv"
    invalid = {
        **_valid_false_positive("invalid"),
        "review_selected": "0",
    }
    _write_manifest(
        manifest,
        [
            dict.fromkeys(FIELDS, ""),
            _valid_false_positive(),
            invalid,
        ],
    )

    report = audit_review_workflow([manifest])

    assert report["mutation_performed"] is False
    assert report["workflow_state_counts"] == {
        "NEEDS_FIX": 2,
        "NEW": 1,
    }
    assert report["summary"] == {
        "scopes_checked": 1,
        "records_checked": 3,
        "blocking_inconsistency_count": 1,
        "diagnostic_record_count": 1,
        "error_count": 0,
    }
    diagnostic = report["inconsistent_records"][0]
    assert diagnostic["sample_id"] == "invalid"
    assert diagnostic["original_fields"]["review_selected"] == "0"
    assert diagnostic["derived_semantics"]["ai_correctness"] == "false_positive"
    assert diagnostic["violations"][0]["suggestion"]


def test_dry_run_includes_job_status_state(tmp_path):
    data_root = tmp_path / "data"
    status = data_root / ".operator_handoff" / "jobs" / "job-1" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps({"job_id": "job-1", "state": "deployed"}),
        encoding="utf-8",
    )

    report = audit_review_workflow([], data_root=data_root)

    assert report["workflow_state_counts"] == {"COMPLETED": 1}
    assert report["summary"]["error_count"] == 0


def test_dry_run_exit_codes(tmp_path, capsys):
    valid = tmp_path / "valid.csv"
    blocking = tmp_path / "blocking.csv"
    _write_manifest(valid, [_valid_false_positive()])
    _write_manifest(
        blocking,
        [{**_valid_false_positive(), "review_selected": "0"}],
    )

    assert main(["--manifest", str(valid)]) == 0
    capsys.readouterr()
    assert main(["--manifest", str(blocking)]) == 2
    capsys.readouterr()
    assert main(["--manifest", str(tmp_path / "missing.csv")]) == 1
