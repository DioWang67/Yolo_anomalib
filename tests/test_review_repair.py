from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.apply_review_repair_plan import main as apply_main
from tools.audit_review_workflow import audit_review_workflow
from tools.generate_review_repair_plan import main as generate_main
from tools.review_repair import (
    RepairPlanBlockedError,
    ReviewRepairError,
    StaleRepairPlanError,
    apply_repair_plan,
    generate_repair_plan,
    repair_audit_exit_code,
    rollback_repair,
    write_repair_plan,
)
from tools.rollback_review_repair import main as rollback_main

FIELDS = [
    "sample_id",
    "image_sha256",
    "review_selected",
    "review_outcome",
    "review_label",
    "failure_category",
    "failure_note",
    "skip_reason",
    "product_verdict",
    "detection_verdict",
    "color_verdict",
    "action_route",
    "training_selected",
    "submission_status",
    "job_status",
]


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "sample_id": "case-1",
        "image_sha256": "a" * 64,
        "review_selected": "0",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "failure_category": "",
        "failure_note": "",
        "skip_reason": "",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
        "submission_status": "",
        "job_status": "",
    }
    row.update(overrides)
    return row


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _draft(tmp_path: Path, row: dict[str, str] | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest, [row or _row()])
    before = manifest.read_bytes()
    plan = generate_repair_plan(manifest)
    plan_path = tmp_path / "plan.json"
    write_repair_plan(plan, plan_path)
    return manifest, before, plan, plan_path


def _approve(
    plan: dict,
    *,
    revision_reason: str = "",
    changes: dict[str, str] | None = None,
) -> dict:
    proposal = plan["proposals"][0]
    proposal["approval_status"] = "approved"
    proposal["approved"] = True
    proposal["reviewer"] = "DOMAIN\\reviewer"
    proposal["decision_reason"] = "Checked against the original inspection"
    proposal["revision_reason"] = revision_reason
    if changes is not None:
        proposal["proposed_field_changes"] = changes
    return plan


def test_dry_run_and_proposal_generation_do_not_modify_source(tmp_path):
    manifest, before, plan, _plan_path = _draft(tmp_path)

    assert manifest.read_bytes() == before
    assert plan["mode"] == "proposal_only"
    assert plan["proposal_count"] == 1
    proposal = plan["proposals"][0]
    assert proposal["scope"] == str(manifest.resolve())
    assert proposal["sample_id"] == "case-1"
    assert proposal["image_sha256"] == "a" * 64
    assert proposal["original_fields"]["review_selected"] == "0"
    assert proposal["derived_semantics"]["ai_correctness"] == "false_positive"
    assert proposal["violation_code"] == "review_without_selection"
    assert proposal["severity"] == "blocking"
    assert proposal["proposed_field_changes"] == {"review_selected": "1"}
    assert proposal["requires_manual_decision"] is True
    assert proposal["approved"] is False


def test_source_sha_prevents_stale_apply(tmp_path):
    manifest, _before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    write_repair_plan(plan, plan_path)
    _write_manifest(manifest, [_row(failure_note="external edit")])

    with pytest.raises(StaleRepairPlanError, match="source manifest SHA"):
        apply_repair_plan(plan_path)


def test_proposal_immutable_basis_detects_scope_tampering(tmp_path):
    _manifest, _before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    plan["proposals"][0]["scope"] = str(tmp_path / "other.csv")
    write_repair_plan(plan, plan_path)

    with pytest.raises(ReviewRepairError, match="immutable basis"):
        apply_repair_plan(plan_path)


def test_unapproved_and_blocking_proposal_cannot_apply(tmp_path):
    manifest, before, _plan, plan_path = _draft(tmp_path)

    with pytest.raises(RepairPlanBlockedError, match="approved or rejected"):
        apply_repair_plan(plan_path)
    assert manifest.read_bytes() == before


def test_apply_validates_and_creates_readable_backup(tmp_path):
    manifest, before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    write_repair_plan(plan, plan_path)

    report = apply_repair_plan(plan_path)

    assert _read_rows(manifest)[0]["review_selected"] == "1"
    assert report["before_audit"]["summary"]["blocking_inconsistency_count"] == 1
    assert report["after_audit"]["summary"]["blocking_inconsistency_count"] == 0
    assert Path(report["backup_path"]).read_bytes() == before
    assert report["changed_records"][0]["before"]["review_selected"] == "0"
    assert report["changed_records"][0]["after"]["review_selected"] == "1"


def test_invalid_approved_changes_are_rejected_before_source_write(tmp_path):
    manifest, before, plan, plan_path = _draft(tmp_path)
    _approve(plan, changes={"training_selected": "not-a-boolean"})
    write_repair_plan(plan, plan_path)

    with pytest.raises(RepairPlanBlockedError, match="Invalid review record"):
        apply_repair_plan(plan_path)
    assert manifest.read_bytes() == before


def test_apply_failure_restores_original_manifest(tmp_path, monkeypatch):
    manifest, before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    write_repair_plan(plan, plan_path)
    import tools.review_repair as repair

    def fail_audit(*_args, **_kwargs):
        raise OSError("simulated append failure")

    monkeypatch.setattr(repair, "_append_audit_event", fail_audit)

    with pytest.raises(ReviewRepairError, match="was restored"):
        apply_repair_plan(plan_path)
    assert manifest.read_bytes() == before


def test_apply_is_idempotent_and_audit_is_append_only(tmp_path):
    manifest, _before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    write_repair_plan(plan, plan_path)

    first = apply_repair_plan(plan_path)
    audit_path = manifest.parent / ".review_repairs" / "repair_audit.jsonl"
    audit_after_first = audit_path.read_bytes()
    second = apply_repair_plan(plan_path)

    assert second == first
    assert audit_path.read_bytes() == audit_after_first
    assert len(audit_path.read_text(encoding="utf-8").splitlines()) == 1


def test_rollback_restores_exact_bytes_and_appends_audit(tmp_path):
    manifest, before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    write_repair_plan(plan, plan_path)
    applied = apply_repair_plan(plan_path)
    report_path = (
        manifest.parent / ".review_repairs" / "reports" / f"{plan['plan_id']}.json"
    )

    rolled_back = rollback_repair(report_path)
    repeated = rollback_repair(report_path)

    assert manifest.read_bytes() == before
    assert rolled_back["event"] == "rolled_back"
    assert repeated["event"] == "rolled_back"
    audit_path = manifest.parent / ".review_repairs" / "repair_audit.jsonl"
    events = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert [event["event"] for event in events] == ["applied", "rolled_back"]
    assert Path(applied["backup_path"]).is_file()


def test_rollback_rejects_post_apply_external_modification(tmp_path):
    manifest, _before, plan, plan_path = _draft(tmp_path)
    _approve(plan)
    write_repair_plan(plan, plan_path)
    apply_repair_plan(plan_path)
    report_path = (
        manifest.parent / ".review_repairs" / "reports" / f"{plan['plan_id']}.json"
    )
    _write_manifest(manifest, [_row(review_selected="1", failure_note="later")])

    with pytest.raises(StaleRepairPlanError, match="changed after repair"):
        rollback_repair(report_path)


def test_cross_scope_plan_does_not_modify_other_manifest(tmp_path):
    manifest_a, _before, plan, plan_path = _draft(tmp_path / "a")
    other_root = tmp_path / "b"
    other_root.mkdir()
    manifest_b = other_root / "review.csv"
    _write_manifest(manifest_b, [_row()])
    before_b = manifest_b.read_bytes()
    _approve(plan)
    plan["source"]["path"] = str(manifest_b.resolve())
    write_repair_plan(plan, plan_path)

    with pytest.raises(ReviewRepairError, match="immutable basis"):
        apply_repair_plan(plan_path)
    assert manifest_b.read_bytes() == before_b
    assert _read_rows(manifest_a)[0]["review_selected"] == "0"


def test_submitted_revision_requires_reason(tmp_path):
    manifest, before, plan, plan_path = _draft(
        tmp_path,
        _row(submission_status="submitted"),
    )
    _approve(plan)
    write_repair_plan(plan, plan_path)

    with pytest.raises(RepairPlanBlockedError, match="revision_reason"):
        apply_repair_plan(plan_path)
    assert manifest.read_bytes() == before

    _approve(plan, revision_reason="Correct legacy selection metadata")
    write_repair_plan(plan, plan_path)
    assert apply_repair_plan(plan_path)["mutation_performed"] is True


def test_completed_snapshot_is_never_mutated(tmp_path):
    manifest, before, plan, plan_path = _draft(
        tmp_path,
        _row(job_status="deployed"),
    )
    proposal = plan["proposals"][0]
    assert proposal["target_mutable"] is False
    _approve(plan, revision_reason="Historical correction request")
    write_repair_plan(plan, plan_path)

    with pytest.raises(RepairPlanBlockedError, match="Immutable snapshot"):
        apply_repair_plan(plan_path)
    assert manifest.read_bytes() == before


def test_uncertain_requires_manual_resolution_and_has_no_silent_change(tmp_path):
    _manifest, _before, plan, _plan_path = _draft(
        tmp_path,
        _row(
            review_outcome="",
            review_label="uncertain",
            product_verdict="unjudgeable",
            detection_verdict="unjudgeable",
            color_verdict="unjudgeable",
            action_route="none",
            training_selected="0",
        ),
    )

    proposal = plan["proposals"][0]
    assert "legacy_uncertain_label" in proposal["violation_codes"]
    assert proposal["requires_manual_decision"] is True
    assert proposal["confidence"] == "none"
    assert proposal["proposed_field_changes"] == {}


def test_inferable_missing_outcome_is_only_a_draft_proposal(tmp_path):
    manifest, before, plan, _plan_path = _draft(
        tmp_path,
        _row(
            review_selected="1",
            review_outcome="",
            review_label="wrong_box",
            failure_category="",
            product_verdict="ng",
            detection_verdict="wrong_box",
        ),
    )

    changes = plan["proposals"][0]["proposed_field_changes"]
    assert changes == {"review_outcome": "fail", "failure_category": "wrong_box"}
    assert manifest.read_bytes() == before


def test_approved_proposal_must_resolve_its_legacy_warnings(tmp_path):
    manifest, before, plan, plan_path = _draft(
        tmp_path,
        _row(review_outcome=""),
    )
    _approve(plan)
    write_repair_plan(plan, plan_path)

    with pytest.raises(RepairPlanBlockedError, match="legacy_review_outcome_missing"):
        apply_repair_plan(plan_path)
    assert manifest.read_bytes() == before


def test_human_annotation_conflict_requires_selection_and_preserves_labels(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest, [{**_row(), "review_selected": "1"}])
    first_label = tmp_path / "first.txt"
    second_label = tmp_path / "second.txt"
    first_label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    second_label.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    import tools.review_repair as repair

    first_hash = repair._normalized_label_sha256(first_label)
    second_hash = repair._normalized_label_sha256(second_label)
    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "conflicts": [
                    {
                        "image_sha256": "b" * 64,
                        "sample_ids": ["human-a", "human-b"],
                        "label_sha256s": [first_hash, second_hash],
                        "label_paths": [str(first_label), str(second_label)],
                        "source_types": ["human_annotation", "human_annotation"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    original_labels = (first_label.read_bytes(), second_label.read_bytes())
    plan = generate_repair_plan(manifest, conflict_report_paths=[conflict])
    annotation = next(
        item for item in plan["proposals"] if item["target_kind"] == "annotation_conflict"
    )
    annotation.update(
        {
            "approval_status": "approved",
            "approved": True,
            "reviewer": "reviewer",
            "decision_reason": "Human-a matches the physical evidence",
            "revision_reason": "Resolve submitted annotation conflict",
        }
    )
    annotation["resolution"] = {
        "mode": "selected_sample",
        "selected_sample_id": "human-a",
        "new_annotation_path": "",
        "new_label_sha256": "",
    }
    plan_path = tmp_path / "annotation-plan.json"
    write_repair_plan(plan, plan_path)

    report = apply_repair_plan(plan_path)

    resolution_path = Path(report["annotation_resolution_paths"][0])
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert resolution["resolution"]["selected_sample_id"] == "human-a"
    assert resolution["old_annotations_preserved"] is True
    assert (first_label.read_bytes(), second_label.read_bytes()) == original_labels


def test_needs_reannotation_never_selects_an_old_label(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest, [{**_row(), "review_selected": "1"}])
    label_a = tmp_path / "a.txt"
    label_b = tmp_path / "b.txt"
    label_a.write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    label_b.write_text("1 0.5 0.5 0.1 0.1", encoding="utf-8")
    import tools.review_repair as repair

    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "conflicts": [
                    {
                        "image_sha256": "c" * 64,
                        "sample_ids": ["a", "b"],
                        "label_sha256s": [
                            repair._normalized_label_sha256(label_a),
                            repair._normalized_label_sha256(label_b),
                        ],
                        "label_paths": [str(label_a), str(label_b)],
                        "source_types": ["human_annotation", "human_annotation"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = generate_repair_plan(manifest, conflict_report_paths=[conflict])
    for proposal in plan["proposals"]:
        proposal["reviewer"] = "reviewer"
        proposal["decision_reason"] = "Reviewed"
        if proposal["target_kind"] == "workflow_record":
            proposal["approval_status"] = "rejected"
        else:
            proposal["approval_status"] = "approved"
            proposal["approved"] = True
            proposal["revision_reason"] = "Both old labels are wrong"
            proposal["resolution"] = {
                "mode": "needs_reannotation",
                "selected_sample_id": "",
                "new_annotation_path": "",
                "new_label_sha256": "",
            }
    plan_path = tmp_path / "plan.json"
    write_repair_plan(plan, plan_path)

    resolution = json.loads(
        Path(apply_repair_plan(plan_path)["annotation_resolution_paths"][0]).read_text()
    )
    assert resolution["resolution"]["mode"] == "needs_reannotation"
    assert resolution["resolution"]["selected_sample_id"] == ""


def test_annotation_resolution_rollback_appends_revocation(tmp_path):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest, [{**_row(), "review_selected": "1"}])
    label_a = tmp_path / "a.txt"
    label_b = tmp_path / "b.txt"
    label_a.write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    label_b.write_text("1 0.5 0.5 0.1 0.1", encoding="utf-8")
    import tools.review_repair as repair

    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "conflicts": [
                    {
                        "image_sha256": "d" * 64,
                        "sample_ids": ["a", "b"],
                        "label_sha256s": [
                            repair._normalized_label_sha256(label_a),
                            repair._normalized_label_sha256(label_b),
                        ],
                        "label_paths": [str(label_a), str(label_b)],
                        "source_types": ["human_annotation", "human_annotation"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    plan = generate_repair_plan(manifest, conflict_report_paths=[conflict])
    proposal = plan["proposals"][0]
    proposal.update(
        {
            "approval_status": "approved",
            "approved": True,
            "reviewer": "reviewer",
            "decision_reason": "Both labels require a new review",
            "revision_reason": "Reopen submitted annotation",
        }
    )
    proposal["resolution"] = {
        "mode": "needs_reannotation",
        "selected_sample_id": "",
        "new_annotation_path": "",
        "new_label_sha256": "",
    }
    plan_path = tmp_path / "plan.json"
    write_repair_plan(plan, plan_path)
    applied = apply_repair_plan(plan_path)
    applied_report = (
        manifest.parent / ".review_repairs" / "reports" / f"{plan['plan_id']}.json"
    )

    rolled_back = rollback_repair(applied_report)

    resolution_path = Path(applied["annotation_resolution_paths"][0])
    assert resolution_path.is_file()
    revocation = Path(rolled_back["annotation_revocation_paths"][0])
    assert revocation.is_file()
    assert json.loads(revocation.read_text())["event"] == "annotation_resolution_revoked"


def test_reaudit_exit_code_changes_after_repair(tmp_path):
    manifest, _before, plan, plan_path = _draft(tmp_path)
    before_report = audit_review_workflow([manifest])
    _approve(plan)
    write_repair_plan(plan, plan_path)
    apply_repair_plan(plan_path)
    after_report = audit_review_workflow([manifest])

    assert repair_audit_exit_code(before_report) == 2
    assert repair_audit_exit_code(after_report) == 0
    assert repair_audit_exit_code({"summary": {"error_count": 1}}) == 1
    assert repair_audit_exit_code({"not_summary": {}}) == 1


def test_cli_generate_apply_and_rollback_exit_codes(tmp_path, capsys):
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest, [_row()])
    plan_path = tmp_path / "plan.json"

    assert generate_main(
        ["--manifest", str(manifest), "--output", str(plan_path)]
    ) == 2
    capsys.readouterr()
    assert apply_main(["--plan", str(plan_path)]) == 2
    capsys.readouterr()

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _approve(plan)
    write_repair_plan(plan, plan_path)
    assert apply_main(["--plan", str(plan_path)]) == 0
    capsys.readouterr()

    report = (
        manifest.parent
        / ".review_repairs"
        / "reports"
        / f"{plan['plan_id']}.json"
    )
    assert rollback_main(["--report", str(report)]) == 0
