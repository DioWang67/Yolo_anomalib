from __future__ import annotations

import base64
import csv
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from app.gui.historical_cleanup_view_model import HistoricalCleanupViewModel
from tools.audit_historical_cleanup import main as audit_main
from tools.historical_cleanup import (
    CleanupConfidence,
    CleanupDecisionError,
    HistoricalCleanupAnalyzer,
    HistoricalCleanupSession,
    cleanup_audit_exit_code,
)

FIELDS = [
    "sample_id",
    "timestamp",
    "product",
    "area",
    "machine_id",
    "work_order",
    "camera_id",
    "model_version",
    "config_snapshot_path",
    "original_path",
    "preprocessed_path",
    "annotated_path",
    "detections_json",
    "class_names_json",
    "output_label",
    "annotation_status",
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

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _row(sample_id: str, image: Path, config: Path, **changes: str) -> dict[str, str]:
    row = {
        "sample_id": sample_id,
        "timestamp": "2026-07-21T00:00:00Z",
        "product": "Cable1",
        "area": "A",
        "machine_id": "M1",
        "work_order": "WO1",
        "camera_id": "CAM1",
        "model_version": "yolo11_test_v1",
        "config_snapshot_path": str(config),
        "original_path": str(image),
        "preprocessed_path": str(image),
        "annotated_path": str(image),
        "detections_json": "[]",
        "class_names_json": '["defect"]',
        "output_label": "",
        "annotation_status": "",
        "review_selected": "0",
        "review_outcome": "",
        "review_label": "",
        "failure_category": "",
        "failure_note": "",
        "skip_reason": "",
        "product_verdict": "",
        "detection_verdict": "",
        "color_verdict": "",
        "action_route": "",
        "training_selected": "1",
        "submission_status": "",
        "job_status": "",
    }
    row.update(changes)
    return row


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> tuple[Path, list[dict[str, str]]]:
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_1X1)
    config = tmp_path / "snapshot.json"
    config.write_text("{}", encoding="utf-8")
    rows = [
        _row("manual", image, config),
        _row(
            "ambiguous",
            image,
            config,
            review_label="false_positive",
            product_verdict="ok",
            detection_verdict="false_positive",
            color_verdict="not_applicable",
            action_route="yolo",
            training_selected="0",
        ),
        _row(
            "uncertain",
            image,
            config,
            review_label="uncertain",
            product_verdict="unjudgeable",
            detection_verdict="unjudgeable",
            color_verdict="unjudgeable",
            action_route="none",
            training_selected="0",
        ),
        _row(
            "inferable",
            image,
            config,
            review_label="false_negative",
            product_verdict="ng",
            detection_verdict="missed",
            color_verdict="not_applicable",
            action_route="yolo",
            training_selected="0",
        ),
        _row(
            "excluded",
            image,
            config,
            review_selected="1",
            review_outcome="pass",
            review_label="confirmed_ok",
            product_verdict="ok",
            detection_verdict="correct",
            color_verdict="not_applicable",
            action_route="none",
            training_selected="0",
        ),
    ]
    manifest = tmp_path / "review.csv"
    _write_manifest(manifest, rows)
    return manifest, rows


def _analysis(tmp_path: Path):
    manifest, rows = _fixture(tmp_path)
    return manifest, rows, HistoricalCleanupAnalyzer().analyze(manifest)


def test_grouping_classifies_all_planner_blockers_and_confidence(tmp_path):
    _manifest, _rows, analysis = _analysis(tmp_path)

    assert analysis.planner_statistics == {
        "ready_count": 0,
        "annotation_count": 0,
        "color_count": 0,
        "blocking_count": 4,
        "excluded_count": 1,
        "manual_review_count": 1,
    }
    assert sum(group.record_count for group in analysis.groups) == 4
    groups = {
        (group.root_cause, group.confidence): group for group in analysis.groups
    }
    assert groups[("legacy_review_outcome_missing", "NONE")].record_count == 1
    assert groups[("legacy_uncertain", "NONE")].record_count == 1
    assert groups[("review_without_selection", "LOW")].record_count == 1
    inferable = groups[("review_without_selection", "MEDIUM")]
    assert inferable.record_count == 1
    assert inferable.batch_approvable is True
    assert inferable.blocking_removed_estimate == 1


def test_high_confidence_requires_explicit_outcome_and_no_other_blocker(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_1X1)
    config = tmp_path / "snapshot.json"
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    _write_manifest(
        manifest,
        [
            _row(
                "high",
                image,
                config,
                review_outcome="pass",
                review_label="false_positive",
                product_verdict="ok",
                detection_verdict="false_positive",
                color_verdict="not_applicable",
                action_route="yolo",
                training_selected="0",
            )
        ],
    )

    analysis = HistoricalCleanupAnalyzer().analyze(manifest)

    assert len(analysis.groups) == 1
    assert analysis.groups[0].confidence == CleanupConfidence.HIGH.value
    assert analysis.groups[0].batch_approvable is True


def test_proposal_and_json_csv_exports_are_read_only(tmp_path):
    manifest, _rows, analysis = _analysis(tmp_path)
    original = manifest.read_bytes()
    session = HistoricalCleanupSession(analysis)
    json_path = session.export_json(tmp_path / "proposal.json")
    csv_path = session.export_csv(tmp_path / "proposal.csv")

    exported = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported["mode"] == "dry_run"
    assert exported["manifest_sha256"] == analysis.manifest_sha256
    assert len(exported["groups"]) == 4
    assert "root_cause" in csv_path.read_text(encoding="utf-8-sig")
    assert manifest.read_bytes() == original


def test_group_and_single_decisions_require_identity_and_safe_proposal(tmp_path):
    _manifest, _rows, analysis = _analysis(tmp_path)
    session = HistoricalCleanupSession(analysis)
    groups = {(item.root_cause, item.confidence): item for item in analysis.groups}
    medium = groups[("review_without_selection", "MEDIUM")]
    low = groups[("review_without_selection", "LOW")]

    with pytest.raises(CleanupDecisionError, match="Reviewer"):
        session.approve_group(medium.group_id, reviewer="", reason="checked")
    session.approve_group(medium.group_id, reviewer="reviewer", reason="checked")
    assert session.decision_for(medium.record_ids[0])["status"] == "approved"
    with pytest.raises(CleanupDecisionError, match="not batch approvable"):
        session.approve_group(low.group_id, reviewer="reviewer", reason="checked")
    with pytest.raises(CleanupDecisionError, match="no complete Phase 1C-safe"):
        session.approve_record(low.record_ids[0], reviewer="reviewer", reason="checked")
    session.reject_record(low.record_ids[0], reviewer="reviewer", reason="ambiguous")
    assert session.decision_for(low.record_ids[0])["status"] == "rejected"


def test_apply_delegates_to_phase1c_then_audits_planner_and_rollback(tmp_path):
    manifest, _rows, analysis = _analysis(tmp_path)
    original = manifest.read_bytes()
    session = HistoricalCleanupSession(analysis)
    groups = {(item.root_cause, item.confidence): item for item in analysis.groups}
    session.approve_group(
        groups[("review_without_selection", "MEDIUM")].group_id,
        reviewer="reviewer",
        reason="Verified legacy false-negative mapping",
    )
    session.reject_group(
        groups[("review_without_selection", "LOW")].group_id,
        reviewer="reviewer",
        reason="Outcome is ambiguous",
    )
    session.reject_group(
        groups[("legacy_uncertain", "NONE")].group_id,
        reviewer="reviewer",
        reason="Requires a new review",
    )

    report = session.apply()

    assert report["before"]["planner"]["blocking_count"] == 4
    assert report["after"]["planner_statistics"]["blocking_count"] == 3
    assert report["remaining_blocking"] == 3
    assert report["repair_ids"]
    assert report["report_sha256"]
    assert report["manifest_sha256"] != analysis.manifest_sha256
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        repaired = list(csv.DictReader(handle))
    assert repaired[3]["review_selected"] == "1"
    assert repaired[3]["review_outcome"] == "fail"
    assert repaired[3]["failure_category"] == "missed_detection"

    rollback = session.rollback()

    assert rollback["phase1c_rollback"]["event"] == "rolled_back"
    assert rollback["restored"]["planner_statistics"]["blocking_count"] == 4
    assert manifest.read_bytes() == original


def test_stale_manifest_sha_blocks_before_phase1c_apply(tmp_path):
    manifest, rows, analysis = _analysis(tmp_path)
    session = HistoricalCleanupSession(analysis)
    groups = {(item.root_cause, item.confidence): item for item in analysis.groups}
    session.approve_group(
        groups[("review_without_selection", "MEDIUM")].group_id,
        reviewer="reviewer",
        reason="checked",
    )
    session.reject_group(
        groups[("review_without_selection", "LOW")].group_id,
        reviewer="reviewer",
        reason="ambiguous",
    )
    session.reject_group(
        groups[("legacy_uncertain", "NONE")].group_id,
        reviewer="reviewer",
        reason="uncertain",
    )
    rows[0]["failure_note"] = "external edit"
    _write_manifest(manifest, rows)

    with pytest.raises(CleanupDecisionError, match="manifest SHA changed"):
        session.apply()


def test_view_model_pages_and_exposes_proposals_without_routing_logic(tmp_path):
    _manifest, _rows, analysis = _analysis(tmp_path)
    view_model = HistoricalCleanupViewModel(analysis, page_size=1)
    group = next(item for item in view_model.groups if item.record_count == 1)

    page = view_model.records_page(group.group_id)

    assert page.total_count == 1
    assert len(page.records) == 1
    assert "Suggested Fix:" in view_model.group_detail(group.group_id)
    assert "Phase 1C Ready:" in view_model.record_detail(page.records[0].record_id)


def test_dry_run_exit_codes(tmp_path, capsys):
    manifest, _rows, analysis = _analysis(tmp_path)
    assert cleanup_audit_exit_code(analysis) == 2
    assert audit_main([str(manifest)]) == 2
    capsys.readouterr()
    assert audit_main([str(tmp_path / "missing.csv")]) == 1


def test_missing_and_invalid_annotation_evidence_override_generic_manual_group(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_1X1)
    config = tmp_path / "snapshot.json"
    config.write_text("{}", encoding="utf-8")
    bad_bbox = tmp_path / "bad-bbox.txt"
    bad_bbox.write_text("0 0.5 0.5 1.2 0.1\n", encoding="utf-8")
    bad_class = tmp_path / "bad-class.txt"
    bad_class.write_text("4 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    missing = tmp_path / "missing.png"
    manifest = tmp_path / "evidence.csv"
    _write_manifest(
        manifest,
        [
            _row("missing", missing, config),
            _row(
                "bbox",
                image,
                config,
                annotation_status="verified_annotation",
                output_label=str(bad_bbox),
            ),
            _row(
                "class",
                image,
                config,
                annotation_status="verified_annotation",
                output_label=str(bad_class),
            ),
        ],
    )

    analysis = HistoricalCleanupAnalyzer().analyze(manifest)

    assert analysis.root_cause_counts == {
        "invalid_bbox": 1,
        "invalid_class": 1,
        "missing_image": 1,
    }
    assert all(group.confidence == "LOW" for group in analysis.groups)


def test_annotation_conflict_is_manual_only_and_affects_exit_code(tmp_path):
    image = tmp_path / "frame.png"
    image.write_bytes(PNG_1X1)
    config = tmp_path / "snapshot.json"
    config.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "review.csv"
    _write_manifest(
        manifest,
        [
            _row(
                "excluded",
                image,
                config,
                review_selected="1",
                review_outcome="pass",
                review_label="confirmed_ok",
                product_verdict="ok",
                detection_verdict="correct",
                color_verdict="not_applicable",
                action_route="none",
                training_selected="0",
            )
        ],
    )
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    second.write_text("0 0.4 0.4 0.1 0.1\n", encoding="utf-8")
    import tools.review_repair as repair

    conflict = tmp_path / "conflict.json"
    conflict.write_text(
        json.dumps(
            {
                "conflicts": [
                    {
                        "image_sha256": hashlib.sha256(PNG_1X1).hexdigest(),
                        "sample_ids": ["human-a", "human-b"],
                        "label_sha256s": [
                            repair._normalized_label_sha256(first),
                            repair._normalized_label_sha256(second),
                        ],
                        "label_paths": [str(first), str(second)],
                        "source_types": ["human_annotation", "human_annotation"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    analysis = HistoricalCleanupAnalyzer().analyze(
        manifest, conflict_report_paths=[conflict]
    )

    assert analysis.planner_statistics["blocking_count"] == 0
    assert analysis.root_cause_counts == {"annotation_conflict": 1}
    assert analysis.groups[0].confidence == "NONE"
    assert analysis.groups[0].batch_approvable is False
    assert cleanup_audit_exit_code(analysis) == 2


def test_cleanup_apply_is_only_a_facade_over_phase1c(tmp_path, monkeypatch):
    manifest, _rows, analysis = _analysis(tmp_path)
    original = manifest.read_bytes()
    session = HistoricalCleanupSession(analysis)
    groups = {(item.root_cause, item.confidence): item for item in analysis.groups}
    session.approve_group(
        groups[("review_without_selection", "MEDIUM")].group_id,
        reviewer="reviewer",
        reason="checked",
    )
    session.reject_group(
        groups[("review_without_selection", "LOW")].group_id,
        reviewer="reviewer",
        reason="ambiguous",
    )
    session.reject_group(
        groups[("legacy_uncertain", "NONE")].group_id,
        reviewer="reviewer",
        reason="uncertain",
    )
    called = []

    def fake_phase1c(plan_path):
        called.append(Path(plan_path))
        return {
            "plan_path": str(plan_path),
            "repair_id": "repair-delegated",
            "backup_path": str(tmp_path / "backup.csv"),
        }

    monkeypatch.setattr("tools.historical_cleanup.apply_repair_plan", fake_phase1c)

    session.apply()

    assert called == [session.adjudicated_plan_path]
    assert manifest.read_bytes() == original
    source = inspect.getsource(HistoricalCleanupSession.apply)
    assert source.count("apply_repair_plan(") == 1
