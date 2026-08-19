"""Safe, approval-gated repair workflow for legacy review manifests.

Repair plans are intentionally external JSON artifacts.  Existing CSV columns
remain the source of truth, while every mutation is protected by source and
record hashes, validated through :mod:`tools.review_workflow`, backed up, and
recorded in an append-only JSONL audit.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import uuid
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.audit_review_workflow import audit_review_workflow
from tools.review_workflow import (
    ReviewWorkflowValidationError,
    derive_review_semantics,
    ensure_record_consistent,
    ensure_transition_valid,
    record_identity,
    validate_record_consistency,
)

PLAN_SCHEMA_VERSION = 1
REPAIR_ARTIFACT_DIR = ".review_repairs"
APPROVAL_STATUSES = frozenset({"pending", "approved", "rejected"})
ANNOTATION_RESOLUTION_MODES = frozenset(
    {"selected_sample", "new_annotation_revision", "needs_reannotation"}
)
COMPLETED_JOB_STATUSES = frozenset({"deployed", "completed"})
SUBMITTED_STATUSES = frozenset(
    {
        "submitted",
        "queued",
        "waiting_feedback",
        "waiting_annotation",
        "preparing_dataset",
        "training",
        "evaluating",
        "deploying",
        "deployed",
        "completed",
        "failed",
    }
)


class ReviewRepairError(RuntimeError):
    """Base error for malformed artifacts or failed repair I/O."""


class RepairPlanBlockedError(ReviewRepairError):
    """The plan is readable but cannot be applied without human correction."""


class StaleRepairPlanError(RepairPlanBlockedError):
    """The source or target record changed after proposal generation."""


def generate_repair_plan(
    manifest_path: str | Path,
    *,
    conflict_report_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    """Create a non-mutating draft plan from Phase 1B diagnostics."""
    manifest = Path(manifest_path).expanduser().resolve()
    fieldnames, rows, source_bytes = _read_manifest(manifest)
    source_sha256 = _sha256(source_bytes)
    report = audit_review_workflow([manifest])
    if report["summary"]["error_count"]:
        raise ReviewRepairError(
            f"Cannot generate repair plan because audit failed: {report['errors']}"
        )

    created_at = _utc_now()
    proposals: list[dict[str, Any]] = []
    for diagnostic in report["inconsistent_records"]:
        row_number = int(diagnostic["row_number"])
        row_index = row_number - 2
        if not 0 <= row_index < len(rows):
            raise ReviewRepairError(
                f"Audit returned invalid row number {row_number} for {manifest}"
            )
        row = rows[row_index]
        proposals.append(
            _workflow_proposal(
                manifest=manifest,
                row=row,
                row_number=row_number,
                diagnostic=diagnostic,
                created_at=created_at,
            )
        )

    conflict_reports: list[dict[str, str]] = []
    for conflict_path_value in conflict_report_paths:
        conflict_path = Path(conflict_path_value).expanduser().resolve()
        conflict_bytes = _read_bytes(conflict_path)
        conflict_payload = _read_json_object(conflict_path)
        conflict_reports.append(
            {"path": str(conflict_path), "sha256": _sha256(conflict_bytes)}
        )
        conflicts = conflict_payload.get("conflicts")
        if not isinstance(conflicts, list):
            raise ReviewRepairError(
                f"Conflict report has no conflicts list: {conflict_path}"
            )
        for conflict_index, conflict in enumerate(conflicts):
            if not isinstance(conflict, dict):
                raise ReviewRepairError(
                    f"Conflict entry {conflict_index} is not an object"
                )
            proposals.append(
                _annotation_conflict_proposal(
                    conflict_path=conflict_path,
                    conflict_report_sha256=_sha256(conflict_bytes),
                    conflict_index=conflict_index,
                    conflict=conflict,
                    created_at=created_at,
                )
            )

    plan_id = f"repair-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:12]}"
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": plan_id,
        "mode": "proposal_only",
        "created_at": created_at,
        "source": {
            "type": "review_manifest",
            "path": str(manifest),
            "sha256": source_sha256,
            "fieldnames": fieldnames,
            "record_count": len(rows),
        },
        "conflict_reports": conflict_reports,
        "proposal_count": len(proposals),
        "proposals": proposals,
        "approval_instructions": {
            "required_statuses": ["approved", "rejected"],
            "approved_flag_required": True,
            "reviewer_required": True,
            "decision_reason_required": True,
            "revision_reason_required_for_submitted_or_completed": True,
        },
    }
    plan["plan_basis_sha256"] = _plan_basis_sha256(plan)
    return plan


def write_repair_plan(plan: Mapping[str, Any], destination: str | Path) -> Path:
    """Atomically write a proposal without modifying any source data."""
    path = Path(destination).expanduser().resolve()
    _write_json_atomic(path, dict(plan))
    return path


def default_plan_path(plan: Mapping[str, Any]) -> Path:
    source = _plan_source_path(plan)
    return source.parent / REPAIR_ARTIFACT_DIR / "plans" / f"{plan['plan_id']}.json"


def apply_repair_plan(
    plan_path: str | Path,
    *,
    audit_log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply one fully adjudicated plan, or return its existing receipt."""
    plan_file = Path(plan_path).expanduser().resolve()
    plan = _read_json_object(plan_file)
    _validate_plan_structure(plan)
    source = _plan_source_path(plan)
    audit_path = _audit_log_path(source, audit_log_path)
    prior = _latest_plan_event(audit_path, str(plan["plan_id"]))
    if prior and prior.get("event") == "applied":
        report_path = Path(str(prior["report_path"]))
        if report_path.is_file():
            return _read_json_object(report_path)
        raise ReviewRepairError(
            f"Applied audit exists but its report is missing: {report_path}"
        )

    approved, annotation_approved = _validate_approvals(plan)
    source_lock = source.with_name(f".{source.name}.lock")
    with _exclusive_lock(source_lock):
        fieldnames, rows, source_bytes = _read_manifest(source)
        current_sha = _sha256(source_bytes)
        expected_sha = str(plan["source"]["sha256"])
        if current_sha != expected_sha:
            raise StaleRepairPlanError(
                "Repair plan is stale: source manifest SHA changed "
                f"from {expected_sha} to {current_sha}"
            )
        if fieldnames != list(plan["source"]["fieldnames"]):
            raise StaleRepairPlanError("Repair plan is stale: CSV header changed")
        if len(rows) != int(plan["source"]["record_count"]):
            raise StaleRepairPlanError("Repair plan is stale: record count changed")

        before_report = audit_review_workflow([source])
        updated_rows = [dict(row) for row in rows]
        changed_records: list[dict[str, Any]] = []
        for proposal in approved:
            row_index = _validate_record_locator(proposal, source, updated_rows)
            before = dict(updated_rows[row_index])
            changes = proposal.get("proposed_field_changes")
            if not isinstance(changes, dict) or not changes:
                raise RepairPlanBlockedError(
                    f"Approved proposal {proposal['proposal_id']} has no field changes"
                )
            unknown_fields = sorted(set(changes) - set(fieldnames))
            if unknown_fields:
                raise RepairPlanBlockedError(
                    f"Proposal {proposal['proposal_id']} would change unknown CSV fields: "
                    + ", ".join(unknown_fields)
                )
            after = {**before, **{str(key): str(value) for key, value in changes.items()}}
            before_context = {**before, **_proposal_lifecycle_context(proposal)}
            after_context = {
                **after,
                **_proposal_lifecycle_context(proposal),
                "revision_reason": str(proposal.get("revision_reason") or ""),
            }
            try:
                ensure_record_consistent(after_context)
                ensure_transition_valid(before_context, after_context)
            except ReviewWorkflowValidationError as exc:
                raise RepairPlanBlockedError(str(exc)) from exc
            unresolved_codes = {
                violation.code for violation in validate_record_consistency(after_context)
            } & {str(code) for code in proposal.get("violation_codes", [])}
            if unresolved_codes:
                raise RepairPlanBlockedError(
                    f"Approved proposal {proposal['proposal_id']} does not resolve: "
                    + ", ".join(sorted(unresolved_codes))
                )
            _require_revision_reason(proposal)
            updated_rows[row_index] = after
            changed_records.append(
                {
                    "proposal_id": proposal["proposal_id"],
                    "scope": proposal["scope"],
                    "sample_id": proposal["sample_id"],
                    "row_number": proposal["record_locator"]["row_number"],
                    "before": before,
                    "after": after,
                    "reviewer": proposal["reviewer"],
                    "decision_reason": proposal["decision_reason"],
                    "revision_reason": str(proposal.get("revision_reason") or ""),
                }
            )

        repair_root = source.parent / REPAIR_ARTIFACT_DIR
        repair_id = str(plan["plan_id"])
        backup_path = repair_root / "backups" / repair_id / source.name
        report_path = repair_root / "reports" / f"{repair_id}.json"
        resolution_paths: list[Path] = []
        new_bytes = _serialize_manifest(
            fieldnames,
            updated_rows,
            include_bom=source_bytes.startswith(b"\xef\xbb\xbf"),
        )
        _create_backup(backup_path, source_bytes)
        try:
            if changed_records:
                _write_bytes_atomic(source, new_bytes)
            for proposal in annotation_approved:
                resolution_paths.append(
                    _write_annotation_resolution(proposal, repair_root)
                )
            after_report = audit_review_workflow([source])
            report = {
                "schema_version": 1,
                "repair_id": repair_id,
                "event": "applied",
                "applied_at": _utc_now(),
                "plan_path": str(plan_file),
                "plan_basis_sha256": plan["plan_basis_sha256"],
                "source_manifest": str(source),
                "before_sha256": current_sha,
                "after_sha256": _sha256(_read_bytes(source)),
                "backup_path": str(backup_path),
                "changed_records": changed_records,
                "annotation_resolution_paths": [str(path) for path in resolution_paths],
                "rejected_proposals": [
                    proposal["proposal_id"]
                    for proposal in plan["proposals"]
                    if proposal["approval_status"] == "rejected"
                ],
                "before_audit": before_report,
                "after_audit": after_report,
                "mutation_performed": bool(changed_records or resolution_paths),
            }
            _write_json_atomic(report_path, report)
            _append_audit_event(
                audit_path,
                {
                    "schema_version": 1,
                    "event": "applied",
                    "repair_id": repair_id,
                    "plan_id": repair_id,
                    "timestamp": report["applied_at"],
                    "source_manifest": str(source),
                    "before_sha256": report["before_sha256"],
                    "after_sha256": report["after_sha256"],
                    "backup_path": str(backup_path),
                    "report_path": str(report_path),
                    "changed_record_count": len(changed_records),
                    "annotation_resolution_count": len(resolution_paths),
                },
            )
            return report
        except RepairPlanBlockedError:
            if changed_records and backup_path.is_file():
                _write_bytes_atomic(source, _read_bytes(backup_path))
            for resolution_path in resolution_paths:
                resolution_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            if changed_records and backup_path.is_file():
                _write_bytes_atomic(source, _read_bytes(backup_path))
            for resolution_path in resolution_paths:
                resolution_path.unlink(missing_ok=True)
            report_path.unlink(missing_ok=True)
            raise ReviewRepairError(
                f"Repair apply failed and source manifest was restored: {exc}"
            ) from exc


def rollback_repair(
    report_path: str | Path,
    *,
    audit_log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Restore the exact pre-repair bytes when the applied state is unchanged."""
    applied_report_path = Path(report_path).expanduser().resolve()
    applied = _read_json_object(applied_report_path)
    if applied.get("event") != "applied":
        raise ReviewRepairError("Rollback input is not an applied repair report")
    source = Path(str(applied["source_manifest"])).resolve()
    backup = Path(str(applied["backup_path"])).resolve()
    audit_path = _audit_log_path(source, audit_log_path)
    rollback_report_path = applied_report_path.with_name(
        f"{applied['repair_id']}.rollback.json"
    )
    lock_path = source.with_name(f".{source.name}.lock")
    with _exclusive_lock(lock_path):
        current = _read_bytes(source)
        current_sha = _sha256(current)
        before_sha = str(applied["before_sha256"])
        after_sha = str(applied["after_sha256"])
        annotation_paths = [
            Path(str(value)).resolve()
            for value in applied.get("annotation_resolution_paths", [])
        ]
        if rollback_report_path.is_file():
            return _read_json_object(rollback_report_path)
        if current_sha == before_sha and not annotation_paths:
            return {
                "schema_version": 1,
                "repair_id": applied["repair_id"],
                "event": "rolled_back",
                "idempotent": True,
                "source_manifest": str(source),
                "after_sha256": before_sha,
            }
        if current_sha != after_sha:
            raise StaleRepairPlanError(
                "Rollback refused: source changed after repair "
                f"(expected {after_sha}, got {current_sha})"
            )
        backup_bytes = _read_bytes(backup)
        if _sha256(backup_bytes) != before_sha:
            raise ReviewRepairError("Rollback backup SHA does not match repair report")

        pre_rollback_backup = backup.with_name(f"{source.name}.pre-rollback")
        _create_backup(pre_rollback_backup, current)
        revocation_paths: list[Path] = []
        try:
            if current_sha != before_sha:
                _write_bytes_atomic(source, backup_bytes)
            for resolution_path in annotation_paths:
                if not resolution_path.is_file():
                    raise ReviewRepairError(
                        f"Annotation resolution is missing: {resolution_path}"
                    )
                revocation_path = resolution_path.with_suffix(
                    f"{resolution_path.suffix}.rollback.json"
                )
                _write_json_atomic(
                    revocation_path,
                    {
                        "schema_version": 1,
                        "event": "annotation_resolution_revoked",
                        "revoked_at": _utc_now(),
                        "repair_id": applied["repair_id"],
                        "resolution_path": str(resolution_path),
                        "resolution_sha256": _sha256(_read_bytes(resolution_path)),
                    },
                )
                revocation_paths.append(revocation_path)
            audit = audit_review_workflow([source])
            rollback_report = {
                "schema_version": 1,
                "repair_id": applied["repair_id"],
                "event": "rolled_back",
                "rolled_back_at": _utc_now(),
                "source_manifest": str(source),
                "from_sha256": current_sha,
                "after_sha256": _sha256(_read_bytes(source)),
                "restored_backup_path": str(backup),
                "pre_rollback_backup_path": str(pre_rollback_backup),
                "annotation_revocation_paths": [
                    str(path) for path in revocation_paths
                ],
                "audit": audit,
                "idempotent": False,
            }
            _write_json_atomic(rollback_report_path, rollback_report)
            _append_audit_event(
                audit_path,
                {
                    "schema_version": 1,
                    "event": "rolled_back",
                    "repair_id": applied["repair_id"],
                    "plan_id": applied["repair_id"],
                    "timestamp": rollback_report["rolled_back_at"],
                    "source_manifest": str(source),
                    "before_sha256": current_sha,
                    "after_sha256": rollback_report["after_sha256"],
                    "report_path": str(rollback_report_path),
                },
            )
            return rollback_report
        except Exception as exc:
            _write_bytes_atomic(source, current)
            for revocation_path in revocation_paths:
                revocation_path.unlink(missing_ok=True)
            rollback_report_path.unlink(missing_ok=True)
            raise ReviewRepairError(
                f"Rollback failed and the applied state was restored: {exc}"
            ) from exc


def repair_audit_exit_code(report: Mapping[str, Any]) -> int:
    """Use the same 0/1/2 contract as the Phase 1B audit CLI."""
    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return 1
    if int(summary.get("error_count", 0)):
        return 1
    return 2 if int(summary.get("blocking_inconsistency_count", 0)) else 0


def _workflow_proposal(
    *,
    manifest: Path,
    row: dict[str, str],
    row_number: int,
    diagnostic: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    violations = [dict(item) for item in diagnostic["violations"]]
    codes = [str(item["code"]) for item in violations]
    changes, confidence, requires_manual = _suggest_field_changes(row, codes)
    blocking = any(bool(item.get("blocking")) for item in violations)
    requires_manual = requires_manual or blocking
    immutable_target = _is_immutable_scope(manifest, row)
    if immutable_target:
        changes = {}
        requires_manual = True
    proposal: dict[str, Any] = {
        "proposal_id": f"workflow-row-{row_number}-{uuid.uuid4().hex[:10]}",
        "target_kind": "workflow_record",
        "scope": str(manifest),
        "sample_id": record_identity(row),
        "image_sha256": _image_sha256(row),
        "record_locator": {
            "row_number": row_number,
            "sample_id": record_identity(row),
            "record_sha256": _record_sha256(row),
        },
        "original_fields": dict(row),
        "derived_semantics": derive_review_semantics(row).to_dict(),
        "violation_code": codes[0],
        "violation_codes": codes,
        "violations": violations,
        "severity": "blocking" if blocking else "warning",
        "suggested_action": " ".join(
            str(item.get("suggestion") or "") for item in violations
        ).strip(),
        "proposed_field_changes": changes,
        "confidence": confidence,
        "requires_manual_decision": requires_manual,
        "target_mutable": not immutable_target,
        "approval_status": "pending",
        "approved": False,
        "reviewer": "",
        "decision_reason": "",
        "revision_reason": "",
        "created_at": created_at,
        "lifecycle_context": {
            "submission_status": str(row.get("submission_status") or ""),
            "job_status": str(row.get("job_status") or ""),
        },
    }
    proposal["proposal_basis_sha256"] = _proposal_basis_sha256(proposal)
    return proposal


def _annotation_conflict_proposal(
    *,
    conflict_path: Path,
    conflict_report_sha256: str,
    conflict_index: int,
    conflict: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    sample_ids = _string_list(conflict.get("sample_ids"))
    label_sha256s = _string_list(conflict.get("label_sha256s"))
    label_paths = _string_list(conflict.get("label_paths"))
    if len(sample_ids) < 2 or len(label_sha256s) != len(sample_ids):
        raise ReviewRepairError(
            f"Invalid human annotation conflict in {conflict_path}: sample/hash mismatch"
        )
    source_types = _string_list(conflict.get("source_types"))
    violation_code = (
        "human_annotation_conflict"
        if source_types and all(value.startswith("human_") for value in source_types)
        else "canonical_label_conflict"
    )
    conflicting_sample_ids = (
        sample_ids
        if len(set(sample_ids)) == len(sample_ids)
        else [
            f"{sample_id}@label-{label_sha[:12]}"
            for sample_id, label_sha in zip(sample_ids, label_sha256s, strict=True)
        ]
    )
    original_sample_ids = _string_list(conflict.get("original_sample_ids")) or sample_ids
    proposal: dict[str, Any] = {
        "proposal_id": f"annotation-conflict-{conflict_index}-{uuid.uuid4().hex[:10]}",
        "target_kind": "annotation_conflict",
        "scope": str(conflict_path),
        "sample_id": ",".join(conflicting_sample_ids),
        "image_sha256": str(conflict.get("image_sha256") or ""),
        "original_fields": dict(conflict),
        "derived_semantics": {},
        "violation_code": violation_code,
        "violation_codes": [violation_code],
        "violations": [
            {
                "code": violation_code,
                "blocking": True,
                "message": "Equally authoritative human annotations conflict",
                "suggestion": (
                    "Select one sample, create a new annotation revision, or mark "
                    "the image as needing reannotation."
                ),
            }
        ],
        "severity": "blocking",
        "suggested_action": (
            "Choose selected_sample, new_annotation_revision, or needs_reannotation."
        ),
        "proposed_field_changes": {},
        "confidence": "none",
        "requires_manual_decision": True,
        "target_mutable": False,
        "approval_status": "pending",
        "approved": False,
        "reviewer": "",
        "decision_reason": "",
        "revision_reason": "",
        "created_at": created_at,
        "conflict_source": {
            "path": str(conflict_path),
            "sha256": conflict_report_sha256,
            "index": conflict_index,
        },
        "conflicting_sample_ids": conflicting_sample_ids,
        "original_sample_ids": original_sample_ids,
        "old_label_sha256s": label_sha256s,
        "label_paths": label_paths,
        "resolution": {
            "mode": "",
            "selected_sample_id": "",
            "new_annotation_path": "",
            "new_label_sha256": "",
        },
        "lifecycle_context": {"submission_status": "submitted", "job_status": ""},
    }
    proposal["proposal_basis_sha256"] = _proposal_basis_sha256(proposal)
    return proposal


def _suggest_field_changes(
    row: Mapping[str, Any], codes: list[str]
) -> tuple[dict[str, str], str, bool]:
    changes: dict[str, str] = {}
    confidence = "high"
    manual = False
    code_set = set(codes)
    if "legacy_review_selection_missing" in code_set:
        changes["review_selected"] = "1"
    if "review_without_selection" in code_set:
        changes["review_selected"] = "1"
        confidence = "medium"
        manual = True
    if "legacy_review_outcome_missing" in code_set:
        label = str(row.get("review_label") or "").strip().lower()
        inferred = {
            "confirmed_ok": ("pass", ""),
            "verified_empty": ("pass", ""),
            "false_negative": ("fail", "missed_detection"),
            "wrong_box": ("fail", "wrong_box"),
            "wrong_class": ("fail", "wrong_class"),
        }.get(label)
        if inferred is None:
            confidence = "low"
            manual = True
        else:
            outcome, category = inferred
            changes["review_outcome"] = outcome
            if category and not str(row.get("failure_category") or "").strip():
                changes["failure_category"] = category
    if "legacy_uncertain_label" in code_set:
        changes = {}
        confidence = "none"
        manual = True
    if "skip_selected_for_training" in code_set:
        changes["training_selected"] = "0"
    if "annotation_fix_without_correction_route" in code_set:
        changes["action_route"] = "yolo"
        confidence = "medium"
    if "missing_annotation_direct_training" in code_set:
        changes["action_route"] = "yolo"
        confidence = "medium"
    if "color_only_sent_to_yolo" in code_set:
        changes["action_route"] = "color"
        confidence = "medium"
    known = {
        "legacy_review_selection_missing",
        "review_without_selection",
        "legacy_review_outcome_missing",
        "legacy_uncertain_label",
        "skip_selected_for_training",
        "annotation_fix_without_correction_route",
        "missing_annotation_direct_training",
        "color_only_sent_to_yolo",
    }
    if code_set - known:
        manual = True
        if not changes:
            confidence = "none"
    return changes, confidence, manual


def _validate_plan_structure(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ReviewRepairError("Unsupported repair plan schema version")
    if not str(plan.get("plan_id") or ""):
        raise ReviewRepairError("Repair plan has no plan_id")
    proposals = plan.get("proposals")
    if not isinstance(proposals, list):
        raise ReviewRepairError("Repair plan proposals must be a list")
    expected_plan_basis = _plan_basis_sha256(plan)
    if plan.get("plan_basis_sha256") != expected_plan_basis:
        raise ReviewRepairError("Repair plan immutable basis was modified")
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ReviewRepairError("Repair proposal must be an object")
        expected = _proposal_basis_sha256(proposal)
        if proposal.get("proposal_basis_sha256") != expected:
            raise ReviewRepairError(
                f"Proposal immutable basis was modified: {proposal.get('proposal_id')}"
            )


def _validate_approvals(
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workflow: list[dict[str, Any]] = []
    annotation: list[dict[str, Any]] = []
    pending: list[str] = []
    for raw in plan["proposals"]:
        proposal = dict(raw)
        status = str(proposal.get("approval_status") or "pending")
        if status not in APPROVAL_STATUSES:
            raise RepairPlanBlockedError(
                f"Invalid approval_status for {proposal['proposal_id']}: {status}"
            )
        if status == "pending":
            pending.append(str(proposal["proposal_id"]))
            continue
        reviewer = str(proposal.get("reviewer") or "").strip()
        reason = str(proposal.get("decision_reason") or "").strip()
        if not reviewer or not reason:
            raise RepairPlanBlockedError(
                f"Adjudicated proposal {proposal['proposal_id']} requires reviewer and decision_reason"
            )
        if status == "rejected":
            if proposal.get("approved") is True:
                raise RepairPlanBlockedError(
                    f"Rejected proposal {proposal['proposal_id']} cannot have approved=true"
                )
            continue
        if proposal.get("approved") is not True:
            raise RepairPlanBlockedError(
                f"Proposal {proposal['proposal_id']} requires approved=true"
            )
        if proposal["target_kind"] == "workflow_record":
            if proposal.get("target_mutable") is not True:
                raise RepairPlanBlockedError(
                    f"Immutable snapshot proposal cannot be applied: {proposal['proposal_id']}"
                )
            workflow.append(proposal)
        elif proposal["target_kind"] == "annotation_conflict":
            annotation.append(proposal)
        else:
            raise ReviewRepairError(
                f"Unsupported proposal target_kind: {proposal['target_kind']}"
            )
    if pending:
        raise RepairPlanBlockedError(
            "Every proposal must be explicitly approved or rejected before apply: "
            + ", ".join(pending[:10])
        )
    if not workflow and not annotation:
        raise RepairPlanBlockedError("Repair plan contains no approved changes")
    return workflow, annotation


def _validate_record_locator(
    proposal: Mapping[str, Any], source: Path, rows: list[dict[str, str]]
) -> int:
    if Path(str(proposal["scope"])).resolve() != source:
        raise StaleRepairPlanError(
            f"Proposal scope does not match source manifest: {proposal['scope']}"
        )
    locator = proposal.get("record_locator")
    if not isinstance(locator, Mapping):
        raise ReviewRepairError("Workflow proposal has no record_locator")
    row_index = int(locator["row_number"]) - 2
    if not 0 <= row_index < len(rows):
        raise StaleRepairPlanError("Proposal row is outside the source manifest")
    row = rows[row_index]
    if record_identity(row) != str(locator["sample_id"]):
        raise StaleRepairPlanError("Target record identity changed")
    if _record_sha256(row) != str(locator["record_sha256"]):
        raise StaleRepairPlanError("Target record fields changed")
    return row_index


def _require_revision_reason(proposal: Mapping[str, Any]) -> None:
    context = _proposal_lifecycle_context(proposal)
    submission = str(context.get("submission_status") or "").lower()
    job = str(context.get("job_status") or "").lower()
    if (submission in SUBMITTED_STATUSES or job in COMPLETED_JOB_STATUSES) and not str(
        proposal.get("revision_reason") or ""
    ).strip():
        raise RepairPlanBlockedError(
            f"Submitted/completed proposal {proposal['proposal_id']} requires revision_reason"
        )


def _write_annotation_resolution(
    proposal: Mapping[str, Any], repair_root: Path
) -> Path:
    _require_revision_reason(proposal)
    source = proposal.get("conflict_source")
    if not isinstance(source, Mapping):
        raise ReviewRepairError("Annotation proposal has no conflict_source")
    conflict_path = Path(str(source["path"])).resolve()
    if _sha256(_read_bytes(conflict_path)) != str(source["sha256"]):
        raise StaleRepairPlanError("Annotation conflict report changed")
    resolution = proposal.get("resolution")
    if not isinstance(resolution, Mapping):
        raise RepairPlanBlockedError("Annotation conflict has no resolution")
    mode = str(resolution.get("mode") or "")
    if mode not in ANNOTATION_RESOLUTION_MODES:
        raise RepairPlanBlockedError(
            f"Unsupported annotation conflict resolution mode: {mode!r}"
        )
    sample_ids = _string_list(proposal.get("conflicting_sample_ids"))
    original_sample_ids = _string_list(proposal.get("original_sample_ids"))
    old_hashes = _string_list(proposal.get("old_label_sha256s"))
    label_paths = _string_list(proposal.get("label_paths"))
    original_conflict = proposal.get("original_fields")
    normalized_labels = (
        original_conflict.get("normalized_labels", [])
        if isinstance(original_conflict, Mapping)
        else []
    )
    if label_paths and len(label_paths) == len(old_hashes):
        for index, (path_value, expected) in enumerate(
            zip(label_paths, old_hashes, strict=True)
        ):
            label_path = Path(path_value)
            if label_path.is_file():
                actual = _normalized_label_sha256(label_path)
            elif index < len(normalized_labels) and isinstance(
                normalized_labels[index], list
            ):
                actual = _sha256(
                    "\n".join(str(line) for line in normalized_labels[index]).encode(
                        "utf-8"
                    )
                )
            else:
                raise StaleRepairPlanError(
                    f"Conflicting annotation evidence is unavailable: {path_value}"
                )
            if actual != expected:
                raise StaleRepairPlanError(
                    f"Conflicting annotation changed since proposal: {path_value}"
                )
    selected = str(resolution.get("selected_sample_id") or "")
    new_path_value = str(resolution.get("new_annotation_path") or "")
    new_hash = str(resolution.get("new_label_sha256") or "")
    if mode == "selected_sample":
        if selected not in sample_ids:
            raise RepairPlanBlockedError(
                "selected_sample resolution must name one conflicting sample"
            )
    elif mode == "new_annotation_revision":
        if not new_path_value or not new_hash:
            raise RepairPlanBlockedError(
                "new_annotation_revision requires path and label SHA"
            )
        new_path = Path(new_path_value).expanduser().resolve()
        if not new_path.is_file() or _normalized_label_sha256(new_path) != new_hash:
            raise RepairPlanBlockedError(
                "New annotation revision is missing or its label SHA is invalid"
            )
    elif selected or new_path_value:
        raise RepairPlanBlockedError(
            "needs_reannotation cannot silently select or replace an old annotation"
        )
    destination = (
        repair_root
        / "annotation_resolutions"
        / f"{proposal['image_sha256'] or 'unknown'}-{proposal['proposal_id']}.json"
    )
    payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "proposal_id": proposal["proposal_id"],
        "image_sha256": proposal["image_sha256"],
        "conflicting_sample_ids": sample_ids,
        "original_sample_ids": original_sample_ids,
        "old_label_sha256s": old_hashes,
        "label_paths": label_paths,
        "resolution": dict(resolution),
        "reviewer": proposal["reviewer"],
        "decision_reason": proposal["decision_reason"],
        "revision_reason": proposal["revision_reason"],
        "old_annotations_preserved": True,
    }
    _write_json_atomic(destination, payload)
    return destination


def _read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]], bytes]:
    raw = _read_bytes(path)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ReviewRepairError(f"Manifest is not UTF-8: {path}") from exc
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if not reader.fieldnames:
            raise ReviewRepairError(f"Manifest header is missing: {path}")
        rows = [dict(row) for row in reader]
    except csv.Error as exc:
        raise ReviewRepairError(f"Manifest CSV is invalid: {path}: {exc}") from exc
    return list(reader.fieldnames), rows, raw


def _serialize_manifest(
    fieldnames: list[str], rows: list[dict[str, str]], *, include_bom: bool
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    encoding = "utf-8-sig" if include_bom else "utf-8"
    return buffer.getvalue().encode(encoding)


def _create_backup(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_bytes(path) != content:
            raise ReviewRepairError(f"Backup already exists with different content: {path}")
        return
    try:
        with path.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if _read_bytes(path) != content:
            raise ReviewRepairError(
                f"Backup race produced different content: {path}"
            ) from None


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _write_bytes_atomic(path, serialized)


def _append_audit_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.lock")
    line = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    with _exclusive_lock(lock):
        with path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _latest_plan_event(path: Path, plan_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest: dict[str, Any] | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ReviewRepairError(
                    f"Repair audit line {line_number} is not an object"
                )
            if str(payload.get("plan_id") or "") == plan_id:
                latest = dict(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewRepairError(f"Repair audit is unreadable: {path}: {exc}") from exc
    return latest


@contextmanager
def _exclusive_lock(lock_path: Path, timeout_seconds: float = 5.0):
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Repair target is locked: {lock_path}") from None
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _proposal_basis_sha256(proposal: Mapping[str, Any]) -> str:
    mutable = {
        "proposal_basis_sha256",
        "proposed_field_changes",
        "approval_status",
        "approved",
        "reviewer",
        "decision_reason",
        "revision_reason",
        "resolution",
    }
    basis = {key: value for key, value in proposal.items() if key not in mutable}
    return _canonical_sha256(basis)


def _plan_basis_sha256(plan: Mapping[str, Any]) -> str:
    proposals = plan.get("proposals")
    proposal_basis = [
        str(item.get("proposal_basis_sha256") or "")
        for item in proposals
        if isinstance(item, Mapping)
    ] if isinstance(proposals, list) else []
    return _canonical_sha256(
        {
            "schema_version": plan.get("schema_version"),
            "plan_id": plan.get("plan_id"),
            "created_at": plan.get("created_at"),
            "source": plan.get("source"),
            "conflict_reports": plan.get("conflict_reports"),
            "proposal_basis_sha256s": proposal_basis,
        }
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def _record_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_sha256({str(key): str(value or "") for key, value in row.items()})


def _image_sha256(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("image_sha256") or "").strip().lower()
    if explicit:
        return explicit
    for field in ("original_path", "source_image", "output_image"):
        path_value = str(row.get(field) or "").strip()
        if path_value:
            path = Path(path_value)
            if path.is_file():
                return _sha256(_read_bytes(path))
    return ""


def _normalized_label_sha256(path: Path) -> str:
    try:
        lines = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    except (OSError, UnicodeDecodeError) as exc:
        raise ReviewRepairError(f"Cannot read annotation label: {path}: {exc}") from exc
    return _sha256("\n".join(lines).encode("utf-8"))


def _proposal_lifecycle_context(proposal: Mapping[str, Any]) -> dict[str, str]:
    raw = proposal.get("lifecycle_context")
    if not isinstance(raw, Mapping):
        return {}
    return {
        "submission_status": str(raw.get("submission_status") or ""),
        "job_status": str(raw.get("job_status") or ""),
    }


def _is_immutable_scope(path: Path, row: Mapping[str, Any]) -> bool:
    lowered = {part.lower() for part in path.parts}
    job = str(row.get("job_status") or "").strip().lower()
    return bool(
        "jobs" in lowered
        or "submissions" in lowered
        or job in COMPLETED_JOB_STATUSES
    )


def _plan_source_path(plan: Mapping[str, Any]) -> Path:
    source = plan.get("source")
    if not isinstance(source, Mapping) or not str(source.get("path") or ""):
        raise ReviewRepairError("Repair plan source path is missing")
    return Path(str(source["path"])).expanduser().resolve()


def _audit_log_path(source: Path, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return source.parent / REPAIR_ARTIFACT_DIR / "repair_audit.jsonl"


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewRepairError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewRepairError(f"JSON artifact root is not an object: {path}")
    return dict(payload)


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ReviewRepairError(f"Cannot read repair source {path}: {exc}") from exc


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "RepairPlanBlockedError",
    "ReviewRepairError",
    "StaleRepairPlanError",
    "apply_repair_plan",
    "default_plan_path",
    "generate_repair_plan",
    "repair_audit_exit_code",
    "rollback_repair",
    "write_repair_plan",
]
