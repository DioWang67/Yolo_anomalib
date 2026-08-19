"""RC-1 historical cleanup analysis and approval facade.

The module is deliberately split from Qt and never writes a review manifest.
Formal mutations and rollback are delegated to the Phase 1C repair framework.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import tempfile
import threading
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from tools.audit_review_workflow import AUDITED_FIELDS, audit_review_workflow
from tools.export_review_dataset import prepare_annotation_draft
from tools.processing_pipeline import (
    ProcessingPlanner,
    RoutingDecisionType,
)
from tools.review_repair import (
    apply_repair_plan,
    generate_repair_plan,
    rollback_repair,
    write_repair_plan,
)
from tools.review_workflow import (
    RequiredAction,
    derive_review_semantics,
    validate_record_consistency,
)

CLEANUP_SCHEMA_VERSION = 1
CLEANUP_ARTIFACT_DIR = ".review_repairs/cleanup"
DECISION_STATUSES = frozenset({"pending", "approved", "rejected", "skipped"})


class BlockingRootCause(str, Enum):
    REVIEW_WITHOUT_SELECTION = "review_without_selection"
    LEGACY_SELECTION_MISSING = "legacy_selection_missing"
    LEGACY_REVIEW_OUTCOME_MISSING = "legacy_review_outcome_missing"
    LEGACY_UNCERTAIN = "legacy_uncertain"
    ANNOTATION_CONFLICT = "annotation_conflict"
    CANONICAL_CONFLICT = "canonical_conflict"
    REANNOTATION_REQUIRED = "reannotation_required"
    STALE_ANNOTATION = "stale_annotation"
    STALE_LABEL = "stale_label"
    STALE_IMAGE = "stale_image"
    MISSING_LABEL = "missing_label"
    MISSING_IMAGE = "missing_image"
    INVALID_BBOX = "invalid_bbox"
    INVALID_CLASS = "invalid_class"
    CONFIG_STALE = "config_stale"
    COLOR_SCOPE_MISSING = "color_scope_missing"
    OTHER = "other"


class CleanupConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class HistoricalCleanupError(RuntimeError):
    """Base error for cleanup analysis or external artifact I/O."""


class CleanupDecisionError(HistoricalCleanupError):
    """A requested approval is unsafe or incomplete."""


@dataclass(frozen=True)
class CleanupRecord:
    record_id: str
    sample_id: str
    source_index: int
    root_cause: str
    contributing_causes: tuple[str, ...]
    violation_codes: tuple[str, ...]
    reason: str
    confidence: str
    confidence_reason: str
    suggested_fix: str
    potential_risk: str
    blocking_removed_estimate: int
    apply_ready: bool
    phase1c_proposal_id: str
    proposed_field_changes: Mapping[str, str]
    original_fields: Mapping[str, str]
    derived_semantics: Mapping[str, str]
    open_targets: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contributing_causes"] = list(self.contributing_causes)
        value["violation_codes"] = list(self.violation_codes)
        return value


@dataclass(frozen=True)
class CleanupGroup:
    group_id: str
    root_cause: str
    title: str
    record_ids: tuple[str, ...]
    record_count: int
    confidence: str
    confidence_reason: str
    suggested_fix: str
    potential_risk: str
    blocking_removed_estimate: int
    batch_approvable: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["record_ids"] = list(self.record_ids)
        return value


@dataclass(frozen=True)
class CleanupAnalysis:
    analysis_id: str
    created_at: str
    manifest_path: str
    manifest_sha256: str
    sample_count: int
    audit_summary: Mapping[str, Any]
    planner_statistics: Mapping[str, int]
    root_cause_counts: Mapping[str, int]
    groups: tuple[CleanupGroup, ...]
    records: tuple[CleanupRecord, ...]
    warnings: tuple[str, ...]
    phase1c_plan: Mapping[str, Any]
    conflict_report_paths: tuple[str, ...] = ()

    def to_dict(self, *, include_phase1c_plan: bool = False) -> dict[str, Any]:
        payload = {
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "mode": "dry_run",
            "mutation_performed": False,
            "analysis_id": self.analysis_id,
            "created_at": self.created_at,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "sample_count": self.sample_count,
            "audit_summary": dict(self.audit_summary),
            "planner_statistics": dict(self.planner_statistics),
            "root_cause_counts": dict(self.root_cause_counts),
            "groups": [group.to_dict() for group in self.groups],
            "records": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
            "conflict_report_paths": list(self.conflict_report_paths),
        }
        if include_phase1c_plan:
            payload["phase1c_plan"] = copy.deepcopy(dict(self.phase1c_plan))
        return payload


class HistoricalCleanupAnalyzer:
    """Build one immutable, read-only cleanup analysis for a complete manifest."""

    def __init__(self, planner: ProcessingPlanner | None = None) -> None:
        self._planner = planner or ProcessingPlanner()

    def analyze(
        self,
        manifest_path: str | Path,
        *,
        conflict_report_paths: Iterable[str | Path] = (),
        operator: str = "audit",
    ) -> CleanupAnalysis:
        manifest = Path(manifest_path).expanduser().resolve()
        rows, raw = _read_manifest(manifest)
        manifest_sha = _sha256(raw)
        audit = audit_review_workflow([manifest])
        if int(audit["summary"].get("error_count", 0)):
            raise HistoricalCleanupError(
                f"Review workflow audit failed: {audit.get('errors', [])}"
            )
        repair_plan = generate_repair_plan(
            manifest,
            conflict_report_paths=conflict_report_paths,
        )
        if str(repair_plan["source"].get("sha256") or "") != manifest_sha:
            raise HistoricalCleanupError(
                "Manifest changed while the cleanup audit was running; run it again."
            )
        processing_plan = self._planner.create_plan(
            list(enumerate(rows)),
            operator=operator,
            source_manifest_sha=manifest_sha,
        )
        workflow_proposals = {
            int(proposal["record_locator"]["row_number"]) - 2: proposal
            for proposal in repair_plan["proposals"]
            if proposal.get("target_kind") == "workflow_record"
        }
        records: list[CleanupRecord] = []
        decisions = {item.source_index: item for item in processing_plan.routing_decisions}
        for source_index, row in enumerate(rows):
            decision = decisions[source_index]
            if decision.decision not in {
                RoutingDecisionType.BLOCKED,
                RoutingDecisionType.MANUAL_REVIEW_REQUIRED,
            }:
                continue
            proposal = workflow_proposals.get(source_index)
            records.append(
                _classify_workflow_record(
                    row=row,
                    source_index=source_index,
                    violation_codes=(
                        tuple(proposal.get("violation_codes", ()))
                        if proposal is not None
                        else decision.violation_codes
                    ),
                    proposal=proposal,
                )
            )
        for proposal in repair_plan["proposals"]:
            if proposal.get("target_kind") == "annotation_conflict":
                records.append(_classify_conflict_record(proposal))
        groups = _build_groups(records)
        warnings: list[str] = []
        if not repair_plan.get("conflict_reports"):
            warnings.append(
                "No annotation conflict report was supplied; conflict classification "
                "covers only evidence present in the manifest."
            )
        return CleanupAnalysis(
            analysis_id=f"cleanup-audit-{uuid.uuid4().hex[:12]}",
            created_at=_utc_now(),
            manifest_path=str(manifest),
            manifest_sha256=manifest_sha,
            sample_count=len(rows),
            audit_summary=dict(audit["summary"]),
            planner_statistics=_statistics_dict(processing_plan.statistics),
            root_cause_counts=dict(
                sorted(Counter(record.root_cause for record in records).items())
            ),
            groups=tuple(groups),
            records=tuple(records),
            warnings=tuple(warnings),
            phase1c_plan=copy.deepcopy(repair_plan),
            conflict_report_paths=tuple(
                str(item.get("path") or "")
                for item in repair_plan.get("conflict_reports", ())
            ),
        )


class HistoricalCleanupSession:
    """Persist human decisions and delegate formal apply/rollback to Phase 1C."""

    def __init__(
        self,
        analysis: CleanupAnalysis,
        *,
        session_path: str | Path | None = None,
    ) -> None:
        self.analysis = analysis
        self._lock = threading.RLock()
        manifest = Path(analysis.manifest_path)
        session_id = f"cleanup-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:10]}"
        root = manifest.parent / CLEANUP_ARTIFACT_DIR / "sessions" / session_id
        self.session_path = (
            Path(session_path).expanduser().resolve()
            if session_path is not None
            else root / "session.json"
        )
        self.base_plan_path = self.session_path.parent / "phase1c-base-plan.json"
        self.adjudicated_plan_path = self.session_path.parent / "phase1c-adjudicated-plan.json"
        self.cleanup_report_path = self.session_path.parent / "cleanup-report.json"
        self._records = {record.record_id: record for record in analysis.records}
        self._groups = {group.group_id: group for group in analysis.groups}
        self._payload: dict[str, Any] = {
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "session_id": session_id,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "source_manifest": analysis.manifest_path,
            "source_manifest_sha256": analysis.manifest_sha256,
            "analysis_id": analysis.analysis_id,
            "base_phase1c_plan_path": str(self.base_plan_path),
            "decisions": {
                record.record_id: {
                    "status": "pending",
                    "reviewer": "",
                    "reason": "",
                    "decided_at": "",
                }
                for record in analysis.records
            },
            "last_apply_report": "",
            "last_cleanup_report": "",
            "last_rollback_report": "",
        }
        write_repair_plan(analysis.phase1c_plan, self.base_plan_path)
        self._persist()

    @property
    def payload(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._payload)

    def decision_for(self, record_id: str) -> Mapping[str, str]:
        return dict(self._decision(record_id))

    def approve_group(self, group_id: str, *, reviewer: str, reason: str) -> None:
        group = self._group(group_id)
        if not group.batch_approvable:
            raise CleanupDecisionError(
                f"Group {group_id} is not batch approvable at {group.confidence} confidence"
            )
        for record_id in group.record_ids:
            self._ensure_approvable(self._record(record_id))
        self._set_many(group.record_ids, "approved", reviewer, reason)

    def reject_group(self, group_id: str, *, reviewer: str, reason: str) -> None:
        self._set_many(self._group(group_id).record_ids, "rejected", reviewer, reason)

    def skip_group(self, group_id: str, *, reviewer: str, reason: str) -> None:
        self._set_many(self._group(group_id).record_ids, "skipped", reviewer, reason)

    def approve_record(self, record_id: str, *, reviewer: str, reason: str) -> None:
        self._ensure_approvable(self._record(record_id))
        self._set_many((record_id,), "approved", reviewer, reason)

    def reject_record(self, record_id: str, *, reviewer: str, reason: str) -> None:
        self._set_many((record_id,), "rejected", reviewer, reason)

    def skip_record(self, record_id: str, *, reviewer: str, reason: str) -> None:
        self._set_many((record_id,), "skipped", reviewer, reason)

    def export_json(self, destination: str | Path) -> Path:
        payload = self.analysis.to_dict()
        payload["session"] = copy.deepcopy(self._payload)
        path = Path(destination).expanduser().resolve()
        _write_json_atomic(path, payload)
        return path

    def export_csv(self, destination: str | Path) -> Path:
        path = Path(destination).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = (
            "group_id",
            "record_id",
            "sample_id",
            "root_cause",
            "contributing_causes",
            "violation_codes",
            "confidence",
            "suggested_fix",
            "blocking_removed_estimate",
            "apply_ready",
            "decision",
            "reviewer",
            "decision_reason",
        )
        group_by_record = {
            record_id: group.group_id
            for group in self.analysis.groups
            for record_id in group.record_ids
        }
        rows = []
        for record in self.analysis.records:
            decision = self._decision(record.record_id)
            rows.append(
                {
                    "group_id": group_by_record[record.record_id],
                    "record_id": record.record_id,
                    "sample_id": record.sample_id,
                    "root_cause": record.root_cause,
                    "contributing_causes": "|".join(record.contributing_causes),
                    "violation_codes": "|".join(record.violation_codes),
                    "confidence": record.confidence,
                    "suggested_fix": record.suggested_fix,
                    "blocking_removed_estimate": record.blocking_removed_estimate,
                    "apply_ready": "1" if record.apply_ready else "0",
                    "decision": decision["status"],
                    "reviewer": decision["reviewer"],
                    "decision_reason": decision["reason"],
                }
            )
        _write_csv_atomic(path, columns, rows)
        return path

    def apply(self) -> dict[str, Any]:
        """Materialize decisions, then call the Phase 1C apply implementation."""
        with self._lock:
            current_sha = _sha256(Path(self.analysis.manifest_path).read_bytes())
            if current_sha != self.analysis.manifest_sha256:
                raise CleanupDecisionError(
                    "Cleanup analysis is stale: manifest SHA changed; run audit again."
                )
            plan = copy.deepcopy(dict(self.analysis.phase1c_plan))
            records_by_proposal = {
                record.phase1c_proposal_id: record
                for record in self.analysis.records
                if record.phase1c_proposal_id
            }
            approved_count = 0
            for proposal in plan["proposals"]:
                record = records_by_proposal.get(str(proposal["proposal_id"]))
                if record is None:
                    raise CleanupDecisionError(
                        f"No cleanup decision maps to Phase 1C proposal {proposal['proposal_id']}"
                    )
                decision = self._decision(record.record_id)
                status = str(decision["status"])
                if status == "pending":
                    raise CleanupDecisionError(
                        "Every Phase 1C proposal must be approved, rejected, or skipped "
                        f"before apply; pending record: {record.record_id}"
                    )
                proposal["reviewer"] = decision["reviewer"]
                proposal["decision_reason"] = decision["reason"]
                proposal["revision_reason"] = decision["reason"]
                if status == "approved":
                    self._ensure_approvable(record)
                    proposal["approval_status"] = "approved"
                    proposal["approved"] = True
                    approved_count += 1
                else:
                    proposal["approval_status"] = "rejected"
                    proposal["approved"] = False
            if not approved_count:
                raise CleanupDecisionError("At least one safe proposal must be approved")
            write_repair_plan(plan, self.adjudicated_plan_path)
            phase1c_report = apply_repair_plan(self.adjudicated_plan_path)
            cleanup_report = self._build_cleanup_report(phase1c_report)
            _write_json_atomic(self.cleanup_report_path, cleanup_report)
            self._payload["last_apply_report"] = str(
                Path(self.analysis.manifest_path).parent
                / ".review_repairs"
                / "reports"
                / f"{phase1c_report['repair_id']}.json"
            )
            self._payload["last_cleanup_report"] = str(self.cleanup_report_path)
            self._payload["updated_at"] = _utc_now()
            self._persist()
            return cleanup_report

    def rollback(self, report_path: str | Path | None = None) -> dict[str, Any]:
        """Delegate rollback to Phase 1C and report the restored planner state."""
        target = str(report_path or self._payload.get("last_apply_report") or "")
        if not target:
            raise CleanupDecisionError("No applied Phase 1C report is available")
        phase1c_rollback = rollback_repair(target)
        snapshot = build_cleanup_snapshot(self.analysis.manifest_path)
        report = {
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "event": "cleanup_rolled_back",
            "created_at": _utc_now(),
            "session_id": self._payload["session_id"],
            "phase1c_rollback": phase1c_rollback,
            "restored": snapshot,
        }
        destination = self.session_path.parent / "cleanup-rollback-report.json"
        report["report_sha256"] = _canonical_sha256(report)
        _write_json_atomic(destination, report)
        self._payload["last_rollback_report"] = str(destination)
        self._payload["updated_at"] = _utc_now()
        self._persist()
        return report

    def _build_cleanup_report(self, phase1c_report: Mapping[str, Any]) -> dict[str, Any]:
        after = build_cleanup_snapshot(self.analysis.manifest_path)
        decisions = self._payload["decisions"]
        applied_groups: list[str] = []
        rejected_groups: list[str] = []
        for group in self.analysis.groups:
            statuses = {decisions[item]["status"] for item in group.record_ids}
            if statuses == {"approved"}:
                applied_groups.append(group.group_id)
            elif statuses <= {"rejected", "skipped"}:
                rejected_groups.append(group.group_id)
        report = {
            "schema_version": CLEANUP_SCHEMA_VERSION,
            "event": "cleanup_applied",
            "created_at": _utc_now(),
            "session_id": self._payload["session_id"],
            "before": {
                "manifest_sha256": self.analysis.manifest_sha256,
                "audit": dict(self.analysis.audit_summary),
                "planner": dict(self.analysis.planner_statistics),
            },
            "after": after,
            "remaining_blocking": after["planner_statistics"]["blocking_count"],
            "applied_groups": applied_groups,
            "rejected_groups": rejected_groups,
            "rollback_reference": phase1c_report.get("backup_path", ""),
            "repair_ids": [phase1c_report.get("repair_id", "")],
            "phase1c_report": dict(phase1c_report),
            "manifest_sha256": after["manifest_sha256"],
        }
        report["report_sha256"] = _canonical_sha256(report)
        return report

    def _set_many(
        self,
        record_ids: Sequence[str],
        status: str,
        reviewer: str,
        reason: str,
    ) -> None:
        reviewer = reviewer.strip()
        reason = reason.strip()
        if status not in DECISION_STATUSES or status == "pending":
            raise CleanupDecisionError(f"Unsupported decision: {status}")
        if not reviewer or not reason:
            raise CleanupDecisionError("Reviewer and decision reason are required")
        with self._lock:
            for record_id in record_ids:
                self._record(record_id)
            decided_at = _utc_now()
            for record_id in record_ids:
                self._payload["decisions"][record_id] = {
                    "status": status,
                    "reviewer": reviewer,
                    "reason": reason,
                    "decided_at": decided_at,
                }
            self._payload["updated_at"] = decided_at
            self._persist()

    def _persist(self) -> None:
        _write_json_atomic(self.session_path, self._payload)

    def _record(self, record_id: str) -> CleanupRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise CleanupDecisionError(f"Unknown cleanup record: {record_id}") from exc

    def _group(self, group_id: str) -> CleanupGroup:
        try:
            return self._groups[group_id]
        except KeyError as exc:
            raise CleanupDecisionError(f"Unknown cleanup group: {group_id}") from exc

    def _decision(self, record_id: str) -> Mapping[str, str]:
        self._record(record_id)
        return self._payload["decisions"][record_id]

    @staticmethod
    def _ensure_approvable(record: CleanupRecord) -> None:
        if not record.apply_ready or not record.phase1c_proposal_id:
            raise CleanupDecisionError(
                f"Record {record.record_id} has no complete Phase 1C-safe repair; "
                "open it for manual review."
            )


def build_cleanup_snapshot(manifest_path: str | Path) -> dict[str, Any]:
    """Re-run the existing audit and Planner without modifying their rules."""
    manifest = Path(manifest_path).expanduser().resolve()
    rows, raw = _read_manifest(manifest)
    audit = audit_review_workflow([manifest])
    plan = ProcessingPlanner().create_plan(
        list(enumerate(rows)),
        operator="historical-cleanup-audit",
        source_manifest_sha=_sha256(raw),
    )
    return {
        "manifest_path": str(manifest),
        "manifest_sha256": _sha256(raw),
        "sample_count": len(rows),
        "audit_summary": dict(audit["summary"]),
        "planner_statistics": _statistics_dict(plan.statistics),
    }


def cleanup_audit_exit_code(analysis: CleanupAnalysis | Mapping[str, Any]) -> int:
    if isinstance(analysis, CleanupAnalysis):
        statistics = analysis.planner_statistics
        record_count = len(analysis.records)
    else:
        statistics = analysis.get("planner_statistics", {})
        records = analysis.get("records", ())
        record_count = len(records) if isinstance(records, (list, tuple)) else 0
    if not isinstance(statistics, Mapping):
        return 1
    return 2 if int(statistics.get("blocking_count", 0)) or record_count else 0


def write_cleanup_audit(analysis: CleanupAnalysis, destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    payload = analysis.to_dict()
    payload["report_sha256"] = _canonical_sha256(payload)
    _write_json_atomic(path, payload)
    return path


def _classify_workflow_record(
    *,
    row: Mapping[str, str],
    source_index: int,
    violation_codes: Sequence[str],
    proposal: Mapping[str, Any] | None,
) -> CleanupRecord:
    evidence = _inspect_evidence(row)
    codes = tuple(str(code) for code in violation_codes)
    contributing = list(_root_causes_from_codes(codes))
    for cause in evidence["causes"]:
        if cause not in contributing:
            contributing.append(cause)
    root = _primary_root_cause(row, codes, contributing)
    changes = {
        str(key): str(value)
        for key, value in (proposal or {}).get("proposed_field_changes", {}).items()
    }
    apply_ready = _proposal_is_apply_ready(row, proposal)
    confidence, confidence_reason, strategy, risk = _strategy(
        root=root,
        row=row,
        codes=codes,
        apply_ready=apply_ready,
        changes=changes,
    )
    record_id = f"manifest-row-{source_index + 2}"
    semantics = derive_review_semantics(row)
    return CleanupRecord(
        record_id=record_id,
        sample_id=_sample_id(row, record_id),
        source_index=source_index,
        root_cause=root.value,
        contributing_causes=tuple(cause.value for cause in contributing),
        violation_codes=codes,
        reason=_reason_for(root, codes),
        confidence=confidence.value,
        confidence_reason=confidence_reason,
        suggested_fix=strategy,
        potential_risk=risk,
        blocking_removed_estimate=1 if apply_ready else 0,
        apply_ready=apply_ready,
        phase1c_proposal_id=str((proposal or {}).get("proposal_id") or ""),
        proposed_field_changes=changes,
        original_fields=_review_fields(row),
        derived_semantics=semantics.to_dict(),
        open_targets={
            "sample": str(row.get("config_snapshot_path") or ""),
            "image": str(
                row.get("original_path") or row.get("preprocessed_path") or ""
            ),
            "annotation": str(row.get("output_label") or ""),
            "review": "",
            "conflict_report": "",
        },
    )


def _classify_conflict_record(proposal: Mapping[str, Any]) -> CleanupRecord:
    code = str(proposal.get("violation_code") or "canonical_label_conflict")
    if code == "human_annotation_conflict":
        root = BlockingRootCause.ANNOTATION_CONFLICT
    else:
        root = BlockingRootCause.CANONICAL_CONFLICT
    source = proposal.get("conflict_source") or {}
    return CleanupRecord(
        record_id=f"conflict-{proposal['proposal_id']}",
        sample_id=str(proposal.get("sample_id") or "unknown"),
        source_index=-1,
        root_cause=root.value,
        contributing_causes=(root.value,),
        violation_codes=(code,),
        reason="Conflicting annotations have equal or unresolved authority.",
        confidence=CleanupConfidence.NONE.value,
        confidence_reason="Canonical selection requires direct human evidence.",
        suggested_fix="Open the conflict report and choose a Phase 1C resolution manually.",
        potential_risk="Automatic selection could silently preserve the wrong label.",
        blocking_removed_estimate=0,
        apply_ready=False,
        phase1c_proposal_id=str(proposal["proposal_id"]),
        proposed_field_changes={},
        original_fields={
            str(key): str(value) for key, value in proposal.get("original_fields", {}).items()
        },
        derived_semantics={},
        open_targets={
            "sample": "",
            "image": "",
            "annotation": "",
            "review": "",
            "conflict_report": str(source.get("path") or ""),
        },
    )


def _build_groups(records: Sequence[CleanupRecord]) -> list[CleanupGroup]:
    buckets: dict[tuple[str, str, str, bool], list[CleanupRecord]] = {}
    for record in records:
        key = (
            record.root_cause,
            record.confidence,
            record.suggested_fix,
            record.apply_ready,
        )
        buckets.setdefault(key, []).append(record)
    groups: list[CleanupGroup] = []
    for key, items in buckets.items():
        root, confidence, suggested_fix, apply_ready = key
        digest = hashlib.sha256("|".join(map(str, key)).encode("utf-8")).hexdigest()[:10]
        groups.append(
            CleanupGroup(
                group_id=f"{root}-{digest}",
                root_cause=root,
                title=f"{root} ({confidence})",
                record_ids=tuple(item.record_id for item in items),
                record_count=len(items),
                confidence=confidence,
                confidence_reason=items[0].confidence_reason,
                suggested_fix=suggested_fix,
                potential_risk=items[0].potential_risk,
                blocking_removed_estimate=sum(
                    item.blocking_removed_estimate for item in items
                ),
                batch_approvable=apply_ready
                and confidence in {
                    CleanupConfidence.HIGH.value,
                    CleanupConfidence.MEDIUM.value,
                },
            )
        )
    return sorted(
        groups,
        key=lambda group: (
            -group.record_count,
            group.root_cause,
            group.confidence,
        ),
    )


def _strategy(
    *,
    root: BlockingRootCause,
    row: Mapping[str, str],
    codes: Sequence[str],
    apply_ready: bool,
    changes: Mapping[str, str],
) -> tuple[CleanupConfidence, str, str, str]:
    if root == BlockingRootCause.REVIEW_WITHOUT_SELECTION:
        if (
            str(row.get("review_outcome") or "").strip()
            and codes == ("review_without_selection",)
            and apply_ready
        ):
            return (
                CleanupConfidence.HIGH,
                "An explicit review outcome exists, semantics are unique, and no other blocker remains.",
                "Confirm the scope and set review_selected=1 through Phase 1C.",
                "The record may have been intentionally outside the original review scope.",
            )
        if apply_ready and "review_outcome" in changes:
            return (
                CleanupConfidence.MEDIUM,
                "The legacy label has one Phase 1B-compatible outcome mapping, but human approval is still required.",
                "Approve the complete Phase 1C proposal for review selection and inferred outcome.",
                "Legacy PASS/FAIL intent may differ from the inferred mapping.",
            )
        return (
            CleanupConfidence.LOW,
            "A review label exists, but the missing outcome cannot be reconstructed uniquely.",
            "Open the current review and explicitly choose an outcome before creating a revised Phase 1C proposal.",
            "Setting only review_selected would leave the Phase 1C proposal inconsistent.",
        )
    if root == BlockingRootCause.LEGACY_SELECTION_MISSING and apply_ready:
        return (
            CleanupConfidence.HIGH,
            "The reviewed record is complete and only its compatibility selection flag is missing.",
            "Set review_selected=1 through Phase 1C.",
            "Confirm the record belonged to the historical review scope.",
        )
    if root == BlockingRootCause.LEGACY_REVIEW_OUTCOME_MISSING:
        return (
            CleanupConfidence.NONE if not str(row.get("review_label") or "").strip() else CleanupConfidence.LOW,
            "No completed human outcome exists in the historical record.",
            "Open the sample in the existing Review UI and perform an explicit human review.",
            "Inventing an outcome could turn unknown evidence into training truth.",
        )
    if root == BlockingRootCause.LEGACY_UNCERTAIN:
        return (
            CleanupConfidence.NONE,
            "The historical value explicitly records uncertainty.",
            "Perform a new manual review; do not infer PASS, FAIL, or training eligibility.",
            "Any automatic choice would erase the recorded uncertainty.",
        )
    if root in {
        BlockingRootCause.ANNOTATION_CONFLICT,
        BlockingRootCause.CANONICAL_CONFLICT,
        BlockingRootCause.REANNOTATION_REQUIRED,
    }:
        return (
            CleanupConfidence.NONE,
            "Competing annotation evidence has no auditable automatic winner.",
            "Resolve the conflict manually using the existing Phase 1C resolution fields.",
            "Silent selection can train on an incorrect canonical label.",
        )
    if root in {
        BlockingRootCause.STALE_ANNOTATION,
        BlockingRootCause.STALE_LABEL,
        BlockingRootCause.STALE_IMAGE,
        BlockingRootCause.MISSING_LABEL,
        BlockingRootCause.MISSING_IMAGE,
        BlockingRootCause.INVALID_BBOX,
        BlockingRootCause.INVALID_CLASS,
        BlockingRootCause.CONFIG_STALE,
        BlockingRootCause.COLOR_SCOPE_MISSING,
    }:
        return (
            CleanupConfidence.LOW,
            "The evidence is incomplete or stale and cannot establish truth by metadata alone.",
            "Open the affected evidence and rebuild or re-review it through the owning workflow.",
            "Repairing paths or labels without evidence may associate the wrong artifact.",
        )
    return (
        CleanupConfidence.NONE,
        "No safe, unique cleanup rule covers this blocker.",
        "Inspect the record and create an explicit Phase 1C repair decision.",
        "A generic repair could alter business meaning.",
    )


def _proposal_is_apply_ready(
    row: Mapping[str, str], proposal: Mapping[str, Any] | None
) -> bool:
    if not proposal or proposal.get("target_mutable") is not True:
        return False
    changes = proposal.get("proposed_field_changes")
    if not isinstance(changes, Mapping) or not changes:
        return False
    after = {**row, **{str(key): str(value) for key, value in changes.items()}}
    remaining = {issue.code for issue in validate_record_consistency(after)}
    proposed_codes = {str(code) for code in proposal.get("violation_codes", [])}
    return not (remaining & proposed_codes)


def _primary_root_cause(
    row: Mapping[str, str],
    codes: Sequence[str],
    causes: Sequence[BlockingRootCause],
) -> BlockingRootCause:
    priority = (
        BlockingRootCause.CANONICAL_CONFLICT,
        BlockingRootCause.ANNOTATION_CONFLICT,
        BlockingRootCause.MISSING_IMAGE,
        BlockingRootCause.MISSING_LABEL,
        BlockingRootCause.INVALID_CLASS,
        BlockingRootCause.INVALID_BBOX,
        BlockingRootCause.STALE_IMAGE,
        BlockingRootCause.STALE_LABEL,
        BlockingRootCause.STALE_ANNOTATION,
        BlockingRootCause.CONFIG_STALE,
        BlockingRootCause.COLOR_SCOPE_MISSING,
    )
    for root in priority:
        if root in causes:
            return root
    if "legacy_uncertain_label" in codes:
        return BlockingRootCause.LEGACY_UNCERTAIN
    if "review_without_selection" in codes:
        return BlockingRootCause.REVIEW_WITHOUT_SELECTION
    if "legacy_review_selection_missing" in codes:
        return BlockingRootCause.LEGACY_SELECTION_MISSING
    if "legacy_review_outcome_missing" in codes:
        return BlockingRootCause.LEGACY_REVIEW_OUTCOME_MISSING
    if not str(row.get("review_label") or "").strip() and not str(
        row.get("review_outcome") or ""
    ).strip():
        return BlockingRootCause.LEGACY_REVIEW_OUTCOME_MISSING
    return causes[0] if causes else BlockingRootCause.OTHER


def _root_causes_from_codes(codes: Sequence[str]) -> tuple[BlockingRootCause, ...]:
    mapping = {
        "review_without_selection": BlockingRootCause.REVIEW_WITHOUT_SELECTION,
        "legacy_review_selection_missing": BlockingRootCause.LEGACY_SELECTION_MISSING,
        "legacy_review_outcome_missing": BlockingRootCause.LEGACY_REVIEW_OUTCOME_MISSING,
        "legacy_uncertain_label": BlockingRootCause.LEGACY_UNCERTAIN,
        "human_annotation_conflict": BlockingRootCause.ANNOTATION_CONFLICT,
        "canonical_label_conflict": BlockingRootCause.CANONICAL_CONFLICT,
        "needs_reannotation": BlockingRootCause.REANNOTATION_REQUIRED,
    }
    return tuple(dict.fromkeys(mapping.get(code, BlockingRootCause.OTHER) for code in codes))


def _inspect_evidence(row: Mapping[str, str]) -> dict[str, Any]:
    causes: list[BlockingRootCause] = []
    original = Path(str(row.get("original_path") or ""))
    processed = Path(str(row.get("preprocessed_path") or ""))
    if not original.is_file() and not processed.is_file():
        causes.append(BlockingRootCause.MISSING_IMAGE)
    config_value = str(row.get("config_snapshot_path") or "").strip()
    if config_value and not Path(config_value).is_file():
        causes.append(BlockingRootCause.CONFIG_STALE)
    label_value = str(row.get("output_label") or "").strip()
    annotation_status = str(row.get("annotation_status") or "").strip()
    if annotation_status == "verified_annotation" and (
        not label_value or not Path(label_value).is_file()
    ):
        causes.append(BlockingRootCause.MISSING_LABEL)
    semantics = derive_review_semantics(row)
    if semantics.required_action == RequiredAction.COLOR_CALIBRATION and (
        not str(row.get("product") or "").strip()
        or not str(row.get("area") or "").strip()
    ):
        causes.append(BlockingRootCause.COLOR_SCOPE_MISSING)
    if BlockingRootCause.MISSING_IMAGE not in causes:
        try:
            _image, label_text = prepare_annotation_draft(row)
            label_cause = _validate_yolo_text(label_text, row)
            if label_cause is not None:
                causes.append(label_cause)
        except (OSError, ValueError, json.JSONDecodeError):
            causes.append(BlockingRootCause.MISSING_IMAGE)
    return {"causes": tuple(dict.fromkeys(causes))}


def _validate_yolo_text(
    value: str, row: Mapping[str, str]
) -> BlockingRootCause | None:
    try:
        class_names = json.loads(str(row.get("class_names_json") or "[]"))
    except json.JSONDecodeError:
        return BlockingRootCause.INVALID_CLASS
    for line in value.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            return BlockingRootCause.INVALID_BBOX
        try:
            class_id = int(parts[0])
            bbox = tuple(float(item) for item in parts[1:])
        except ValueError:
            return BlockingRootCause.INVALID_BBOX
        if class_id < 0 or (class_names and class_id >= len(class_names)):
            return BlockingRootCause.INVALID_CLASS
        if (
            any(item < 0 or item > 1 for item in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            return BlockingRootCause.INVALID_BBOX
    return None


def _reason_for(root: BlockingRootCause, codes: Sequence[str]) -> str:
    if codes:
        return f"Root cause {root.value}; workflow diagnostics: {', '.join(codes)}."
    return f"Root cause {root.value}; Planner requires an explicit human review."


def _review_fields(row: Mapping[str, str]) -> dict[str, str]:
    fields = tuple(AUDITED_FIELDS) + (
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
        "output_label",
        "annotation_status",
    )
    return {field: str(row.get(field) or "") for field in dict.fromkeys(fields)}


def _sample_id(row: Mapping[str, str], fallback: str) -> str:
    for field in ("sample_id", "inspection_id", "config_snapshot_path", "original_path"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    return fallback


def _statistics_dict(statistics: Any) -> dict[str, int]:
    return {
        "ready_count": int(statistics.ready_count),
        "annotation_count": int(statistics.annotation_count),
        "color_count": int(statistics.color_count),
        "blocking_count": int(statistics.blocking_count),
        "excluded_count": int(statistics.excluded_count),
        "manual_review_count": int(statistics.manual_review_count),
    }


def _read_manifest(path: Path) -> tuple[list[dict[str, str]], bytes]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if not reader.fieldnames:
            raise HistoricalCleanupError(f"Manifest header is missing: {path}")
        return [dict(row) for row in reader], raw
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise HistoricalCleanupError(f"Cannot read manifest {path}: {exc}") from exc


def _write_csv_atomic(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
