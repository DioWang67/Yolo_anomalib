"""Explicit approval, activation, and follow-up replan for color packages."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from tools.color_calibration_packages import ColorCalibrationPackage, _write_json_atomic, load_color_calibration_package
from tools.color_calibration_service import ColorCalibrationError, ColorCalibrationScope, canonical_sha256, sha256_file
from tools.color_configuration_revisions import ColorConfigurationRevisionStore
from tools.processing_execution import CancellationToken
from tools.processing_pipeline import ProcessingPlan, ProcessingPlanner


@dataclass(frozen=True)
class ColorCalibrationApproval:
    scope_hash: str
    decision: str
    reviewer: str
    decision_reason: str
    decided_at: datetime
    proposal_sha256: str
    preview_sha256: str
    metrics_sha256: str
    current_config_sha256: str
    proposed_config_sha256: str


@dataclass(frozen=True)
class ColorCalibrationCompletion:
    completion_report_id: str
    package_id: str
    status: str
    completed_at: datetime
    revision_ids: tuple[str, ...]
    activated_scope_hashes: tuple[str, ...]
    rejected_scope_hashes: tuple[str, ...]
    failures: tuple[Mapping[str, str], ...]
    follow_up_plan: ProcessingPlan | None
    report_path: Path


class ColorCalibrationApprovalService:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_generator = id_generator or (lambda: str(uuid4()))

    def decide(
        self,
        package_path: str | Path,
        scope_hash: str,
        *,
        approved: bool,
        reviewer: str,
        reason: str,
    ) -> ColorCalibrationApproval:
        package = load_color_calibration_package(package_path)
        reviewer = reviewer.strip()
        reason = reason.strip()
        if not reviewer or not reason:
            raise ColorCalibrationError("COLOR_APPROVAL_REQUIRED", "Reviewer and decision reason are required.")
        if approved and reviewer.casefold() == package.operator.strip().casefold():
            raise ColorCalibrationError("COLOR_SELF_APPROVAL_FORBIDDEN", "Package creator cannot approve their own color proposal.")
        proposal, preview, gate = _scope_artifacts(package, scope_hash)
        if approved and not gate.approval_allowed:
            raise ColorCalibrationError("COLOR_GATE_FAILED", "This scope did not pass the color calibration gate.")
        proposed_path = package.root / "proposed_configs" / f"{scope_hash}.json"
        metrics = dict(preview.metrics)
        approval = ColorCalibrationApproval(
            scope_hash=scope_hash,
            decision="APPROVED" if approved else "REJECTED",
            reviewer=reviewer,
            decision_reason=reason,
            decided_at=self.clock(),
            proposal_sha256=proposal.proposal_sha256,
            preview_sha256=preview.preview_sha256,
            metrics_sha256=canonical_sha256(metrics),
            current_config_sha256=proposal.current_config_sha256,
            proposed_config_sha256=sha256_file(proposed_path),
        )
        path = package.root / "approval.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        decisions = payload.setdefault("decisions", {})
        existing = decisions.get(scope_hash)
        serialized = _approval_dict(approval)
        if existing is not None:
            if existing == serialized:
                return approval
            raise ColorCalibrationError("COLOR_APPROVAL_ALREADY_RECORDED", "A final decision already exists for this scope.")
        decisions[scope_hash] = serialized
        _write_json_atomic(path, payload)
        _write_package_event(
            package,
            "COLOR_SCOPE_APPROVED" if approved else "COLOR_SCOPE_REJECTED",
            event_id=self.id_generator(),
            created_at=approval.decided_at,
            metadata={
                "scope_hash": scope_hash,
                "reviewer": reviewer,
                "proposal_sha256": approval.proposal_sha256,
            },
        )
        return approval


class ColorCalibrationResumeService:
    def __init__(
        self,
        *,
        revision_store: ColorConfigurationRevisionStore,
        planner: ProcessingPlanner,
        current_config_resolver: Callable[[ColorCalibrationScope], tuple[Path, str]],
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
    ) -> None:
        self.revision_store = revision_store
        self.planner = planner
        self.current_config_resolver = current_config_resolver
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_generator = id_generator or (lambda: str(uuid4()))

    def resume(
        self,
        package_path: str | Path,
        original_plan: ProcessingPlan,
        *,
        current_entries: Sequence[tuple[int, Mapping[str, Any]]],
        current_manifest_sha: str,
        operator: str,
        cancellation_token: CancellationToken | None = None,
    ) -> ColorCalibrationCompletion:
        package = load_color_calibration_package(package_path)
        if package.plan_id != original_plan.plan_id or package.source_manifest_sha != current_manifest_sha:
            raise ColorCalibrationError("COLOR_PACKAGE_STALE", "Manifest or plan changed after color proposal.", retryable=True)
        approvals = _load_approvals(package)
        missing = {scope.scope_hash for scope in package.scopes} - set(approvals)
        if missing:
            raise ColorCalibrationError("COLOR_APPROVAL_PENDING", f"Scopes still require a decision: {', '.join(sorted(missing))}")
        revisions: list[str] = []
        activated: list[str] = []
        rejected: list[str] = []
        failures: list[Mapping[str, str]] = []
        for scope in package.scopes:
            if cancellation_token is not None and cancellation_token.is_cancelled:
                failures.append(
                    {
                        "scope_hash": scope.scope_hash,
                        "code": "COLOR_CALIBRATION_CANCELLED",
                        "message": cancellation_token.reason,
                    }
                )
                continue
            approval = approvals[scope.scope_hash]
            if approval.decision == "REJECTED":
                rejected.append(scope.scope_hash)
                continue
            proposal, preview, gate = _scope_artifacts(package, scope.scope_hash)
            if not gate.approval_allowed:
                failures.append({"scope_hash": scope.scope_hash, "code": "COLOR_GATE_FAILED"})
                continue
            if (
                approval.proposal_sha256 != proposal.proposal_sha256
                or approval.preview_sha256 != preview.preview_sha256
                or approval.metrics_sha256 != canonical_sha256(dict(preview.metrics))
            ):
                failures.append({"scope_hash": scope.scope_hash, "code": "COLOR_APPROVAL_SHA_MISMATCH"})
                continue
            current_path, current_sha = self.current_config_resolver(scope)
            if current_sha != approval.current_config_sha256 or not current_path.is_file():
                failures.append({"scope_hash": scope.scope_hash, "code": "CURRENT_CONFIG_STALE"})
                continue
            proposed_path = package.root / "proposed_configs" / f"{scope.scope_hash}.json"
            if sha256_file(proposed_path) != approval.proposed_config_sha256:
                failures.append({"scope_hash": scope.scope_hash, "code": "PROPOSED_CONFIG_SHA_MISMATCH"})
                continue
            proposed = json.loads(proposed_path.read_text(encoding="utf-8"))
            active = self.revision_store.read_active_pointer(scope)
            parent_id = str(active.get("revision_id") or "") if active else ""
            try:
                revision = self.revision_store.commit(
                    package, scope, operator=operator, reason=approval.decision_reason,
                    proposal_sha256=approval.proposal_sha256, preview_sha256=approval.preview_sha256,
                    proposed_config=proposed, metrics=dict(preview.metrics), parent_revision_id=parent_id,
                    parent_config_sha256=current_sha,
                )
                revisions.append(revision.revision_id)
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    failures.append(
                        {
                            "scope_hash": scope.scope_hash,
                            "code": "COLOR_CANCELLED_AFTER_REVISION_COMMIT",
                            "message": "Immutable revision was preserved but not activated.",
                        }
                    )
                    continue
                self.revision_store.activate(
                    revision, operator=operator, reason=approval.decision_reason,
                    expected_current_sha256=current_sha,
                )
                activated.append(scope.scope_hash)
                if cancellation_token is not None and cancellation_token.is_cancelled:
                    failures.append(
                        {
                            "scope_hash": scope.scope_hash,
                            "code": "COLOR_CANCELLATION_TOO_LATE",
                            "message": "Activation already completed; no automatic rollback was attempted.",
                        }
                    )
            except ColorCalibrationError as exc:
                failures.append({"scope_hash": scope.scope_hash, "code": exc.code, "message": str(exc)})
        completion_id = str(self.id_generator())
        completed_at = self.clock()
        status = "PARTIAL_FAILURE" if failures else ("PARTIAL_SUCCESS" if rejected else "COMPLETED")
        follow_up = None
        if activated:
            follow_up = build_follow_up_plan_from_color_calibration(
                original_plan, package, current_entries=current_entries,
                current_manifest_sha=current_manifest_sha, completion_report_id=completion_id,
                revision_ids=tuple(revisions), planner=self.planner, operator=operator,
            )
        report_path = package.root / f"completion-{completion_id}.json"
        payload = {
            "schema_version": 1, "completion_report_id": completion_id,
            "parent_report_id": package.report_id, "package_id": package.package_id,
            "status": status, "completed_at": completed_at.isoformat(),
            "revision_ids": revisions, "activated_scope_hashes": activated,
            "rejected_scope_hashes": rejected, "failures": failures,
            "follow_up_plan_id": follow_up.plan_id if follow_up else None,
            "training_started": False, "dataset_started": False,
        }
        _write_json_atomic(report_path, payload)
        _write_json_atomic(package.root / "completion.json", payload)
        _write_package_event(
            package,
            "COLOR_PACKAGE_PARTIAL" if status != "COMPLETED" else "COLOR_PACKAGE_COMPLETED",
            event_id=str(self.id_generator()),
            created_at=completed_at,
            metadata={
                "completion_report_id": completion_id,
                "revision_ids": revisions,
                "failure_codes": [str(item.get("code") or "") for item in failures],
            },
        )
        return ColorCalibrationCompletion(
            completion_id, package.package_id, status, completed_at, tuple(revisions),
            tuple(activated), tuple(rejected), tuple(failures), follow_up, report_path,
        )


def build_follow_up_plan_from_color_calibration(
    original_plan: ProcessingPlan,
    package: ColorCalibrationPackage,
    *,
    current_entries: Sequence[tuple[int, Mapping[str, Any]]],
    current_manifest_sha: str,
    completion_report_id: str,
    revision_ids: tuple[str, ...],
    planner: ProcessingPlanner,
    operator: str,
) -> ProcessingPlan:
    if current_manifest_sha != original_plan.source_manifest_sha:
        raise ColorCalibrationError("COLOR_PACKAGE_STALE", "Manifest changed before follow-up planning.")
    planned = planner.create_plan(
        current_entries, operator=operator, execution_mode=original_plan.execution_mode,
        source_manifest_sha=current_manifest_sha,
        review_revision=original_plan.review_revision,
    )
    return replace(
        planned,
        retry_source_plan_id=original_plan.plan_id,
        retry_source_report_id=completion_report_id,
        attempt=original_plan.attempt + 1,
        color_source_plan_id=original_plan.plan_id,
        color_source_report_id=package.report_id,
        color_source_package_id=package.package_id,
        color_revision_ids=revision_ids,
    )


def _scope_artifacts(package: ColorCalibrationPackage, scope_hash: str):
    for proposal, preview, gate in zip(
        package.proposals, package.previews, package.gates, strict=True
    ):
        if proposal.scope.scope_hash == scope_hash:
            return proposal, preview, gate
    raise ColorCalibrationError("CALIBRATION_SCOPE_MISSING", f"Unknown color scope: {scope_hash}")


def _load_approvals(package: ColorCalibrationPackage) -> dict[str, ColorCalibrationApproval]:
    raw = json.loads((package.root / "approval.json").read_text(encoding="utf-8"))
    result: dict[str, ColorCalibrationApproval] = {}
    for key, value in raw.get("decisions", {}).items():
        result[str(key)] = ColorCalibrationApproval(
            scope_hash=str(value["scope_hash"]), decision=str(value["decision"]), reviewer=str(value["reviewer"]),
            decision_reason=str(value["decision_reason"]), decided_at=datetime.fromisoformat(str(value["decided_at"])),
            proposal_sha256=str(value["proposal_sha256"]), preview_sha256=str(value["preview_sha256"]),
            metrics_sha256=str(value["metrics_sha256"]), current_config_sha256=str(value["current_config_sha256"]),
            proposed_config_sha256=str(value["proposed_config_sha256"]),
        )
    return result


def _approval_dict(value: ColorCalibrationApproval) -> dict[str, Any]:
    result = asdict(value)
    result["decided_at"] = value.decided_at.isoformat()
    return result


def _write_package_event(
    package: ColorCalibrationPackage,
    event_type: str,
    *,
    event_id: str,
    created_at: datetime,
    metadata: Mapping[str, Any],
) -> Path:
    safe_id = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in str(event_id)
    ).strip(".-")
    if not safe_id:
        raise ColorCalibrationError("COLOR_EVENT_ID_INVALID", "Color event ID is invalid.")
    path = package.root / "events" / f"{safe_id}.json"
    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "event_id": safe_id,
            "event_type": event_type,
            "package_id": package.package_id,
            "plan_id": package.plan_id,
            "report_id": package.report_id,
            "created_at": created_at.isoformat(),
            "metadata": dict(metadata),
        },
    )
    return path
