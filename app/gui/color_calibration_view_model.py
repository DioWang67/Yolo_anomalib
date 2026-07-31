"""Presentation-only adapter for Phase 3C3 color calibration packages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.color_calibration_packages import load_color_calibration_package
from tools.color_configuration_revisions import ColorConfigurationRevisionSummary
from tools.processing_reports import ProcessingReport


@dataclass(frozen=True)
class ColorScopeViewModel:
    scope_hash: str
    label: str
    proposal_status: str
    gate_status: str
    regression_count: int


@dataclass(frozen=True)
class ColorRevisionViewModel:
    scope_hash: str
    scope_label: str
    revision_id: str
    display_version: str
    parent_display_version: str
    active: bool
    created_at: str
    operator: str
    evidence_level: str
    changes: str


class ColorCalibrationViewModel:
    def __init__(self) -> None:
        self.visible = False
        self.package_path = ""
        self.preview_path = ""
        self.completion_path = ""
        self.scopes: tuple[ColorScopeViewModel, ...] = ()
        self.pending_count = 0
        self.approved_count = 0
        self.rejected_count = 0
        self.active_revision_ids: tuple[str, ...] = ()
        self.revisions: tuple[ColorRevisionViewModel, ...] = ()
        self._decider = None
        self._resume = None
        self._rollback = None
        self._history = None

    @classmethod
    def from_report(
        cls,
        report: ProcessingReport,
        *,
        decider=None,
        resume=None,
        rollback=None,
        history=None,
    ):
        result = cls()
        package_paths = {
            str(item.metadata.get("package_path") or "")
            for item in report.sample_results
            if item.action and "COLOR_CALIBRATION_PROPOSED" in item.action
        } - {""}
        if not package_paths:
            return result
        package = load_color_calibration_package(sorted(package_paths)[0])
        result.visible = True
        result.package_path = str(package.package_path)
        result.preview_path = str(package.root / "previews")
        result.scopes = tuple(
            ColorScopeViewModel(
                proposal.scope.scope_hash,
                proposal.scope.key,
                proposal.status.value,
                "PASSED" if gate.passed else "BLOCKED",
                len(gate.regression_samples),
            )
            for proposal, gate in zip(package.proposals, package.gates, strict=True)
        )
        result._decider = decider
        result._resume = resume
        result._rollback = rollback
        result._history = history
        result.refresh()
        result.refresh_revisions()
        return result

    def refresh(self) -> None:
        if not self.package_path:
            return
        import json
        approval_path = Path(self.package_path).parent / "approval.json"
        decisions = json.loads(approval_path.read_text(encoding="utf-8")).get("decisions", {})
        self.approved_count = sum(value.get("decision") == "APPROVED" for value in decisions.values())
        self.rejected_count = sum(value.get("decision") == "REJECTED" for value in decisions.values())
        self.pending_count = len(self.scopes) - self.approved_count - self.rejected_count

    def decide(self, scope_hash: str, *, approved: bool, reviewer: str, reason: str):
        if self._decider is None:
            raise RuntimeError("Color approval service is unavailable")
        outcome = self._decider(self.package_path, scope_hash, approved, reviewer, reason)
        self.refresh()
        return outcome

    def apply_approved(self):
        if self._resume is None:
            raise RuntimeError("Color activation service is unavailable")
        outcome = self._resume(self.package_path)
        self.completion_path = str(outcome.report_path)
        self.active_revision_ids = outcome.revision_ids
        self.refresh_revisions()
        return outcome

    def rollback(self, scope_hash: str, target_revision_id: str, operator: str, reason: str):
        if self._rollback is None:
            raise RuntimeError("Color rollback service is unavailable")
        outcome = self._rollback(
            self.package_path, scope_hash, target_revision_id, operator, reason
        )
        self.refresh_revisions()
        return outcome

    def refresh_revisions(self) -> None:
        if self._history is None or not self.package_path:
            self.revisions = ()
            return
        self.revisions = tuple(
            _revision_view_model(item)
            for item in self._history(self.package_path)
        )

    def revisions_for_scope(
        self, scope_hash: str
    ) -> tuple[ColorRevisionViewModel, ...]:
        return tuple(
            item for item in self.revisions if item.scope_hash == scope_hash
        )


def _revision_view_model(
    value: ColorConfigurationRevisionSummary,
) -> ColorRevisionViewModel:
    return ColorRevisionViewModel(
        scope_hash=value.scope_hash,
        scope_label=value.scope_label,
        revision_id=value.revision_id,
        display_version=value.display_version,
        parent_display_version=value.parent_display_version,
        active=value.active,
        created_at=value.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        operator=value.operator,
        evidence_level=value.evidence_level,
        changes="; ".join(value.changes),
    )
