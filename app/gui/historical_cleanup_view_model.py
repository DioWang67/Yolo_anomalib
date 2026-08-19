"""Presentation model for the RC-1 Historical Cleanup Assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.historical_cleanup import (
    CleanupAnalysis,
    CleanupGroup,
    CleanupRecord,
    HistoricalCleanupAnalyzer,
    HistoricalCleanupSession,
)


@dataclass(frozen=True)
class CleanupGroupViewModel:
    group_id: str
    title: str
    record_count: int
    confidence: str
    blocking_removed_estimate: int
    batch_approvable: bool


@dataclass(frozen=True)
class CleanupRecordViewModel:
    record_id: str
    sample_id: str
    root_cause: str
    confidence: str
    decision: str
    apply_ready: bool


@dataclass(frozen=True)
class CleanupPageViewModel:
    records: tuple[CleanupRecordViewModel, ...]
    page: int
    page_count: int
    total_count: int


class HistoricalCleanupViewModel:
    """Expose paged display data and approval commands without Qt dependencies."""

    def __init__(
        self,
        analysis: CleanupAnalysis,
        *,
        session: HistoricalCleanupSession | None = None,
        analyzer: HistoricalCleanupAnalyzer | None = None,
        page_size: int = 50,
    ) -> None:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.analysis = analysis
        self.session = session or HistoricalCleanupSession(analysis)
        self._analyzer = analyzer or HistoricalCleanupAnalyzer()
        self.page_size = page_size
        self._groups = {group.group_id: group for group in analysis.groups}
        self._records = {record.record_id: record for record in analysis.records}
        self.groups = tuple(self._group_view(group) for group in analysis.groups)
        self.manifest_path = analysis.manifest_path
        self.manifest_sha256 = analysis.manifest_sha256
        self.sample_count = analysis.sample_count
        self.planner_statistics = dict(analysis.planner_statistics)
        self.warnings = analysis.warnings

    def group_detail(self, group_id: str) -> str:
        group = self._group(group_id)
        return "\n".join(
            (
                f"Root Cause: {group.root_cause}",
                f"Records: {group.record_count}",
                f"Confidence: {group.confidence}",
                f"Reason: {group.confidence_reason}",
                f"Suggested Fix: {group.suggested_fix}",
                f"Blocking Removed Estimate: {group.blocking_removed_estimate}",
                f"Potential Risk: {group.potential_risk}",
            )
        )

    def records_page(self, group_id: str, page: int = 0) -> CleanupPageViewModel:
        group = self._group(group_id)
        total = group.record_count
        page_count = max(1, (total + self.page_size - 1) // self.page_size)
        effective_page = min(max(0, page), page_count - 1)
        start = effective_page * self.page_size
        selected = group.record_ids[start : start + self.page_size]
        return CleanupPageViewModel(
            records=tuple(self._record_view(self._record(item)) for item in selected),
            page=effective_page,
            page_count=page_count,
            total_count=total,
        )

    def record_detail(self, record_id: str) -> str:
        record = self._record(record_id)
        decision = self.session.decision_for(record_id)
        return "\n".join(
            (
                f"Sample: {record.sample_id}",
                f"Root Cause: {record.root_cause}",
                f"Contributing: {', '.join(record.contributing_causes) or '-'}",
                f"Violation Codes: {', '.join(record.violation_codes) or '-'}",
                f"Confidence: {record.confidence}",
                f"Confidence Reason: {record.confidence_reason}",
                f"Suggested Fix: {record.suggested_fix}",
                f"Phase 1C Ready: {'Yes' if record.apply_ready else 'No'}",
                f"Proposed Changes: {json.dumps(record.proposed_field_changes, ensure_ascii=False)}",
                f"Decision: {decision['status']}",
                f"Risk: {record.potential_risk}",
            )
        )

    def approve_group(self, group_id: str, reviewer: str, reason: str) -> None:
        self.session.approve_group(group_id, reviewer=reviewer, reason=reason)

    def reject_group(self, group_id: str, reviewer: str, reason: str) -> None:
        self.session.reject_group(group_id, reviewer=reviewer, reason=reason)

    def skip_group(self, group_id: str, reviewer: str, reason: str) -> None:
        self.session.skip_group(group_id, reviewer=reviewer, reason=reason)

    def approve_record(self, record_id: str, reviewer: str, reason: str) -> None:
        self.session.approve_record(record_id, reviewer=reviewer, reason=reason)

    def reject_record(self, record_id: str, reviewer: str, reason: str) -> None:
        self.session.reject_record(record_id, reviewer=reviewer, reason=reason)

    def skip_record(self, record_id: str, reviewer: str, reason: str) -> None:
        self.session.skip_record(record_id, reviewer=reviewer, reason=reason)

    def export_json(self, destination: str | Path) -> Path:
        return self.session.export_json(destination)

    def export_csv(self, destination: str | Path) -> Path:
        return self.session.export_csv(destination)

    def open_target(self, record_id: str, target: str) -> str:
        return str(self._record(record_id).open_targets.get(target) or "")

    def current_review_json(self, record_id: str) -> str:
        record = self._record(record_id)
        return json.dumps(
            {
                "original_fields": record.original_fields,
                "derived_semantics": record.derived_semantics,
                "proposal": record.proposed_field_changes,
                "decision": self.session.decision_for(record_id),
            },
            ensure_ascii=False,
            indent=2,
        )

    def run_audit_again(self) -> CleanupAnalysis:
        return self._analyzer.analyze(
            self.manifest_path,
            conflict_report_paths=self.analysis.conflict_report_paths,
            operator="cleanup-ui",
        )

    def apply(self) -> dict[str, Any]:
        return self.session.apply()

    def rollback(self) -> dict[str, Any]:
        return self.session.rollback()

    def _group_view(self, group: CleanupGroup) -> CleanupGroupViewModel:
        return CleanupGroupViewModel(
            group_id=group.group_id,
            title=group.title,
            record_count=group.record_count,
            confidence=group.confidence,
            blocking_removed_estimate=group.blocking_removed_estimate,
            batch_approvable=group.batch_approvable,
        )

    def _record_view(self, record: CleanupRecord) -> CleanupRecordViewModel:
        return CleanupRecordViewModel(
            record_id=record.record_id,
            sample_id=record.sample_id,
            root_cause=record.root_cause,
            confidence=record.confidence,
            decision=str(self.session.decision_for(record.record_id)["status"]),
            apply_ready=record.apply_ready,
        )

    def _group(self, group_id: str) -> CleanupGroup:
        try:
            return self._groups[group_id]
        except KeyError as exc:
            raise ValueError(f"Unknown cleanup group: {group_id}") from exc

    def _record(self, record_id: str) -> CleanupRecord:
        try:
            return self._records[record_id]
        except KeyError as exc:
            raise ValueError(f"Unknown cleanup record: {record_id}") from exc
