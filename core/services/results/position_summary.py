from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

POSITION_FAIL_STATES = {"WRONG", "UNEXPECTED", "INVALID", "ERROR", "FAIL"}
POSITION_OK_STATES = {"CORRECT", "OK"}
POSITION_SKIP_STATES = {"", "UNKNOWN", "DISABLED", None}


@dataclass(frozen=True)
class PositionIssue:
    """Normalized position-check record for one detected item."""

    label: str
    status: str
    error: float | None
    dx: float | None
    dy: float | None


@dataclass(frozen=True)
class PositionSummary:
    """Aggregated, product-facing summary of position inspection."""

    total_with_position: int
    correct_count: int
    fail_count: int
    skipped_count: int
    issues: list[PositionIssue]
    likely_fixture_shift: bool
    average_dx: float | None
    average_dy: float | None


def summarize_position_records(records: list[dict[str, Any]]) -> PositionSummary:
    """Summarize per-item position outcomes into product-facing diagnostics."""
    issues: list[PositionIssue] = []
    correct_count = 0
    skipped_count = 0
    statuses: list[str] = []
    offsets: list[tuple[float, float]] = []

    for record in records or []:
        status = str(record.get("position_status") or "").upper()
        if status in POSITION_SKIP_STATES:
            skipped_count += 1
            continue

        statuses.append(status)
        if status in POSITION_OK_STATES:
            correct_count += 1
        elif status in POSITION_FAIL_STATES:
            dx, dy = _extract_offset(record)
            error = _coerce_float(record.get("position_error"))
            issue = PositionIssue(
                label=str(record.get("class") or record.get("label") or "?"),
                status=status,
                error=error,
                dx=dx,
                dy=dy,
            )
            issues.append(issue)
            if dx is not None and dy is not None:
                offsets.append((dx, dy))

    total_with_position = len(statuses)
    fail_count = len(issues)
    average_dx = None
    average_dy = None
    if offsets:
        average_dx = sum(dx for dx, _ in offsets) / len(offsets)
        average_dy = sum(dy for _, dy in offsets) / len(offsets)

    return PositionSummary(
        total_with_position=total_with_position,
        correct_count=correct_count,
        fail_count=fail_count,
        skipped_count=skipped_count,
        issues=issues,
        likely_fixture_shift=_looks_like_fixture_shift(total_with_position, issues),
        average_dx=average_dx,
        average_dy=average_dy,
    )


def format_fixture_shift_hint(summary: PositionSummary) -> str | None:
    """Return a concise operator-facing fixture-shift hint when applicable."""
    if not summary.likely_fixture_shift:
        return None
    if summary.average_dx is not None and summary.average_dy is not None:
        return (
            "可能是治具整體偏移 "
            f"(avg dx={summary.average_dx:+.1f}, dy={summary.average_dy:+.1f})"
        )
    return "可能是治具整體偏移，請檢查定位或重新校正。"


def _extract_offset(record: dict[str, Any]) -> tuple[float | None, float | None]:
    offset = record.get("position_offset") or {}
    if not isinstance(offset, dict):
        return None, None
    return _coerce_float(offset.get("dx")), _coerce_float(offset.get("dy"))


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_fixture_shift(
    total_with_position: int,
    issues: list[PositionIssue],
) -> bool:
    if total_with_position < 2 or len(issues) != total_with_position:
        return False

    directional = [
        (issue.dx, issue.dy)
        for issue in issues
        if issue.dx is not None and issue.dy is not None
    ]
    if len(directional) < 2:
        return True

    dxs = [dx for dx, _ in directional]
    dys = [dy for _, dy in directional]
    avg_dx = sum(dxs) / len(dxs)
    avg_dy = sum(dys) / len(dys)
    spread = max(_stddev(dxs), _stddev(dys))
    magnitude = sqrt(avg_dx**2 + avg_dy**2)
    return magnitude >= 3.0 and spread <= max(4.0, magnitude * 0.35)


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sqrt(sum((value - mean) ** 2 for value in values) / len(values))
