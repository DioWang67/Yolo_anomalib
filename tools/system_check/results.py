"""Result types and exception isolation for the preflight checker.

Every check returns a :class:`CheckResult`. No check is allowed to abort the
run: :func:`run_check` converts any unexpected exception into an ``UNKNOWN``
result carrying the exception text, so a machine missing a driver, an API, or a
permission still produces a complete report.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    """Outcome of a single check.

    ``SKIP`` is an addition to the four requested states. It marks a check that
    does not apply to *this* configuration — server sync disabled, anomalib
    backend disabled — and is kept distinct from ``UNKNOWN`` because
    "not required here" and "could not be determined" lead an operator to two
    different actions.
    """

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"
    SKIP = "SKIP"


# Ordering used to roll individual checks up into one verdict.
_SEVERITY: dict[Status, int] = {
    Status.PASS: 0,
    Status.SKIP: 0,
    Status.UNKNOWN: 1,
    Status.WARNING: 2,
    Status.FAIL: 3,
}


class Confidence(str, Enum):
    """How a requirement threshold was established."""

    #: Read directly out of the repository (config default, pin, code path).
    CONFIRMED = "CONFIRMED"
    #: Derived by this tool from repository facts; advisory, not a contract.
    SUGGESTED = "SUGGESTED"
    #: The repository does not state it and it cannot be inferred.
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class CheckResult:
    """One preflight finding.

    Args:
        check_id: Stable machine key, used by the JSON report and comparisons.
        title: Human-readable one-line label.
        status: Outcome.
        detail: Operator-facing explanation of *why* this status was chosen.
        requirement: The threshold applied, if any.
        measured: What was actually observed, if any.
        confidence: Provenance of ``requirement``.
        source: ``path:line`` the requirement came from, for auditability.
        remedy: What to do about a WARNING/FAIL, when there is a known action.
        data: Structured payload for the JSON report and baseline comparison.
    """

    check_id: str
    title: str
    status: Status
    detail: str
    requirement: str | None = None
    measured: str | None = None
    confidence: Confidence = Confidence.CONFIRMED
    source: str | None = None
    remedy: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "check_id": self.check_id,
            "title": self.title,
            "status": self.status.value,
            "detail": self.detail,
            "requirement": self.requirement,
            "measured": self.measured,
            "confidence": self.confidence.value,
            "source": self.source,
            "remedy": self.remedy,
            "data": dict(self.data),
        }


def run_check(
    check_id: str,
    title: str,
    func: Callable[[], CheckResult | list[CheckResult]],
) -> list[CheckResult]:
    """Run one check function with full exception isolation.

    A check that raises is reported as ``UNKNOWN`` rather than killing the run.
    ``KeyboardInterrupt`` and ``SystemExit`` derive from ``BaseException`` and
    are intentionally not caught, so the operator can still abort the tool.

    Args:
        check_id: Stable key used if the check itself fails.
        title: Label used if the check itself fails.
        func: Callable returning one result or a list of results.

    Returns:
        The check's results, or a single ``UNKNOWN`` result describing the
        failure.
    """
    try:
        produced = func()
    except Exception as exc:  # noqa: BLE001 - isolation is the whole point
        return [
            CheckResult(
                check_id=check_id,
                title=title,
                status=Status.UNKNOWN,
                detail=f"Check raised {type(exc).__name__}: {exc}",
                confidence=Confidence.UNKNOWN,
                remedy="Re-run with --debug to capture the traceback.",
                data={"traceback": traceback.format_exc(limit=8)},
            )
        ]
    if isinstance(produced, CheckResult):
        return [produced]
    return list(produced)


def overall_status(results: Iterable[CheckResult]) -> Status:
    """Roll individual results up into one verdict.

    Any ``FAIL`` fails the machine. ``WARNING`` and ``UNKNOWN`` both degrade the
    verdict to a warning: an undetermined requirement is not evidence of
    compliance.
    """
    worst = Status.PASS
    for result in results:
        if _SEVERITY[result.status] > _SEVERITY[worst]:
            worst = result.status
    if worst is Status.FAIL:
        return Status.FAIL
    if worst in (Status.WARNING, Status.UNKNOWN):
        return Status.WARNING
    return Status.PASS


def overall_label(status: Status) -> str:
    """Return the operator-facing wording for a rolled-up verdict."""
    if status is Status.FAIL:
        return "FAIL"
    if status is Status.WARNING:
        return "PASS WITH WARNINGS"
    return "PASS"
