"""Replay a saved inspection snapshot's verdict with and without duplicate suppression.

Why this exists
---------------
Section 10 of ``docs/pilot/CROSS_CLASS_DUPLICATE_DETECTION_PROPOSAL.md`` makes an
A/B report on the ``report_only`` versus ``suppress`` verdict difference a
blocking gate before Cable1/A may leave report-only. Nothing produced one.
``audit_cross_class_duplicates.py`` reports which *boxes* a policy would
suppress; the named approvers in section 16 sign for what happens to the
*board*, and a box list does not tell them that.

Fidelity over independence
--------------------------
This module holds no copy of the count, sequence, colour-retraction or finalize
rules. It drives the production ``CrossClassDuplicateFilterStep``,
``CountCheckStep``, ``SequenceCheckStep`` and ``finalize_status`` over a
snapshot-backed context, because a second implementation would drift from the
line the first time anyone edits ``core/pipeline/steps.py`` -- and an A/B report
that disagrees with the line is worse than no report at all.

Every snapshot is first replayed in the mode it was *recorded* under, and the
result compared against its recorded status. A snapshot whose recorded status
cannot be reproduced is excluded from the verdict ledger: either it was produced
by code that no longer exists, or this replay is incomplete, and neither may be
handed to an approver as evidence about today's code.

What this does not re-derive
----------------------------
Colours are read from the snapshot's recorded ``color_result``, never re-measured
from the image. Whether suppression changes verdicts is the policy question, and
the measurement is already immutable evidence -- but this report therefore cannot
detect a colour-measurement change and must not be presented as a full pipeline
re-run.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import resolve_position_check_enabled  # noqa: E402
from core.pipeline.context import DetectionContext  # noqa: E402
from core.pipeline.finalize import finalize_status  # noqa: E402
from core.pipeline.registry import validate_duplicate_filter_order  # noqa: E402
from core.pipeline.steps import (  # noqa: E402
    CountCheckStep,
    CrossClassDuplicateFilterStep,
    SequenceCheckStep,
)
from core.services.cross_class_duplicate_filter import (  # noqa: E402
    DuplicateFilterMode,
)
from core.services.decision_engine import collect_fail_reasons  # noqa: E402

#: Version of this comparison contract. Bump when a field's meaning changes, so
#: an archived report is never reinterpreted under newer rules.
AB_POLICY_VERSION = 1

MODE_REPORT_ONLY = DuplicateFilterMode.REPORT_ONLY.value
MODE_SUPPRESS = DuplicateFilterMode.SUPPRESS.value

#: Verdict transitions, kept apart and never netted into one "changed" count: a
#: board the line would newly accept and one it would newly reject carry opposite
#: risks, and section 10 gates on them separately.
TRANSITION_UNCHANGED = "unchanged"
TRANSITION_REASONS_ONLY = "reasons_changed_verdict_unchanged"
TRANSITION_FAIL_TO_PASS = "fail_to_pass"
TRANSITION_PASS_TO_FAIL = "pass_to_fail"

#: Statuses that record an aborted inspection. They carry no comparable verdict.
TERMINAL_STATUSES = frozenset({"INFERENCE_ERROR", "ERROR", "CANCELED"})

_PASS_STATUS = "PASS"

# The replay drives production steps that log at info/warning level. Routing
# them to a silent, non-propagating logger keeps a 1500-snapshot report readable
# and its output identical between runs.
_REPLAY_LOGGER = logging.getLogger(__name__ + ".replay")
_REPLAY_LOGGER.addHandler(logging.NullHandler())
_REPLAY_LOGGER.propagate = False


class ReplayError(ValueError):
    """Raised when a snapshot cannot be replayed into a trustworthy verdict."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


class _ReplaySnapshotConfig:
    """Minimal config view over the runtime configuration a replay runs against.

    Mirrors the production ``Config`` accessors the replayed steps call --
    including raising on a missing or malformed mapping. ``CountCheckStep`` and
    ``CrossClassDuplicateFilterStep`` both fail closed on a lookup error, so a
    replay that quietly substituted an empty expectation would report a PASS the
    line would never have given.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = payload
        self.color_fail_closed = bool(payload.get("color_fail_closed", True))
        self.fail_on_unexpected = bool(payload.get("fail_on_unexpected", True))

    def get_items_by_area(self, product: str, area: str) -> list[str] | None:
        expected: Any = self._payload["expected_items"].get(product, {}).get(area)
        # Handed over unvalidated, exactly as production does: ``CountCheckStep``
        # normalizes the list itself and fails closed on anything it cannot read,
        # so checking here would only move the same rejection earlier under a
        # different error. The cast records that this type is the caller's
        # expectation, not a guarantee this data has been checked against.
        return cast("list[str] | None", expected)

    def is_position_check_enabled(self, product: str, area: str) -> bool:
        return resolve_position_check_enabled(
            self._payload["position_config"], product, area
        )


@dataclass(frozen=True)
class ReplayVerdict:
    """One board verdict, recomputed under a single duplicate-filter mode."""

    mode: str
    status: str
    fail_reasons: tuple[str, ...]
    duplicate_filter_status: str
    candidate_count: int
    raw_count: int
    effective_count: int
    suppressed_count: int
    color_is_ok: bool | None
    count_is_ok: bool | None
    count_status: str
    sequence_is_ok: bool | None
    sequence_reason: str
    missing_items: tuple[str, ...]
    unexpected_items: tuple[str, ...]

    @property
    def is_pass(self) -> bool:
        return self.status == _PASS_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "fail_reasons": list(self.fail_reasons),
            "duplicate_filter_status": self.duplicate_filter_status,
            "candidate_count": self.candidate_count,
            "raw_count": self.raw_count,
            "effective_count": self.effective_count,
            "suppressed_count": self.suppressed_count,
            "color_is_ok": self.color_is_ok,
            "count_is_ok": self.count_is_ok,
            "count_status": self.count_status,
            "sequence_is_ok": self.sequence_is_ok,
            "sequence_reason": self.sequence_reason,
            "missing_items": list(self.missing_items),
            "unexpected_items": list(self.unexpected_items),
        }


@dataclass(frozen=True)
class FidelityCheck:
    """Whether current code reproduces what the snapshot actually recorded.

    ``status_reproduced`` is the hard gate. ``reasons_reproduced`` is advisory:
    the 2026-08-19 colour-retraction fix deliberately changes which reasons a
    suppressed board reports without changing its verdict, so a reason drift is
    evidence of that fix rather than of a broken replay.
    """

    recorded_mode: str
    recorded_status: str
    recorded_fail_reasons: tuple[str, ...]
    replayed_status: str
    replayed_fail_reasons: tuple[str, ...]
    status_reproduced: bool
    reasons_reproduced: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "recorded_mode": self.recorded_mode,
            "recorded_status": self.recorded_status,
            "recorded_fail_reasons": list(self.recorded_fail_reasons),
            "replayed_status": self.replayed_status,
            "replayed_fail_reasons": list(self.replayed_fail_reasons),
            "status_reproduced": self.status_reproduced,
            "reasons_reproduced": self.reasons_reproduced,
        }


@dataclass(frozen=True)
class AbComparison:
    """One snapshot's verdict under report_only versus suppress."""

    fidelity: FidelityCheck
    report_only: ReplayVerdict
    suppress: ReplayVerdict
    transition: str
    cleared_reasons: tuple[str, ...]
    added_reasons: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        """Whether this comparison may be presented as evidence about today's code."""
        return self.fidelity.status_reproduced

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": AB_POLICY_VERSION,
            "fidelity": self.fidelity.to_dict(),
            "usable_as_evidence": self.is_usable,
            "report_only": self.report_only.to_dict(),
            "suppress": self.suppress.to_dict(),
            "transition": self.transition,
            "cleared_reasons": list(self.cleared_reasons),
            "added_reasons": list(self.added_reasons),
        }


def recorded_filter_mode(payload: Mapping[str, Any]) -> str:
    """Return the duplicate-filter mode the snapshot was recorded under.

    Raises:
        ReplayError: When the snapshot records no replayable mode. Such a
            snapshot predates the filter, so there is nothing to anchor the
            fidelity check to and it cannot serve as A/B evidence.
    """
    duplicate_filter = payload.get("duplicate_filter")
    if not isinstance(duplicate_filter, Mapping):
        raise ReplayError(
            "snapshot records no duplicate_filter metadata",
            code="recorded_mode_unavailable",
        )
    raw_mode = str(duplicate_filter.get("mode") or "").strip().lower()
    if raw_mode not in {MODE_REPORT_ONLY, MODE_SUPPRESS}:
        raise ReplayError(
            "snapshot duplicate_filter.mode is not a replayable mode: "
            + repr(raw_mode),
            code="recorded_mode_unavailable",
        )
    return raw_mode


def replay_verdict(
    payload: Mapping[str, Any],
    *,
    mode: str,
    filter_options: Mapping[str, Any],
    config_payload: Mapping[str, Any],
) -> ReplayVerdict:
    """Recompute one board verdict from a snapshot under ``mode``.

    Args:
        payload: The saved snapshot mapping.
        mode: ``report_only`` or ``suppress``.
        filter_options: Duplicate-filter policy options. ``mode`` and ``enabled``
            are overridden here so both sides of the A/B share one threshold set.
        config_payload: Runtime configuration to replay against -- the snapshot's
            own ``config``, or an operator-supplied model config.

    Raises:
        ReplayError: When the snapshot or configuration cannot support a
            trustworthy replay.
    """
    if mode not in {MODE_REPORT_ONLY, MODE_SUPPRESS}:
        raise ReplayError("unsupported replay mode: " + repr(mode), code="unsupported_mode")

    product = str(payload.get("product") or "").strip()
    area = str(payload.get("area") or "").strip()
    if not product or not area:
        raise ReplayError(
            "snapshot product and area are required to replay a verdict",
            code="invalid_scope",
        )

    recorded_status = str(payload.get("status") or "").strip().upper()
    if recorded_status in TERMINAL_STATUSES:
        raise ReplayError(
            "snapshot records an aborted inspection: " + recorded_status,
            code="terminal_status",
        )

    pipeline = config_payload.get("pipeline")
    if not isinstance(pipeline, list):
        raise ReplayError(
            "replay configuration must define a pipeline list",
            code="invalid_pipeline",
        )
    step_names = [str(name).strip().lower() for name in pipeline]
    # A pipeline the production registry would reject is not a pipeline any
    # verdict may be attributed to.
    validate_duplicate_filter_order(step_names)

    steps_config = config_payload.get("steps")
    if steps_config is None:
        steps_config = {}
    if not isinstance(steps_config, Mapping):
        raise ReplayError(
            "replay configuration steps must be a mapping", code="invalid_steps"
        )

    detections = _replay_detections(payload)
    color_result = payload.get("color_result")
    if color_result is not None and not isinstance(color_result, Mapping):
        raise ReplayError(
            "snapshot color_result must be a mapping", code="invalid_color_result"
        )

    result: dict[str, Any] = {"detections": detections}
    # YOLO-side signals the replayed steps do not recompute. finalize_status
    # consumes them, so dropping one would silently narrow the verdict.
    for key in ("missing_items", "unexpected_items", "slot_mismatches"):
        if isinstance(payload.get(key), list):
            result[key] = deepcopy(payload[key])
    for key in ("alignment_quality", "is_anomaly"):
        if payload.get(key) is not None:
            result[key] = deepcopy(payload[key])

    config = _ReplaySnapshotConfig(config_payload)
    ctx = DetectionContext(
        product=product,
        area=area,
        inference_type=str(payload.get("detector") or ""),
        # No image is loaded: colours come from the snapshot's recorded
        # measurements, and none of the three replayed steps reads a frame.
        # Loading the annotated evidence image here would make the report
        # depend on files that retention is allowed to delete.
        frame=cast(Any, None),
        processed_image=cast(Any, None),
        result=result,
        # Deliberately non-terminal: finalize_status recomputes the verdict from
        # the corrected signals and returns early on a terminal status.
        status=_PASS_STATUS,
        color_result=deepcopy(dict(color_result)) if color_result is not None else None,
        config=config,
    )

    if "cross_class_duplicate_filter" in step_names:
        options = dict(filter_options)
        options["mode"] = mode
        options["enabled"] = True
        CrossClassDuplicateFilterStep(_REPLAY_LOGGER, options).run(ctx)
    if "count_check" in step_names:
        CountCheckStep(
            _REPLAY_LOGGER, product, area, _step_options(steps_config, "count_check")
        ).run(ctx)
    if "sequence_check" in step_names:
        SequenceCheckStep(
            _REPLAY_LOGGER, product, area, _step_options(steps_config, "sequence_check")
        ).run(ctx)

    finalize_status(ctx, fail_on_unexpected=config.fail_on_unexpected)

    fail_reasons = collect_fail_reasons(
        status=ctx.status,
        decision=ctx.result.get("decision"),
        color_result=ctx.color_result,
        sequence_check=ctx.result.get("sequence_check"),
        detector=str(payload.get("detector") or ""),
        anomaly_score=payload.get("anomaly_score"),
        error_message=payload.get("error_message"),
    )

    duplicate_filter = ctx.result.get("duplicate_filter") or {}
    count_check = ctx.result.get("count_check") or {}
    sequence_check = ctx.result.get("sequence_check") or {}
    effective = ctx.result.get("detections") or []
    return ReplayVerdict(
        mode=mode,
        status=str(ctx.status),
        fail_reasons=tuple(fail_reasons),
        duplicate_filter_status=str(duplicate_filter.get("status") or ""),
        candidate_count=int(duplicate_filter.get("candidate_count") or 0),
        raw_count=int(duplicate_filter.get("raw_count") or len(detections)),
        effective_count=int(duplicate_filter.get("effective_count") or len(effective)),
        suppressed_count=int(duplicate_filter.get("suppressed_count") or 0),
        color_is_ok=_optional_flag(ctx.color_result, "is_ok"),
        count_is_ok=_optional_flag(count_check, "is_ok"),
        count_status=str(count_check.get("status") or ""),
        sequence_is_ok=_optional_flag(sequence_check, "is_ok"),
        sequence_reason=str(sequence_check.get("reason") or ""),
        missing_items=tuple(str(item) for item in ctx.result.get("missing_items") or []),
        unexpected_items=tuple(
            str(item) for item in ctx.result.get("unexpected_items") or []
        ),
    )


def compare_snapshot(
    payload: Mapping[str, Any],
    *,
    filter_options: Mapping[str, Any],
    config_payload: Mapping[str, Any],
) -> AbComparison:
    """Replay one snapshot under both modes and classify the difference.

    The fidelity anchor is the *recorded* mode, not ``report_only``: a snapshot
    saved by a suppress-mode run recorded the suppress verdict, so anchoring on
    report-only would report a mismatch for every such snapshot and bury a
    genuine one among them.
    """
    recorded_mode = recorded_filter_mode(payload)
    recorded_status = str(payload.get("status") or "").strip().upper()
    recorded_reasons = tuple(str(reason) for reason in (payload.get("fail_reasons") or []))

    report_only = replay_verdict(
        payload,
        mode=MODE_REPORT_ONLY,
        filter_options=filter_options,
        config_payload=config_payload,
    )
    suppress = replay_verdict(
        payload,
        mode=MODE_SUPPRESS,
        filter_options=filter_options,
        config_payload=config_payload,
    )
    anchor = report_only if recorded_mode == MODE_REPORT_ONLY else suppress

    fidelity = FidelityCheck(
        recorded_mode=recorded_mode,
        recorded_status=recorded_status,
        recorded_fail_reasons=recorded_reasons,
        replayed_status=anchor.status,
        replayed_fail_reasons=anchor.fail_reasons,
        status_reproduced=anchor.status == recorded_status,
        reasons_reproduced=set(anchor.fail_reasons) == set(recorded_reasons),
    )

    before = set(report_only.fail_reasons)
    after = set(suppress.fail_reasons)
    return AbComparison(
        fidelity=fidelity,
        report_only=report_only,
        suppress=suppress,
        transition=classify_transition(report_only, suppress),
        cleared_reasons=tuple(sorted(before - after)),
        added_reasons=tuple(sorted(after - before)),
    )


def classify_transition(report_only: ReplayVerdict, suppress: ReplayVerdict) -> str:
    """Classify how enabling suppression would change one board's outcome."""
    if report_only.is_pass and not suppress.is_pass:
        return TRANSITION_PASS_TO_FAIL
    if not report_only.is_pass and suppress.is_pass:
        return TRANSITION_FAIL_TO_PASS
    if set(report_only.fail_reasons) != set(suppress.fail_reasons):
        return TRANSITION_REASONS_ONLY
    return TRANSITION_UNCHANGED


def _replay_detections(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the pre-filter detections, deep-copied so a replay cannot mutate evidence.

    ``raw_detections`` is only written when suppression actually removed a box;
    otherwise ``detections`` already *is* the raw set. Reading the effective set
    when a raw one exists would feed the suppress replay boxes a previous run had
    already removed, and report no change where there was one.
    """
    raw = payload.get("raw_detections")
    detections = raw if raw is not None else payload.get("detections")
    if detections is None:
        detections = []
    if not isinstance(detections, list) or not all(
        isinstance(item, Mapping) for item in detections
    ):
        raise ReplayError(
            "snapshot detections must be a list of mappings", code="invalid_detections"
        )
    return [dict(deepcopy(dict(item))) for item in detections]


def _step_options(steps_config: Mapping[str, Any], name: str) -> dict[str, Any]:
    options = steps_config.get(name)
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ReplayError(
            "replay configuration steps." + name + " must be a mapping",
            code="invalid_steps",
        )
    return dict(deepcopy(dict(options)))


def _optional_flag(payload: Any, key: str) -> bool | None:
    """Return a check's boolean verdict, or None when the check published none.

    None and False are different facts here: a check that never ran must not be
    rendered as a check that failed.
    """
    if not isinstance(payload, Mapping) or key not in payload:
        return None
    return bool(payload.get(key))
