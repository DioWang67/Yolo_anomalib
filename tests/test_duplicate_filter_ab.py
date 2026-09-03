"""Tests for the report_only versus suppress A/B evidence report.

The fixtures here describe states ``ColorCheckerService`` can really produce.
The 2026-08-19 regression (section 17.3 of the pilot proposal) was hidden for
fourteen days by a fixture that paired a ``Red`` detector class with an
``Orange`` measurement and ``is_ok=True`` -- a combination the service cannot
emit -- so a test asserting an impossible state is treated here as no test at
all.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tools.duplicate_filter_ab import (
    MODE_REPORT_ONLY,
    MODE_SUPPRESS,
    TRANSITION_FAIL_TO_PASS,
    TRANSITION_PASS_TO_FAIL,
    TRANSITION_REASONS_ONLY,
    TRANSITION_UNCHANGED,
    ReplayError,
    ReplayVerdict,
    classify_transition,
    compare_snapshot,
    recorded_filter_mode,
    replay_verdict,
)
from tools.duplicate_filter_ab_report import main

FILTER_OPTIONS = {
    "mode": "report_only",
    "iou_threshold": 0.90,
    "center_distance_ratio_max": 0.10,
    "area_similarity_min": 0.80,
    "require_same_verified_class": True,
    "require_color_check_pass": True,
    "require_different_raw_class": True,
    "require_position_disabled": True,
}


def _replay_config() -> dict[str, Any]:
    return {
        "pipeline": [
            "color_check",
            "cross_class_duplicate_filter",
            "count_check",
            "sequence_check",
            "save_results",
        ],
        "steps": {
            "count_check": {"strict": True},
            "sequence_check": {
                "direction": "left_to_right",
                "expected": ["Green", "Orange"],
            },
            "cross_class_duplicate_filter": dict(FILTER_OPTIONS),
        },
        "expected_items": {"Cable1": {"A": ["Green", "Orange"]}},
        "position_config": {"Cable1": {"A": {"enabled": False}}},
        "fail_on_unexpected": True,
        "color_fail_closed": True,
    }


def _duplicate_snapshot(
    *,
    recorded_mode: str = MODE_REPORT_ONLY,
    recorded_status: str = "DETECTION_FAIL",
    recorded_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """A good board carrying one cross-class duplicate over a single orange wire.

    Both duplicate boxes measure Orange, so the box the detector called ``Red``
    has ``is_ok=False``: one measured colour cannot agree with two different
    detector classes. Suppressing it leaves the board genuinely good, which is
    the case the pilot gate exists to protect.
    """
    return {
        "inspection_id": "inspection-duplicate",
        "timestamp": "2026-08-20T08:00:00",
        "status": recorded_status,
        "fail_reasons": (
            recorded_reasons
            if recorded_reasons is not None
            else ["UNEXPECTED_COMPONENT", "COLOR_MISMATCH", "SEQUENCE_MISMATCH"]
        ),
        "product": "Cable1",
        "area": "A",
        "detector": "yolo",
        "model_info": {"model_version": "1.0.6"},
        "config_hash": "hash-1",
        "duplicate_filter": {"mode": recorded_mode},
        "detections": [
            {
                "class": "Green",
                "verified_class": "Green",
                "confidence": 0.90,
                "bbox": [0, 0, 20, 20],
            },
            {
                "class": "Orange",
                "verified_class": "Orange",
                "confidence": 0.664,
                "bbox": [100, 353, 141, 395],
            },
            {
                "class": "Red",
                "verified_class": "Orange",
                "confidence": 0.808,
                "bbox": [100, 353, 141, 396],
            },
        ],
        "color_result": {
            "is_ok": False,
            "status": "evaluated",
            "items": [
                {
                    "index": 0,
                    "class_name": "Green",
                    "best_color": "Green",
                    "is_ok": True,
                    "measurement_is_ok": True,
                },
                {
                    "index": 1,
                    "class_name": "Orange",
                    "best_color": "Orange",
                    "is_ok": True,
                    "measurement_is_ok": True,
                },
                {
                    "index": 2,
                    "class_name": "Red",
                    "best_color": "Orange",
                    "is_ok": False,
                    "measurement_is_ok": True,
                },
            ],
        },
        "config": _replay_config(),
    }


def _clean_snapshot() -> dict[str, Any]:
    """A board with no duplicate at all: both modes must agree it passes."""
    return {
        "inspection_id": "inspection-clean",
        "timestamp": "2026-08-20T09:00:00",
        "status": "PASS",
        "fail_reasons": [],
        "product": "Cable1",
        "area": "A",
        "detector": "yolo",
        "model_info": {"model_version": "1.0.6"},
        "duplicate_filter": {"mode": MODE_REPORT_ONLY},
        "detections": [
            {
                "class": "Green",
                "verified_class": "Green",
                "confidence": 0.90,
                "bbox": [0, 0, 20, 20],
            },
            {
                "class": "Orange",
                "verified_class": "Orange",
                "confidence": 0.88,
                "bbox": [100, 353, 141, 395],
            },
        ],
        "color_result": {
            "is_ok": True,
            "status": "evaluated",
            "items": [
                {
                    "index": 0,
                    "class_name": "Green",
                    "best_color": "Green",
                    "is_ok": True,
                    "measurement_is_ok": True,
                },
                {
                    "index": 1,
                    "class_name": "Orange",
                    "best_color": "Orange",
                    "is_ok": True,
                    "measurement_is_ok": True,
                },
            ],
        },
        "config": _replay_config(),
    }


def _replay(payload: dict[str, Any], mode: str) -> ReplayVerdict:
    return replay_verdict(
        payload,
        mode=mode,
        filter_options=FILTER_OPTIONS,
        config_payload=payload["config"],
    )


def _verdict(
    *, status: str, reasons: tuple[str, ...] = (), mode: str = MODE_REPORT_ONLY
) -> ReplayVerdict:
    return ReplayVerdict(
        mode=mode,
        status=status,
        fail_reasons=reasons,
        duplicate_filter_status="",
        candidate_count=0,
        raw_count=0,
        effective_count=0,
        suppressed_count=0,
        color_is_ok=None,
        count_is_ok=None,
        count_status="",
        sequence_is_ok=None,
        sequence_reason="",
        missing_items=(),
        unexpected_items=(),
    )


# --------------------------------------------------------------------------
# Replay behaviour
# --------------------------------------------------------------------------


def test_report_only_replay_keeps_the_duplicate_and_its_derived_failures() -> None:
    verdict = _replay(_duplicate_snapshot(), MODE_REPORT_ONLY)

    assert verdict.status == "DETECTION_FAIL"
    assert verdict.duplicate_filter_status == "reported"
    assert verdict.suppressed_count == 0
    assert verdict.effective_count == 3
    assert verdict.unexpected_items == ("Orange",)
    assert verdict.sequence_reason == "length_mismatch"
    assert verdict.color_is_ok is False


def test_suppress_replay_clears_the_derived_failures_and_passes_the_good_board() -> None:
    verdict = _replay(_duplicate_snapshot(), MODE_SUPPRESS)

    assert verdict.status == "PASS"
    assert verdict.duplicate_filter_status == "suppressed"
    assert verdict.suppressed_count == 1
    assert verdict.effective_count == 2
    assert verdict.unexpected_items == ()
    assert verdict.count_is_ok is True
    assert verdict.sequence_is_ok is True
    # Section 6.3 rule 7: the suppressed box's colour failure is retracted.
    assert verdict.color_is_ok is True
    assert verdict.fail_reasons == ()


def test_suppression_keeps_the_box_the_measurement_agrees_with() -> None:
    """Section 6.3 rule 2: consistency outranks confidence.

    The ``Red`` box here carries the *higher* confidence (0.808 against 0.664),
    so a confidence-ordered rule would keep the label the pixels refuted and
    fail a good board on the detector's own error.
    """
    payload = _duplicate_snapshot()
    verdict = _replay(payload, MODE_SUPPRESS)

    assert verdict.status == "PASS"
    assert verdict.suppressed_count == 1


def test_replay_reads_raw_detections_when_a_previous_run_already_suppressed() -> None:
    payload = _duplicate_snapshot()
    payload["raw_detections"] = deepcopy(payload["detections"])
    # What a suppress-mode run persisted as its effective set.
    payload["detections"] = [payload["raw_detections"][0], payload["raw_detections"][1]]

    verdict = _replay(payload, MODE_SUPPRESS)

    assert verdict.raw_count == 3
    assert verdict.suppressed_count == 1


def test_replay_does_not_mutate_the_snapshot_evidence() -> None:
    payload = _duplicate_snapshot()
    original = json.dumps(payload, sort_keys=True)

    _replay(payload, MODE_SUPPRESS)

    assert json.dumps(payload, sort_keys=True) == original


def test_unreadable_expected_items_fails_closed_instead_of_passing() -> None:
    payload = _duplicate_snapshot()
    del payload["config"]["expected_items"]

    verdict = _replay(payload, MODE_SUPPRESS)

    assert verdict.status == "DETECTION_FAIL"
    assert verdict.count_status == "expected_items_lookup_failed"


def test_position_check_enabled_blocks_suppression() -> None:
    """Section 4.1 requirement 8: an enabled position check must fail closed."""
    payload = _duplicate_snapshot()
    payload["config"]["position_config"]["Cable1"]["A"]["enabled"] = True

    verdict = _replay(payload, MODE_SUPPRESS)

    assert verdict.duplicate_filter_status == "blocked_position_enabled"
    assert verdict.suppressed_count == 0
    assert verdict.status == "DETECTION_FAIL"


def test_terminal_status_snapshot_is_refused() -> None:
    payload = _duplicate_snapshot(recorded_status="INFERENCE_ERROR")

    with pytest.raises(ReplayError) as excinfo:
        _replay(payload, MODE_SUPPRESS)

    assert excinfo.value.code == "terminal_status"


def test_malformed_detections_are_refused() -> None:
    payload = _duplicate_snapshot()
    payload["detections"] = ["not-a-mapping"]

    with pytest.raises(ReplayError) as excinfo:
        _replay(payload, MODE_SUPPRESS)

    assert excinfo.value.code == "invalid_detections"


def test_pipeline_without_color_check_is_refused() -> None:
    payload = _duplicate_snapshot()
    payload["config"]["pipeline"] = ["cross_class_duplicate_filter", "count_check"]

    with pytest.raises(ValueError):
        _replay(payload, MODE_SUPPRESS)


def test_unsupported_mode_is_refused() -> None:
    with pytest.raises(ReplayError) as excinfo:
        _replay(_duplicate_snapshot(), "off")

    assert excinfo.value.code == "unsupported_mode"


# --------------------------------------------------------------------------
# Transition classification and fidelity
# --------------------------------------------------------------------------


def test_classify_transition_separates_the_two_risk_directions() -> None:
    passing = _verdict(status="PASS")
    failing = _verdict(status="DETECTION_FAIL", reasons=("SEQUENCE_MISMATCH",))

    assert classify_transition(failing, passing) == TRANSITION_FAIL_TO_PASS
    assert classify_transition(passing, failing) == TRANSITION_PASS_TO_FAIL
    assert classify_transition(failing, failing) == TRANSITION_UNCHANGED
    assert (
        classify_transition(
            failing,
            _verdict(status="DETECTION_FAIL", reasons=("COLOR_MISMATCH",)),
        )
        == TRANSITION_REASONS_ONLY
    )


def test_comparison_reports_the_fail_to_pass_transition_and_cleared_reasons() -> None:
    comparison = compare_snapshot(
        _duplicate_snapshot(),
        filter_options=FILTER_OPTIONS,
        config_payload=_replay_config(),
    )

    assert comparison.transition == TRANSITION_FAIL_TO_PASS
    assert comparison.is_usable
    assert set(comparison.cleared_reasons) == {
        "UNEXPECTED_COMPONENT",
        "COLOR_MISMATCH",
        "SEQUENCE_MISMATCH",
    }
    assert comparison.added_reasons == ()


def test_fidelity_anchors_on_the_recorded_mode_not_on_report_only() -> None:
    """A suppress-recorded snapshot recorded the suppress verdict.

    Anchoring the fidelity check on report_only would flag every suppress-mode
    snapshot as unreproducible and bury a genuine mismatch among them.
    """
    payload = _duplicate_snapshot(
        recorded_mode=MODE_SUPPRESS, recorded_status="PASS", recorded_reasons=[]
    )

    comparison = compare_snapshot(
        payload, filter_options=FILTER_OPTIONS, config_payload=_replay_config()
    )

    assert comparison.fidelity.recorded_mode == MODE_SUPPRESS
    assert comparison.fidelity.status_reproduced
    assert comparison.fidelity.reasons_reproduced


def test_unreproducible_recorded_status_is_not_usable_as_evidence() -> None:
    payload = _duplicate_snapshot(recorded_status="PASS", recorded_reasons=[])

    comparison = compare_snapshot(
        payload, filter_options=FILTER_OPTIONS, config_payload=_replay_config()
    )

    assert not comparison.fidelity.status_reproduced
    assert not comparison.is_usable


def test_reason_drift_alone_stays_usable_as_evidence() -> None:
    """The colour-retraction fix changes reasons without changing the verdict."""
    payload = _duplicate_snapshot(
        recorded_mode=MODE_SUPPRESS,
        recorded_status="PASS",
        recorded_reasons=["COLOR_MISMATCH"],
    )

    comparison = compare_snapshot(
        payload, filter_options=FILTER_OPTIONS, config_payload=_replay_config()
    )

    assert comparison.fidelity.status_reproduced
    assert not comparison.fidelity.reasons_reproduced
    assert comparison.is_usable


@pytest.mark.parametrize(
    "duplicate_filter",
    [None, {}, {"mode": "off"}, {"mode": ""}, "suppress"],
)
def test_snapshot_without_a_replayable_recorded_mode_is_refused(
    duplicate_filter: Any,
) -> None:
    payload = _duplicate_snapshot()
    if duplicate_filter is None:
        del payload["duplicate_filter"]
    else:
        payload["duplicate_filter"] = duplicate_filter

    with pytest.raises(ReplayError) as excinfo:
        recorded_filter_mode(payload)

    assert excinfo.value.code == "recorded_mode_unavailable"


# --------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------


def _write_snapshot(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    path = directory / f"{name}_config_snapshot.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _run_report(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = main(argv)
    return code, capsys.readouterr().out


def test_report_counts_only_report_only_runs_toward_the_pilot_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    _write_snapshot(snapshots, "a", _duplicate_snapshot())
    _write_snapshot(
        snapshots,
        "b",
        _duplicate_snapshot(
            recorded_mode=MODE_SUPPRESS, recorded_status="PASS", recorded_reasons=[]
        ),
    )
    output = tmp_path / "report.json"

    code, _ = _run_report(
        [
            str(snapshots),
            "--product",
            "Cable1",
            "--area",
            "A",
            "--code-revision",
            "abc1234",
            "--output-json",
            str(output),
        ],
        capsys,
    )

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["recorded_mode_counts"] == {MODE_REPORT_ONLY: 1, MODE_SUPPRESS: 1}
    assert report["gate"]["report_only_inspection_count"] == 1
    assert report["ab"]["comparable_snapshot_count"] == 2


def test_gate_never_reports_satisfied_without_a_shift_attestation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Section 15 takes whichever condition completes later.

    Reaching the inspection count is therefore not sufficient, and the report
    must not present it as if it were.
    """
    _write_snapshot(tmp_path, "a", _duplicate_snapshot())

    code, out = _run_report(
        [str(tmp_path), "--product", "Cable1", "--area", "A", "--required-inspections", "1"],
        capsys,
    )

    assert code == 0
    assert "satisfied=False" in out
    assert "full shift coverage requires a named attestation" in out


def test_report_states_when_no_report_only_evidence_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_snapshot(
        tmp_path,
        "a",
        _duplicate_snapshot(
            recorded_mode=MODE_SUPPRESS, recorded_status="PASS", recorded_reasons=[]
        ),
    )

    code, out = _run_report([str(tmp_path), "--product", "Cable1", "--area", "A"], capsys)

    assert code == 0
    assert "has not produced any report-only evidence yet" in out


def test_report_says_when_no_snapshot_exercised_the_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A zero transition count on unexercised data must not read as a clean sheet."""
    _write_snapshot(tmp_path, "clean", _clean_snapshot())

    code, out = _run_report([str(tmp_path), "--product", "Cable1", "--area", "A"], capsys)

    assert code == 0
    assert "the policy is unexercised here" in out


def test_report_lists_the_verdict_changing_board(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_snapshot(tmp_path, "a", _duplicate_snapshot())

    code, out = _run_report([str(tmp_path), "--product", "Cable1", "--area", "A"], capsys)

    assert code == 0
    assert "fail_to_pass=1" in out
    assert "FAIL_TO_PASS" in out


def test_unreproducible_snapshot_is_excluded_and_fails_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_snapshot(
        tmp_path, "a", _duplicate_snapshot(recorded_status="PASS", recorded_reasons=[])
    )

    code, out = _run_report([str(tmp_path), "--product", "Cable1", "--area", "A"], capsys)

    assert code == 1
    assert "status_mismatch=1" in out
    assert "FIDELITY |" in out


def test_unreplayable_snapshot_is_named_rather_than_silently_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _duplicate_snapshot()
    del payload["duplicate_filter"]
    _write_snapshot(tmp_path, "a", payload)

    code, out = _run_report([str(tmp_path), "--product", "Cable1", "--area", "A"], capsys)

    assert code == 0
    assert "EXCLUDED | recorded_mode_unavailable | 1" in out


def test_writing_a_report_requires_a_declared_code_revision(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Snapshots cannot prove which code produced them.

    The 2026-08-05 regression survived because config looked right while code
    behaviour had changed, so an evidence file with no revision pin is refused.
    """
    _write_snapshot(tmp_path, "a", _duplicate_snapshot())

    code = main([str(tmp_path), "--output-json", str(tmp_path / "out" / "r.json")])

    assert code == 1
    assert not (tmp_path / "out" / "r.json").exists()


def test_report_records_the_revision_as_a_declaration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    _write_snapshot(snapshots, "a", _duplicate_snapshot())
    output = tmp_path / "report.json"

    code, _ = _run_report(
        [str(snapshots), "--code-revision", "abc1234", "--output-json", str(output)],
        capsys,
    )

    assert code == 0
    window = json.loads(output.read_text(encoding="utf-8"))["evidence_window"]
    assert window["declared_code_revision"] == "abc1234"
    assert window["code_revision_is_operator_declared"] is True


def test_report_destination_may_not_sit_inside_the_scanned_directory(
    tmp_path: Path,
) -> None:
    _write_snapshot(tmp_path, "a", _duplicate_snapshot())

    code = main(
        [str(tmp_path), "--code-revision", "abc1234", "--output-json", str(tmp_path / "r.json")]
    )

    assert code == 1
    assert not (tmp_path / "r.json").exists()


def test_empty_scope_is_reported_as_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_snapshot(tmp_path, "a", _duplicate_snapshot())

    code, out = _run_report([str(tmp_path), "--product", "Other"], capsys)

    assert code == 1
    assert "no snapshots matched the selected evidence window" in out
