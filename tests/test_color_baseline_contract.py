"""The baseline contract: what an artifact claims, and what that claim allows.

These read like small tests because the module is small. It is worth its own
file anyway: three separate gates -- the acceptance picker, the publication
builder and the runtime loader -- act on nothing but these two functions, so a
silent change of meaning here changes what every one of them lets through.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core import color_baseline_contract as contract
from core.color_baseline_contract import (
    BASELINE_ALGORITHM_VERSION,
    baseline_compatibility_failure,
    color_model_algorithm,
    color_model_compatibility_failure,
)
from core.stats_color_checker import ColorDecisionTuning

_RESOLVED_TUNING = ColorDecisionTuning().to_dict()


def test_v5_provenance_schema_matches_the_effective_tuning_fields() -> None:
    """A tuning field change without a version-contract change must fail CI."""
    assert set(_RESOLVED_TUNING) == contract._REQUIRED_DECISION_TUNING_KEYS


def _write(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_algorithm_is_read_from_the_artifacts_own_record(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "color_stats.json",
        {"summary": {"Black": {"count": 6}}, "recalibration": {"algorithm": "  v7  "}},
    )

    assert color_model_algorithm(path) == "v7"


@pytest.mark.parametrize(
    "payload",
    [
        {"summary": {}},
        {"summary": {}, "recalibration": {}},
        {"summary": {}, "recalibration": {"algorithm": ""}},
        {"summary": {}, "recalibration": {"algorithm": "   "}},
        {"summary": {}, "recalibration": {"algorithm": 4}},
        {"summary": {}, "recalibration": "stats-robust-v4"},
        ["stats-robust-v4"],
    ],
    ids=[
        "no-provenance-block",
        "empty-block",
        "empty-string",
        "blank-string",
        "not-a-string",
        "block-is-not-a-mapping",
        "payload-is-not-a-mapping",
    ],
)
def test_an_unusable_record_reads_as_no_record(tmp_path: Path, payload: object) -> None:
    """Every way of failing to say it collapses to the same answer.

    The callers treat "cannot be established" identically, so distinguishing a
    malformed record from an absent one here would only be discarded at each of
    them -- and a raised exception would be swallowed three times over.
    """
    assert color_model_algorithm(_write(tmp_path / "stats.json", payload)) is None


def test_an_unreadable_file_reads_as_no_record(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"recalibration": {"algorithm"', encoding="utf-8")

    assert color_model_algorithm(missing) is None
    assert color_model_algorithm(truncated) is None


def test_the_current_algorithm_is_the_only_compatible_one() -> None:
    assert baseline_compatibility_failure(BASELINE_ALGORITHM_VERSION) == ""


def test_a_superseded_algorithm_names_both_versions() -> None:
    """The operator has to be able to tell which of their baselines this is."""
    failure = baseline_compatibility_failure("stats-robust-v2")

    assert "stats-robust-v2" in failure
    assert BASELINE_ALGORITHM_VERSION in failure


def test_no_record_is_refused_separately_from_a_stale_one() -> None:
    """Distinct because the remedy differs.

    A stale baseline is rebuilt. One that never recorded an algorithm may be
    perfectly good and merely undocumented, and telling the operator it was
    built by a superseded algorithm would be a claim nothing supports.
    """
    unrecorded = baseline_compatibility_failure(None)

    assert unrecorded != ""
    assert unrecorded != baseline_compatibility_failure("stats-robust-v2")
    assert BASELINE_ALGORITHM_VERSION in unrecorded


def test_compatibility_reads_the_file_when_given_a_path(tmp_path: Path) -> None:
    current = _write(
        tmp_path / "current.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
            }
        },
    )
    stale = _write(
        tmp_path / "stale.json", {"recalibration": {"algorithm": "stats-robust-v2"}}
    )

    assert color_model_compatibility_failure(current) == ""
    assert "stats-robust-v2" in color_model_compatibility_failure(stale)


def test_runtime_roi_policy_must_match_the_artifact_geometry(tmp_path: Path) -> None:
    runtime_policy = {
        "inset_x_ratio": 0.2,
        "inset_y_ratio": 0.0,
        "min_size": 8,
    }
    matching = _write(
        tmp_path / "matching.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_roi_policy": runtime_policy,
                "color_decision_tuning": _RESOLVED_TUNING,
            }
        },
    )
    missing = _write(
        tmp_path / "missing.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
            }
        },
    )
    different = _write(
        tmp_path / "different.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
                "color_roi_policy": {
                    "inset_x_ratio": 0.1,
                    "inset_y_ratio": 0.0,
                    "min_size": 8,
                },
            }
        },
    )

    assert (
        color_model_compatibility_failure(
            matching, expected_roi_policy=runtime_policy
        )
        == ""
    )
    assert "未記錄顏色 ROI policy" in color_model_compatibility_failure(
        missing, expected_roi_policy=runtime_policy
    )
    assert "ROI policy 不一致" in color_model_compatibility_failure(
        different, expected_roi_policy=runtime_policy
    )


def test_preserved_colors_must_come_from_the_same_roi_geometry(tmp_path: Path) -> None:
    runtime_policy = {
        "inset_x_ratio": 0.2,
        "inset_y_ratio": 0.0,
        "min_size": 8,
    }
    path = _write(
        tmp_path / "mixed.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_roi_policy": runtime_policy,
                "color_decision_tuning": _RESOLVED_TUNING,
                "preserved_colors": ["Black"],
                "base_algorithm": BASELINE_ALGORITHM_VERSION,
                "base_color_decision_tuning": _RESOLVED_TUNING,
                "base_color_roi_policy": {
                    "inset_x_ratio": 0.0,
                    "inset_y_ratio": 0.0,
                    "min_size": 8,
                },
            }
        },
    )

    failure = color_model_compatibility_failure(
        path, expected_roi_policy=runtime_policy
    )

    assert "Black" in failure
    assert "不同顏色 ROI policy" in failure


def test_the_rebuilder_stamps_the_contract_version() -> None:
    """One version, so what is written is what the three gates require.

    The rebuilder keeps its own name for it, and that alias drifting would let
    every rebuild produce artifacts the runtime then refuses.
    """
    from core.services.color_baseline_recalibration import ALGORITHM_VERSION

    assert ALGORITHM_VERSION == BASELINE_ALGORITHM_VERSION


def test_preserved_colors_cannot_borrow_the_current_label(tmp_path: Path) -> None:
    """A rebuild that preserves a color copies that color's numbers from the base.

    The file then honestly records that this run used the current algorithm
    while part of its statistics were measured by whatever produced the base.
    Trusting the stamp alone let a rebuild whose five colors were all preserved
    come out byte-identical to the deployed baseline and still pass every gate.
    """
    path = _write(
        tmp_path / "mixed.json",
        {
            "summary": {},
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
                "preserved_colors": ["Black", "Red"],
                "base_algorithm": "stats-robust-v2",
            },
        },
    )

    failure = color_model_compatibility_failure(path)

    assert "Black" in failure and "Red" in failure
    assert "stats-robust-v2" in failure


def test_a_mixture_on_a_current_base_is_accepted(tmp_path: Path) -> None:
    """Otherwise a color that ran short of evidence would block every rebuild.

    Preserving from a base built by the same algorithm keeps one geometry
    throughout, which is the ordinary case and must not be refused.
    """
    path = _write(
        tmp_path / "mixed.json",
        {
            "summary": {},
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
                "preserved_colors": ["Green"],
                "base_algorithm": BASELINE_ALGORITHM_VERSION,
                "base_color_decision_tuning": _RESOLVED_TUNING,
            },
        },
    )

    assert color_model_compatibility_failure(path) == ""


def test_a_base_with_no_record_is_not_treated_as_agreement(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "mixed.json",
        {
            "summary": {},
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
                "preserved_colors": ["Black"],
            },
        },
    )

    assert "未記錄" in color_model_compatibility_failure(path)


def test_the_older_field_name_still_reaches_the_check(tmp_path: Path) -> None:
    """The file that exposed this hole predates the combined list.

    Reading only the new field would have left that candidate passing, which is
    the one case the check exists for.
    """
    path = _write(
        tmp_path / "legacy.json",
        {
            "summary": {},
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
                "preserved_by_safety": ["Black", "Green", "Orange", "Red", "Yellow"],
            },
        },
    )

    assert color_model_compatibility_failure(path) != ""


def test_a_fully_rebuilt_candidate_is_unaffected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "rebuilt.json",
        {
            "summary": {},
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
                "preserved_colors": [],
                "base_algorithm": "stats-robust-v2",
            },
        },
    )

    assert color_model_compatibility_failure(path) == ""


def test_resolved_tuning_must_be_recorded_and_match_runtime(tmp_path: Path) -> None:
    matching = _write(
        tmp_path / "matching.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": _RESOLVED_TUNING,
            }
        },
    )
    missing = _write(
        tmp_path / "missing.json",
        {"recalibration": {"algorithm": BASELINE_ALGORITHM_VERSION}},
    )
    partial = _write(
        tmp_path / "partial.json",
        {
            "recalibration": {
                "algorithm": BASELINE_ALGORITHM_VERSION,
                "color_decision_tuning": {"yellow_h_max": 35.0},
            }
        },
    )
    changed = dict(_RESOLVED_TUNING)
    changed["yellow_h_max"] += 1.0

    assert (
        color_model_compatibility_failure(
            matching,
            expected_decision_tuning=_RESOLVED_TUNING,
        )
        == ""
    )
    assert "未記錄完整" in color_model_compatibility_failure(missing)
    assert "未記錄完整" in color_model_compatibility_failure(partial)
    assert "tuning 不一致" in color_model_compatibility_failure(
        matching,
        expected_decision_tuning=changed,
    )


def test_v4_artifact_is_rejected_by_v5_runtime(tmp_path: Path) -> None:
    legacy = _write(
        tmp_path / "v4.json",
        {
            "recalibration": {
                "algorithm": "stats-robust-v4",
                "color_decision_tuning": _RESOLVED_TUNING,
            }
        },
    )

    failure = color_model_compatibility_failure(legacy)

    assert "stats-robust-v4" in failure
    assert BASELINE_ALGORITHM_VERSION in failure
