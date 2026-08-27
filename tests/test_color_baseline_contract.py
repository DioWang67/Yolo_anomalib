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

from core.color_baseline_contract import (
    BASELINE_ALGORITHM_VERSION,
    baseline_compatibility_failure,
    color_model_algorithm,
    color_model_compatibility_failure,
)


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
        {"recalibration": {"algorithm": BASELINE_ALGORITHM_VERSION}},
    )
    stale = _write(
        tmp_path / "stale.json", {"recalibration": {"algorithm": "stats-robust-v2"}}
    )

    assert color_model_compatibility_failure(current) == ""
    assert "stats-robust-v2" in color_model_compatibility_failure(stale)


def test_the_rebuilder_stamps_the_contract_version() -> None:
    """One version, so what is written is what the three gates require.

    The rebuilder keeps its own name for it, and that alias drifting would let
    every rebuild produce artifacts the runtime then refuses.
    """
    from core.services.color_baseline_recalibration import ALGORITHM_VERSION

    assert ALGORITHM_VERSION == BASELINE_ALGORITHM_VERSION
