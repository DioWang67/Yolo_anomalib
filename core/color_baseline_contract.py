"""The contract between whoever writes a color baseline and whoever trusts one.

A stored baseline is not a free-standing table of numbers. ``coverage_mean`` and
the sampled envelopes are only meaningful against the ROI geometry they were
measured on, so a baseline is a table *plus* the algorithm that produced it.
Pairing statistics from one geometry with a runtime that measures another does
not fail loudly -- it shifts every score by whatever the two crops differ by,
which reads as a working system with a mysteriously generous or harsh color
check.

This module therefore owns two things and nothing else: which algorithm is
current, and the single way to read an artifact's own record of the algorithm
that built it. Three parties consult it -- the acceptance picker, the
publication gate and the runtime loader -- and they must never drift into
disagreeing about which baselines are usable, which is exactly what happens
when each grows its own copy of the check.

It deliberately sits below the rebuilder: the rebuilder imports the checker, so
the version cannot live in the rebuilder without the runtime having to import
the whole recalibration service to find out what it requires.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

#: Bumped whenever a rebuild changes what the stored statistics mean -- the ROI
#: geometry, the sampling mask, or the definition of a recorded quantity. A bump
#: invalidates comparison against every earlier baseline; it is not a changelog
#: for the rebuilder's internals.
BASELINE_ALGORITHM_VERSION = "stats-robust-v4"

#: Where a color model records the algorithm that produced it.
_PROVENANCE_SECTION = "recalibration"
_PROVENANCE_KEY = "algorithm"


def color_model_algorithm(stats_path: str | Path) -> str | None:
    """Return the algorithm a color model records for itself.

    ``None`` covers every way the answer can be absent -- no provenance block,
    an unreadable file, a malformed one -- because the callers all treat "cannot
    be established" identically and a raised exception here would only be
    swallowed at each of them.
    """
    try:
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    section = payload.get(_PROVENANCE_SECTION)
    if not isinstance(section, Mapping):
        return None
    recorded = section.get(_PROVENANCE_KEY)
    if not isinstance(recorded, str):
        return None
    return recorded.strip() or None


def baseline_compatibility_failure(algorithm: str | None) -> str:
    """Return why a baseline built by ``algorithm`` cannot be trusted, or ``""``.

    Phrased as the cause rather than as one caller's consequence, so the picker,
    the publication gate and the runtime log can all state the same reason.
    """
    if algorithm == BASELINE_ALGORITHM_VERSION:
        return ""
    if not algorithm:
        return (
            f"未記錄重建演算法（目前為 {BASELINE_ALGORITHM_VERSION}）："
            "無法確認裁切座標空間是否相同"
        )
    return (
        f"演算法 {algorithm}（目前為 {BASELINE_ALGORITHM_VERSION}）："
        "裁切座標空間不同，統計量不可比較"
    )


def color_model_compatibility_failure(stats_path: str | Path) -> str:
    """Return why the color model at ``stats_path`` cannot be trusted, or ``""``."""
    return baseline_compatibility_failure(color_model_algorithm(stats_path))
