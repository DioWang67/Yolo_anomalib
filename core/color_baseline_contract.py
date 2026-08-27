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
#: Colors whose statistics were copied from the base rather than measured by the
#: run that wrote the file.
_PRESERVED_KEY = "preserved_colors"
#: What the base that those colors came from claimed for itself.
_BASE_ALGORITHM_KEY = "base_algorithm"
#: What the same idea was called before both preservation reasons shared a list.
_LEGACY_PRESERVED_KEY = "preserved_by_safety"


def _provenance(stats_path: str | Path) -> Mapping[str, object]:
    """Return an artifact's provenance block, or an empty mapping."""
    try:
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(payload, Mapping):
        return {}
    section = payload.get(_PROVENANCE_SECTION)
    return section if isinstance(section, Mapping) else {}


def _recorded_string(section: Mapping[str, object], key: str) -> str | None:
    recorded = section.get(key)
    if not isinstance(recorded, str):
        return None
    return recorded.strip() or None


def color_model_algorithm(stats_path: str | Path) -> str | None:
    """Return the algorithm a color model records for itself.

    ``None`` covers every way the answer can be absent -- no provenance block,
    an unreadable file, a malformed one -- because the callers all treat "cannot
    be established" identically and a raised exception here would only be
    swallowed at each of them.
    """
    return _recorded_string(_provenance(stats_path), _PROVENANCE_KEY)


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
    """Return why the color model at ``stats_path`` cannot be trusted, or ``""``.

    Stricter than the algorithm alone, because a rebuild that preserves a color
    copies that color's numbers from its base. Such a file honestly records that
    this run used the current algorithm while part of its statistics were
    measured by whatever produced the base, so the stamp alone would let the
    base's geometry borrow a label it did not earn. A mixture is acceptable only
    when the base claimed the current algorithm as well -- which is the ordinary
    case of a later rebuild preserving a color that ran short of evidence.
    """
    section = _provenance(stats_path)
    failure = baseline_compatibility_failure(
        _recorded_string(section, _PROVENANCE_KEY)
    )
    if failure:
        return failure
    preserved = section.get(_PRESERVED_KEY)
    if not isinstance(preserved, (list, tuple)):
        # Artifacts written before the single list existed recorded only the
        # safety-rejected colors. Falling back to that is what makes this check
        # reach the file that exposed the hole, rather than only future ones.
        preserved = section.get(_LEGACY_PRESERVED_KEY)
    if not isinstance(preserved, (list, tuple)) or not preserved:
        return ""
    base_algorithm = _recorded_string(section, _BASE_ALGORITHM_KEY)
    if base_algorithm == BASELINE_ALGORITHM_VERSION:
        return ""
    names = "、".join(str(color) for color in preserved)
    return (
        f"{names} 沿用舊基準統計（舊基準演算法 "
        f"{base_algorithm or '未記錄'}，目前為 {BASELINE_ALGORITHM_VERSION}）："
        "裁切座標空間不同，統計量不可比較"
    )
