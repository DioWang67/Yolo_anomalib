"""The contract between whoever writes a color baseline and whoever trusts one.

A stored baseline is not a free-standing table of numbers. ``coverage_mean`` and
the sampled envelopes are only meaningful against the ROI geometry they were
measured on, so a baseline is a table *plus* the algorithm that produced it.
Pairing statistics from one geometry with a runtime that measures another does
not fail loudly -- it shifts every score by whatever the two crops differ by,
which reads as a working system with a mysteriously generous or harsh color
check.

This module therefore owns the current algorithm and the single compatibility
decision over an artifact's provenance, including the ROI geometry used to
measure it. Three parties consult it -- the acceptance picker, the publication
gate and the runtime loader -- and they must never drift into disagreeing about
which baselines are usable, which is exactly what happens when each grows its
own copy of the check.

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
BASELINE_ALGORITHM_VERSION = "stats-robust-v5"

#: Where a color model records the algorithm that produced it.
_PROVENANCE_SECTION = "recalibration"
#: Where a stats artifact keeps its per-color statistics, one key per color.
_SUMMARY_SECTION = "summary"
_PROVENANCE_KEY = "algorithm"
#: Colors whose statistics were copied from the base rather than measured by the
#: run that wrote the file.
_PRESERVED_KEY = "preserved_colors"
#: What the base that those colors came from claimed for itself.
_BASE_ALGORITHM_KEY = "base_algorithm"
#: Sampling geometry used by the run that wrote the artifact.
_ROI_POLICY_KEY = "color_roi_policy"
#: Sampling geometry of the base supplying any preserved colors.
_BASE_ROI_POLICY_KEY = "base_color_roi_policy"
#: Complete resolved classifier tuning used for rebuild validation.
_DECISION_TUNING_KEY = "color_decision_tuning"
#: Resolved tuning used by the base supplying any preserved colors.
_BASE_DECISION_TUNING_KEY = "base_color_decision_tuning"
#: v5 classifier fields. Adding, removing or redefining one requires a new
#: baseline algorithm version; a partial mapping is not resolved provenance.
_REQUIRED_DECISION_TUNING_KEYS = frozenset(
    {
        "sat_threshold",
        "yellow_h_min",
        "yellow_h_max",
        "yellow_s_min",
        "yellow_v_min",
        "orange_red_tie_margin",
        "center_margin_ratio",
        "red_h_low_max",
        "red_h_high_min",
        "red_s_min",
        "red_v_min",
        "orange_h_min",
        "orange_h_max",
        "orange_s_min",
        "orange_v_min",
        "green_h_min",
        "green_h_max",
        "green_s_min",
        "green_v_min",
        "green_v_max",
    }
)
#: What the same idea was called before both preservation reasons shared a list.
_LEGACY_PRESERVED_KEY = "preserved_by_safety"


def _payload(stats_path: str | Path) -> Mapping[str, object] | None:
    """Return an artifact's parsed contents, or ``None`` when unreadable."""
    try:
        payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _provenance(stats_path: str | Path) -> Mapping[str, object]:
    """Return an artifact's provenance block, or an empty mapping."""
    payload = _payload(stats_path)
    if payload is None:
        return {}
    section = payload.get(_PROVENANCE_SECTION)
    return section if isinstance(section, Mapping) else {}


def color_model_vocabulary(stats_path: str | Path) -> frozenset[str] | None:
    """Return the colors a stats artifact carries statistics for, casefolded.

    ``None`` means the artifact could not be read or has no summary at all --
    which a caller must not treat as "an artifact that scores no colors", or a
    corrupt file would read as a configuration error about every color.
    """
    payload = _payload(stats_path)
    if payload is None:
        return None
    summary = payload.get(_SUMMARY_SECTION)
    if not isinstance(summary, Mapping):
        return None
    return frozenset(
        normalized.casefold()
        for color in summary
        if (normalized := str(color or "").strip())
    )


def _recorded_string(section: Mapping[str, object], key: str) -> str | None:
    recorded = section.get(key)
    if not isinstance(recorded, str):
        return None
    return recorded.strip() or None


def _canonical_roi_policy(value: object) -> tuple[float, float, int] | None:
    """Normalize the geometry fields needed for an exact contract comparison."""
    if not isinstance(value, Mapping):
        return None
    try:
        inset_x = float(value["inset_x_ratio"])
        inset_y = float(value["inset_y_ratio"])
        raw_min_size = value["min_size"]
        if isinstance(raw_min_size, bool):
            return None
        min_size = int(raw_min_size)
        if float(raw_min_size) != min_size:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return inset_x, inset_y, min_size


def _format_roi_policy(policy: tuple[float, float, int]) -> str:
    return f"inset_x={policy[0]:g}, inset_y={policy[1]:g}, min_size={policy[2]}"


def _canonical_tuning(value: object) -> tuple[tuple[str, float], ...] | None:
    """Normalize a complete resolved tuning mapping for exact comparison."""
    if not isinstance(value, Mapping) or not value:
        return None
    normalized: list[tuple[str, float]] = []
    try:
        for key, raw_value in value.items():
            if not isinstance(key, str) or not key.strip() or isinstance(raw_value, bool):
                return None
            number = float(raw_value)
            if not (-float("inf") < number < float("inf")):
                return None
            normalized.append((key.strip(), number))
    except (TypeError, ValueError):
        return None
    if {key for key, _value in normalized} != _REQUIRED_DECISION_TUNING_KEYS:
        return None
    return tuple(sorted(normalized))


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


def color_model_compatibility_failure(
    stats_path: str | Path,
    *,
    expected_roi_policy: Mapping[str, object] | None = None,
    expected_decision_tuning: Mapping[str, object] | None = None,
) -> str:
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
    recorded_tuning = _canonical_tuning(section.get(_DECISION_TUNING_KEY))
    if recorded_tuning is None:
        return "未記錄完整 resolved color decision tuning：無法確認驗證與執行期 classifier 相同"
    expected_tuning = (
        _canonical_tuning(expected_decision_tuning)
        if expected_decision_tuning is not None
        else None
    )
    if expected_decision_tuning is not None and expected_tuning is None:
        return "執行期 color decision tuning 格式無效，無法確認 classifier 契約"
    if expected_tuning is not None and recorded_tuning != expected_tuning:
        return "color decision tuning 不一致：基準驗證與執行期 classifier 不同"
    expected_geometry = (
        _canonical_roi_policy(expected_roi_policy)
        if expected_roi_policy is not None
        else None
    )
    if expected_roi_policy is not None and expected_geometry is None:
        return "執行期顏色 ROI policy 格式無效，無法確認量測幾何"
    if expected_geometry is not None:
        recorded_geometry = _canonical_roi_policy(section.get(_ROI_POLICY_KEY))
        if recorded_geometry is None:
            return "未記錄顏色 ROI policy：無法確認基準與執行期量測幾何相同"
        if recorded_geometry != expected_geometry:
            return (
                "顏色 ROI policy 不一致（基準："
                f"{_format_roi_policy(recorded_geometry)}；執行期："
                f"{_format_roi_policy(expected_geometry)}）"
            )
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
        base_tuning = _canonical_tuning(section.get(_BASE_DECISION_TUNING_KEY))
        if base_tuning is None:
            return (
                f"{'、'.join(str(color) for color in preserved)} 沿用的舊基準"
                "未記錄完整 resolved color decision tuning"
            )
        if base_tuning != recorded_tuning:
            return (
                f"{'、'.join(str(color) for color in preserved)} 沿用不同 color "
                "decision tuning 的舊基準"
            )
        if expected_geometry is None:
            return ""
        base_geometry = _canonical_roi_policy(section.get(_BASE_ROI_POLICY_KEY))
        if base_geometry is None:
            return (
                f"{'、'.join(str(color) for color in preserved)} 沿用的舊基準"
                "未記錄顏色 ROI policy：無法確認量測幾何相同"
            )
        if base_geometry == expected_geometry:
            return ""
        return (
            f"{'、'.join(str(color) for color in preserved)} 沿用不同顏色 ROI policy "
            f"的舊基準（舊基準：{_format_roi_policy(base_geometry)}；執行期："
            f"{_format_roi_policy(expected_geometry)}）"
        )
    names = "、".join(str(color) for color in preserved)
    return (
        f"{names} 沿用舊基準統計（舊基準演算法 "
        f"{base_algorithm or '未記錄'}，目前為 {BASELINE_ALGORITHM_VERSION}）："
        "裁切座標空間不同，統計量不可比較"
    )
