"""The color measurement settings a live station actually runs with.

Three parties need the same two answers -- what geometry the station measures
color in, and which decision tuning it scores with: the rebuilder, which must
build a baseline in that geometry; the runtime, which refuses a baseline built
in any other; and the acceptance and publication gates, which decide whether an
artifact may be offered at all. When only some of them could answer, a gate
could green-light a baseline the line then rejected.

Both answers come from the live station config, never from a model version
snapshot. A snapshot records what some other day's deployment carried, and
these fields belong to the physical station.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.services.color_preflight import DEFAULT_MINIMUM_MARGIN_RETENTION
from core.services.slot_roi import ColorRoiPolicy
from core.stats_color_checker import ColorDecisionTuning

#: Station config keys this module owns.
ROI_POLICY_KEY = "color_roi_policy"
DECISION_TUNING_KEY = "color_decision_tuning"
EXPECTED_ITEMS_KEY = "expected_items"
PREFLIGHT_KEY = "color_preflight"

#: Detector classes that name a part rather than a color, so they are not
#: expected in the color baseline's vocabulary. Defined here because every
#: party that reads a station's expected items has to draw the same line: the
#: runtime excluding one while config validation rejected it would take down a
#: station that is configured correctly.
DEFAULT_GENERIC_DETECTOR_CLASSES = ("LED",)


class StationColorSettingsError(ValueError):
    """Raised when a station's color settings cannot be read or are invalid."""


def station_config_path(
    models_root: str | Path, product: str, area: str, model_type: str
) -> Path:
    """Return the live station config for a scope.

    One definition of the shape, because every party that reads these settings
    has to read them from the same file to mean the same thing.
    """
    return Path(models_root) / product / area / model_type / "config.yaml"


def _station_payload(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise StationColorSettingsError(f"站點模型設定不存在：{path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise StationColorSettingsError(
            f"站點模型設定無法讀取：{path}（{exc}）"
        ) from exc
    if not isinstance(payload, dict):
        raise StationColorSettingsError(f"站點模型設定格式錯誤：{path}")
    return payload


def station_color_roi_policy(config_path: str | Path) -> ColorRoiPolicy:
    """Return the bbox policy the runtime measures color with.

    Must be the live station config, not a model version snapshot: the runtime
    reads this same field from this same file, so taking it from anywhere else
    lets a baseline be built in a geometry production does not use, and nothing
    downstream can tell -- every statistic is well-formed, only measured
    elsewhere.
    """
    payload = _station_payload(config_path)
    try:
        return ColorRoiPolicy.from_mapping(payload.get(ROI_POLICY_KEY))
    except (TypeError, ValueError) as exc:
        raise StationColorSettingsError(f"顏色 ROI 設定無效：{exc}") from exc


@dataclass(frozen=True)
class StationColorPreflight:
    """The station's recorded reference for the pre-shift color check.

    Recorded once on a known-good board and compared against every shift, the
    way ``calibration.target_luma`` is. It is advisory evidence, not part of the
    baseline contract: it changes no score, so a change here cannot invalidate a
    deployed baseline. It is still station-local -- the reference belongs to
    this fixture under this light -- so a deployment or a version rollback must
    preserve it rather than publish another day's numbers.
    """

    reference_margins: Mapping[str, float]
    minimum_margin_retention: float
    recorded_at: str
    recorded_by: str
    #: SHA-256 of the baseline these margins were measured against. A margin is
    #: a distance above a threshold that the baseline defines, so the pair is
    #: only comparable while the baseline is the same file. Recording the hash
    #: is what makes a reference expire with its baseline instead of quietly
    #: outliving it.
    baseline_sha256: str = ""

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_margins)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "minimum_margin_retention": float(self.minimum_margin_retention),
            "reference_margins": {
                name: float(value)
                for name, value in sorted(self.reference_margins.items())
            },
        }
        if self.baseline_sha256:
            payload["baseline_sha256"] = self.baseline_sha256
        if self.recorded_at:
            payload["recorded_at"] = self.recorded_at
        if self.recorded_by:
            payload["recorded_by"] = self.recorded_by
        return payload

    @classmethod
    def from_mapping(
        cls, value: object | None
    ) -> "StationColorPreflight":
        if value is None:
            return cls({}, DEFAULT_MINIMUM_MARGIN_RETENTION, "", "")
        if not isinstance(value, Mapping):
            raise StationColorSettingsError(
                f"{PREFLIGHT_KEY} 必須是 mapping。"
            )
        raw_margins = value.get("reference_margins")
        if raw_margins is None:
            margins: dict[str, float] = {}
        elif isinstance(raw_margins, Mapping):
            margins = {}
            for name, margin in raw_margins.items():
                key = str(name or "").strip()
                if not key:
                    continue
                try:
                    margins[key] = float(margin)
                except (TypeError, ValueError) as exc:
                    raise StationColorSettingsError(
                        f"{PREFLIGHT_KEY}.reference_margins.{key} 不是數值。"
                    ) from exc
        else:
            raise StationColorSettingsError(
                f"{PREFLIGHT_KEY}.reference_margins 必須是 mapping。"
            )
        raw_retention = value.get("minimum_margin_retention")
        if raw_retention is None:
            retention = DEFAULT_MINIMUM_MARGIN_RETENTION
        else:
            try:
                retention = float(raw_retention)
            except (TypeError, ValueError) as exc:
                raise StationColorSettingsError(
                    f"{PREFLIGHT_KEY}.minimum_margin_retention 不是數值。"
                ) from exc
            if not 0.0 < retention <= 1.0:
                # Above 1.0 would demand a board better than the reference it
                # was recorded from, so every shift would fail.
                raise StationColorSettingsError(
                    f"{PREFLIGHT_KEY}.minimum_margin_retention 必須介於 0 與 1。"
                )
        return cls(
            reference_margins=margins,
            minimum_margin_retention=retention,
            recorded_at=str(value.get("recorded_at") or "").strip(),
            recorded_by=str(value.get("recorded_by") or "").strip(),
            baseline_sha256=str(value.get("baseline_sha256") or "").strip(),
        )


def station_color_preflight(config_path: str | Path) -> StationColorPreflight:
    """Return the station's recorded pre-shift color reference."""
    payload = _station_payload(config_path)
    return StationColorPreflight.from_mapping(payload.get(PREFLIGHT_KEY))


def expected_color_positions(
    station_config: Mapping[str, Any],
    product: str,
    area: str,
    *,
    generic_classes: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return one entry per color position the station inspects, in order.

    Repeats are kept: a board carrying two black wires expects black twice, and
    a caller counting what it read needs that count. Generic detector classes
    are dropped, because they name a part and carry no baseline statistics.
    """
    expected = station_config.get(EXPECTED_ITEMS_KEY)
    if not isinstance(expected, Mapping):
        return ()
    areas = expected.get(product)
    if not isinstance(areas, Mapping):
        return ()
    items = areas.get(area)
    if isinstance(items, str) or not isinstance(items, (list, tuple, set)):
        return ()
    generic = {
        name.casefold()
        for value in (
            DEFAULT_GENERIC_DETECTOR_CLASSES
            if generic_classes is None
            else generic_classes
        )
        if (name := str(value or "").strip())
    }
    return tuple(
        name
        for item in items
        if (name := str(item or "").strip())
        and name.casefold() not in generic
    )


def expected_color_names(
    station_config: Mapping[str, Any],
    product: str,
    area: str,
    *,
    generic_classes: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Return the distinct colors a station inspects, in config order.

    Repeats are collapsed, for callers that need the color vocabulary rather
    than the board layout -- the rebuilder rejects a repeated color request,
    and config validation only has to name each unknown color once. Use
    ``expected_color_positions`` when the count per color matters.
    """
    seen: set[str] = set()
    colors: list[str] = []
    for name in expected_color_positions(
        station_config, product, area, generic_classes=generic_classes
    ):
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        colors.append(name)
    return tuple(colors)


def station_expected_color_names(
    config_path: str | Path,
    product: str,
    area: str,
    *,
    generic_classes: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """``expected_color_names`` for a station config on disk."""
    return expected_color_names(
        _station_payload(config_path),
        product,
        area,
        generic_classes=generic_classes,
    )


def station_expected_color_positions(
    config_path: str | Path,
    product: str,
    area: str,
    *,
    generic_classes: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """``expected_color_positions`` for a station config on disk."""
    return expected_color_positions(
        _station_payload(config_path),
        product,
        area,
        generic_classes=generic_classes,
    )


def station_color_decision_tuning(
    config_path: str | Path,
) -> ColorDecisionTuning:
    """Return the complete effective tuning the live station scores with."""
    payload = _station_payload(config_path)
    raw_tuning = payload.get(DECISION_TUNING_KEY)
    if raw_tuning is not None and not isinstance(raw_tuning, dict):
        raise StationColorSettingsError(
            f"{DECISION_TUNING_KEY} 必須是 mapping。"
        )
    try:
        return ColorDecisionTuning.from_dict(raw_tuning)
    except (TypeError, ValueError) as exc:
        raise StationColorSettingsError(
            f"顏色 decision tuning 無效：{exc}"
        ) from exc
