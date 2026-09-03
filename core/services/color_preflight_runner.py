"""Locate a reference-board inspection and judge it, for the GUI and the CLI.

One evaluation path, two front ends. The CLI and the dialog ask the same
question at the same moment in the same ritual, so a second copy of "find the
snapshot, resolve what the station expects, check the baseline's provenance"
would be a copy free to drift -- which is the failure this whole area has
already paid for once.

Nothing here captures a frame. The station triggers its own inspection of the
golden sample through the normal manual path, and this reads what that
inspection saved. Keeping the check out of the capture path is also what lets
the same code answer the retrospective question against older snapshots.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:  # pragma: no cover - import cost is the point
    from core.services.color_preflight_samples import ColorGamutSample

from core.color_baseline_contract import color_model_compatibility_failure
from core.services.color_preflight import (
    ColorPreflightReport,
    evaluate_color_preflight,
)
from core.services.station_color_settings import (
    StationColorSettingsError,
    station_color_decision_tuning,
    station_color_preflight,
    station_color_roi_policy,
    station_expected_color_positions,
)

_STATS_CHECKER_TYPE = "stats"


class ColorPreflightUnavailable(RuntimeError):
    """Raised when there is nothing to judge, or nothing to judge it against.

    Distinct from a failing check: a station with no reference board inspection
    has not been measured, which is not the same as having been measured and
    found wanting.
    """


@dataclass(frozen=True)
class ColorPreflightSource:
    """Which inspection was judged, and when it was taken."""

    snapshot_path: Path
    taken_at: str
    model_version: str
    #: SHA-256 of the baseline this reading was scored against, so a reference
    #: recorded from it can be bound to that file and expire with it.
    baseline_sha256: str = ""

    @property
    def age(self) -> str:
        """How long ago the snapshot was written, in words an operator reads.

        Shown because a stale snapshot is the one real trap in reading a saved
        inspection: an operator who forgot to trigger one would otherwise be
        judging this shift on last week's board.
        """
        try:
            taken = datetime.fromisoformat(self.taken_at)
        except ValueError:
            return ""
        now = datetime.now(taken.tzinfo) if taken.tzinfo else datetime.now()
        seconds = max(0, int((now - taken).total_seconds()))
        if seconds < 90:
            return f"{seconds} 秒前"
        minutes = seconds // 60
        if minutes < 90:
            return f"{minutes} 分鐘前"
        hours = minutes // 60
        if hours < 48:
            return f"{hours} 小時前"
        return f"{hours // 24} 天前"


def latest_inspection_snapshot(
    results_root: str | Path, product: str, area: str, model_type: str
) -> Path | None:
    """Newest saved inspection for a scope, by file modification time.

    Modification time rather than the file name: the stem carries a wall-clock
    time but not a date, so sorting names mixes yesterday's late shift with
    this morning's.
    """
    root = Path(results_root).expanduser()
    if not root.is_dir():
        return None
    pattern = (
        f"*/{product}/{area}/*/metadata/{model_type}/*_config_snapshot.json"
    )
    candidates = [path for path in root.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ColorPreflightUnavailable(
            f"檢測快照無法讀取：{path}（{exc}）"
        ) from exc
    if not isinstance(payload, dict):
        raise ColorPreflightUnavailable(f"檢測快照格式錯誤：{path}")
    return payload


def read_station_config(config_path: str | Path) -> dict[str, Any]:
    """Load a station config as a plain mapping."""
    path = Path(config_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ColorPreflightUnavailable(f"站點模型設定不存在：{path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ColorPreflightUnavailable(
            f"站點模型設定無法讀取：{path}（{exc}）"
        ) from exc
    if not isinstance(payload, dict):
        raise ColorPreflightUnavailable(f"站點模型設定格式錯誤：{path}")
    return payload


def resolve_color_model(config_path: Path, raw_value: str) -> Path | None:
    """Locate the deployed baseline a station config names."""
    if not raw_value:
        return None
    name = Path(raw_value)
    for candidate in (
        config_path.parent / name.name,
        config_path.parent / name,
        name,
    ):
        resolved = candidate.expanduser()
        if resolved.is_file():
            return resolved.resolve()
    return None


def _baseline_identity(config_path: Path, station_config: dict[str, Any]) -> str:
    """SHA-256 of the baseline this station scores against, or ``""``.

    Identity by content rather than by the recorded algorithm label: a rebuild
    that keeps the same label still produces different statistics, and a
    reference recorded against the old numbers would go on being compared
    against the new ones.
    """
    model_path = resolve_color_model(
        config_path, str(station_config.get("color_model_path") or "")
    )
    if model_path is None:
        return ""
    try:
        return hashlib.sha256(model_path.read_bytes()).hexdigest()
    except OSError:
        return ""


def baseline_provenance_failure(
    config_path: str | Path, station_config: dict[str, Any]
) -> str:
    """Ask what the runtime asks when it loads this station's baseline.

    Surfaced in the report because a preflight that passed against statistics
    measured on another geometry has proven nothing -- and under ``warn`` a
    station loads exactly that with one log line.
    """
    path = Path(config_path)
    checker = str(
        station_config.get("color_checker_type") or ""
    ).strip().casefold()
    if checker != _STATS_CHECKER_TYPE:
        return ""
    model_path = resolve_color_model(
        path, str(station_config.get("color_model_path") or "")
    )
    if model_path is None:
        return "找不到已部署的顏色基準檔案"
    return color_model_compatibility_failure(
        model_path,
        expected_roi_policy=station_color_roi_policy(path).to_dict(),
        expected_decision_tuning=station_color_decision_tuning(path).to_dict(),
    )


def gamut_samples_for(
    config_path: str | Path, snapshot_path: str | Path
) -> dict[str, "ColorGamutSample"]:
    """Load the crops and colour clouds behind one reading.

    Separate from ``run_color_preflight`` because it is only worth its cost
    where there is a screen: the verdict and the margins need no images, so the
    command-line front end never pays for decoding them.

    Returns an empty mapping rather than raising when the pictures cannot be
    built -- a missing crop file is a reason to show fewer pictures, never a
    reason to withhold the reading they illustrate.
    """
    from core.services.color_preflight_samples import build_gamut_samples

    path = Path(config_path).expanduser()
    try:
        station_config = read_station_config(path)
        roi_policy = station_color_roi_policy(path)
        tuning = station_color_decision_tuning(path)
        payload = _read_snapshot(Path(snapshot_path))
    except (ColorPreflightUnavailable, StationColorSettingsError):
        return {}
    model_path = resolve_color_model(
        path, str(station_config.get("color_model_path") or "")
    )
    summary: dict[str, Any] = {}
    if model_path is not None:
        try:
            baseline = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            baseline = {}
        if isinstance(baseline, dict) and isinstance(
            baseline.get("summary"), dict
        ):
            summary = baseline["summary"]
    artifacts = payload.get("artifacts")
    crop_paths = (
        artifacts.get("cropped_paths") or []
        if isinstance(artifacts, dict)
        else []
    )
    color_result = payload.get("color_result") or {}
    try:
        return build_gamut_samples(
            crop_paths=[str(item) for item in crop_paths],
            detections=payload.get("detections") or [],
            color_items=color_result.get("items") or [],
            baseline_summary=summary,
            roi_policy=roi_policy,
            center_margin_ratio=tuning.center_margin_ratio,
            sat_threshold=tuning.sat_threshold,
        )
    except (OSError, ValueError, TypeError):
        return {}


def run_color_preflight(
    *,
    config_path: str | Path,
    results_root: str | Path,
    product: str,
    area: str,
    model_type: str,
    snapshot_path: str | Path | None = None,
) -> tuple[ColorPreflightReport, ColorPreflightSource]:
    """Judge the newest -- or a named -- reference-board inspection.

    Raises ``ColorPreflightUnavailable`` when the station cannot be judged at
    all: no declared board, no saved inspection, or an inspection that carries
    no color measurements.
    """
    path = Path(config_path).expanduser()
    try:
        # Positions, not distinct colors: a board with two black wires has to
        # be read as two blacks, or the second reads as a surplus.
        expected_colors = station_expected_color_positions(path, product, area)
        preflight = station_color_preflight(path)
        station_config = read_station_config(path)
        provenance = baseline_provenance_failure(path, station_config)
        baseline_id = _baseline_identity(path, station_config)
    except StationColorSettingsError as exc:
        raise ColorPreflightUnavailable(str(exc)) from exc

    if not expected_colors:
        raise ColorPreflightUnavailable(
            f"{product}/{area} 的 expected_items 沒有列出任何色別，"
            "無法判斷這片板子應該讀到什麼。"
        )

    if snapshot_path is not None:
        snapshot = Path(snapshot_path).expanduser()
        if not snapshot.is_file():
            raise ColorPreflightUnavailable(f"找不到檢測快照：{snapshot}")
    else:
        found = latest_inspection_snapshot(
            results_root, product, area, model_type
        )
        if found is None:
            raise ColorPreflightUnavailable(
                "找不到這個工位的檢測快照。請先放金板跑一次手動檢測。"
            )
        snapshot = found

    payload = _read_snapshot(snapshot)
    color_result = payload.get("color_result") or {}
    items = color_result.get("items") or []
    if not items:
        raise ColorPreflightUnavailable(
            "這筆檢測沒有顏色量測結果（status="
            f"{color_result.get('status')!r}）。"
        )

    report = evaluate_color_preflight(
        items,
        expected_colors,
        preflight.reference_margins,
        minimum_margin_retention=preflight.minimum_margin_retention,
        baseline_provenance_failure=provenance,
        reference_baseline_id=preflight.baseline_sha256,
        current_baseline_id=baseline_id,
    )
    model_info = payload.get("model_info") or {}
    source = ColorPreflightSource(
        snapshot_path=snapshot,
        taken_at=(
            str(payload.get("timestamp") or "")
            or datetime.fromtimestamp(snapshot.stat().st_mtime)
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        model_version=str(model_info.get("model_version") or ""),
        baseline_sha256=baseline_id,
    )
    return report, source
