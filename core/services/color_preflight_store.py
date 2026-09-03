"""Persistence for the pre-shift color check: its ledger and its reference.

Two different kinds of writing live here, and they are deliberately not the
same operation:

* **The ledger** is append-only evidence. Every run lands in it, pass or fail,
  so "the margin has been sliding for a fortnight" is answerable rather than
  remembered. Nothing reads it back to make a decision.
* **The reference** is what shifts are judged against, and writing it is an
  explicit, named act -- the same shape as recording ``calibration.target_luma``
  on a known-good board. It is never written as a side effect of a run: a check
  that quietly re-recorded its own reading would rise with the drift it exists
  to detect and always report OK.

Neither one touches the color baseline. A reference margin changes no score.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from core.services.color_preflight import ColorPreflightReport
from core.services.station_color_settings import (
    PREFLIGHT_KEY,
    StationColorPreflight,
    StationColorSettingsError,
)

LEDGER_SCHEMA_VERSION = 1


class ColorPreflightStoreError(RuntimeError):
    """Raised when a preflight record or reference cannot be persisted."""


@dataclass(frozen=True)
class ColorPreflightRecord:
    """One pre-shift run, as it is written to the ledger."""

    recorded_at: str
    operator: str
    product: str
    area: str
    model_type: str
    model_version: str
    report: ColorPreflightReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "recorded_at": self.recorded_at,
            "operator": self.operator,
            "product": self.product,
            "area": self.area,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "status": self.report.status,
            "baseline_provenance_failure": (
                self.report.baseline_provenance_failure
            ),
            "minimum_margin_retention": self.report.minimum_margin_retention,
            "reference_is_missing": self.report.reference_is_missing,
            "colors": [
                {
                    "color": item.color,
                    "expected_count": item.expected_count,
                    "observed_count": item.observed_count,
                    "margin": item.margin,
                    "reference_margin": item.reference_margin,
                    "retention": item.retention,
                    "state": item.state,
                }
                for item in self.report.colors
            ],
        }


def _utc_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ColorPreflightLedger:
    """Append-only, one file per station scope, newest line last."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser()

    def path_for(self, product: str, area: str, model_type: str) -> Path:
        return (
            self._root
            / _safe_segment(product)
            / _safe_segment(area)
            / f"{_safe_segment(model_type)}.jsonl"
        )

    def append(self, record: ColorPreflightRecord) -> Path:
        path = self.path_for(record.product, record.area, record.model_type)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            raise ColorPreflightStoreError(
                f"顏色開線紀錄寫入失敗：{path}（{exc}）"
            ) from exc
        return path

    def read(
        self, product: str, area: str, model_type: str, *, limit: int | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Return past runs, oldest first, skipping any unreadable line.

        A corrupt line is skipped rather than fatal: this is a trend log, and
        losing today's history because one line was truncated by a power cut
        would be a worse outcome than a gap.
        """
        path = self.path_for(product, area, model_type)
        if not path.is_file():
            return ()
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            raise ColorPreflightStoreError(
                f"顏色開線紀錄無法讀取：{path}（{exc}）"
            ) from exc
        records: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        if limit is not None and limit >= 0:
            records = records[-limit:]
        return tuple(records)


def _safe_segment(value: str) -> str:
    """Keep a scope name usable as one path segment.

    Product and area come from config, so they are not attacker-controlled, but
    a stray separator would silently scatter a station's history across
    directories instead of failing.
    """
    name = str(value or "").strip()
    if not name or name in {".", ".."} or any(sep in name for sep in "/\\"):
        raise ColorPreflightStoreError(f"無效的範圍名稱：{value!r}")
    return name


def record_reference_margins(
    config_path: str | Path,
    report: ColorPreflightReport,
    *,
    operator: str,
    baseline_sha256: str = "",
    minimum_margin_retention: float | None = None,
) -> StationColorPreflight:
    """Write today's measured margins as the bar future shifts are judged by.

    Refuses a report that is not clean. Recording a board whose colors misread,
    went unmeasured, or already sat below their thresholds would make the fault
    the target, and every later shift would agree with it.

    A baseline that cannot be shown to be current is *not* a refusal. The
    reference is a self-comparison against whatever baseline the station runs
    today, and ``baseline_sha256`` binds it to that file, so it expires the
    moment the baseline is rebuilt. Refusing instead would leave a station
    unable to watch its own drift until an unrelated migration finished --
    which is exactly the period when drift goes unseen.
    """
    unusable = tuple(
        item.state
        for item in report.colors
        if item.state
        not in {"OK", "MARGIN_LOW", "NO_REFERENCE", "REFERENCE_STALE"}
    )
    if unusable:
        raise ColorPreflightStoreError(
            "這次讀數有色別未正確判讀("
            + "、".join(sorted(set(unusable)))
            + "),不能作為參考餘裕。請先處理光源或治具再重錄。"
        )
    margins = report.measured_margins()
    if not margins:
        raise ColorPreflightStoreError("這次讀數沒有可記錄的餘裕。")

    name = str(operator or "").strip()
    if not name:
        # The reference decides what every later shift is compared against, so
        # it carries who chose it, exactly as the calibration log requires.
        raise ColorPreflightStoreError("建立參考餘裕必須具名。")

    path = Path(config_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ColorPreflightStoreError(f"站點模型設定不存在：{path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ColorPreflightStoreError(
            f"站點模型設定無法讀取：{path}（{exc}）"
        ) from exc
    if not isinstance(payload, dict):
        raise ColorPreflightStoreError(f"站點模型設定格式錯誤：{path}")

    try:
        existing = StationColorPreflight.from_mapping(payload.get(PREFLIGHT_KEY))
    except StationColorSettingsError as exc:
        raise ColorPreflightStoreError(str(exc)) from exc

    retention = (
        existing.minimum_margin_retention
        if minimum_margin_retention is None
        else float(minimum_margin_retention)
    )
    recorded = StationColorPreflight(
        reference_margins=margins,
        minimum_margin_retention=retention,
        recorded_at=_utc_now(),
        recorded_by=name,
        baseline_sha256=str(baseline_sha256 or "").strip()
        or existing.baseline_sha256,
    )
    payload[PREFLIGHT_KEY] = recorded.to_dict()
    _write_yaml_with_backup(path, payload)
    return recorded


def _write_yaml_with_backup(path: Path, payload: dict[str, Any]) -> None:
    """Replace a station config atomically, keeping the previous copy.

    The same care the illumination calibration takes when it writes exposure
    back: a half-written station config is a station that will not start.
    """
    try:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_bytes(path.read_bytes())
        text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name,
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        os.replace(temp_path, path)
    except OSError as exc:
        raise ColorPreflightStoreError(
            f"站點模型設定寫入失敗：{path}（{exc}）"
        ) from exc
