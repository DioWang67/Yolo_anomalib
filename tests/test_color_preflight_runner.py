from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from core.color_baseline_contract import BASELINE_ALGORITHM_VERSION
from core.services.color_preflight_runner import (
    ColorPreflightSource,
    ColorPreflightUnavailable,
    latest_inspection_snapshot,
    run_color_preflight,
)
from core.services.slot_roi import ColorRoiPolicy
from core.stats_color_checker import ColorDecisionTuning

CABLE1_A = ["Red", "Green", "Orange", "Yellow", "Black", "Black"]


def _write_snapshot(
    results_root: Path,
    date: str,
    stem: str,
    *,
    items: list[dict],
    status: str = "evaluated",
    timestamp: str = "2026-09-02T07:41:00+08:00",
) -> Path:
    path = (
        results_root
        / date
        / "Cable1"
        / "A"
        / "PASS"
        / "metadata"
        / "yolo"
        / f"{stem}_config_snapshot.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "model_info": {"model_version": "1.0.6"},
                "color_result": {"status": status, "items": items},
            }
        ),
        encoding="utf-8",
    )
    return path


def _item(color: str, *, margin: float) -> dict:
    score, threshold = 0.5 + margin, 0.5
    return {
        "index": 0,
        "class_name": color,
        "class": color,
        "bbox": [0, 0, 10, 10],
        "best_color": color,
        "diff": max(0.0, 1.0 - score),
        "threshold": max(0.0, 1.0 - threshold),
        "is_ok": True,
        "measurement_is_ok": True,
    }


def _board(margin: float = 0.3) -> list[dict]:
    return [_item(color, margin=margin) for color in CABLE1_A]


def _station(tmp_path: Path, *, extra: str = "", with_baseline: bool = True) -> Path:
    station = tmp_path / "models" / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True, exist_ok=True)
    if with_baseline:
        (station / "color_stats.json").write_text(
            json.dumps(
                {
                    "summary": {
                        color: {
                            "count": 30,
                            "hsv_min": [0.0, 0.0, 0.0],
                            "hsv_max": [180.0, 255.0, 255.0],
                        }
                        for color in set(CABLE1_A)
                    },
                    "recalibration": {
                        "algorithm": BASELINE_ALGORITHM_VERSION,
                        "color_decision_tuning": ColorDecisionTuning().to_dict(),
                        "color_roi_policy": ColorRoiPolicy().to_dict(),
                    },
                }
            ),
            encoding="utf-8",
        )
    config = station / "config.yaml"
    config.write_text(
        "enable_color_check: true\n"
        "color_checker_type: stats\n"
        "color_model_path: color_stats.json\n"
        "expected_items:\n"
        "  Cable1:\n"
        "    A:\n" + "".join(f"    - {color}\n" for color in CABLE1_A) + extra,
        encoding="utf-8",
    )
    return config


def test_the_newest_snapshot_wins_across_dates(tmp_path: Path) -> None:
    """Names carry a wall-clock time but not a date, so sort by mtime.

    Sorting by name mixes yesterday's late shift with this morning's early one,
    which is exactly the reading an operator must not be handed.
    """
    results = tmp_path / "Result"
    yesterday = _write_snapshot(results, "20260901", "yolo_Cable1_A_235900", items=_board())
    time.sleep(0.01)
    today = _write_snapshot(results, "20260902", "yolo_Cable1_A_074100", items=_board())
    older = datetime.now() - timedelta(days=1)
    os.utime(yesterday, (older.timestamp(), older.timestamp()))

    assert latest_inspection_snapshot(results, "Cable1", "A", "yolo") == today


def test_another_scope_is_not_borrowed(tmp_path: Path) -> None:
    results = tmp_path / "Result"
    _write_snapshot(results, "20260902", "yolo_Cable1_A_074100", items=_board())

    assert latest_inspection_snapshot(results, "Cable2", "A", "yolo") is None
    assert latest_inspection_snapshot(results, "Cable1", "B", "yolo") is None
    assert latest_inspection_snapshot(results, "Cable1", "A", "anomalib") is None


def test_a_run_reads_the_board_the_station_declares(tmp_path: Path) -> None:
    config = _station(tmp_path)
    results = tmp_path / "Result"
    snapshot = _write_snapshot(
        results, "20260902", "yolo_Cable1_A_074100", items=_board(0.3)
    )

    report, source = run_color_preflight(
        config_path=config,
        results_root=results,
        product="Cable1",
        area="A",
        model_type="yolo",
    )

    assert source.snapshot_path == snapshot
    assert source.model_version == "1.0.6"
    # Two black positions, read as two blacks rather than one and a surplus.
    black = next(item for item in report.colors if item.color == "Black")
    assert (black.expected_count, black.observed_count) == (2, 2)
    assert report.baseline_provenance_failure == ""


def test_a_station_geometry_the_baseline_never_saw_is_surfaced(
    tmp_path: Path,
) -> None:
    """The same question the runtime asks when it loads the baseline."""
    config = _station(
        tmp_path, extra="color_roi_policy:\n  inset_x_ratio: 0.2\n"
    )
    results = tmp_path / "Result"
    _write_snapshot(results, "20260902", "yolo_Cable1_A_074100", items=_board())

    report, _source = run_color_preflight(
        config_path=config,
        results_root=results,
        product="Cable1",
        area="A",
        model_type="yolo",
    )

    assert "ROI policy" in report.baseline_provenance_failure
    # Reported, but WARN: it is a standing station condition the shift cannot
    # fix, and NG every shift would teach operators to ignore NG.
    assert report.status == "WARN"


def test_a_missing_baseline_file_is_reported_not_ignored(tmp_path: Path) -> None:
    config = _station(tmp_path, with_baseline=False)
    results = tmp_path / "Result"
    _write_snapshot(results, "20260902", "yolo_Cable1_A_074100", items=_board())

    report, _source = run_color_preflight(
        config_path=config,
        results_root=results,
        product="Cable1",
        area="A",
        model_type="yolo",
    )

    assert "找不到已部署的顏色基準" in report.baseline_provenance_failure


def test_no_saved_inspection_is_unavailable_rather_than_a_pass(
    tmp_path: Path,
) -> None:
    config = _station(tmp_path)

    with pytest.raises(ColorPreflightUnavailable, match="手動檢測"):
        run_color_preflight(
            config_path=config,
            results_root=tmp_path / "Result",
            product="Cable1",
            area="A",
            model_type="yolo",
        )


def test_an_inspection_with_no_colour_measurements_is_unavailable(
    tmp_path: Path,
) -> None:
    config = _station(tmp_path)
    results = tmp_path / "Result"
    _write_snapshot(
        results,
        "20260902",
        "yolo_Cable1_A_074100",
        items=[],
        status="no_detections",
    )

    with pytest.raises(ColorPreflightUnavailable, match="no_detections"):
        run_color_preflight(
            config_path=config,
            results_root=results,
            product="Cable1",
            area="A",
            model_type="yolo",
        )


def test_a_station_that_declares_no_colours_cannot_be_judged(
    tmp_path: Path,
) -> None:
    station = tmp_path / "models" / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    config = station / "config.yaml"
    config.write_text("color_checker_type: stats\n", encoding="utf-8")

    with pytest.raises(ColorPreflightUnavailable, match="expected_items"):
        run_color_preflight(
            config_path=config,
            results_root=tmp_path / "Result",
            product="Cable1",
            area="A",
            model_type="yolo",
        )


@pytest.mark.parametrize(
    ("delta", "expected"),
    (
        (timedelta(seconds=20), "20 秒前"),
        (timedelta(minutes=12), "12 分鐘前"),
        (timedelta(hours=5), "5 小時前"),
        (timedelta(days=9), "9 天前"),
    ),
)
def test_snapshot_age_is_shown_because_a_stale_board_is_the_real_trap(
    delta: timedelta, expected: str, tmp_path: Path
) -> None:
    """An operator who forgot to trigger an inspection judges last week's board."""
    taken = datetime.now().astimezone() - delta
    source = ColorPreflightSource(
        snapshot_path=tmp_path / "snapshot.json",
        taken_at=taken.isoformat(timespec="seconds"),
        model_version="1.0.6",
    )

    assert source.age == expected


def test_an_unparsable_timestamp_shows_no_age_rather_than_a_wrong_one(
    tmp_path: Path,
) -> None:
    source = ColorPreflightSource(
        snapshot_path=tmp_path / "snapshot.json",
        taken_at="not-a-timestamp",
        model_version="",
    )

    assert source.age == ""
