from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.models import ColorCheckItemResult
from core.services.color_preflight import evaluate_color_preflight
from core.services.color_preflight_store import (
    ColorPreflightLedger,
    ColorPreflightRecord,
    ColorPreflightStoreError,
    record_reference_margins,
)
from core.services.station_color_settings import station_color_preflight

CABLE1_A = ("Red", "Green", "Orange", "Yellow", "Black", "Black")


def _item(best_color: str, *, margin: float) -> ColorCheckItemResult:
    score, threshold = 0.5 + margin, 0.5
    return ColorCheckItemResult(
        index=0,
        class_name=best_color,
        bbox=[0, 0, 10, 10],
        best_color=best_color,
        diff=max(0.0, 1.0 - score),
        threshold=max(0.0, 1.0 - threshold),
        is_ok=margin >= 0,
        measurement_is_ok=margin >= 0,
    )


def _clean_report(**margins: float):
    items = [_item(color, margin=margins[color.casefold()]) for color in CABLE1_A]
    return evaluate_color_preflight(items, CABLE1_A, None)


def _station_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "weights: models/Cable1/A/yolo/weights/best.onnx\n"
        "enable_color_check: true\n"
        "color_checker_type: stats\n" + extra,
        encoding="utf-8",
    )
    return path


def test_recording_a_reference_is_named_and_reversible(tmp_path: Path) -> None:
    config_path = _station_config(tmp_path)
    report = _clean_report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )

    recorded = record_reference_margins(
        config_path, report, operator="line-lead-a"
    )

    assert recorded.recorded_by == "line-lead-a"
    assert recorded.reference_margins["Black"] == pytest.approx(0.12)
    # Readable back through the same accessor the check itself uses.
    stored = station_color_preflight(config_path)
    assert stored.reference_margins == recorded.reference_margins
    assert stored.recorded_by == "line-lead-a"
    # The previous config is kept, because a station config that fails to write
    # is a station that will not start.
    assert (tmp_path / "config.yaml.bak").is_file()


def test_recording_preserves_the_rest_of_the_station_config(tmp_path: Path) -> None:
    config_path = _station_config(
        tmp_path, "color_roi_policy:\n  inset_x_ratio: 0.2\n"
    )
    report = _clean_report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )

    record_reference_margins(config_path, report, operator="line-lead-a")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert payload["color_roi_policy"] == {"inset_x_ratio": 0.2}
    assert payload["color_checker_type"] == "stats"


def test_an_unnamed_reference_is_refused(tmp_path: Path) -> None:
    """The reference decides what every later shift is judged against."""
    config_path = _station_config(tmp_path)
    report = _clean_report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )

    with pytest.raises(ColorPreflightStoreError, match="具名"):
        record_reference_margins(config_path, report, operator="   ")


def test_a_board_that_misread_cannot_become_the_reference(tmp_path: Path) -> None:
    """Recording a faulty board makes the fault the target for every shift after."""
    config_path = _station_config(tmp_path)
    items = [_item(color, margin=0.3) for color in CABLE1_A]
    items[2] = _item("Red", margin=0.4)  # the Orange position reads Red
    report = evaluate_color_preflight(items, CABLE1_A, None)

    with pytest.raises(ColorPreflightStoreError, match="未正確判讀"):
        record_reference_margins(config_path, report, operator="line-lead-a")

    assert "color_preflight" not in yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )


def test_a_reference_is_bound_to_the_baseline_it_was_measured_on(
    tmp_path: Path,
) -> None:
    """Binding, not refusing.

    A baseline that predates the current contract used to block recording
    outright, which left every station unable to watch its own drift until an
    unrelated migration finished -- exactly the period when drift goes unseen.
    The reference now records the baseline's hash, so it expires the moment the
    baseline is rebuilt.
    """
    config_path = _station_config(tmp_path)
    items = [_item(color, margin=0.3) for color in CABLE1_A]
    report = evaluate_color_preflight(
        items,
        CABLE1_A,
        None,
        baseline_provenance_failure="未記錄重建演算法",
    )

    recorded = record_reference_margins(
        config_path, report, operator="line-lead-a", baseline_sha256="a" * 64
    )

    assert recorded.baseline_sha256 == "a" * 64
    assert station_color_preflight(config_path).baseline_sha256 == "a" * 64


def test_a_recorded_reference_survives_being_read_as_settings(tmp_path: Path) -> None:
    """Round-trip through YAML, since that is how the next shift sees it."""
    config_path = _station_config(tmp_path)
    report = _clean_report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )
    record_reference_margins(
        config_path, report, operator="line-lead-a", minimum_margin_retention=0.5
    )

    stored = station_color_preflight(config_path)

    assert stored.minimum_margin_retention == pytest.approx(0.5)
    assert stored.has_reference is True
    # And it drives the next evaluation without further translation.
    next_shift = evaluate_color_preflight(
        [_item(color, margin=0.05) for color in CABLE1_A],
        CABLE1_A,
        stored.reference_margins,
        minimum_margin_retention=stored.minimum_margin_retention,
    )
    assert next_shift.status == "WARN"


def test_the_ledger_appends_and_reads_back_oldest_first(tmp_path: Path) -> None:
    ledger = ColorPreflightLedger(tmp_path / ".color_preflight")
    report = _clean_report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )
    for index in range(3):
        ledger.append(
            ColorPreflightRecord(
                recorded_at=f"2026-09-0{index + 1}T07:40:00+08:00",
                operator="line-lead-a",
                product="Cable1",
                area="A",
                model_type="yolo",
                model_version="1.0.6",
                report=report,
            )
        )

    records = ledger.read("Cable1", "A", "yolo")

    assert [item["recorded_at"] for item in records] == [
        "2026-09-01T07:40:00+08:00",
        "2026-09-02T07:40:00+08:00",
        "2026-09-03T07:40:00+08:00",
    ]
    assert records[0]["colors"][0]["state"] == "NO_REFERENCE"
    assert ledger.read("Cable1", "A", "yolo", limit=1)[0]["recorded_at"].endswith(
        "03T07:40:00+08:00"
    )


def test_a_truncated_line_costs_one_run_not_the_whole_history(
    tmp_path: Path,
) -> None:
    """A power cut mid-write must not erase a fortnight of trend."""
    ledger = ColorPreflightLedger(tmp_path / ".color_preflight")
    report = _clean_report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )
    ledger.append(
        ColorPreflightRecord(
            recorded_at="2026-09-01T07:40:00+08:00",
            operator="line-lead-a",
            product="Cable1",
            area="A",
            model_type="yolo",
            model_version="1.0.6",
            report=report,
        )
    )
    path = ledger.path_for("Cable1", "A", "yolo")
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"recorded_at": "2026-09-02T07\n')

    records = ledger.read("Cable1", "A", "yolo")

    assert len(records) == 1
    assert records[0]["recorded_at"] == "2026-09-01T07:40:00+08:00"


def test_a_scope_name_that_would_escape_its_directory_is_refused(
    tmp_path: Path,
) -> None:
    ledger = ColorPreflightLedger(tmp_path / ".color_preflight")

    with pytest.raises(ColorPreflightStoreError):
        ledger.path_for("../elsewhere", "A", "yolo")
