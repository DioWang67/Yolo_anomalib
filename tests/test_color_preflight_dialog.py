from __future__ import annotations

import os
from pathlib import Path

import pytest
from PyQt5.QtWidgets import QApplication

from app.gui.color_preflight_dialog import ColorPreflightDialog
from app.gui.i18n import tr
from core.models import ColorCheckItemResult
from core.services.color_preflight import evaluate_color_preflight
from core.services.color_preflight_runner import (
    ColorPreflightSource,
    ColorPreflightUnavailable,
)
from core.services.color_preflight_store import (
    ColorPreflightLedger,
    ColorPreflightStoreError,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

CABLE1_A = ("Red", "Green", "Orange", "Yellow", "Black", "Black")


def _rows(dialog) -> list:
    """The visual rows currently on the panel, excluding the trailing stretch."""
    from app.gui.color_gamut_view import ColorGamutRow

    return [
        widget
        for index in range(dialog._rows_layout.count())
        if isinstance(
            widget := dialog._rows_layout.itemAt(index).widget(), ColorGamutRow
        )
    ]


def _row_count(dialog) -> int:
    return len(_rows(dialog))


@pytest.fixture(scope="module")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture(autouse=True)
def _no_modals(monkeypatch: pytest.MonkeyPatch):
    """Keep the dialog's notifications from opening a real modal.

    An offscreen QMessageBox waiting for a click takes the whole process down,
    so the seam has to be closed for every test rather than remembered in the
    ones that happen to trigger it.
    """
    for name in ("information", "warning"):
        monkeypatch.setattr(
            f"app.gui.color_preflight_dialog.QMessageBox.{name}",
            lambda *args, **kwargs: None,
        )


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


def _source(tmp_path: Path) -> ColorPreflightSource:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    return ColorPreflightSource(
        snapshot_path=snapshot,
        taken_at="2026-09-02T07:41:00+08:00",
        model_version="1.0.6",
    )


def _dialog(
    tmp_path: Path,
    report,
    *,
    record_fn=None,
    operator_prompt_fn=None,
    ledger=None,
) -> ColorPreflightDialog:
    def run_fn(**_kwargs):
        return report, _source(tmp_path)

    return ColorPreflightDialog(
        config_path=tmp_path / "config.yaml",
        results_root=tmp_path / "Result",
        product="Cable1",
        area="A",
        model_type="yolo",
        ledger=ledger or ColorPreflightLedger(tmp_path / ".color_preflight"),
        run_fn=run_fn,
        record_fn=record_fn or (lambda *a, **k: None),
        operator_prompt_fn=operator_prompt_fn or (lambda: ("tester", True)),
        language="zh",
    )


def _report(**margins: float):
    items = [_item(color, margin=margins[color.casefold()]) for color in CABLE1_A]
    return evaluate_color_preflight(
        items,
        CABLE1_A,
        {"Red": 0.57, "Green": 0.49, "Orange": 0.45, "Yellow": 0.40, "Black": 0.126},
    )


def test_a_clean_board_shows_every_colour_and_offers_to_be_recorded(
    qapp, tmp_path: Path
) -> None:
    report = _report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )

    dialog = _dialog(tmp_path, report)

    assert _row_count(dialog) == 5
    assert dialog._verdict_label.text() == tr("zh", "preflight_verdict_ok")
    assert dialog._record_btn.isEnabled() is True
    # The remedy note is guidance for a failure, so it stays out of the way.
    # isHidden, not isVisible: an unshown dialog makes every child invisible.
    assert dialog._remedy_label.isHidden() is True
    dialog.deleteLater()


def test_an_eroded_margin_is_shown_before_anything_misjudges(
    qapp, tmp_path: Path
) -> None:
    report = _report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.028
    )

    dialog = _dialog(tmp_path, report)

    assert dialog._verdict_label.text() == tr("zh", "preflight_verdict_warn")
    black = next(
        row for row in _rows(dialog) if row._reading.color == "Black"
    )
    # The state reads as words, not as the identifier the ledger records.
    assert black._state_text == tr("zh", "preflight_state_margin_low")
    assert black._reading.retention == pytest.approx(0.22, abs=0.01)
    # A row that needs attention is washed and emphasised, so it is spotted
    # rather than read: the row knows it is not OK, and paints accordingly.
    assert black._reading.is_ok is False
    assert all(row._reading.is_ok for row in _rows(dialog) if row is not black)
    dialog.deleteLater()


def test_a_board_that_cannot_become_a_reference_does_not_offer_to(
    qapp, tmp_path: Path
) -> None:
    """The store would refuse it, so the button must not invite the refusal."""
    items = [_item(color, margin=0.3) for color in CABLE1_A]
    items[2] = _item("Red", margin=0.4)
    report = evaluate_color_preflight(items, CABLE1_A, None)

    dialog = _dialog(tmp_path, report)

    assert dialog._verdict_label.text() == tr("zh", "preflight_verdict_ng")
    assert dialog._record_btn.isEnabled() is False
    assert dialog._remedy_label.isHidden() is False
    dialog.deleteLater()


def test_an_unapproved_baseline_is_reported_but_still_recordable(
    qapp, tmp_path: Path
) -> None:
    """The board read correctly, so the station may still record its drift bar.

    Blocking here was the wrong call: it left every station unable to watch its
    own drift until the baseline migration finished. The reference is bound to
    this baseline's hash instead, so it expires when the baseline is rebuilt.
    """
    items = [_item(color, margin=0.3) for color in CABLE1_A]
    report = evaluate_color_preflight(
        items,
        CABLE1_A,
        None,
        baseline_provenance_failure="未記錄重建演算法",
    )

    dialog = _dialog(tmp_path, report)

    assert dialog._record_btn.isEnabled() is True
    assert "未記錄重建演算法" in dialog._notice_label.toolTip()
    assert dialog._notice_label.isHidden() is False
    # The actionable step is to record a reference, and the advice also says
    # the baseline note does not stand in the way of doing so.
    # The lead sentence is on screen and the full wording is on hover, so the
    # panel stays readable without losing the reasoning.
    assert dialog._remedy_label.text().endswith("。")
    assert len(dialog._remedy_label.text()) < 40
    hover = dialog._remedy_label.toolTip()
    assert tr("zh", "preflight_remedy_record") in hover
    assert tr("zh", "preflight_remedy_baseline") in hover
    # Not a word about the lighting, which is fine.
    assert tr("zh", "preflight_remedy_misread") not in hover
    dialog.deleteLater()


def test_a_stale_reference_says_so_rather_than_comparing_anyway(
    qapp, tmp_path: Path
) -> None:
    report = evaluate_color_preflight(
        [_item(color, margin=0.3) for color in CABLE1_A],
        CABLE1_A,
        {"Red": 0.5, "Green": 0.5, "Orange": 0.5, "Yellow": 0.5, "Black": 0.5},
        reference_baseline_id="a" * 64,
        current_baseline_id="b" * 64,
    )

    dialog = _dialog(tmp_path, report)

    assert dialog._verdict_label.text() == tr("zh", "preflight_verdict_warn")
    assert tr("zh", "preflight_reference_stale") in dialog._notice_label.toolTip()
    assert tr("zh", "preflight_remedy_stale") in dialog._remedy_label.toolTip()
    assert dialog._record_btn.isEnabled() is True
    dialog.deleteLater()


def test_recording_is_named_and_can_be_cancelled(qapp, tmp_path: Path) -> None:
    report = _report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )
    calls: list[str] = []

    def record_fn(_config, _report, *, operator, baseline_sha256=""):
        calls.append(operator)
        return None

    dialog = _dialog(
        tmp_path,
        report,
        record_fn=record_fn,
        operator_prompt_fn=lambda: ("", False),
    )
    dialog._on_record()

    assert calls == []

    dialog._operator_prompt_fn = lambda: ("line-lead-a", True)
    dialog._on_record()

    assert calls == ["line-lead-a"]
    dialog.deleteLater()


def test_nothing_to_check_is_reported_rather_than_shown_as_a_pass(
    qapp, tmp_path: Path
) -> None:
    """A station nobody measured is not a station that passed."""

    def run_fn(**_kwargs):
        raise ColorPreflightUnavailable("找不到這個工位的檢測快照。")

    dialog = ColorPreflightDialog(
        config_path=tmp_path / "config.yaml",
        results_root=tmp_path / "Result",
        product="Cable1",
        area="A",
        model_type="yolo",
        ledger=ColorPreflightLedger(tmp_path / ".color_preflight"),
        run_fn=run_fn,
        language="zh",
    )

    assert _row_count(dialog) == 0
    assert dialog._record_btn.isEnabled() is False
    assert "找不到" in dialog._verdict_label.text()
    dialog.deleteLater()


def test_every_run_lands_in_the_ledger(qapp, tmp_path: Path) -> None:
    ledger = ColorPreflightLedger(tmp_path / ".color_preflight")
    report = _report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.028
    )

    dialog = _dialog(tmp_path, report, ledger=ledger)
    dialog.refresh()

    records = ledger.read("Cable1", "A", "yolo")
    # Once on construction, once on the explicit re-read.
    assert len(records) == 2
    assert records[0]["status"] == "WARN"
    dialog.deleteLater()


def test_a_failed_ledger_write_still_shows_the_verdict(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The operator needs the reading on screen more than the log line."""

    def explode(_self, _record):
        raise ColorPreflightStoreError("disk full")

    monkeypatch.setattr(ColorPreflightLedger, "append", explode)
    report = _report(
        red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12
    )

    dialog = _dialog(tmp_path, report)

    assert _row_count(dialog) == 5
    assert dialog._verdict_label.text() == tr("zh", "preflight_verdict_ok")
    dialog.deleteLater()
