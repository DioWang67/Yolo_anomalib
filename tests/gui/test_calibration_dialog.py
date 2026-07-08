"""Headless tests for the calibration dialog wiring (no camera, no threads)."""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for GUI tests")
pytestmark = pytest.mark.gui

from app.gui.calibration_dialog import CalibrationDialog
from core.services.auto_calibrator import CalibrationOutcome, CalibrationReason


class FakeSession:
    """Stand-in for CalibrationSession with scripted return values."""

    def __init__(self):
        self.luma = 100.0
        self.recorded = {
            "exposure_time": 51170.0,
            "gain": 23.0,
            "light_brightness": 80,
            "target_luma": 137.5,
        }
        self.after_run = {
            "exposure_time": 42000.0,
            "gain": 23.0,
            "light_brightness": 75,
        }

    def measure(self, roi=None):
        return self.luma

    def record_current(self, roi=None):
        return dict(self.recorded)

    def snapshot_after_run(self):
        return dict(self.after_run)


class SaveSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, config_path, **kwargs):
        self.calls.append((config_path, kwargs))
        return type("R", (), {"backup_path": "cfg.yaml.bak"})()


@pytest.fixture
def dialog(qtbot, tmp_path):
    session = FakeSession()
    save_spy = SaveSpy()
    dlg = CalibrationDialog(
        config_path=tmp_path / "config.yaml",
        session=session,
        save_fn=save_spy,
        initial_target=120.0,
        initial_tolerance=5.0,
        language="en",
    )
    qtbot.addWidget(dlg)
    dlg._timer.stop()  # deterministic: no background luma polling in tests
    return dlg, session, save_spy


def test_live_luma_is_displayed(dialog):
    dlg, session, _ = dialog
    session.luma = 88.0
    dlg._refresh_luma()
    assert dlg._luma_label.text() == "88.0"


def test_initial_target_and_tolerance_prefilled(dialog):
    dlg, _, _ = dialog
    assert dlg._target_spin.value() == pytest.approx(120.0)
    assert dlg._tolerance_spin.value() == pytest.approx(5.0)


def test_record_saves_hardware_and_updates_target(dialog):
    dlg, session, save_spy = dialog
    dlg._on_record()

    assert dlg._target_spin.value() == pytest.approx(137.5)
    assert len(save_spy.calls) == 1
    _, kwargs = save_spy.calls[0]
    assert kwargs["exposure_time"] == 51170.0
    assert kwargs["gain"] == 23.0
    assert kwargs["light_brightness"] == 80
    assert kwargs["target_luma"] == 137.5
    assert kwargs["tolerance"] == pytest.approx(5.0)


def test_auto_success_saves_converged_values(dialog):
    dlg, session, save_spy = dialog
    outcome = CalibrationOutcome(
        success=True,
        reason=CalibrationReason.WITHIN_TOLERANCE,
        iterations=6,
        final_luma=121.0,
        final_exposure=42000.0,
        final_led_brightness=191,
    )
    dlg._on_auto_finished(outcome)

    assert len(save_spy.calls) == 1
    _, kwargs = save_spy.calls[0]
    assert kwargs["exposure_time"] == 42000.0
    assert kwargs["light_brightness"] == 75
    # target unchanged by a run (keeps the operator-set target)
    assert kwargs["target_luma"] == pytest.approx(120.0)


def test_auto_failure_saves_nothing(dialog):
    dlg, _, save_spy = dialog
    outcome = CalibrationOutcome(
        success=False,
        reason=CalibrationReason.EXHAUSTED,
        iterations=20,
        final_luma=90.0,
        final_exposure=100000.0,
        final_led_brightness=255,
    )
    dlg._on_auto_finished(outcome)

    assert save_spy.calls == []
    assert "exhausted" in dlg._status_label.text().lower()
