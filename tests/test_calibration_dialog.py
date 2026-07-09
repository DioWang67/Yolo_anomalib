"""Tests for the calibration dialog's frame->pixmap preview helper."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5.QtWidgets", reason="PyQt5 is required for calibration dialog tests")

from pathlib import Path

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication

import app.gui.calibration_dialog as calibration_dialog_module
from app.gui.calibration_dialog import CalibrationDialog, _frame_to_pixmap
from core.services.auto_calibrator import CalibrationOutcome, CalibrationReason


@pytest.fixture()
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


class FakeSession:
    """Session stub for driving the dialog's auto-calibrate result handlers."""

    def __init__(self) -> None:
        self.snapshot = {
            "exposure_time": 5000.0,
            "gain": 5.0,
            "light_brightness": 40,
        }

    def capture_and_measure(self, roi=None):
        return None, None

    def snapshot_after_run(self) -> dict:
        return dict(self.snapshot)


class SpyMessageBox:
    """Record which QMessageBox variant fired, without showing a modal."""

    calls: list[str] = []

    @staticmethod
    def information(parent, title, text) -> None:
        SpyMessageBox.calls.append("information")

    @staticmethod
    def warning(parent, title, text) -> None:
        SpyMessageBox.calls.append("warning")

    @staticmethod
    def critical(parent, title, text) -> None:
        SpyMessageBox.calls.append("critical")


def _make_dialog(app: QApplication, monkeypatch, tmp_path: Path) -> CalibrationDialog:
    SpyMessageBox.calls = []
    monkeypatch.setattr(calibration_dialog_module, "QMessageBox", SpyMessageBox)
    return CalibrationDialog(
        config_path=tmp_path / "config.yaml",
        session=FakeSession(),
        save_fn=lambda path, **kw: type("R", (), {"backup_path": "b.yaml"})(),
        language="en",
    )


def _outcome(success: bool) -> CalibrationOutcome:
    return CalibrationOutcome(
        success=success,
        reason=CalibrationReason.WITHIN_TOLERANCE if success else CalibrationReason.MAX_ITERATIONS,
        iterations=8,
        final_luma=128.0,
        final_exposure=5000.0,
        final_led_brightness=100,
    )


def test_success_shows_information_box(app, monkeypatch, tmp_path) -> None:
    dialog = _make_dialog(app, monkeypatch, tmp_path)
    dialog._on_auto_finished(_outcome(success=True))
    assert SpyMessageBox.calls == ["information"]


def test_non_convergence_shows_warning_box(app, monkeypatch, tmp_path) -> None:
    dialog = _make_dialog(app, monkeypatch, tmp_path)
    dialog._on_auto_finished(_outcome(success=False))
    assert SpyMessageBox.calls == ["warning"]


def test_worker_failure_shows_critical_box(app, monkeypatch, tmp_path) -> None:
    dialog = _make_dialog(app, monkeypatch, tmp_path)
    dialog._on_auto_failed("camera vanished")
    assert SpyMessageBox.calls == ["critical"]


def test_frame_to_pixmap_scales_bgr_frame_within_target(app: QApplication) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    pixmap = _frame_to_pixmap(frame, QSize(320, 240))

    assert pixmap is not None
    assert not pixmap.isNull()
    # KeepAspectRatio: never exceeds the target box.
    assert pixmap.width() <= 320
    assert pixmap.height() <= 240


def test_frame_to_pixmap_handles_grayscale(app: QApplication) -> None:
    frame = np.zeros((100, 100), dtype=np.uint8)

    pixmap = _frame_to_pixmap(frame, QSize(320, 240))

    assert pixmap is not None
    assert not pixmap.isNull()


def test_frame_to_pixmap_returns_none_for_empty_or_missing(app: QApplication) -> None:
    assert _frame_to_pixmap(None, QSize(320, 240)) is None
    assert _frame_to_pixmap(np.empty((0, 0, 3), dtype=np.uint8), QSize(320, 240)) is None
