"""Tests for CalibrationSession: the camera/LED glue around AutoCalibrator."""

from __future__ import annotations

import numpy as np
import pytest

from core.services.auto_calibrator import CalibrationReason, CalibrationTarget
from core.services.calibration_session import (
    CalibrationSession,
    percent_to_value,
    value_to_percent,
)


class FakeCamera:
    """Linear-response camera: luma = k_exp*exposure + k_led*led (via light)."""

    def __init__(self, light=None, exposure=5000.0, k_exp=0.01, k_led=0.4):
        self.exposure = exposure
        self.gain = 10.0
        self.k_exp = k_exp
        self.k_led = k_led
        self.exposure_min = 100.0
        self.exposure_max = 100000.0
        self._light = light

    def _luma(self) -> float:
        led = self._light.last_brightness if self._light else 0
        return float(min(255.0, self.k_exp * self.exposure + self.k_led * led))

    def capture_frame(self):
        return np.full((8, 8, 3), int(self._luma()), dtype=np.uint8)

    def get_exposure(self):
        return self.exposure

    def get_exposure_range(self):
        return {"current": self.exposure, "min": self.exposure_min, "max": self.exposure_max}

    def set_exposure(self, exposure_time):
        self.exposure = float(exposure_time)
        return True

    def get_gain(self):
        return self.gain


class FakeLight:
    def __init__(self, is_open=True, max_value=255, value=100):
        self.is_open = is_open
        self._max = max_value
        self._value = value

    @property
    def max_value(self):
        return self._max

    @property
    def last_brightness(self):
        return self._value

    def set_brightness(self, value):
        self._value = int(value)


def test_percent_value_roundtrip():
    assert value_to_percent(255, 255) == 100
    assert value_to_percent(0, 255) == 0
    assert percent_to_value(50, 200) == 100
    assert value_to_percent(128, 255) == 50


def test_measure_returns_none_on_capture_failure():
    class DeadCam(FakeCamera):
        def capture_frame(self):
            return None

    session = CalibrationSession(DeadCam(), None, sleep_fn=lambda _s: None)
    assert session.measure() is None


def test_capture_and_measure_returns_frame_and_luma_in_one_capture():
    light = FakeLight(value=204)
    cam = FakeCamera(light=light, exposure=51170.0)
    session = CalibrationSession(cam, light, sleep_fn=lambda _s: None)

    frame, luma = session.capture_and_measure()

    assert frame is not None
    assert frame.shape == (8, 8, 3)
    assert luma == pytest.approx(cam._luma(), abs=0.5)


def test_capture_and_measure_returns_none_pair_on_capture_failure():
    class DeadCam(FakeCamera):
        def capture_frame(self):
            return None

    session = CalibrationSession(DeadCam(), None, sleep_fn=lambda _s: None)
    assert session.capture_and_measure() == (None, None)


def test_record_current_captures_hardware_and_target():
    light = FakeLight(value=204)  # 80%
    cam = FakeCamera(light=light, exposure=51170.0)
    session = CalibrationSession(cam, light, sleep_fn=lambda _s: None)

    snap = session.record_current()

    assert snap["exposure_time"] == 51170.0
    assert snap["gain"] == 10.0
    assert snap["light_brightness"] == 80
    # target luma equals the current measured luma
    assert snap["target_luma"] == pytest.approx(cam._luma(), abs=0.5)


def test_record_current_raises_without_camera_signal():
    class DeadCam(FakeCamera):
        def capture_frame(self):
            return None

    session = CalibrationSession(DeadCam(), None, sleep_fn=lambda _s: None)
    with pytest.raises(RuntimeError):
        session.record_current()


def test_run_auto_converges_and_applies_to_hardware():
    light = FakeLight(value=80)
    cam = FakeCamera(light=light, exposure=3000.0)  # starts dark
    session = CalibrationSession(cam, light, sleep_fn=lambda _s: None)
    target = CalibrationTarget(target_luma=150.0, tolerance=3.0)

    outcome = session.run_auto(target, max_iterations=40)

    assert outcome.success is True
    assert outcome.reason == CalibrationReason.WITHIN_TOLERANCE
    assert abs(cam._luma() - 150.0) <= 3.0
    # Exposure must stay within device bounds
    assert cam.exposure_min <= cam.exposure <= cam.exposure_max


def test_run_auto_reports_progress_with_shrinking_error():
    light = FakeLight(value=80)
    cam = FakeCamera(light=light, exposure=3000.0)  # starts dark
    session = CalibrationSession(cam, light, sleep_fn=lambda _s: None)
    target = CalibrationTarget(target_luma=150.0, tolerance=3.0)

    steps: list[tuple[int, str, float, float]] = []
    session.run_auto(
        target,
        max_iterations=40,
        on_step=lambda i, phase, luma, err: steps.append((i, phase.value, luma, err)),
    )

    assert steps, "expected at least one progress step"
    assert [s[0] for s in steps] == list(range(1, len(steps) + 1))  # 1-based, contiguous
    assert steps[-1][3] < steps[0][3]  # error shrinks overall


def test_run_auto_without_light_uses_exposure_only():
    cam = FakeCamera(light=None, exposure=4000.0, k_led=0.0)
    session = CalibrationSession(cam, None, sleep_fn=lambda _s: None)
    target = CalibrationTarget(target_luma=120.0, tolerance=3.0)

    outcome = session.run_auto(target, max_iterations=40)

    assert outcome.success is True
    assert abs(cam._luma() - 120.0) <= 3.0


def test_read_state_raises_when_exposure_unreadable():
    class NoExp(FakeCamera):
        def get_exposure(self):
            return None

    session = CalibrationSession(NoExp(), None, sleep_fn=lambda _s: None)
    with pytest.raises(RuntimeError):
        session.read_state()


def test_snapshot_after_run_reads_converged_values():
    light = FakeLight(value=128)
    cam = FakeCamera(light=light, exposure=7777.0)
    session = CalibrationSession(cam, light, sleep_fn=lambda _s: None)

    snap = session.snapshot_after_run()

    assert snap["exposure_time"] == 7777.0
    assert snap["gain"] == 10.0
    assert snap["light_brightness"] == 50
    assert "target_luma" not in snap
