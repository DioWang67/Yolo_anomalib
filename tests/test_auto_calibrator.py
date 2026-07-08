"""Unit tests for the pure illumination auto-calibrator."""

from __future__ import annotations

import numpy as np
import pytest

from core.services.auto_calibrator import (
    AutoCalibrator,
    CalibrationPhase,
    CalibrationReason,
    CalibrationTarget,
    HardwareState,
    drive_calibration,
    measure_luma,
)


# ---------------------------------------------------------------- measure_luma
def test_measure_luma_grayscale_is_mean():
    frame = np.full((10, 10), 120, dtype=np.uint8)
    assert measure_luma(frame) == pytest.approx(120.0)


def test_measure_luma_uses_bt601_weights_on_bgr():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 2] = 100  # red channel in BGR
    # 0.299 * 100
    assert measure_luma(frame) == pytest.approx(29.9, abs=1e-3)


def test_measure_luma_roi_is_clamped_and_empty_returns_zero():
    frame = np.full((10, 10, 3), 50, dtype=np.uint8)
    frame[0:5, 0:5] = 200
    # ROI fully inside the bright quadrant
    assert measure_luma(frame, roi=(0, 0, 5, 5)) == pytest.approx(200.0)
    # Degenerate ROI -> 0
    assert measure_luma(frame, roi=(5, 5, 5, 5)) == 0.0
    # Out-of-range ROI clamps rather than raising
    assert measure_luma(frame, roi=(-5, -5, 999, 999)) > 0.0


def test_measure_luma_rejects_bad_shape():
    with pytest.raises(ValueError):
        measure_luma(np.zeros((3, 3, 4), dtype=np.uint8))


# ------------------------------------------------------------------- propose
def _state(**kw) -> HardwareState:
    base = dict(
        exposure=10000.0,
        exposure_min=100.0,
        exposure_max=100000.0,
        led_brightness=128,
        led_max=255,
        led_available=True,
    )
    base.update(kw)
    return HardwareState(**base)


def test_propose_done_when_within_tolerance():
    cal = AutoCalibrator(CalibrationTarget(target_luma=120.0, tolerance=4.0))
    p = cal.propose(122.0, _state())
    assert p.done is True
    assert p.reason == CalibrationReason.WITHIN_TOLERANCE
    assert p.phase == CalibrationPhase.DONE


def test_propose_far_error_uses_led_coarse():
    cal = AutoCalibrator(CalibrationTarget(target_luma=200.0, tolerance=4.0))
    # measured far below target (error 120 >> 4*4) -> LED coarse, brighten
    p = cal.propose(80.0, _state(led_brightness=100))
    assert p.phase == CalibrationPhase.LED_COARSE
    assert p.led_brightness > 100
    assert p.exposure == 10000.0  # exposure untouched in coarse phase


def test_propose_small_error_uses_exposure_fine():
    cal = AutoCalibrator(CalibrationTarget(target_luma=120.0, tolerance=4.0))
    # error 10 < coarse_band(16) -> exposure fine
    p = cal.propose(110.0, _state())
    assert p.phase == CalibrationPhase.EXPOSURE_FINE
    assert p.exposure > 10000.0
    assert p.led_brightness == 128  # LED untouched in fine phase


def test_propose_falls_back_to_led_when_exposure_saturated():
    cal = AutoCalibrator(CalibrationTarget(target_luma=200.0, tolerance=4.0))
    # small-ish error so we're in fine phase, but exposure already at max
    st = _state(exposure=100000.0, exposure_max=100000.0, led_brightness=100)
    p = cal.propose(190.0, st)
    assert p.phase == CalibrationPhase.LED_COARSE
    assert p.led_brightness > 100


def test_propose_exhausted_when_no_actuator_can_move():
    cal = AutoCalibrator(CalibrationTarget(target_luma=250.0, tolerance=4.0))
    st = _state(
        exposure=100000.0,
        exposure_max=100000.0,
        led_brightness=255,
        led_max=255,
    )
    p = cal.propose(150.0, st)
    assert p.done is True
    assert p.reason == CalibrationReason.EXHAUSTED


def test_propose_no_led_uses_exposure_even_when_far():
    cal = AutoCalibrator(CalibrationTarget(target_luma=200.0, tolerance=4.0))
    st = _state(led_available=False)
    p = cal.propose(80.0, st)
    assert p.phase == CalibrationPhase.EXPOSURE_FINE


def test_propose_near_black_pushes_led_to_max():
    cal = AutoCalibrator(CalibrationTarget(target_luma=120.0, tolerance=4.0))
    p = cal.propose(0.0, _state(led_brightness=10))
    assert p.led_brightness == 255


def test_propose_guarantees_movement_on_tiny_ratio():
    # ratio ~1 but outside tolerance -> must still nudge, not stall
    cal = AutoCalibrator(CalibrationTarget(target_luma=100.0, tolerance=1.0))
    st = _state(exposure=10000.0, led_brightness=128)
    p = cal.propose(97.0, st)  # error 3 -> fine phase, ratio ~1.03
    assert p.exposure != 10000.0


# --------------------------------------------------------------- drive loop
class _SimRig:
    """Linear-response camera: luma = k_exp * exposure + k_led * led, clipped."""

    def __init__(self, exposure=5000.0, led=100, k_exp=0.008, k_led=0.4):
        self.exposure = exposure
        self.led = led
        self.k_exp = k_exp
        self.k_led = k_led
        self.exposure_min = 100.0
        self.exposure_max = 100000.0
        self.led_max = 255

    def luma(self) -> float:
        return float(min(255.0, self.k_exp * self.exposure + self.k_led * self.led))

    def state(self) -> HardwareState:
        return HardwareState(
            exposure=self.exposure,
            exposure_min=self.exposure_min,
            exposure_max=self.exposure_max,
            led_brightness=self.led,
            led_max=self.led_max,
            led_available=True,
        )

    def apply(self, exposure: float, led: int) -> None:
        self.exposure = float(exposure)
        self.led = int(led)


def test_drive_calibration_converges_from_dark():
    rig = _SimRig(exposure=3000.0, led=60)  # starts dark (~48 luma)
    target = CalibrationTarget(target_luma=140.0, tolerance=3.0)
    cal = AutoCalibrator(target)

    outcome = drive_calibration(
        cal,
        measure_fn=rig.luma,
        read_state_fn=rig.state,
        apply_fn=rig.apply,
        max_iterations=30,
    )

    assert outcome.success is True
    assert outcome.reason == CalibrationReason.WITHIN_TOLERANCE
    assert abs(outcome.final_luma - 140.0) <= 3.0


def test_drive_calibration_converges_from_bright():
    rig = _SimRig(exposure=90000.0, led=250)  # saturated bright
    target = CalibrationTarget(target_luma=120.0, tolerance=3.0)
    cal = AutoCalibrator(target)

    outcome = drive_calibration(
        cal,
        measure_fn=rig.luma,
        read_state_fn=rig.state,
        apply_fn=rig.apply,
        max_iterations=40,
    )

    assert outcome.success is True
    assert abs(outcome.final_luma - 120.0) <= 3.0


def test_drive_calibration_respects_bounds_and_reports_failure():
    # Target unreachable: max luma achievable < target
    rig = _SimRig(exposure=1000.0, led=100, k_exp=0.0001, k_led=0.05)
    target = CalibrationTarget(target_luma=200.0, tolerance=2.0)
    cal = AutoCalibrator(target)

    outcome = drive_calibration(
        cal,
        measure_fn=rig.luma,
        read_state_fn=rig.state,
        apply_fn=rig.apply,
        max_iterations=25,
    )

    assert outcome.success is False
    assert outcome.final_exposure <= rig.exposure_max
    assert outcome.final_led_brightness <= rig.led_max


def test_drive_calibration_already_on_target_is_zero_iterations():
    rig = _SimRig()
    on_target = rig.luma()
    target = CalibrationTarget(target_luma=on_target, tolerance=2.0)
    cal = AutoCalibrator(target)

    outcome = drive_calibration(
        cal,
        measure_fn=rig.luma,
        read_state_fn=rig.state,
        apply_fn=rig.apply,
        max_iterations=10,
    )

    assert outcome.success is True
    assert outcome.iterations == 0
