"""Illumination auto-calibration: drive image brightness to a recorded target.

Scope and strategy
------------------
The color checker runs open-loop against fixed HSV/LAB ranges, so ambient light
drift silently shifts measured colors. This module closes the loop on the one
axis that matters most and is cheapest to control: overall image brightness
(luma). A recorded ``CalibrationTarget`` says "this product/area/type should
read this mean luma"; the calibrator nudges hardware until a live frame matches
within tolerance.

Two-phase control (LED coarse, exposure fine), per operator decision:
  * LED brightness is a coarse 0..255 actuator — use it to get luma into the
    ballpark quickly.
  * Exposure time is a fine, near-continuous actuator — use it to settle onto
    the target within tolerance.

The decision logic (``AutoCalibrator.propose``) is pure and hardware-free so it
is fully unit-testable. ``drive_calibration`` runs the loop against injected
capture/apply callables (DIP) — the GUI supplies real camera/light functions,
tests supply a simulated linear-response rig.

Concurrency: single-threaded by contract. The caller (GUI worker thread) must
guarantee no inspection runs concurrently; this module holds no shared state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np

# BT.601 luma weights in OpenCV BGR channel order. Only self-consistency
# matters for a brightness target, but we fix the formula for reproducibility.
_LUMA_BGR = np.array([0.114, 0.587, 0.299], dtype=np.float32)

# Proportional-step damping: new = current * (1 + damping * (ratio - 1)).
# < 1.0 avoids overshoot on the (roughly) linear luma response.
_DEFAULT_DAMPING = 0.8
# Error beyond this multiple of tolerance is "far" -> LED coarse phase.
_DEFAULT_COARSE_FACTOR = 4.0
# Minimum relative exposure move to count as progress (avoid stalling).
_MIN_EXPOSURE_STEP_RATIO = 0.02


def measure_luma(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    """Return mean luma (0..255) of a frame or an ROI within it.

    Args:
        frame: ``HxWx3`` uint8 BGR image, or ``HxW`` grayscale.
        roi: Optional ``(x1, y1, x2, y2)`` pixel box; clamped to the frame.
            ``None`` measures the whole frame.

    Returns:
        Mean luma as a float in ``[0, 255]``; ``0.0`` for an empty region.

    Raises:
        ValueError: If ``frame`` is not a 2D or 3-channel image.
    """
    if frame is None or frame.ndim not in (2, 3):
        raise ValueError("frame must be a 2D grayscale or 3-channel image")

    region = frame
    if roi is not None:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = roi
        x1 = max(0, min(int(x1), w))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        region = frame[y1:y2, x1:x2]

    if region.size == 0:
        return 0.0

    if region.ndim == 2:
        return float(np.mean(region))
    if region.shape[2] != 3:
        raise ValueError("color frame must have exactly 3 channels")
    return float(np.mean(region.astype(np.float32) @ _LUMA_BGR))


class CalibrationPhase(str, Enum):
    """Which actuator the calibrator is currently moving."""

    LED_COARSE = "led_coarse"
    EXPOSURE_FINE = "exposure_fine"
    DONE = "done"


class CalibrationReason(str, Enum):
    """Machine-readable outcome of a proposal or a completed run."""

    WITHIN_TOLERANCE = "within_tolerance"
    ADJUST = "adjust"
    EXHAUSTED = "exhausted"          # no actuator can move further toward target
    NO_SIGNAL = "no_signal"          # measured luma ~0, cannot compute a ratio
    MAX_ITERATIONS = "max_iterations"


@dataclass(frozen=True)
class CalibrationTarget:
    """Recorded brightness target for one product/area/type.

    Args:
        target_luma: Desired mean luma in ``[0, 255]``.
        tolerance: Half-width of the acceptance band, in luma units.
        roi: Optional pixel box the target was measured over; None = full frame.
    """

    target_luma: float
    tolerance: float = 4.0
    roi: tuple[int, int, int, int] | None = None

    def is_within(self, measured_luma: float) -> bool:
        """Return True when ``measured_luma`` is inside the acceptance band."""
        return abs(measured_luma - self.target_luma) <= self.tolerance


@dataclass(frozen=True)
class HardwareState:
    """Current actuator values and their reachable bounds.

    Args:
        exposure: Current exposure time (device units, typically microseconds).
        exposure_min: Minimum settable exposure.
        exposure_max: Maximum settable exposure.
        led_brightness: Current LED brightness (0..led_max).
        led_max: Maximum LED brightness (controller-defined, usually 255).
        led_available: Whether the LED controller is connected and adjustable.
    """

    exposure: float
    exposure_min: float
    exposure_max: float
    led_brightness: int = 0
    led_max: int = 255
    led_available: bool = False


@dataclass(frozen=True)
class CalibrationProposal:
    """Next setpoint proposed for a single calibration step.

    ``exposure``/``led_brightness`` are the values to apply next; unchanged
    actuators keep their current value. When ``done`` is True, no change is
    needed and ``reason`` explains why the run should stop.
    """

    done: bool
    reason: CalibrationReason
    phase: CalibrationPhase
    exposure: float
    led_brightness: int


@dataclass(frozen=True)
class CalibrationOutcome:
    """Result of a full ``drive_calibration`` run."""

    success: bool
    reason: CalibrationReason
    iterations: int
    final_luma: float
    final_exposure: float
    final_led_brightness: int


class AutoCalibrator:
    """Pure two-phase brightness controller (LED coarse, exposure fine).

    The controller is stateless across calls: every decision derives from the
    supplied measurement and hardware state, so it is trivially testable and
    free of drift between the object and the real hardware.
    """

    def __init__(
        self,
        target: CalibrationTarget,
        *,
        damping: float = _DEFAULT_DAMPING,
        coarse_factor: float = _DEFAULT_COARSE_FACTOR,
    ) -> None:
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1]")
        if coarse_factor < 1.0:
            raise ValueError("coarse_factor must be >= 1.0")
        self._target = target
        self._damping = float(damping)
        self._coarse_factor = float(coarse_factor)

    @property
    def target(self) -> CalibrationTarget:
        """The brightness target this calibrator drives toward."""
        return self._target

    def propose(
        self,
        measured_luma: float,
        state: HardwareState,
    ) -> CalibrationProposal:
        """Propose the next actuator setpoint for one calibration step.

        Args:
            measured_luma: Mean luma of the latest frame (over the target ROI).
            state: Current actuator values and bounds.

        Returns:
            A proposal whose ``done`` flag signals completion (in tolerance,
            exhausted, or no signal) or a next setpoint to apply.
        """
        if self._target.is_within(measured_luma):
            return self._hold(CalibrationReason.WITHIN_TOLERANCE, state)

        if measured_luma <= 1e-3:
            # Cannot form a meaningful ratio; only escape is more light.
            return self._no_signal_step(state)

        ratio = self._target.target_luma / measured_luma
        need_more = measured_luma < self._target.target_luma
        error = abs(measured_luma - self._target.target_luma)
        coarse_band = self._coarse_factor * self._target.tolerance

        prefer_led = (
            state.led_available
            and error > coarse_band
            and self._led_can_move(state, need_more)
        )
        if prefer_led:
            return self._led_step(state, ratio, need_more)

        # Exposure fine phase, with LED as the fallback actuator.
        if self._exposure_can_move(state, need_more):
            return self._exposure_step(state, ratio, need_more)
        if state.led_available and self._led_can_move(state, need_more):
            return self._led_step(state, ratio, need_more)
        return self._hold(CalibrationReason.EXHAUSTED, state)

    # ------------------------------------------------------------------
    # Step builders
    # ------------------------------------------------------------------
    def _led_step(
        self, state: HardwareState, ratio: float, need_more: bool
    ) -> CalibrationProposal:
        target_led = state.led_brightness * (1.0 + self._damping * (ratio - 1.0))
        new_led = int(round(target_led))
        new_led = max(0, min(new_led, state.led_max))
        new_led = self._ensure_move_int(
            state.led_brightness, new_led, need_more, 0, state.led_max
        )
        return CalibrationProposal(
            done=False,
            reason=CalibrationReason.ADJUST,
            phase=CalibrationPhase.LED_COARSE,
            exposure=state.exposure,
            led_brightness=new_led,
        )

    def _exposure_step(
        self, state: HardwareState, ratio: float, need_more: bool
    ) -> CalibrationProposal:
        target_exp = state.exposure * (1.0 + self._damping * (ratio - 1.0))
        new_exp = max(state.exposure_min, min(target_exp, state.exposure_max))
        new_exp = self._ensure_move_float(
            state.exposure, new_exp, need_more,
            state.exposure_min, state.exposure_max,
        )
        return CalibrationProposal(
            done=False,
            reason=CalibrationReason.ADJUST,
            phase=CalibrationPhase.EXPOSURE_FINE,
            exposure=new_exp,
            led_brightness=state.led_brightness,
        )

    def _no_signal_step(self, state: HardwareState) -> CalibrationProposal:
        """Near-black frame: push the strongest available actuator upward."""
        if state.led_available and state.led_brightness < state.led_max:
            return CalibrationProposal(
                done=False,
                reason=CalibrationReason.ADJUST,
                phase=CalibrationPhase.LED_COARSE,
                exposure=state.exposure,
                led_brightness=state.led_max,
            )
        if state.exposure < state.exposure_max:
            return CalibrationProposal(
                done=False,
                reason=CalibrationReason.ADJUST,
                phase=CalibrationPhase.EXPOSURE_FINE,
                exposure=state.exposure_max,
                led_brightness=state.led_brightness,
            )
        return self._hold(CalibrationReason.NO_SIGNAL, state)

    def _hold(
        self, reason: CalibrationReason, state: HardwareState
    ) -> CalibrationProposal:
        return CalibrationProposal(
            done=True,
            reason=reason,
            phase=CalibrationPhase.DONE,
            exposure=state.exposure,
            led_brightness=state.led_brightness,
        )

    # ------------------------------------------------------------------
    # Movement helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _led_can_move(state: HardwareState, need_more: bool) -> bool:
        return (need_more and state.led_brightness < state.led_max) or (
            not need_more and state.led_brightness > 0
        )

    @staticmethod
    def _exposure_can_move(state: HardwareState, need_more: bool) -> bool:
        return (need_more and state.exposure < state.exposure_max) or (
            not need_more and state.exposure > state.exposure_min
        )

    @staticmethod
    def _ensure_move_int(
        current: int, proposed: int, need_more: bool, lo: int, hi: int
    ) -> int:
        """Guarantee at least a 1-unit move in the needed direction."""
        if proposed != current:
            return proposed
        if need_more and current < hi:
            return current + 1
        if not need_more and current > lo:
            return current - 1
        return current

    @staticmethod
    def _ensure_move_float(
        current: float, proposed: float, need_more: bool, lo: float, hi: float
    ) -> float:
        """Guarantee a minimum relative exposure move in the needed direction."""
        if abs(proposed - current) >= current * _MIN_EXPOSURE_STEP_RATIO:
            return proposed
        step = max(current * _MIN_EXPOSURE_STEP_RATIO, 1.0)
        if need_more and current < hi:
            return min(hi, current + step)
        if not need_more and current > lo:
            return max(lo, current - step)
        return proposed


def drive_calibration(
    calibrator: AutoCalibrator,
    *,
    measure_fn: Callable[[], float],
    read_state_fn: Callable[[], HardwareState],
    apply_fn: Callable[[float, int], None],
    settle_fn: Callable[[], None] = lambda: None,
    max_iterations: int = 20,
    on_step: Callable[[int, CalibrationPhase, float, float], None] | None = None,
) -> CalibrationOutcome:
    """Run the calibration loop against injected hardware callables.

    Args:
        calibrator: Configured controller.
        measure_fn: Capture a frame and return its measured luma.
        read_state_fn: Return the current ``HardwareState``.
        apply_fn: Apply ``(exposure, led_brightness)`` to the hardware.
        settle_fn: Block briefly so the applied change takes effect before the
            next measurement (e.g. sleep + discard a frame). Default: no-op.
        max_iterations: Hard cap on adjustment steps (prevents oscillation).
        on_step: Optional progress hook called once per adjustment step with
            ``(iteration, phase, luma, error)`` before the move is applied, so a
            UI can show convergence. Shrinking ``error`` means it is converging.

    Returns:
        A ``CalibrationOutcome`` summarizing convergence and final settings.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")

    target_luma = calibrator.target.target_luma
    state = read_state_fn()
    luma = measure_fn()

    for iteration in range(1, max_iterations + 1):
        proposal = calibrator.propose(luma, state)
        if proposal.done:
            return CalibrationOutcome(
                success=proposal.reason == CalibrationReason.WITHIN_TOLERANCE,
                reason=proposal.reason,
                iterations=iteration - 1,
                final_luma=luma,
                final_exposure=state.exposure,
                final_led_brightness=state.led_brightness,
            )

        if on_step is not None:
            on_step(iteration, proposal.phase, luma, abs(luma - target_luma))

        apply_fn(proposal.exposure, proposal.led_brightness)
        settle_fn()
        state = read_state_fn()
        luma = measure_fn()

    success = calibrator.target.is_within(luma)
    return CalibrationOutcome(
        success=success,
        reason=(
            CalibrationReason.WITHIN_TOLERANCE
            if success
            else CalibrationReason.MAX_ITERATIONS
        ),
        iterations=max_iterations,
        final_luma=luma,
        final_exposure=state.exposure,
        final_led_brightness=state.led_brightness,
    )
