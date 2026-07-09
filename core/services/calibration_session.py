from __future__ import annotations

"""Hardware-facing glue for brightness auto-calibration.

``AutoCalibrator`` is pure; this module binds it to a real camera and LED
controller so the GUI can (a) read the current live luma, (b) record current
hardware values as the model's target, and (c) run the closed-loop calibration.

The camera/light dependencies are duck-typed protocols and injected, so the
session is fully unit-testable with fakes and carries no Qt or hardware import.

LED brightness has two representations: the controller speaks raw 0..max_value
(usually 255); config and the operator UI speak 0..100 percent. This module is
the single boundary that converts between them.
"""

import time
from typing import Callable, Protocol

import numpy as np

from core.services.auto_calibrator import (
    AutoCalibrator,
    CalibrationOutcome,
    CalibrationPhase,
    CalibrationTarget,
    HardwareState,
    drive_calibration,
    measure_luma,
)


class CameraLike(Protocol):
    """Subset of CameraController the calibration session relies on."""

    def capture_frame(self) -> np.ndarray | None: ...
    def get_exposure(self) -> float | None: ...
    def get_exposure_range(self) -> dict | None: ...
    def set_exposure(self, exposure_time: float) -> bool: ...
    def get_gain(self) -> float | None: ...


class LightLike(Protocol):
    """Subset of LightController the calibration session relies on."""

    is_open: bool
    max_value: int
    last_brightness: int

    def set_brightness(self, value: int) -> None: ...


def value_to_percent(value: int, max_value: int) -> int:
    """Convert a raw LED value (0..max_value) to 0..100 percent."""
    if max_value <= 0:
        return 0
    return int(round(max(0, min(value, max_value)) / max_value * 100))


def percent_to_value(percent: int, max_value: int) -> int:
    """Convert 0..100 percent to a raw LED value (0..max_value)."""
    return int(round(max(0, min(100, percent)) / 100 * max_value))


class CalibrationSession:
    """Bind an AutoCalibrator to a live camera and (optional) LED controller.

    Args:
        camera: Camera exposing capture/get_exposure/get_exposure_range/
            set_exposure/get_gain.
        light: LED controller (``None`` when no light is connected).
        settle_seconds: Delay after applying a change before re-measuring, so
            the sensor/LED settle. Kept small; the loop is bounded anyway.
        sleep_fn: Injectable sleep (tests pass a no-op).
    """

    def __init__(
        self,
        camera: CameraLike,
        light: LightLike | None,
        *,
        settle_seconds: float = 0.2,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._camera = camera
        self._light = light
        self._settle_seconds = max(0.0, float(settle_seconds))
        self._sleep = sleep_fn

    # ------------------------------------------------------------------
    # Live readings
    # ------------------------------------------------------------------
    def measure(self, roi: tuple[int, int, int, int] | None = None) -> float | None:
        """Capture one frame and return its mean luma, or None on capture fail."""
        _, luma = self.capture_and_measure(roi)
        return luma

    def capture_and_measure(
        self, roi: tuple[int, int, int, int] | None = None
    ) -> tuple[np.ndarray | None, float | None]:
        """Capture one frame; return ``(frame, mean luma)``.

        A single capture serves both the live preview and the luma readout so
        the UI never double-captures. Both elements are ``None`` when the
        camera fails to deliver a frame.
        """
        frame = self._camera.capture_frame()
        if frame is None:
            return None, None
        return frame, measure_luma(frame, roi)

    def _led_available(self) -> bool:
        return bool(self._light is not None and getattr(self._light, "is_open", False))

    def read_state(self) -> HardwareState:
        """Snapshot current exposure bounds and LED value into a HardwareState.

        Raises:
            RuntimeError: If exposure or its range cannot be read (the loop
                cannot run safely without device bounds).
        """
        rng = self._camera.get_exposure_range()
        exposure = self._camera.get_exposure()
        if rng is None or exposure is None:
            raise RuntimeError("無法讀取相機曝光參數，請確認相機已連線")

        led_value = 0
        led_max = 255
        if self._led_available():
            led_max = int(getattr(self._light, "max_value", 255))
            led_value = int(getattr(self._light, "last_brightness", 0) or 0)

        return HardwareState(
            exposure=float(exposure),
            exposure_min=float(rng.get("min", exposure)),
            exposure_max=float(rng.get("max", exposure)),
            led_brightness=led_value,
            led_max=led_max,
            led_available=self._led_available(),
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def record_current(
        self,
        roi: tuple[int, int, int, int] | None = None,
    ) -> dict:
        """Read current hardware values and current luma as the target.

        Returns:
            A mapping ready for ``save_calibration_settings``: ``exposure_time``,
            ``gain``, ``light_brightness`` (percent), and ``target_luma``.

        Raises:
            RuntimeError: If a live frame or exposure cannot be read.
        """
        luma = self.measure(roi)
        if luma is None:
            raise RuntimeError("無法取得影像，請確認相機已連線並可取像")
        exposure = self._camera.get_exposure()
        gain = self._camera.get_gain()
        if exposure is None:
            raise RuntimeError("無法讀取曝光值，請確認相機已連線")

        state = self.read_state()
        light_percent = value_to_percent(state.led_brightness, state.led_max)
        return {
            "exposure_time": float(exposure),
            "gain": float(gain) if gain is not None else None,
            "light_brightness": light_percent,
            "target_luma": round(float(luma), 2),
        }

    def run_auto(
        self,
        target: CalibrationTarget,
        *,
        max_iterations: int = 20,
        on_step: Callable[[int, CalibrationPhase, float, float], None] | None = None,
    ) -> CalibrationOutcome:
        """Run the closed-loop calibration against the live hardware.

        Applies exposure to the camera and brightness to the LED (when
        available) until measured luma reaches ``target`` or the loop is
        exhausted / capped. ``on_step`` is forwarded to ``drive_calibration``
        for per-step progress reporting.
        """

        def measure_fn() -> float:
            luma = self.measure(target.roi)
            # A failed capture reads as darkest so the loop reacts (adds light)
            # instead of crashing; a persistent failure hits max_iterations.
            return 0.0 if luma is None else luma

        def apply_fn(exposure: float, led_value: int) -> None:
            self._camera.set_exposure(float(exposure))
            if self._led_available():
                self._light.set_brightness(int(led_value))

        def settle_fn() -> None:
            if self._settle_seconds > 0:
                self._sleep(self._settle_seconds)
            # Discard one frame so the next measurement reflects the new setting.
            self._camera.capture_frame()

        return drive_calibration(
            AutoCalibrator(target),
            measure_fn=measure_fn,
            read_state_fn=self.read_state,
            apply_fn=apply_fn,
            settle_fn=settle_fn,
            max_iterations=max_iterations,
            on_step=on_step,
        )

    def snapshot_after_run(self) -> dict:
        """Read converged hardware values for saving after ``run_auto``.

        Returns:
            Mapping for ``save_calibration_settings`` without ``target_luma``
            (the target itself is unchanged by a run).
        """
        exposure = self._camera.get_exposure()
        gain = self._camera.get_gain()
        state = self.read_state()
        return {
            "exposure_time": float(exposure) if exposure is not None else None,
            "gain": float(gain) if gain is not None else None,
            "light_brightness": value_to_percent(
                state.led_brightness, state.led_max
            ),
        }
