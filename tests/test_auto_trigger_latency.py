"""Trigger latency is counted in camera frames, so every frame must earn its place.

``PRODUCT_APPEAR`` used to spend a whole frame on nothing but a state change:
presence was already confirmed by ``appear_frames`` and stability was not
scored until the next frame. At the measured 55 ms per frame on the Cable1/A
station that was 55 ms of pure latency. These tests pin the fix, and — more
importantly — pin that it bought speed without relaxing any criterion.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from core.auto_trigger import (
    AutoTriggerConfig,
    AutoTriggerStateMachine,
    TriggerState,
)


def _product_frame(blurred: bool = False) -> np.ndarray:
    """A high-contrast frame that reads as present, sharp and motionless."""
    image: np.ndarray = np.zeros((512, 768, 3), dtype=np.uint8)
    cv2.rectangle(image, (150, 120), (620, 400), (240, 240, 240), -1)
    cv2.rectangle(image, (200, 170), (560, 350), (20, 20, 20), -1)
    if blurred:
        blurred_image: np.ndarray = cv2.GaussianBlur(image, (31, 31), 0)
        return blurred_image
    return image


def _machine(appear: int, stable: int) -> AutoTriggerStateMachine:
    return AutoTriggerStateMachine(
        AutoTriggerConfig(
            appear_frames=appear,
            stable_frames=stable,
            product_area_threshold=5000,
            inspection_cooldown_ms=0,
        )
    )


def _run(machine: AutoTriggerStateMachine, frame: np.ndarray, limit: int = 20):
    """Feed identical frames until the trigger fires. Returns (frames, stable_count)."""
    stable_at_trigger = 0
    for count in range(1, limit + 1):
        _, fired = machine.update(frame.copy(), store_frame=frame)
        if fired:
            return count, stable_at_trigger
        stable_at_trigger = machine.get_debug_info().stable_count
    return None, stable_at_trigger


@pytest.mark.parametrize(
    ("appear", "stable", "expected_frames"),
    [(2, 3, 5), (3, 6, 9), (1, 1, 2)],
)
def test_trigger_costs_appear_plus_stable_frames(
    appear: int, stable: int, expected_frames: int
) -> None:
    """No frame is spent purely on a state transition.

    The old machine needed ``appear + 1 + stable`` frames. The extra one was
    the PRODUCT_APPEAR cycle, which scored nothing.
    """
    frames, _ = _run(_machine(appear, stable), _product_frame())
    assert frames == expected_frames


@pytest.mark.parametrize(("appear", "stable"), [(2, 3), (3, 6), (2, 1)])
def test_required_stable_frames_are_still_accumulated(appear: int, stable: int) -> None:
    """Speed must come from removing dead time, not from demanding less evidence."""
    machine = _machine(appear, stable)
    frames, _ = _run(machine, _product_frame())

    assert frames is not None
    assert machine.get_debug_info().stable_count == stable


def test_blurred_frames_never_trigger() -> None:
    """The per-frame quality bar is untouched by the latency fix."""
    frames, _ = _run(_machine(2, 3), _product_frame(blurred=True))
    assert frames is None


def test_product_removed_during_appear_resets_to_wait_empty() -> None:
    """Judging the transition frame must not skip the presence re-check."""
    machine = _machine(2, 3)
    product = _product_frame()
    empty = np.zeros((512, 768, 3), dtype=np.uint8)

    machine.update(product.copy(), store_frame=product)
    machine.update(product.copy(), store_frame=product)
    assert machine.get_debug_info().state is TriggerState.PRODUCT_APPEAR

    machine.update(empty.copy(), store_frame=empty)
    debug = machine.get_debug_info()
    assert debug.state is TriggerState.WAIT_EMPTY
    assert debug.stable_count == 0


def test_motion_resets_the_stable_run() -> None:
    """A moving product still restarts the count, from either entry point."""
    machine = _machine(1, 3)
    product = _product_frame()
    shifted = np.roll(product, 120, axis=1)

    machine.update(product.copy(), store_frame=product)  # appear -> PRODUCT_APPEAR
    machine.update(product.copy(), store_frame=product)  # scored on entry
    assert machine.get_debug_info().stable_count >= 1

    machine.update(shifted.copy(), store_frame=shifted)  # large motion
    assert machine.get_debug_info().stable_count == 0
