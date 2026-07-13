"""Focused tests for auto-trigger latency and candidate-frame retention."""

from unittest.mock import patch

import numpy as np

from core.auto_trigger import AutoTriggerConfig, AutoTriggerStateMachine


def test_auto_trigger_defaults_use_responsive_stability_window():
    """Defaults should avoid an unnecessarily long pre-inspection delay."""
    config = AutoTriggerConfig()

    assert config.frame_buffer_size == 8
    assert config.appear_frames == 3
    assert config.stable_frames == 6


def test_auto_trigger_keeps_full_resolution_candidates_only_for_product_frames():
    """Empty frames must not consume the expensive full-resolution buffer."""
    config = AutoTriggerConfig(product_area_threshold=1)
    state_machine = AutoTriggerStateMachine(config)
    frame = np.zeros((24, 24, 3), dtype=np.uint8)

    with patch("core.auto_trigger.detect_product_presence", return_value=(False, 0.0)):
        state_machine.update(frame, store_frame=frame)

    assert len(state_machine._frame_buffer) == 0

    with patch("core.auto_trigger.detect_product_presence", return_value=(True, 10.0)):
        state_machine.update(frame, store_frame=frame)

    assert len(state_machine._frame_buffer) == 1
