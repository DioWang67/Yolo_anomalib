"""A golden sample measured before the light settles is a light measurement.

Auto-calibration runs at camera-ready and, when it cannot converge, logs,
flashes the status bar and marks the scope so it will not retry. Nothing
stopped the golden-sample check from then running three inspections at the
wrong brightness and reporting the result as a colour deviation at a position
— sending the operator to inspect the board and the colour model, which are
both fine. These tests pin the precondition that closes that gap, and pin that
it refuses to block on anything it does not actually know.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.gui.auto_calibration_handler import CalibrationStatus
from app.gui.color_preflight_handler import (
    ColorPreflightHandlerMixin,
    golden_sample_readiness,
)
from core.services.golden_sample import GoldenSampleError

SCOPE = ("Cable1", "A", "yolo")


def _host(**overrides):
    """A station with every other precondition satisfied."""
    base = {
        "product_combo": SimpleNamespace(currentText=lambda: "Cable1"),
        "area_combo": SimpleNamespace(currentText=lambda: "A"),
        "inference_combo": SimpleNamespace(currentText=lambda: "yolo"),
        "_auto_controller": SimpleNamespace(is_running=lambda: False),
        "use_camera_chk": SimpleNamespace(isChecked=lambda: True),
        "controller": SimpleNamespace(has_system=lambda: True),
        "is_detection_running": lambda: False,
        "start_detection": lambda **kwargs: None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _illumination(host):
    checks = golden_sample_readiness(host, *SCOPE)
    return next(check for check in checks if check.label == "照明已收斂")


def test_unconverged_illumination_blocks_and_names_the_numbers() -> None:
    """The hint has to say it is a brightness problem, with the figures."""
    host = _host(
        _last_autocalibration_status=CalibrationStatus(
            SCOPE, False, "max_iterations", 90.6, 56.0, 2.0
        )
    )
    check = _illumination(host)

    assert not check.ok
    assert "90.6" in check.hint
    assert "56.0" in check.hint
    assert "+34.6" in check.hint
    assert "max_iterations" in check.hint


def test_converged_illumination_does_not_block() -> None:
    host = _host(
        _last_autocalibration_status=CalibrationStatus(
            SCOPE, True, "within_tolerance", 56.4, 56.0, 2.0
        )
    )
    assert _illumination(host).ok


def test_calibration_still_running_blocks() -> None:
    """The window this exists to close: green preconditions mid-convergence."""
    host = _host(_autocalib_worker=object())
    check = _illumination(host)

    assert not check.ok
    assert "進行中" in check.hint


def test_never_calibrated_does_not_block() -> None:
    """Refusing the daily check on a missing reading would stop the line."""
    assert _illumination(_host()).ok


def test_station_without_a_luma_target_does_not_block() -> None:
    """No recorded target means no band to be outside of."""
    host = _host(
        _last_autocalibration_status=CalibrationStatus(SCOPE, True, "no_target")
    )
    assert _illumination(host).ok


def test_a_result_from_another_station_is_ignored() -> None:
    """A stale reading must not block the station now in front of the operator."""
    host = _host(
        _last_autocalibration_status=CalibrationStatus(
            ("PCBA1", "B", "yolo"), False, "max_iterations", 120.0, 56.0, 2.0
        )
    )
    assert _illumination(host).ok


def test_missing_luma_still_blocks_with_the_reason() -> None:
    """A crashed run leaves the light wherever it stopped."""
    host = _host(
        _last_autocalibration_status=CalibrationStatus(
            SCOPE, False, "error", None, 56.0, 2.0
        )
    )
    check = _illumination(host)

    assert not check.ok
    assert "error" in check.hint


def test_the_capture_refuses_exactly_when_the_panel_says_so() -> None:
    """The panel and the capture must not diverge, as for the other four."""
    started: list[dict] = []
    host = _host(
        start_detection=lambda **kwargs: started.append(kwargs),
        _last_autocalibration_status=CalibrationStatus(
            SCOPE, False, "max_iterations", 90.6, 56.0, 2.0
        ),
    )

    with pytest.raises(GoldenSampleError) as raised:
        ColorPreflightHandlerMixin._capture_golden_sample(host, *SCOPE)

    assert "照明已收斂" in str(raised.value)
    assert started == [], "no inspection may run at the wrong brightness"
