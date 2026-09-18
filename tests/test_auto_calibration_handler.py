"""Tests for re-deriving the exposure without anybody asking.

The load-bearing tests here are the refusals. This runs unprompted, while an
operator is walking up to a station, so every way it can decline has to leave
the station exactly as it behaves today -- no modal, no exception out of a Qt
slot, and the configured exposure still in force.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5.QtWidgets", reason="PyQt5 is required for GUI tests")

from PyQt5.QtWidgets import QApplication, QWidget

import app.gui.auto_calibration_handler as auto_calibration_module
from app.gui.auto_calibration_handler import AutoCalibrationHandlerMixin
from app.gui.calibration_handler import CalibrationHandlerMixin
from app.gui.light_handler import LightHandlerMixin
from core.services.auto_calibrator import CalibrationOutcome, CalibrationReason

CABLE1 = ("Cable1", "A", "yolo")


class FakeCombo:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:
        return self._text


class FakeCamera:
    def __init__(self, is_initialized: bool = True) -> None:
        self.is_initialized = is_initialized


class FakeDetectionSystem:
    def __init__(self, camera: FakeCamera | None) -> None:
        self.camera = camera


class FakeController:
    def __init__(self, camera: FakeCamera | None, has_system: bool = True) -> None:
        self.detection_system = FakeDetectionSystem(camera)
        self._has_system = has_system
        self.override_calls: list[tuple[tuple[str, str, str], float]] = []
        self.clear_calls = 0
        self.invalidate_calls = 0

    def has_system(self) -> bool:
        return self._has_system

    def set_runtime_exposure_override(self, scope, exposure_time) -> None:
        self.override_calls.append((scope, exposure_time))

    def clear_runtime_exposure_override(self) -> None:
        self.clear_calls += 1

    def invalidate_applied_camera_settings(self) -> None:
        self.invalidate_calls += 1


class FakeLightController:
    def __init__(self, is_open: bool = True) -> None:
        self.max_value = 255
        self.is_open = is_open
        self.values: list[int] = []

    def set_brightness(self, value: int) -> None:
        self.values.append(value)


class FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self.messages.append(message)


class FakeWorker:
    """Runs the outcome straight through on start(), on this thread."""

    instances: list["FakeWorker"] = []
    outcome: object | None = None
    failure: str | None = None

    def __init__(self, session, target, parent=None) -> None:
        self.session = session
        self.target = target
        self.parent = parent
        self._finished_ok = []
        self._failed = []
        self._finished = []
        self._progress = []
        FakeWorker.instances.append(self)

    class _Signal:
        def __init__(self, sink):
            self._sink = sink

        def connect(self, slot):
            self._sink.append(slot)

    @property
    def finished_ok(self):
        return FakeWorker._Signal(self._finished_ok)

    @property
    def failed(self):
        return FakeWorker._Signal(self._failed)

    @property
    def finished(self):
        return FakeWorker._Signal(self._finished)

    @property
    def progress(self):
        return FakeWorker._Signal(self._progress)

    def start(self) -> None:
        if FakeWorker.failure is not None:
            for slot in self._failed:
                slot(FakeWorker.failure)
        else:
            for slot in self._finished_ok:
                slot(FakeWorker.outcome)
        for slot in self._finished:
            slot()

    def wait(self, timeout_ms: int) -> bool:
        return True


class AutoCalibrationHost(
    QWidget, CalibrationHandlerMixin, AutoCalibrationHandlerMixin, LightHandlerMixin
):
    def __init__(
        self,
        controller: FakeController,
        config_path: Path,
        *,
        detection_running: bool = False,
        inference: str = "yolo",
    ) -> None:
        super().__init__()
        self.controller = controller
        self._catalog = _FakeCatalog(config_path)
        self.current_language = "en"
        self._logger = logging.getLogger(__name__)
        self._light_controller = FakeLightController()
        self.product_combo = FakeCombo("Cable1")
        self.area_combo = FakeCombo("A")
        self.inference_combo = FakeCombo(inference)
        self._detection_running = detection_running
        self.logs: list[str] = []
        self._status = FakeStatusBar()
        self.keepalive: list[str] = []
        self.light_applied: list[tuple] = []
        self._autocalib_worker = None
        self._autocalib_scope = None
        self._autocalib_timer = None
        self._last_autocalibrated_scope = None

    def is_detection_running(self) -> bool:
        return self._detection_running

    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def statusBar(self) -> FakeStatusBar:
        return self._status

    def update_start_enabled(self) -> None:
        pass

    def update_camera_controls(self) -> None:
        pass

    def _stop_light_keepalive(self) -> None:
        self.keepalive.append("stop")

    def _start_light_keepalive(self, value: int) -> None:
        self.keepalive.append(f"start:{value}")

    def _apply_model_light_brightness(self, *scope) -> None:
        self.light_applied.append(scope)


class _FakeCatalog:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def config_path(self, product, area, inference_type) -> Path:
        return self._config_path


@pytest.fixture()
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def fake_worker(monkeypatch):
    FakeWorker.instances = []
    FakeWorker.outcome = CalibrationOutcome(
        success=True,
        reason=CalibrationReason.WITHIN_TOLERANCE,
        iterations=4,
        final_luma=45.2,
        final_exposure=21980.0,
        final_led_brightness=0,
    )
    FakeWorker.failure = None
    monkeypatch.setattr(auto_calibration_module, "AutoCalibrateWorker", FakeWorker)
    monkeypatch.setattr(
        auto_calibration_module, "CalibrationSession", lambda *a, **k: object()
    )
    yield


def station(tmp_path: Path, *, target: float | None = 45.0) -> Path:
    path = tmp_path / "config.yaml"
    body = "conf_thres: 0.5\n"
    if target is not None:
        body += f"calibration:\n  target_luma: {target}\n  tolerance: 2.0\n"
    path.write_text(body, encoding="utf-8")
    return path


def host(tmp_path, app, **kwargs) -> AutoCalibrationHost:
    target = kwargs.pop("target", 45.0)
    camera = kwargs.pop("camera", FakeCamera())
    has_system = kwargs.pop("has_system", True)
    controller = FakeController(camera, has_system=has_system)
    return AutoCalibrationHost(controller, station(tmp_path, target=target), **kwargs)


# -- the refusals -----------------------------------------------------------


def test_nothing_runs_while_the_camera_is_being_used(tmp_path, app):
    """The camera has no lock; the whole codebase serialises on this flag."""
    widget = host(tmp_path, app, detection_running=True)

    widget._run_scheduled_autocalibration()

    assert FakeWorker.instances == []
    assert widget._last_autocalibrated_scope is None


def test_nothing_runs_before_a_station_is_selected(tmp_path, app):
    """Camera-ready can land before the model list has loaded."""
    widget = host(tmp_path, app)
    widget.area_combo = FakeCombo("")

    widget._run_scheduled_autocalibration()

    assert FakeWorker.instances == []
    assert widget._last_autocalibrated_scope is None


def test_a_camera_less_attempt_does_not_consume_the_run(tmp_path, app):
    """The combos can fill before the camera opens; the later call must work."""
    widget = host(tmp_path, app, camera=None)

    widget._run_scheduled_autocalibration()
    assert FakeWorker.instances == []
    assert widget._last_autocalibrated_scope is None

    widget.controller.detection_system.camera = FakeCamera()
    widget._run_scheduled_autocalibration()

    assert len(FakeWorker.instances) == 1


def test_a_station_with_no_recorded_target_is_left_alone(tmp_path, app):
    """The entire rollout gate: no target means nobody has calibrated here.

    Every LED/* and PCBA1/* station takes this branch, so they are untouched
    without needing a config key or a toggle to say so.
    """
    widget = host(tmp_path, app, target=None)

    widget._run_scheduled_autocalibration()

    assert FakeWorker.instances == []
    assert widget.controller.override_calls == []
    # Marked, so it is not retried on every repaint of the combos.
    assert widget._last_autocalibrated_scope == CABLE1


def test_the_same_station_is_not_calibrated_twice(tmp_path, app):
    widget = host(tmp_path, app)

    widget._run_scheduled_autocalibration()
    widget._run_scheduled_autocalibration()

    assert len(FakeWorker.instances) == 1


# -- the run ----------------------------------------------------------------


def test_a_converged_run_is_held_for_the_session_and_named_as_such(tmp_path, app):
    widget = host(tmp_path, app)

    widget._run_scheduled_autocalibration()

    assert widget.controller.override_calls == [(CABLE1, 21980.0)]
    assert widget._last_autocalibrated_scope == CABLE1
    assert "21980" in widget._status.messages[-1]
    # The operator is told the config file was not touched.
    assert "not changed" in widget._status.messages[-1]


def test_the_camera_cache_is_dropped_before_the_loop_touches_hardware(tmp_path, app):
    """So whatever the loop leaves behind, the next apply re-pushes it."""
    widget = host(tmp_path, app)

    widget._run_scheduled_autocalibration()

    assert widget.controller.invalidate_calls == 1


def test_the_led_keepalive_is_suspended_for_the_run(tmp_path, app):
    widget = host(tmp_path, app)

    widget._run_scheduled_autocalibration()

    assert widget.keepalive[0] == "stop"


def test_a_run_that_does_not_converge_changes_nothing(tmp_path, app):
    """The station degrades to exactly today's behaviour, and says so once."""
    FakeWorker.outcome = CalibrationOutcome(
        success=False,
        reason=CalibrationReason.EXHAUSTED,
        iterations=20,
        final_luma=12.0,
        final_exposure=60000.0,
        final_led_brightness=0,
    )
    widget = host(tmp_path, app)

    widget._run_scheduled_autocalibration()

    assert widget.controller.override_calls == []
    assert "exhausted" in widget._status.messages[-1]
    assert widget.light_applied == [CABLE1]
    # Still marked: a station that cannot converge must not retry forever.
    assert widget._last_autocalibrated_scope == CABLE1


def test_a_worker_failure_is_reported_and_never_raised(tmp_path, app):
    FakeWorker.failure = "camera went away"
    widget = host(tmp_path, app)

    widget._run_scheduled_autocalibration()

    assert widget.controller.override_calls == []
    assert any("camera went away" in message for message in widget.logs)


def test_switching_station_stops_applying_the_previous_ones_exposure(tmp_path, app):
    widget = host(tmp_path, app)
    widget._run_scheduled_autocalibration()
    before = widget.controller.clear_calls

    widget.product_combo = FakeCombo("PCBA1")
    widget._run_scheduled_autocalibration()

    assert widget.controller.clear_calls == before + 1


def test_fusion_is_calibrated_as_the_yolo_station_it_reads(tmp_path, app):
    widget = host(tmp_path, app, inference="fusion")

    widget._run_scheduled_autocalibration()

    assert widget.controller.override_calls == [(CABLE1, 21980.0)]


def test_a_run_in_flight_blocks_start_and_then_releases_it(tmp_path, app):
    widget = host(tmp_path, app)
    assert not widget._autocalibration_in_progress()

    widget._autocalib_worker = object()
    assert widget._autocalibration_in_progress()

    widget._finish_autocalibration()
    assert not widget._autocalibration_in_progress()


def test_a_camera_change_forgets_what_the_last_session_measured(tmp_path, app):
    widget = host(tmp_path, app)
    widget._run_scheduled_autocalibration()

    widget._reset_autocalibration_for_camera_change()

    assert widget._last_autocalibrated_scope is None
    assert widget.controller.clear_calls >= 1
