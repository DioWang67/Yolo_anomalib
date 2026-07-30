from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for controller tests")

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtTest import QSignalSpy

from app.gui import auto_inspection_controller as controller_module
from app.gui.auto_inspection_controller import AutoInspectionController


class _Bridge(QObject):
    image_ready = pyqtSignal(object)
    result_ready = pyqtSignal(object)


class _BlockingDetectionSystem:
    def __init__(self) -> None:
        self.camera = SimpleNamespace(is_initialized=True)
        self.detect_entered = threading.Event()
        self.release_detect = threading.Event()
        self.detect_returned = threading.Event()
        self.cancel_callback = None

    def detect(self, *_args, **kwargs):
        self.cancel_callback = kwargs["cancel_cb"]
        self.detect_entered.set()
        self.release_detect.wait(timeout=5.0)
        self.detect_returned.set()
        return SimpleNamespace(status="PASS")


class _ControllablePreviewWorker(QObject):
    frame_ready = pyqtSignal(object)
    trigger_fired = pyqtSignal(object)
    state_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    instances: list[_ControllablePreviewWorker] = []

    def __init__(self, *, camera, config, show_debug_overlay) -> None:
        super().__init__()
        self.camera = camera
        self.config = config
        self.show_debug_overlay = show_debug_overlay
        self.running = False
        self.stop_requested = False
        self.marked_inspecting = 0
        self.shown_results: list[str] = []
        type(self).instances.append(self)

    def start(self) -> None:
        self.running = True

    def isRunning(self) -> bool:  # noqa: N802 - mirrors QThread
        return self.running

    def stop(self) -> None:
        self.stop_requested = True

    def wait(self, *, msecs: int) -> bool:
        del msecs
        return not self.running

    def mark_inspecting(self) -> None:
        self.marked_inspecting += 1

    def mark_result_shown(self, status: str) -> None:
        self.shown_results.append(status)

    def finish(self) -> None:
        if not self.running:
            return
        self.running = False
        self.finished.emit()


def _start_blocked_inspection(monkeypatch, qtbot):
    _ControllablePreviewWorker.instances.clear()
    monkeypatch.setattr(
        controller_module,
        "CameraPreviewWorker",
        _ControllablePreviewWorker,
    )
    system = _BlockingDetectionSystem()
    bridge = _Bridge()
    controller = AutoInspectionController(system, bridge)
    stopped = QSignalSpy(controller.fully_stopped)
    results = QSignalSpy(bridge.result_ready)

    assert controller.start("Cable1", "A", "yolo") is True
    generation = controller.active_generation
    assert generation is not None
    preview = _ControllablePreviewWorker.instances[-1]
    preview.trigger_fired.emit(np.zeros((8, 8, 3), dtype=np.uint8))
    qtbot.waitUntil(system.detect_entered.is_set, timeout=1000)
    return controller, system, bridge, preview, generation, stopped, results


def _finish_preview_workers() -> None:
    for worker in tuple(_ControllablePreviewWorker.instances):
        worker.finish()


def test_fully_stopped_waits_for_inflight_inference_after_preview_finishes(
    qtbot,
    monkeypatch,
) -> None:
    (
        controller,
        system,
        _bridge,
        preview,
        generation,
        stopped,
        results,
    ) = _start_blocked_inspection(monkeypatch, qtbot)

    try:
        assert controller.stop() is False
        preview.finish()
        qtbot.wait(20)

        # A finished capture loop is not a fully stopped controller while the
        # inference it launched can still touch the bridge and state machine.
        assert len(stopped) == 0
    finally:
        system.release_detect.set()

    qtbot.waitUntil(lambda: len(stopped) == 1, timeout=1000)
    assert stopped[0] == [generation]
    assert len(results) == 0


def test_fully_stopped_waits_for_preview_after_inflight_inference_finishes(
    qtbot,
    monkeypatch,
) -> None:
    (
        controller,
        system,
        _bridge,
        preview,
        generation,
        stopped,
        results,
    ) = _start_blocked_inspection(monkeypatch, qtbot)

    try:
        assert controller.stop() is False
        system.release_detect.set()
        assert system.detect_returned.wait(timeout=1.0)
        qtbot.wait(20)

        # Completing the inference first is also insufficient while the camera
        # loop has not emitted QThread.finished.
        assert len(stopped) == 0

        preview.finish()
        qtbot.waitUntil(lambda: len(stopped) == 1, timeout=1000)
        assert stopped[0] == [generation]
        assert len(results) == 0
    finally:
        system.release_detect.set()
        _finish_preview_workers()


def test_start_is_rejected_until_cancelled_inference_and_preview_are_both_done(
    qtbot,
    monkeypatch,
) -> None:
    (
        controller,
        system,
        _bridge,
        preview,
        generation,
        stopped,
        results,
    ) = _start_blocked_inspection(monkeypatch, qtbot)
    first_cancel_event = controller._cancel_event
    assert first_cancel_event is not None

    try:
        assert controller.stop() is False
        preview.finish()
        qtbot.wait(20)

        # start() must not clear the cancellation state of the preceding
        # generation while its detect() call is still unwinding.
        assert controller.start("Cable1", "A", "yolo") is False
        assert system.cancel_callback is not None
        assert system.cancel_callback() is True

        system.release_detect.set()
        qtbot.waitUntil(lambda: len(stopped) == 1, timeout=1000)
        assert stopped[0] == [generation]
        assert len(results) == 0

        assert controller.start("Cable1", "A", "yolo") is True
        second_generation = controller.active_generation
        assert second_generation is not None
        assert second_generation > generation
        assert controller._cancel_event is not first_cancel_event
        assert first_cancel_event.is_set() is True
        assert controller._cancel_event is not None
        assert controller._cancel_event.is_set() is False
        assert len(_ControllablePreviewWorker.instances) == 2

        # A result queued by the prior generation must remain stale even after
        # a later generation owns the controller and has a fresh cancel event.
        controller._inspection_result_ready.emit(
            generation,
            SimpleNamespace(status="STALE"),
        )
        qtbot.wait(20)
        assert len(results) == 0
    finally:
        system.release_detect.set()
        controller.stop()
        _finish_preview_workers()
