from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt5.QtWidgets import QMessageBox

import app.gui.camera_handler as camera_handler


class _Check:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked
        self.enabled = True
        self.blocked: list[bool] = []

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool) -> None:
        self._checked = checked

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def blockSignals(self, blocked: bool) -> None:
        self.blocked.append(blocked)


class _Button:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _Indicator:
    def __init__(self, state: str = "connecting") -> None:
        self.state = state
        self.states: list[str] = []
        self.reconnect_allowed = True

    def set_state(self, state: str) -> None:
        self.state = state
        self.states.append(state)

    def set_reconnect_allowed(self, allowed: bool) -> None:
        self.reconnect_allowed = allowed


class _Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback


class _Host(camera_handler.CameraHandlerMixin):
    def __init__(self) -> None:
        self.current_language = "en"
        self.running = False
        self.system_available = True
        self.camera_connected = True
        self.controller = SimpleNamespace(
            has_system=lambda: self.system_available,
            is_camera_connected=lambda: self.camera_connected,
            reconnect_camera=Mock(return_value=True),
            disconnect_camera=Mock(),
            bridge=SimpleNamespace(end_run=Mock()),
            build_shutdown_worker=self._build_shutdown_worker,
        )
        self.camera_status_indicator = _Indicator()
        self.reconnect_camera_btn = _Button()
        self.disconnect_camera_btn = _Button()
        self.use_camera_chk = _Check(True)
        self.auto_mode_chk = _Check(False)
        self.pick_image_btn = _Button()
        self.clear_image_btn = _Button()
        self.image_path_label = SimpleNamespace(setText=Mock())
        self.selected_image_path = None
        self._camera_check_ts = 0.0
        self._camera_connected_cache = False
        self._run_generation = 2
        self._shutdown_in_progress = False
        self.stats_timer = SimpleNamespace(stop=Mock())
        self.logs: list[str] = []
        self.init_system = Mock()
        self.start_detection = Mock()
        self._reset_ui_state = Mock()

    def is_detection_running(self) -> bool:
        return self.running

    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def _build_shutdown_worker(self):
        return SimpleNamespace(shutdown_complete=_Signal(), start=Mock())


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    monkeypatch.setattr(camera_handler.QMessageBox, "warning", lambda *_args: None)
    monkeypatch.setattr(camera_handler.QMessageBox, "critical", lambda *_args: None)
    monkeypatch.setattr(camera_handler.QMessageBox, "information", lambda *_args: None)


def test_camera_status_sync_preserves_ready_and_reconciles_disconnect() -> None:
    host = _Host()
    host._sync_camera_status(camera_connected=True, running=False)
    assert host.camera_status_indicator.state == "connected"
    host.camera_status_indicator.state = "ready"
    host._sync_camera_status(camera_connected=True, running=True)
    assert host.camera_status_indicator.state == "ready"
    assert not host.camera_status_indicator.reconnect_allowed
    host._sync_camera_status(camera_connected=False, running=False)
    assert host.camera_status_indicator.state == "unavailable"

    host.camera_status_indicator = None
    host._set_camera_status("lost")
    host._sync_camera_status(camera_connected=False, running=False)


def test_update_controls_uses_cached_health_and_image_mode(monkeypatch) -> None:
    host = _Host()
    host.pick_image_btn.enabled = False
    times = iter((10.0, 11.0, 13.1))
    monkeypatch.setattr(camera_handler.time, "monotonic", lambda: next(times))

    host.update_camera_controls()
    assert host.disconnect_camera_btn.enabled
    assert not host.pick_image_btn.enabled

    host.camera_connected = False
    host.update_camera_controls()
    assert host.disconnect_camera_btn.enabled
    host.update_camera_controls()
    assert not host.disconnect_camera_btn.enabled
    assert not host.use_camera_chk.isChecked()
    assert host.pick_image_btn.enabled

    controls_before_failure = (
        host.reconnect_camera_btn.enabled,
        host.disconnect_camera_btn.enabled,
        host.use_camera_chk.enabled,
        host.pick_image_btn.enabled,
        host.clear_image_btn.enabled,
        host.camera_status_indicator.state,
    )
    host.controller.has_system = Mock(side_effect=RuntimeError("controller failed"))
    host.update_camera_controls()
    assert (
        host.reconnect_camera_btn.enabled,
        host.disconnect_camera_btn.enabled,
        host.use_camera_chk.enabled,
        host.pick_image_btn.enabled,
        host.clear_image_btn.enabled,
        host.camera_status_indicator.state,
    ) == controls_before_failure


def test_reconnect_guards_and_success_paths(monkeypatch) -> None:
    host = _Host()
    host.running = True
    host.handle_reconnect_camera()
    host.controller.reconnect_camera.assert_not_called()

    host.running = False
    host.system_available = False
    host.handle_reconnect_camera()
    host.init_system.assert_called_once()

    host.system_available = True
    monkeypatch.setattr(host, "update_camera_controls", Mock())
    toggled: list[bool] = []
    monkeypatch.setattr(host, "on_use_camera_toggled", toggled.append)
    host.handle_reconnect_camera()
    assert host.camera_status_indicator.state == "connected"
    assert toggled == [True]

    host.controller.reconnect_camera.return_value = False
    host.handle_reconnect_camera()
    assert host.camera_status_indicator.state == "unavailable"

    host.controller.reconnect_camera.side_effect = OSError("camera busy")
    host.handle_reconnect_camera()
    assert any("camera busy" in message for message in host.logs)


def test_disconnect_guards_success_and_failure(monkeypatch) -> None:
    host = _Host()
    host.running = True
    host.handle_disconnect_camera()
    host.controller.disconnect_camera.assert_not_called()

    host.running = False
    host.system_available = False
    host.handle_disconnect_camera()
    host.controller.disconnect_camera.assert_not_called()

    host.system_available = True
    monkeypatch.setattr(host, "update_camera_controls", Mock())
    toggled: list[bool] = []
    monkeypatch.setattr(host, "on_use_camera_toggled", toggled.append)
    host.handle_disconnect_camera()
    assert toggled == [False]
    assert host.camera_status_indicator.state == "disconnected"

    host.controller.disconnect_camera.side_effect = OSError("USB failure")
    host.handle_disconnect_camera()
    assert any("USB failure" in message for message in host.logs)


def test_camera_mode_toggle_covers_readiness_and_static_image_controls(monkeypatch) -> None:
    host = _Host()
    host.running = True
    host.on_use_camera_toggled(True)

    host.running = False
    host.camera_connected = False
    host._camera_check_ts = 0.0
    host.on_use_camera_toggled(True)
    assert not host.use_camera_chk.isChecked()
    assert host.camera_status_indicator.state == "unavailable"

    host.camera_connected = True
    host._camera_check_ts = 0.0
    host.use_camera_chk.setChecked(True)
    host.selected_image_path = "old.png"
    host.on_use_camera_toggled(True)
    assert host.selected_image_path is None
    assert not host.pick_image_btn.enabled
    assert host.camera_status_indicator.state == "connected"

    host.selected_image_path = "chosen.png"
    host.on_use_camera_toggled(False)
    assert host.pick_image_btn.enabled
    assert host.clear_image_btn.enabled
    assert host.camera_status_indicator.state == "image_mode"

    update_controls = Mock()
    monkeypatch.setattr(host, "update_camera_controls", update_controls)
    host.controller.is_camera_connected = Mock(side_effect=RuntimeError("health error"))
    host.camera_status_indicator.state = "ready"
    host.on_use_camera_toggled(True)
    assert host.camera_status_indicator.state == "unavailable"
    update_controls.assert_called_once_with()

    update_controls.reset_mock()
    update_controls.side_effect = RuntimeError("ui gone")
    host.controller.is_camera_connected = Mock(return_value=True)
    host.selected_image_path = None
    host.clear_image_btn.enabled = True
    host.image_path_label.setText.reset_mock()
    host.on_use_camera_toggled(False)
    assert host.camera_status_indicator.state == "image_mode"
    assert host.pick_image_btn.enabled
    assert not host.clear_image_btn.enabled
    host.image_path_label.setText.assert_called_once()
    update_controls.assert_called_once_with()


def test_camera_loss_stops_pipeline_then_follows_operator_choice(monkeypatch) -> None:
    host = _Host()
    host._on_camera_disconnected()
    assert host.controller.bridge.end_run.call_args.args == (2,)
    assert host._run_generation == 3
    assert host._shutdown_in_progress
    assert host.camera_status_indicator.state == "lost"
    assert host._shutdown_worker.shutdown_complete.callback == host._on_camera_lost_pipeline_stopped
    host._shutdown_worker.start.assert_called_once()

    reconnect = Mock()
    monkeypatch.setattr(host, "handle_reconnect_camera", reconnect)
    monkeypatch.setattr(camera_handler.QMessageBox, "question", lambda *_args: QMessageBox.Yes)
    host._on_camera_lost_pipeline_stopped()
    reconnect.assert_called_once()
    host.start_detection.assert_called_once()

    host.camera_connected = False
    host._on_camera_lost_pipeline_stopped()
    assert any("manual" in message.lower() for message in host.logs)

    monkeypatch.setattr(camera_handler.QMessageBox, "question", lambda *_args: QMessageBox.No)
    host._on_camera_lost_pipeline_stopped()
    assert host.logs

    host.controller.bridge.end_run.side_effect = RuntimeError("already stopped")
    generation_before_failure = host._run_generation
    host._on_camera_disconnected()
    assert host._run_generation == generation_before_failure
    assert host.camera_status_indicator.state == "lost"
    assert host.stats_timer.stop.call_count == 2
    host._shutdown_worker.start.assert_called_once()
