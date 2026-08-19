from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PyQt5.QtWidgets import QMenu, QWidget

import app.gui.light_handler as light_handler
from core.services.light_controller import LightControlError


class _Preferences:
    def __init__(self, *, port: str = "COM3", brightness: int = 80) -> None:
        self.port = port
        self.brightness = brightness

    def restore_light_port(self) -> str:
        return self.port

    def save_light_port(self, port: str) -> None:
        self.port = port

    def restore_light_brightness(self) -> int:
        return self.brightness

    def save_light_brightness(self, value: int) -> None:
        self.brightness = value


class _Controller:
    def __init__(self, *, is_open: bool = False) -> None:
        self.is_open = is_open
        self.port = "COM3" if is_open else None
        self.max_value = 255
        self.values: list[int] = []
        self.opened: list[str] = []
        self.closed = False
        self.fail_open = False
        self.fail_send = False

    def open(self, port: str) -> None:
        if self.fail_open:
            raise LightControlError("open failed")
        self.opened.append(port)
        self.port = port
        self.is_open = True

    def set_brightness(self, value: int) -> None:
        if self.fail_send:
            raise LightControlError("send failed")
        self.values.append(value)

    def turn_off(self) -> None:
        self.set_brightness(0)

    def close(self) -> None:
        self.closed = True
        self.is_open = False


class _Host(QWidget, light_handler.LightHandlerMixin):
    def __init__(self, controller: _Controller | None = None) -> None:
        super().__init__()
        self.current_language = "en"
        self.preferences = _Preferences()
        self._light_controller = controller
        self._logger = None
        self._status = SimpleNamespace(showMessage=lambda message, _timeout=0: self.statuses.append(message))
        self.statuses: list[str] = []
        self.logs: list[str] = []
        self._catalog = SimpleNamespace(config_path=lambda *_args: Path("config.yaml"))

    def statusBar(self):
        return self._status

    def log_message(self, message: str) -> None:
        self.logs.append(message)


@pytest.fixture(autouse=True)
def no_modal_dialogs(monkeypatch):
    monkeypatch.setattr(light_handler.QMessageBox, "information", lambda *_args: None)
    monkeypatch.setattr(light_handler.QMessageBox, "warning", lambda *_args: None)
    monkeypatch.setattr(light_handler.QMessageBox, "critical", lambda *_args: None)


def test_percent_mapping_and_brightness_dialog_clamp_and_callback(qtbot) -> None:
    assert light_handler._percent_to_value(-1, 255) == 0
    assert light_handler._percent_to_value(50, 255) == 128
    assert light_handler._percent_to_value(101, 255) == 255

    changes: list[int] = []
    dialog = light_handler.BrightnessDialog(150, "en", changes.append)
    qtbot.addWidget(dialog)
    assert dialog.value() == 100
    dialog._slider.setValue(25)
    assert dialog._value_label.text() == "25%"
    assert changes == [25]


def test_controller_creation_and_saved_port_recovery(monkeypatch, qtbot) -> None:
    created = _Controller()
    monkeypatch.setattr(light_handler, "LightController", lambda **_kwargs: created)
    host = _Host(None)
    qtbot.addWidget(host)
    assert host._ensure_light_controller() is created
    assert host._require_open_light_port() is created
    assert created.opened == ["COM3"]

    created.is_open = False
    created.fail_open = True
    monkeypatch.setattr(host, "_light_select_port", lambda: False)
    assert host._require_open_light_port() is None
    assert any("open failed" in message for message in host.logs)

    created.fail_open = False
    host.preferences.port = ""
    monkeypatch.setattr(host, "_light_select_port", lambda: True)
    assert host._require_open_light_port() is None


def test_port_selection_messages_cancel_success_and_failure(monkeypatch, qtbot) -> None:
    host = _Host(_Controller())
    qtbot.addWidget(host)
    monkeypatch.setattr(light_handler, "serial_backend_available", lambda: False)
    assert host._light_no_ports_message()
    monkeypatch.setattr(light_handler, "serial_backend_available", lambda: True)
    assert host._light_no_ports_message()

    monkeypatch.setattr(light_handler, "available_ports", lambda: [])
    assert not host._light_select_port()

    ports = [("COM3", "saved"), ("COM4", "new")]
    monkeypatch.setattr(light_handler, "available_ports", lambda: ports)
    monkeypatch.setattr(light_handler.QInputDialog, "getItem", lambda *_args, **_kwargs: ("", False))
    assert not host._light_select_port()

    monkeypatch.setattr(
        light_handler.QInputDialog,
        "getItem",
        lambda *_args, **_kwargs: ("COM4  —  new", True),
    )
    assert host._light_select_port()
    assert host.preferences.port == "COM4"
    assert host.statuses

    host._light_controller.is_open = False
    host._light_controller.fail_open = True
    assert not host._light_select_port()


def test_turn_on_off_fallback_and_transport_errors(monkeypatch, qtbot) -> None:
    controller = _Controller(is_open=True)
    host = _Host(controller)
    qtbot.addWidget(host)
    host.preferences.brightness = 0
    host._light_turn_on()
    assert controller.values == [255]
    assert host._light_keepalive_value == 255

    host._light_turn_off()
    assert controller.values[-1] == 0
    assert host._light_keepalive_value == 0

    errors: list[str] = []
    monkeypatch.setattr(host, "_report_light_error", lambda exc: errors.append(str(exc)))
    controller.fail_send = True
    host._light_turn_on()
    host._light_turn_off()
    assert errors == ["send failed", "send failed"]

    monkeypatch.setattr(host, "_require_open_light_port", lambda: None)
    host._light_turn_on()
    host._light_turn_off()


@pytest.mark.parametrize("final_percent", (0, 40))
def test_brightness_dialog_applies_live_value_and_persists_final(
    monkeypatch,
    qtbot,
    final_percent: int,
) -> None:
    controller = _Controller(is_open=True)
    host = _Host(controller)
    qtbot.addWidget(host)

    class _Dialog:
        def __init__(self, _start, _language, apply, **_kwargs) -> None:
            self.apply = apply

        def exec_(self) -> None:
            self.apply(25)

        def value(self) -> int:
            return final_percent

    monkeypatch.setattr(light_handler, "BrightnessDialog", _Dialog)
    host._light_open_brightness_dialog()
    assert controller.values == [64]
    assert host.preferences.brightness == final_percent
    assert host._light_keepalive_value == light_handler._percent_to_value(final_percent, 255)

    controller.fail_send = True
    reported: list[str] = []
    monkeypatch.setattr(host, "_report_light_error", lambda exc: reported.append(str(exc)))
    host._light_open_brightness_dialog()
    assert reported == ["send failed"]


def test_model_brightness_validation_and_best_effort_apply(monkeypatch, qtbot) -> None:
    controller = _Controller(is_open=True)
    host = _Host(controller)
    qtbot.addWidget(host)
    monkeypatch.setattr(host, "_active_release_model_config", lambda *_args: None)

    monkeypatch.setattr(light_handler, "load_model_config", lambda _path: {})
    assert host._read_model_light_brightness("Cable1", "A", "fusion") is None
    monkeypatch.setattr(light_handler, "load_model_config", lambda _path: {"light_brightness": "bad"})
    assert host._read_model_light_brightness("Cable1", "A", "yolo") is None
    monkeypatch.setattr(light_handler, "load_model_config", lambda _path: {"light_brightness": 101})
    assert host._read_model_light_brightness("Cable1", "A", "yolo") is None
    monkeypatch.setattr(light_handler, "load_model_config", lambda _path: (_ for _ in ()).throw(OSError("bad yaml")))
    assert host._read_model_light_brightness("Cable1", "A", "yolo") is None

    monkeypatch.setattr(host, "_read_model_light_brightness", lambda *_args: None)
    host._apply_model_light_brightness("Cable1", "A", "yolo")
    monkeypatch.setattr(host, "_read_model_light_brightness", lambda *_args: 0)
    monkeypatch.setattr(host, "_open_light_controller_if_available", lambda: None)
    host._light_keepalive_value = 10
    host._apply_model_light_brightness("Cable1", "A", "yolo")
    assert host._light_keepalive_value == 0

    monkeypatch.setattr(host, "_read_model_light_brightness", lambda *_args: 20)
    monkeypatch.setattr(host, "_open_light_controller_if_available", lambda: controller)
    controller.fail_send = True
    host._apply_model_light_brightness("Cable1", "A", "yolo")
    assert any("send failed" in message for message in host.logs)


def test_non_prompting_open_and_dynamic_port_menu(monkeypatch, qtbot) -> None:
    controller = _Controller()
    host = _Host(controller)
    qtbot.addWidget(host)
    host.preferences.port = ""
    assert host._open_light_controller_if_available() is None

    host.preferences.port = "COM5"
    assert host._open_light_controller_if_available() is controller
    controller.is_open = False
    controller.fail_open = True
    assert host._open_light_controller_if_available() is None

    menu = QMenu()
    qtbot.addWidget(menu)
    monkeypatch.setattr(light_handler, "available_ports", lambda: [])
    host.populate_light_port_menu(menu)
    assert len(menu.actions()) == 1
    assert not menu.actions()[0].isEnabled()

    controller.fail_open = False
    controller.is_open = True
    controller.port = "COM3"
    monkeypatch.setattr(light_handler, "available_ports", lambda: [("COM3", "active"), ("COM4", "other")])
    opened: list[str] = []
    monkeypatch.setattr(host, "_light_open_specific_port", opened.append)
    host.populate_light_port_menu(menu)
    assert len(menu.actions()) == 2
    assert menu.actions()[0].isChecked()
    menu.actions()[1].trigger()
    assert opened == ["COM4"]


def test_direct_port_open_error_reporting_keepalive_and_shutdown(monkeypatch, qtbot) -> None:
    controller = _Controller()
    host = _Host(controller)
    qtbot.addWidget(host)
    host._light_open_specific_port("COM7")
    assert host.preferences.port == "COM7"

    controller.fail_open = True
    host._light_open_specific_port("COM8")
    assert host.preferences.port == "COM7"

    host._report_light_error(RuntimeError("protocol error"))
    assert any("protocol error" in message for message in host.logs)

    host._start_light_keepalive(0)
    assert host._light_keepalive_value == 0
    host._send_light_keepalive()
    host._start_light_keepalive(42)
    host._send_light_keepalive()
    assert controller.values[-1] == 42

    controller.fail_send = True
    host._send_light_keepalive()
    assert host._light_keepalive_value == 0
    host.shutdown_light()
    assert controller.closed

    host._light_controller = None
    monkeypatch.setattr(host, "_stop_light_keepalive", lambda: None)
    host.shutdown_light()


def test_active_release_lookup_without_project_root_is_absent(qtbot) -> None:
    host = _Host(_Controller())
    qtbot.addWidget(host)
    assert host._active_release_model_config("Cable1", "A", "yolo", "yolo") is None
