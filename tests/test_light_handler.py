"""Tests for GUI light keepalive behavior."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5.QtWidgets", reason="PyQt5 is required for light GUI tests")

from PyQt5.QtWidgets import QApplication, QWidget

from app.gui.light_handler import LIGHT_KEEPALIVE_INTERVAL_MS, LightHandlerMixin
from core.services.light_controller import LightControlError


class FakePreferences:
    """Minimal preferences used by the light handler tests."""

    def __init__(self, brightness: int = 80) -> None:
        self.brightness = brightness

    def restore_light_brightness(self) -> int:
        return self.brightness

    def restore_light_port(self) -> str:
        return "COM3"

    def save_light_brightness(self, percent: int) -> None:
        self.brightness = percent


class FakeCatalog:
    """Return one model config path for light-sync tests."""

    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def config_path(self, product: str, area: str, inference_type: str) -> Path:
        return self._config_path


class FakeStatusBar:
    """Capture status messages without a full main window."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self.messages.append(message)


class FakeLightController:
    """Small controller fake for exercising LightHandlerMixin."""

    def __init__(self) -> None:
        self.max_value = 255
        self.is_open = True
        self.port = "COM3"
        self.values: list[int] = []
        self.fail_next = False
        self.closed = False

    def open(self, port: str) -> None:
        self.port = port
        self.is_open = True

    def set_brightness(self, value: int) -> None:
        if self.fail_next:
            self.fail_next = False
            raise LightControlError("simulated disconnect")
        self.values.append(value)

    def turn_off(self) -> None:
        self.set_brightness(0)

    def close(self) -> None:
        self.closed = True
        self.is_open = False


class LightHost(QWidget, LightHandlerMixin):
    """QObject host for the mixin so QTimer can be parented correctly."""

    def __init__(self) -> None:
        super().__init__()
        self.preferences = FakePreferences()
        self.current_language = "en"
        self._light_controller = FakeLightController()
        self._status = FakeStatusBar()
        self.logs: list[str] = []

    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def statusBar(self) -> FakeStatusBar:
        return self._status


@pytest.fixture()
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_light_turn_on_starts_keepalive_timer(app: QApplication) -> None:
    host = LightHost()

    host._light_turn_on()

    assert host._light_controller.values == [204]
    assert host._light_keepalive_value == 204
    assert host._light_keepalive_timer.interval() == LIGHT_KEEPALIVE_INTERVAL_MS
    assert host._light_keepalive_timer.isActive() is True


def test_light_turn_off_stops_keepalive_timer(app: QApplication) -> None:
    host = LightHost()
    host._light_turn_on()

    host._light_turn_off()

    assert host._light_controller.values[-1] == 0
    assert host._light_keepalive_value == 0
    assert host._light_keepalive_timer.isActive() is False


def test_keepalive_failure_stops_timer_without_modal_error(app: QApplication) -> None:
    host = LightHost()
    host._light_turn_on()
    host._light_controller.fail_next = True

    host._send_light_keepalive()

    assert host._light_keepalive_value == 0
    assert host._light_keepalive_timer.isActive() is False
    assert any("simulated disconnect" in message for message in host.logs)


def test_model_zero_brightness_stops_stale_keepalive(
    app: QApplication, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("light_brightness: 0\n", encoding="utf-8")
    host = LightHost()
    host._catalog = FakeCatalog(config_path)
    host._light_turn_on()

    host._apply_model_light_brightness("Cable1", "A", "yolo")

    assert host._light_controller.values[-1] == 0
    assert host._light_keepalive_value == 0
    assert host._light_keepalive_timer.isActive() is False


def test_model_positive_brightness_updates_keepalive_without_changing_preference(
    app: QApplication, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("light_brightness: 25\n", encoding="utf-8")
    host = LightHost()
    host._catalog = FakeCatalog(config_path)

    host._apply_model_light_brightness("Cable1", "A", "yolo")

    assert host._light_controller.values == [64]
    assert host._light_keepalive_value == 64
    assert host._light_keepalive_timer.isActive() is True
    assert host.preferences.brightness == 80
