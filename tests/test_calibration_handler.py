"""Tests for the illumination calibration dialog wiring."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5.QtWidgets", reason="PyQt5 is required for calibration GUI tests")

from PyQt5.QtWidgets import QApplication, QWidget

import app.gui.calibration_handler as calibration_handler_module
from app.gui.calibration_handler import CalibrationHandlerMixin
from app.gui.light_handler import LightHandlerMixin


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
    def __init__(self, camera: FakeCamera | None) -> None:
        self.detection_system = FakeDetectionSystem(camera)
        self.reload_calls: list[tuple[str, str, str]] = []
        self.raise_on_reload = False

    def has_system(self) -> bool:
        return True

    def reload_model_settings(self, product: str, area: str, inference_type: str) -> None:
        if self.raise_on_reload:
            raise RuntimeError("reload boom")
        self.reload_calls.append((product, area, inference_type))


class FakeCatalog:
    def __init__(self, config_path: Path) -> None:
        self._config_path = config_path

    def config_path(self, product: str, area: str, inference_type: str) -> Path:
        return self._config_path


class FakePreferences:
    def __init__(self, brightness: int = 80) -> None:
        self.brightness = brightness

    def restore_light_brightness(self) -> int:
        return self.brightness

    def restore_light_port(self) -> str:
        return "COM3"

    def save_light_brightness(self, percent: int) -> None:
        self.brightness = percent


class FakeLightController:
    def __init__(self) -> None:
        self.max_value = 255
        self.is_open = True
        self.port = "COM3"
        self.values: list[int] = []

    def open(self, port: str) -> None:
        self.port = port
        self.is_open = True

    def set_brightness(self, value: int) -> None:
        self.values.append(value)

    def turn_off(self) -> None:
        self.set_brightness(0)

    def close(self) -> None:
        self.is_open = False


class FakeMessageBox:
    calls: list[tuple[str, str, str]] = []

    @staticmethod
    def warning(parent, title, text) -> None:
        FakeMessageBox.calls.append(("warning", title, text))

    @staticmethod
    def critical(parent, title, text) -> None:
        FakeMessageBox.calls.append(("critical", title, text))


class FakeDialog:
    instances: list["FakeDialog"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        FakeDialog.instances.append(self)

    def exec_(self) -> None:
        pass


class FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def showMessage(self, message: str, timeout: int = 0) -> None:
        self.messages.append(message)


class CalibrationHost(QWidget, CalibrationHandlerMixin, LightHandlerMixin):
    def __init__(self, controller: FakeController, catalog: FakeCatalog, detection_running: bool = False) -> None:
        super().__init__()
        self.controller = controller
        self._catalog = catalog
        self.current_language = "en"
        self.preferences = FakePreferences()
        self._light_controller = FakeLightController()
        self.product_combo = FakeCombo("Cable1")
        self.area_combo = FakeCombo("A")
        self.inference_combo = FakeCombo("yolo")
        self._detection_running = detection_running
        self.logs: list[str] = []
        self._status = FakeStatusBar()

    def is_detection_running(self) -> bool:
        return self._detection_running

    def log_message(self, message: str) -> None:
        self.logs.append(message)

    def statusBar(self) -> FakeStatusBar:
        return self._status


@pytest.fixture()
def app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def patched_ui(monkeypatch):
    FakeMessageBox.calls = []
    FakeDialog.instances = []
    monkeypatch.setattr(calibration_handler_module, "QMessageBox", FakeMessageBox)
    monkeypatch.setattr(calibration_handler_module, "CalibrationDialog", FakeDialog)
    yield


def test_blocked_when_detection_running(app: QApplication, tmp_path: Path) -> None:
    host = CalibrationHost(
        FakeController(FakeCamera()), FakeCatalog(tmp_path / "config.yaml"), detection_running=True
    )

    host.open_calibration_dialog()

    assert FakeDialog.instances == []
    assert FakeMessageBox.calls == [("warning", "Illumination Calibration", "Stop detection before calibrating.")]


def test_blocked_when_selection_missing(app: QApplication, tmp_path: Path) -> None:
    host = CalibrationHost(FakeController(FakeCamera()), FakeCatalog(tmp_path / "config.yaml"))
    host.area_combo = FakeCombo("")

    host.open_calibration_dialog()

    assert FakeDialog.instances == []
    assert FakeMessageBox.calls == [
        ("warning", "Illumination Calibration", "Select a product, area, and inference type first.")
    ]


def test_blocked_when_no_camera(app: QApplication, tmp_path: Path) -> None:
    host = CalibrationHost(FakeController(None), FakeCatalog(tmp_path / "config.yaml"))

    host.open_calibration_dialog()

    assert FakeDialog.instances == []
    assert FakeMessageBox.calls == [
        ("warning", "Illumination Calibration", "Camera is not connected; connect it before calibrating.")
    ]


def test_keepalive_suspended_during_dialog_and_restarted_after(
    app: QApplication, tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("light_brightness: 25\n", encoding="utf-8")
    controller = FakeController(FakeCamera())
    host = CalibrationHost(controller, FakeCatalog(config_path))
    host._light_turn_on()
    assert host._light_keepalive_timer.isActive() is True

    seen_active_during_dialog = []

    class HookedDialog(FakeDialog):
        def exec_(self) -> None:
            seen_active_during_dialog.append(host._light_keepalive_timer.isActive())

    monkeypatch.setattr(calibration_handler_module, "CalibrationDialog", HookedDialog)

    host.open_calibration_dialog()

    assert seen_active_during_dialog == [False]
    assert controller.reload_calls == [("Cable1", "A", "yolo")]
    assert host._light_keepalive_timer.isActive() is True
    assert host._light_keepalive_value == 64


def test_reload_failure_is_logged_not_raised(app: QApplication, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("light_brightness: 25\n", encoding="utf-8")
    controller = FakeController(FakeCamera())
    controller.raise_on_reload = True
    host = CalibrationHost(controller, FakeCatalog(config_path))

    host.open_calibration_dialog()

    assert any("reload boom" in message for message in host.logs)
