"""Unit tests for the serial LED light controller (hardware-free)."""

from __future__ import annotations

import pytest

from core.services.light_controller import (
    DEFAULT_CHANNEL_COUNT,
    LightControlError,
    LightController,
)


class FakeSerial:
    """In-memory stand-in for ``serial.Serial`` used to capture writes."""

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.is_open = True
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise OSError("port closed")
        self.writes.append(bytes(data))
        return len(data)

    def close(self) -> None:
        self.is_open = False
        self.closed = True


def _make(factory_record: list[FakeSerial] | None = None) -> LightController:
    def factory(port: str, baudrate: int, timeout: float) -> FakeSerial:
        fake = FakeSerial(port, baudrate, timeout)
        if factory_record is not None:
            factory_record.append(fake)
        return fake

    return LightController(serial_factory=factory)


def test_off_frame_matches_protocol():
    frame = _make().build_frame(0)
    assert frame.hex() == "5a541d0001120000000000000000000000000000000000000000000000000000000d0a"
    assert len(frame) == 6 + DEFAULT_CHANNEL_COUNT + 2


def test_full_brightness_frame_matches_protocol():
    frame = _make().build_frame(255)
    assert frame.hex() == "5a541d000112ffffffffffffffffffffffffffffffffffffffffffffffffffffff0d0a"


def test_brightness_is_clamped_to_range():
    controller = _make()
    assert controller.build_frame(-50) == controller.build_frame(0)
    assert controller.build_frame(9999) == controller.build_frame(255)


def test_turn_on_off_write_expected_frames():
    record: list[FakeSerial] = []
    controller = _make(record)
    controller.open("COM7")
    controller.turn_on()
    controller.turn_off()

    fake = record[0]
    assert fake.port == "COM7"
    assert fake.baudrate == 115200
    assert fake.writes[0] == controller.build_frame(255)
    assert fake.writes[1] == controller.build_frame(0)


def test_set_brightness_without_open_raises():
    controller = _make()
    with pytest.raises(LightControlError):
        controller.set_brightness(128)


def test_open_replaces_previous_port():
    record: list[FakeSerial] = []
    controller = _make(record)
    controller.open("COM1")
    controller.open("COM2")
    assert record[0].closed is True
    assert controller.port == "COM2"
    assert controller.is_open is True


def test_open_failure_wraps_error_and_stays_closed():
    def failing_factory(port: str, baudrate: int, timeout: float):
        raise OSError("device busy")

    controller = LightController(serial_factory=failing_factory)
    with pytest.raises(LightControlError):
        controller.open("COM9")
    assert controller.is_open is False
    assert controller.port is None


def test_write_failure_raises_light_control_error():
    record: list[FakeSerial] = []
    controller = _make(record)
    controller.open("COM3")
    record[0].is_open = False  # simulate cable yanked
    with pytest.raises(LightControlError):
        controller.turn_on()


def test_close_is_idempotent():
    controller = _make()
    controller.close()  # never opened
    controller.open("COM4")
    controller.close()
    controller.close()
    assert controller.is_open is False
