"""Serial (COM-port) controlled LED light source.

The inspection rig uses a serial LED controller whose protocol is a fixed
35-byte frame::

    5A 54 1D 00 01 12 <27 brightness bytes> 0D 0A

* ``5A 54``        — frame header
* ``1D 00``        — payload length (0x1D = 29 = the 2 command bytes + 27 data)
* ``01 12``        — command (set white-channel brightness)
* 27 data bytes    — per-channel brightness; all set to the same value for
                     a uniform white output (0x00 = off, 0xFF = full)
* ``0D 0A``        — frame trailer

The protocol parameters are constructor defaults so they can be overridden or
faked in tests; the serial backend is injected via ``serial_factory`` so the
class carries no hard dependency on hardware for unit testing (DIP).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Protocol

# Default frame parts for the installed LED controller.
DEFAULT_HEADER = bytes.fromhex("5A541D000112")
DEFAULT_TRAILER = bytes.fromhex("0D0A")
DEFAULT_CHANNEL_COUNT = 27
DEFAULT_BAUDRATE = 115200
MAX_BRIGHTNESS = 255


class LightControlError(RuntimeError):
    """Raised when a serial light operation fails (open/write/close)."""


class _SerialPort(Protocol):
    """Minimal serial interface the controller relies on."""

    is_open: bool

    def write(self, data: bytes) -> int: ...

    def close(self) -> None: ...


def _default_serial_factory(port: str, baudrate: int, timeout: float) -> _SerialPort:
    """Create a real ``pyserial`` port. Imported lazily so tests stay hardware-free."""
    import serial  # type: ignore[import-untyped]

    return serial.Serial(port=port, baudrate=baudrate, timeout=timeout, write_timeout=timeout)


def serial_backend_available() -> bool:
    """Return True if the ``pyserial`` backend can be imported."""
    try:
        import serial.tools.list_ports  # type: ignore[import-untyped]  # noqa: F401

        return True
    except Exception:
        return False


def available_ports() -> list[tuple[str, str]]:
    """Return ``(device, description)`` for every detected serial port.

    Returns an empty list if ``pyserial`` is unavailable so the caller can show
    a friendly message rather than crashing. Use :func:`serial_backend_available`
    to tell "backend missing" apart from "genuinely no ports".
    """
    try:
        from serial.tools import list_ports  # type: ignore[import-untyped]
    except Exception:
        logging.getLogger(__name__).warning(
            "pyserial is not importable in this interpreter; cannot enumerate "
            "serial ports. Install it with `pip install pyserial`."
        )
        return []
    return [(p.device, p.description or p.device) for p in list_ports.comports()]


class LightController:
    """Thread-safe wrapper that drives a serial LED light source.

    A single :class:`threading.Lock` serialises every access to the shared
    serial handle so a UI-thread click and a background brightness update can
    never interleave a write on the same port.
    """

    def __init__(
        self,
        *,
        baudrate: int = DEFAULT_BAUDRATE,
        header: bytes = DEFAULT_HEADER,
        trailer: bytes = DEFAULT_TRAILER,
        channel_count: int = DEFAULT_CHANNEL_COUNT,
        max_value: int = MAX_BRIGHTNESS,
        write_timeout: float = 1.0,
        serial_factory: Callable[[str, int, float], _SerialPort] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if not 0 < max_value <= 255:
            raise ValueError("max_value must be within 1..255")
        self._baudrate = baudrate
        self._header = header
        self._trailer = trailer
        self._channel_count = channel_count
        self._max_value = max_value
        self._write_timeout = write_timeout
        self._serial_factory = serial_factory or _default_serial_factory
        self._logger = logger or logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._serial: _SerialPort | None = None
        self._port: str | None = None
        self._last_brightness = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._serial is not None and bool(getattr(self._serial, "is_open", True))

    @property
    def port(self) -> str | None:
        with self._lock:
            return self._port

    @property
    def max_value(self) -> int:
        return self._max_value

    @property
    def last_brightness(self) -> int:
        with self._lock:
            return self._last_brightness

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def open(self, port: str) -> None:
        """Open *port*, replacing any previously opened handle.

        Raises:
            LightControlError: if the port cannot be opened.
        """
        if not port:
            raise LightControlError("No serial port specified.")
        with self._lock:
            self._close_locked()
            try:
                self._serial = self._serial_factory(port, self._baudrate, self._write_timeout)
            except Exception as exc:  # pyserial raises SerialException
                self._serial = None
                self._port = None
                raise LightControlError(f"Failed to open {port}: {exc}") from exc
            self._port = port
            self._logger.info("Light serial port opened: %s @ %d", port, self._baudrate)

    def close(self) -> None:
        """Close the serial port if open (idempotent, never raises)."""
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        """Close the handle. Caller must hold ``self._lock``."""
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception as exc:  # pragma: no cover - best-effort cleanup
                self._logger.warning("Error closing light serial port: %s", exc)
            finally:
                self._serial = None
                self._port = None

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def build_frame(self, value: int) -> bytes:
        """Build the wire frame for brightness *value* (clamped to 0..max_value)."""
        clamped = max(0, min(self._max_value, int(value)))
        return self._header + bytes([clamped]) * self._channel_count + self._trailer

    def set_brightness(self, value: int) -> None:
        """Send a frame setting all channels to *value* (0..max_value).

        Raises:
            LightControlError: if no port is open or the write fails.
        """
        frame = self.build_frame(value)
        clamped = max(0, min(self._max_value, int(value)))
        with self._lock:
            if self._serial is None:
                raise LightControlError("Light port is not connected.")
            try:
                self._serial.write(frame)
            except Exception as exc:
                raise LightControlError(f"Failed to send light command: {exc}") from exc
            self._last_brightness = clamped
        self._logger.debug("Light brightness set to %d", clamped)

    def turn_on(self) -> None:
        """Turn the light on at full brightness."""
        self.set_brightness(self._max_value)

    def turn_off(self) -> None:
        """Turn the light off (brightness 0)."""
        self.set_brightness(0)
