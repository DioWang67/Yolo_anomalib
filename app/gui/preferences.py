from __future__ import annotations

from PyQt5.QtCore import QByteArray, QSettings


class PreferencesManager:
    """Wrapper around QSettings for persisting GUI preferences."""

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings()

    @property
    def settings(self) -> QSettings:
        return self._settings

    def restore_window_state(self) -> tuple[QByteArray | None, QByteArray | None]:
        geometry = self._settings.value("geometry")
        window_state = self._settings.value("windowState")
        return geometry, window_state

    def save_window_state(self, geometry: QByteArray, window_state: QByteArray) -> None:
        self._settings.setValue("geometry", geometry)
        self._settings.setValue("windowState", window_state)

    def restore_last_selection(self) -> tuple[str, str, str]:
        product = str(self._settings.value("last_product", ""))
        area = str(self._settings.value("last_area", ""))
        inference = str(self._settings.value("last_infer", ""))
        return product, area, inference

    def save_last_selection(self, product: str, area: str, inference: str) -> None:
        self._settings.setValue("last_product", product)
        self._settings.setValue("last_area", area)
        self._settings.setValue("last_infer", inference)

    def restore_show_detection_boxes(self) -> bool:
        """Return whether result images should show detection boxes."""
        value = self._settings.value("show_detection_boxes", True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off"}

    def save_show_detection_boxes(self, enabled: bool) -> None:
        """Persist the result-image detection box visibility preference."""
        self._settings.setValue("show_detection_boxes", bool(enabled))

    def restore_language(self) -> str:
        """Return the persisted GUI language code."""
        value = str(self._settings.value("language", "en"))
        return value if value in {"en", "zh"} else "en"

    def save_language(self, language: str) -> None:
        """Persist the GUI language code."""
        self._settings.setValue("language", language if language in {"en", "zh"} else "en")

    def restore_light_port(self) -> str:
        """Return the last serial port used for the LED light ('' if none)."""
        return str(self._settings.value("light_port", ""))

    def save_light_port(self, port: str) -> None:
        """Persist the serial port used for the LED light."""
        self._settings.setValue("light_port", port or "")

    def restore_light_brightness(self) -> int:
        """Return the persisted LED brightness percentage (0..100, default 100)."""
        try:
            value = int(self._settings.value("light_brightness", 100))
        except (TypeError, ValueError):
            return 100
        return max(0, min(100, value))

    def save_light_brightness(self, percent: int) -> None:
        """Persist the LED brightness percentage (0..100)."""
        self._settings.setValue("light_brightness", max(0, min(100, int(percent))))
