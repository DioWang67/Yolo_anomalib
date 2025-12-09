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
