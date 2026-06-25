"""Mixin and dialog for the serial (COM-port) LED light control menu."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QSlider,
    QVBoxLayout,
)

from app.gui.i18n import tr
from core.services.light_controller import (
    LightControlError,
    LightController,
    available_ports,
    serial_backend_available,
)


def _percent_to_value(percent: int, max_value: int) -> int:
    """Map a 0..100 percentage onto the 0..max_value brightness range."""
    return round(max(0, min(100, percent)) / 100 * max_value)


class BrightnessDialog(QDialog):
    """Modal slider dialog that emits brightness changes live via *on_change*."""

    def __init__(self, percent: int, language: str, on_change, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr(language, "light_brightness_title"))
        self.setMinimumWidth(320)
        self._on_change = on_change

        layout = QVBoxLayout(self)
        self._value_label = QLabel()
        self._value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._value_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("0%"))
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 100)
        self._slider.setValue(max(0, min(100, percent)))
        self._slider.setTracking(True)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(QLabel("100%"))
        layout.addLayout(row)

        self._slider.valueChanged.connect(self._handle_change)
        self._update_label(self._slider.value())

    def _update_label(self, value: int) -> None:
        self._value_label.setText(f"{value}%")

    def _handle_change(self, value: int) -> None:
        self._update_label(value)
        self._on_change(value)

    def value(self) -> int:
        """Return the current slider percentage (0..100)."""
        return self._slider.value()


class LightHandlerMixin:
    """Serial LED light control wired to the 'Lighting' menu.

    Relies on host attributes provided by ``DetectionSystemGUI``:
    ``preferences``, ``current_language``, ``log_message``, ``statusBar`` and
    the ``_t`` translation helper (shared with ``CameraHandlerMixin``).
    """

    def _t(self, key: str, **kwargs: object) -> str:  # pragma: no cover - trivial
        text = tr(getattr(self, "current_language", "en"), key)
        return text.format(**kwargs) if kwargs else text

    # ------------------------------------------------------------------
    # Controller lifecycle
    # ------------------------------------------------------------------
    def _ensure_light_controller(self) -> LightController:
        controller = getattr(self, "_light_controller", None)
        if controller is None:
            controller = LightController(logger=getattr(self, "_logger", None))
            self._light_controller = controller
        return controller

    def _require_open_light_port(self) -> LightController | None:
        """Return an open controller, transparently reopening the saved port.

        Shows a guidance message and returns ``None`` when no port can be
        opened, so callers can simply bail out.
        """
        controller = self._ensure_light_controller()
        if controller.is_open:
            return controller

        saved_port = self.preferences.restore_light_port()
        if saved_port:
            try:
                controller.open(saved_port)
                return controller
            except LightControlError as exc:
                self.log_message(self._t("light_port_open_failed", port=saved_port, error=exc))

        # No usable saved port — ask the operator to pick one.
        if self._light_select_port():
            return controller if controller.is_open else None
        QMessageBox.information(
            self,
            self._t("lighting_menu"),
            self._t("light_select_port_first"),
        )
        return None

    # ------------------------------------------------------------------
    # Menu actions
    # ------------------------------------------------------------------
    def _light_no_ports_message(self) -> str:
        """Pick the right empty-list message: backend missing vs. no hardware."""
        if not serial_backend_available():
            return self._t("light_backend_missing")
        return self._t("light_no_ports")

    def _light_select_port(self) -> bool:
        """Prompt the operator to choose a COM port and open it.

        Returns ``True`` if a port was successfully opened.
        """
        ports = available_ports()
        if not ports:
            QMessageBox.warning(
                self,
                self._t("lighting_menu"),
                self._light_no_ports_message(),
            )
            return False

        labels = [f"{device}  —  {desc}" for device, desc in ports]
        current = self._ensure_light_controller().port or self.preferences.restore_light_port()
        current_idx = next(
            (i for i, (device, _) in enumerate(ports) if device == current), 0
        )
        choice, accepted = QInputDialog.getItem(
            self,
            self._t("light_port"),
            self._t("light_select_port_prompt"),
            labels,
            current_idx,
            editable=False,
        )
        if not accepted:
            return False

        device = ports[labels.index(choice)][0]
        controller = self._ensure_light_controller()
        try:
            controller.open(device)
        except LightControlError as exc:
            QMessageBox.critical(
                self,
                self._t("lighting_menu"),
                self._t("light_port_open_failed", port=device, error=exc),
            )
            return False

        self.preferences.save_light_port(device)
        self.log_message(self._t("light_connected", port=device))
        self.statusBar().showMessage(self._t("light_connected", port=device), 3000)
        return True

    def _light_turn_on(self) -> None:
        controller = self._require_open_light_port()
        if controller is None:
            return
        percent = self.preferences.restore_light_brightness()
        value = _percent_to_value(percent, controller.max_value)
        # 0% saved brightness would make "on" a no-op; fall back to full.
        if value <= 0:
            value = controller.max_value
            percent = 100
        try:
            controller.set_brightness(value)
        except LightControlError as exc:
            self._report_light_error(exc)
            return
        self.log_message(self._t("light_status_on", percent=percent))
        self.statusBar().showMessage(self._t("light_status_on", percent=percent), 3000)

    def _light_turn_off(self) -> None:
        controller = self._require_open_light_port()
        if controller is None:
            return
        try:
            controller.turn_off()
        except LightControlError as exc:
            self._report_light_error(exc)
            return
        self.log_message(self._t("light_status_off"))
        self.statusBar().showMessage(self._t("light_status_off"), 3000)

    def _light_open_brightness_dialog(self) -> None:
        controller = self._require_open_light_port()
        if controller is None:
            return

        start_percent = self.preferences.restore_light_brightness()

        def apply(percent: int) -> None:
            try:
                controller.set_brightness(_percent_to_value(percent, controller.max_value))
            except LightControlError as exc:
                self._report_light_error(exc)

        dialog = BrightnessDialog(
            start_percent, self.current_language, apply, parent=self
        )
        dialog.exec_()

        final_percent = dialog.value()
        self.preferences.save_light_brightness(final_percent)
        self.log_message(self._t("light_brightness_set", percent=final_percent))

    def populate_light_port_menu(self, menu) -> None:
        """Rebuild the dynamic 'Port' submenu with currently available ports."""
        from PyQt5.QtWidgets import QAction, QActionGroup

        menu.clear()
        ports = available_ports()
        if not ports:
            placeholder = QAction(self._light_no_ports_message(), menu)
            placeholder.setEnabled(False)
            menu.addAction(placeholder)
            return

        current = self._ensure_light_controller().port
        group = QActionGroup(menu)
        group.setExclusive(True)
        for device, desc in ports:
            action = QAction(f"{device}  —  {desc}", menu, checkable=True)
            action.setChecked(device == current)
            action.triggered.connect(lambda _checked, d=device: self._light_open_specific_port(d))
            group.addAction(action)
            menu.addAction(action)

    def _light_open_specific_port(self, device: str) -> None:
        """Open *device* directly from the dynamic port submenu."""
        controller = self._ensure_light_controller()
        try:
            controller.open(device)
        except LightControlError as exc:
            QMessageBox.critical(
                self,
                self._t("lighting_menu"),
                self._t("light_port_open_failed", port=device, error=exc),
            )
            return
        self.preferences.save_light_port(device)
        self.log_message(self._t("light_connected", port=device))
        self.statusBar().showMessage(self._t("light_connected", port=device), 3000)

    def _report_light_error(self, exc: Exception) -> None:
        self.log_message(self._t("light_send_failed", error=exc))
        QMessageBox.critical(
            self,
            self._t("lighting_menu"),
            self._t("light_send_failed", error=exc),
        )

    def shutdown_light(self) -> None:
        """Close the serial port during application shutdown (best effort)."""
        controller = getattr(self, "_light_controller", None)
        if controller is not None:
            controller.close()
