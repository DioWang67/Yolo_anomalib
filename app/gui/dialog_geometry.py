"""Screen-aware sizing helpers for desktop dialogs."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QApplication, QDialog, QWidget


@dataclass(frozen=True)
class DialogGeometry:
    """Calculated initial and minimum sizes in device-independent pixels."""

    initial: QSize
    minimum: QSize


def calculate_dialog_geometry(
    *,
    preferred: QSize,
    minimum: QSize,
    available: QSize,
    margin: int = 32,
) -> DialogGeometry:
    """Bound a dialog to one screen while retaining a usable minimum size."""
    usable_width = max(320, available.width() - max(0, margin))
    usable_height = max(280, available.height() - max(0, margin))
    initial = QSize(
        min(max(preferred.width(), 320), usable_width),
        min(max(preferred.height(), 280), usable_height),
    )
    safe_minimum = QSize(
        min(max(minimum.width(), 320), initial.width()),
        min(max(minimum.height(), 280), initial.height()),
    )
    return DialogGeometry(initial=initial, minimum=safe_minimum)


def configure_responsive_dialog(
    dialog: QDialog,
    *,
    preferred: tuple[int, int],
    minimum: tuple[int, int] = (560, 420),
    parent: QWidget | None = None,
) -> DialogGeometry:
    """Apply a DPI-aware size bounded by the dialog's current screen."""
    screen = None
    anchor = parent or dialog.parentWidget()
    if anchor is not None:
        screen = QApplication.screenAt(anchor.frameGeometry().center())
    if screen is None:
        screen = QApplication.primaryScreen()
    available = screen.availableGeometry().size() if screen is not None else QSize(1280, 720)
    geometry = calculate_dialog_geometry(
        preferred=QSize(*preferred),
        minimum=QSize(*minimum),
        available=available,
    )
    dialog.setMinimumSize(geometry.minimum)
    dialog.resize(geometry.initial)
    return geometry
