"""Backward-compatible entrypoint for launching the PyQt GUI."""

from __future__ import annotations

# Expose DetectionSystem for tests that patch GUI.DetectionSystem
from core.detection_system import DetectionSystem  # type: ignore
from app.gui import DetectionSystemGUI, main

__all__ = ["DetectionSystemGUI", "DetectionSystem", "main"]


if __name__ == "__main__":
    main()
