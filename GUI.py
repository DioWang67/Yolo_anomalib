"""Backward-compatible entrypoint for launching the PyQt GUI."""

from __future__ import annotations

from app.gui import DetectionSystemGUI, main

# Expose DetectionSystem for tests that patch GUI.DetectionSystem
from core.detection_system import DetectionSystem  # type: ignore

__all__ = ["DetectionSystemGUI", "DetectionSystem", "main"]


if __name__ == "__main__":
    main()
