"""Backward-compatible entrypoint for launching the PyQt GUI."""

from __future__ import annotations

from app.gui import DetectionSystemGUI, main

__all__ = ["DetectionSystemGUI", "main"]


if __name__ == "__main__":
    main()
