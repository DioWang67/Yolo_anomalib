"""Backward-compatible entrypoint for launching the PyQt GUI."""

from __future__ import annotations

import multiprocessing
import os
import sys

# PyInstaller freeze support must be called before any other code runs.
# Without this, torch's use of multiprocessing spawns a second process on
# Windows that re-executes this script, causing a duplicate GUI window.
multiprocessing.freeze_support()

# When running as a PyInstaller bundle, point TORCH_HOME at the bundled
# timm weight cache so Patchcore backbone loads offline (no internet needed).
if getattr(sys, "frozen", False):
    _timm_cache = os.path.join(getattr(sys, "_MEIPASS", ""), "timm_cache")
    if os.path.isdir(_timm_cache):
        os.environ.setdefault("TORCH_HOME", _timm_cache)

from app.gui import DetectionSystemGUI, main  # noqa: E402

# Expose DetectionSystem for tests that patch GUI.DetectionSystem
from core.detection_system import DetectionSystem  # noqa: E402

__all__ = ["DetectionSystemGUI", "DetectionSystem", "main"]


if __name__ == "__main__":
    main()
