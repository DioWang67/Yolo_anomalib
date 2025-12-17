"""GUI package exposing the PyQt main window entrypoints."""

from .controller import DetectionController
from .main_window import DetectionSystemGUI, main

# ModelCatalog is now in core.services, not exposed here to encourage direct import
__all__ = ["DetectionController", "DetectionSystemGUI", "main"]
