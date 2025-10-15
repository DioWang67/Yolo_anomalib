"""GUI package exposing the PyQt main window entrypoints."""

from .controller import DetectionController, ModelCatalog
from .main_window import DetectionSystemGUI, main

__all__ = ["DetectionController", "DetectionSystemGUI", "ModelCatalog", "main"]
