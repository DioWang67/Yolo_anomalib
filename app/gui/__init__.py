"""GUI package exposing the PyQt main window entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .controller import DetectionController
    from .main_window import DetectionSystemGUI, main

# ModelCatalog is now in core.services, not exposed here to encourage direct import
__all__ = ["DetectionController", "DetectionSystemGUI", "main"]


def __getattr__(name: str) -> Any:
    """Load heavy GUI entrypoints only when callers explicitly request them."""
    if name == "DetectionController":
        from .controller import DetectionController

        value = DetectionController
    elif name in {"DetectionSystemGUI", "main"}:
        from .main_window import DetectionSystemGUI, main

        value = DetectionSystemGUI if name == "DetectionSystemGUI" else main
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
