"""Camera preview package with Qt window, metrics, and CLI helpers."""

from .app import main, parse_args
from .metrics import AdaptiveCalibrator, compute_change_metrics
from .source import CameraSource
from .window import CameraPreviewWindow

__all__ = [
    "AdaptiveCalibrator",
    "CameraPreviewWindow",
    "CameraSource",
    "compute_change_metrics",
    "main",
    "parse_args",
]
