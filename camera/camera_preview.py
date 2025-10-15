"""Backward-compatible entrypoint for the camera preview CLI/app."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.preview import (
    AdaptiveCalibrator,
    CameraPreviewWindow,
    CameraSource,
    compute_change_metrics,
    main,
    parse_args,
)

__all__ = [
    "AdaptiveCalibrator",
    "CameraPreviewWindow",
    "CameraSource",
    "compute_change_metrics",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    raise SystemExit(main())
