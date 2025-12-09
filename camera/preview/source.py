from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from core.config import DetectionConfig

from ..camera_controller import CameraController


class CameraSource:
    """Unified wrapper around CameraController / OpenCV capture."""

    def __init__(
        self,
        use_opencv: bool,
        config_path: Path,
        camera_index: int,
        logger: logging.Logger,
    ) -> None:
        self._controller: CameraController | None = None
        self._capture: cv2.VideoCapture | None = None
        self._use_opencv = use_opencv
        self._camera_index = camera_index
        self._config_path = config_path
        self._logger = logger

        self._open()

    def _open(self) -> None:
        if self._use_opencv:
            self._logger.info("Opening OpenCV camera index %s", self._camera_index)
            cap = cv2.VideoCapture(self._camera_index)
            if not cap or not cap.isOpened():
                raise RuntimeError(f"Failed to open OpenCV camera index {self._camera_index}")
            self._capture = cap
        else:
            self._logger.info("Opening CameraController with config: %s", self._config_path)
            config = DetectionConfig.from_yaml(str(self._config_path))
            controller = CameraController(config)
            controller.initialize()
            self._controller = controller

    def reopen(self) -> None:
        self.shutdown()
        self._open()

    def read(self) -> np.ndarray | None:
        """Return the latest frame in BGR format, or None if unavailable."""
        if self._controller:
            return self._controller.capture_frame()
        if self._capture:
            ok, frame = self._capture.read()
            if not ok:
                return None
            return frame
        return None

    def shutdown(self) -> None:
        if self._controller:
            try:
                self._controller.shutdown()
            except Exception:  # pragma: no cover
                pass
            self._controller = None
        if self._capture:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover
                pass
            self._capture = None
