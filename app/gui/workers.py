from __future__ import annotations

import threading
import time
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.types import DetectionResult

if TYPE_CHECKING:
    from app.gui.controller import DetectionController
    from core.detection_system import DetectionSystem
    from core.services.model_catalog import ModelCatalog


class DetectionWorker(QThread):
    """
    Background detection runner that delegates to DetectionSystem.
    Strictly emits DetectionResult for logic and np.ndarray for UI preview.
    """

    result_ready = pyqtSignal(object)  # Emits DetectionResult
    image_ready = pyqtSignal(object)   # Emits np.ndarray (BGR)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detection_system: DetectionSystem,
        product: str,
        area: str,
        inference_type: str,
        frame: np.ndarray | None = None,
        continuous: bool = False,
    ) -> None:
        super().__init__()
        self._detection_system = detection_system
        self._product = product
        self._area = area
        self._inference_type = inference_type
        self._frame = frame
        self._continuous = continuous
        self._stop_event = threading.Event()

    def cancel(self) -> None:
        """Atomic stop request."""
        self._stop_event.set()

    def run(self) -> None:
        try:
            # Re-init system with selected config
            self._detection_system.load_model_configs(self._product, self._area, self._inference_type)
            frame_id = 0
            while not self._stop_event.is_set():
                if self._frame is not None:
                    # Single shot with provided image
                    image = self._frame
                    # For single shot provided image, we stop after one iteration
                    self.cancel()
                else:
                    # Camera capture
                    image = self._detection_system.capture_image()
                    if image is None:
                        if not self._continuous:
                            self.error_occurred.emit("無法擷取影像")
                        time.sleep(0.1)
                        continue

                # 1. Emit Image immediately for UI Preview
                self.image_ready.emit(image)

                # 2. Run Detection — returns DetectionResult directly
                result = self._detection_system.detect(
                    self._product,
                    self._area,
                    self._inference_type,
                    frame=image
                )
                result.frame_id = frame_id

                # 3. Emit Result
                self.result_ready.emit(result)

                frame_id += 1

                if not self._continuous and self._frame is None:
                    break

        except Exception:
            self.error_occurred.emit(traceback.format_exc())

        finally:
            self.finished.emit()


class ModelLoaderWorker(QThread):
    """Background worker to load model catalog."""

    models_ready = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, catalog: ModelCatalog) -> None:
        super().__init__()
        self._catalog = catalog

    def run(self) -> None:
        try:
            self._catalog.refresh()
            self.models_ready.emit()
        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class CameraInitWorker(QThread):
    """Background worker to initialize camera."""

    finished = pyqtSignal(bool)

    def __init__(self, controller: DetectionController) -> None:
        super().__init__()
        self._controller = controller

    def run(self) -> None:
        try:
            # Trigger system lazy load / init
            if not self._controller.has_system():
                _ = self._controller.detection_system

            # Explicitly call init camera if such method exists,
            # or rely on system init.
            # Here we assume checking connection status might trigger reconnect if auto-connect is on,
            # or we explicitly call reconnect.
            sys = self._controller.detection_system

            # Check if camera is already connected via system status or controller check
            if not self._controller.is_camera_connected():
                if hasattr(sys, "reconnect_camera"):
                     sys.reconnect_camera()

            connected = self._controller.is_camera_connected()
            self.finished.emit(connected)
        except Exception:
            self.finished.emit(False)
