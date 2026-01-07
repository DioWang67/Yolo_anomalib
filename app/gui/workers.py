from __future__ import annotations

import threading
import time
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.types import DetectionItem, DetectionResult, DetectionStatus

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

                # 2. Run Detection
                t0 = time.time()
                # Assuming system.detect still returns dict for now, we wrap it.
                # In a full refactor, system.detect should return DetectionResult directly.
                # For now, we adapt here.
                # Assuming system.detect still returns dict for now, we wrap it.
                # In a full refactor, system.detect should return DetectionResult directly.
                # For now, we adapt here.
                result_dict = self._detection_system.detect(
                    self._product,
                    self._area,
                    self._inference_type,
                    frame=image
                )
                latency = time.time() - t0

                # 3. Adapt to Strong Types
                status_str = result_dict.get("status", "ERROR")
                # Map string to Literal safely
                status: DetectionStatus
                if status_str == "PASS":
                    status = "PASS"
                elif status_str == "FAIL":
                    status = "FAIL"
                else:
                    status = "ERROR"

                items = []
                for d in result_dict.get("detections", []):
                    bbox = d.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    if isinstance(bbox, list):
                        bbox = tuple(bbox)
                    items.append(DetectionItem(
                        label=d.get("class", "unknown"),
                        confidence=float(d.get("confidence", 0.0)),
                        bbox_xyxy=bbox,
                        metadata=d # Keep raw dict in metadata if needed
                    ))

                res = DetectionResult(
                    status=status,
                    items=items,
                    latency=latency,
                    timestamp=time.time(),
                    frame_id=frame_id,
                    image_path=result_dict.get("image_path"),
                    metadata={k: v for k, v in result_dict.items()
                              if k not in ["detections", "status", "image_path"]}
                )

                # 4. Emit Result
                self.result_ready.emit(res)

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
