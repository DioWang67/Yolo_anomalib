from __future__ import annotations

import threading
import traceback
from typing import Any, TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

if TYPE_CHECKING:
    from core.detection_system import DetectionSystem


class DetectionWorker(QThread):
    """Background detection runner that delegates to DetectionSystem."""

    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detection_system: "DetectionSystem",
        product: str,
        area: str,
        inference_type: str,
        frame: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self._detection_system = detection_system
        self._product = product
        self._area = area
        self._inference_type = inference_type
        self._frame = frame
        self._stop_event = threading.Event()

    def cancel(self) -> None:
        try:
            self._stop_event.set()
        except Exception:
            pass

    def run(self) -> None:
        try:
            result = self._detection_system.detect(
                self._product,
                self._area,
                self._inference_type,
                frame=self._frame,
                cancel_cb=self._stop_event.is_set,
            )
            if not self._stop_event.is_set():
                self.result_ready.emit(result)
        except Exception:
            self.error_occurred.emit(traceback.format_exc())
