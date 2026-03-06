"""GUI workers for the detection pipeline.

This module provides three worker categories:

1. **PipelineBridge** — QObject Signal-Slot bridge that converts core
   pipeline callbacks (from ``threading.Thread`` workers) into PyQt
   signals safe for the GUI main thread.

2. **ShutdownWorker** — QThread that calls the blocking
   ``stop_pipeline()`` off the main thread so the UI stays responsive.

3. **DetectionWorker** — QThread that calls the non-blocking
   ``start_pipeline()`` with bridge callbacks injected.

4. **ModelLoaderWorker / CameraInitWorker** — Unchanged utility workers.

Concurrency Safety
------------------
``PipelineBridge`` callbacks are invoked on **core worker threads**
(AcquisitionWorker, StorageWorker). They MUST only emit signals — no
GUI manipulation, no blocking I/O. The signal-slot mechanism guarantees
delivery to the main thread via Qt's event loop.
"""

from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING

import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from core.types import DetectionItem, DetectionResult, DetectionTask

if TYPE_CHECKING:
    from app.gui.controller import DetectionController
    from core.detection_system import DetectionSystem
    from core.services.model_catalog import ModelCatalog

logger = logging.getLogger(__name__)


# ======================================================================
# Pipeline Bridge (Core → GUI signal router)
# ======================================================================

class PipelineBridge(QObject):
    """Converts core pipeline callbacks into thread-safe PyQt signals.

    Instantiated **once** by ``DetectionController`` and injected into
    ``start_pipeline()`` as ``on_task_captured`` / ``on_task_processed``.

    Signals
    -------
    image_ready : np.ndarray
        Emitted when AcquisitionWorker captures a new BGR frame.
    result_ready : DetectionTask
        Emitted when StorageWorker finishes persisting a result.
        The ``DetectionTask.result`` dict is populated at this point.
    error_occurred : str
        Emitted on unrecoverable pipeline errors.
    """

    image_ready = pyqtSignal(object)    # np.ndarray (BGR) for ImagePanel
    result_ready = pyqtSignal(object)   # DetectionTask with .result populated
    error_occurred = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # Callback hooks (called on core worker threads — emit only!)
    # ------------------------------------------------------------------

    def on_task_captured(self, task: DetectionTask) -> None:
        """Hook injected into AcquisitionWorker.

        Emits the raw BGR ndarray via ``image_ready`` for ImagePanel.
        """
        if task.frame is None:
            return
        try:
            self.image_ready.emit(task.frame)
        except Exception:
            logger.error("PipelineBridge.on_task_captured failed", exc_info=True)

    def on_task_processed(self, task: DetectionTask) -> None:
        """Hook injected into StorageWorker.

        Emits the entire ``DetectionTask`` so the GUI can extract
        ``task.result`` (the raw dict) and build a ``DetectionResult``
        for UI display.
        """
        if task.result is None:
            return
        try:
            self.result_ready.emit(task)
        except Exception:
            logger.error("PipelineBridge.on_task_processed failed", exc_info=True)



# ======================================================================
# Shutdown Worker (blocking stop_pipeline → QThread)
# ======================================================================

class ShutdownWorker(QThread):
    """Executes ``stop_pipeline()`` on a background thread.

    ``stop_pipeline()`` contains ``join()`` calls that block until all
    pipeline workers finish draining.  Running it on the main thread
    would freeze the UI for up to ``timeout`` seconds.

    Signals
    -------
    shutdown_complete : —
        Emitted (from the worker thread) after ``stop_pipeline()``
        returns, regardless of success or failure.
    """

    shutdown_complete = pyqtSignal()

    def __init__(
        self,
        detection_system: DetectionSystem,
        timeout: float = 10.0,
    ) -> None:
        super().__init__()
        self._system = detection_system
        self._timeout = timeout

    def run(self) -> None:
        try:
            self._system.stop_pipeline(timeout=self._timeout)
        except Exception:
            logger.error("ShutdownWorker failed", exc_info=True)
        finally:
            self.shutdown_complete.emit()


# ======================================================================
# Detection Worker (pipeline start proxy)
# ======================================================================

class DetectionWorker(QThread):
    """Lightweight proxy that calls ``start_pipeline()`` with bridge hooks.

    ``start_pipeline()`` itself is non-blocking (it spawns daemon threads
    and returns immediately), so this QThread finishes almost instantly
    after the call.  It exists to catch rare init-time exceptions
    (e.g. camera not found) without crashing the main thread.

    Signals
    -------
    finished : —
        Emitted after ``start_pipeline()`` returns (success or failure).
    error_occurred : str
        Emitted if ``start_pipeline()`` raises an exception.
    """

    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        detection_system: DetectionSystem,
        product: str,
        area: str,
        inference_type: str,
        capture_interval: float = 0.5,
        bridge: PipelineBridge | None = None,
    ) -> None:
        super().__init__()
        self._system = detection_system
        self._product = product
        self._area = area
        self._inference_type = inference_type
        self._interval = capture_interval
        self._bridge = bridge

    def run(self) -> None:
        try:
            self._system.start_pipeline(
                self._product,
                self._area,
                self._inference_type,
                capture_interval=self._interval,
                on_task_captured=(
                    self._bridge.on_task_captured if self._bridge else None
                ),
                on_task_processed=(
                    self._bridge.on_task_processed if self._bridge else None
                ),
            )
        except Exception:
            self.error_occurred.emit(traceback.format_exc())
        finally:
            self.finished.emit()


# ======================================================================
# Utility Workers (unchanged)
# ======================================================================

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
            if not self._controller.has_system():
                _ = self._controller.detection_system

            sys = self._controller.detection_system
            if not self._controller.is_camera_connected():
                if hasattr(sys, "reconnect_camera"):
                    sys.reconnect_camera()

            connected = self._controller.is_camera_connected()
            self.finished.emit(connected)
        except Exception:
            self.finished.emit(False)
