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
import threading
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
    camera_disconnected = pyqtSignal()  # Camera lost during pipeline

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active_run_id: int | None = None

    def begin_run(self, run_id: int) -> None:
        """Mark a pipeline run as the only run allowed to emit GUI updates."""
        with self._lock:
            self._active_run_id = run_id

    def end_run(self, run_id: int | None = None) -> None:
        """Invalidate callbacks from a stopped pipeline run."""
        with self._lock:
            if run_id is None or self._active_run_id == run_id:
                self._active_run_id = None

    def _accepts_run(self, run_id: int | None) -> bool:
        """Return True if a worker callback belongs to the active run."""
        with self._lock:
            return run_id is not None and self._active_run_id == run_id

    # ------------------------------------------------------------------
    # Callback hooks (called on core worker threads — emit only!)
    # ------------------------------------------------------------------

    def on_task_captured(self, task: DetectionTask, run_id: int | None = None) -> None:
        """Hook injected into AcquisitionWorker.

        Emits the raw BGR ndarray via ``image_ready`` for ImagePanel.
        """
        if not self._accepts_run(run_id):
            return
        if task.frame is None:
            return
        try:
            self.image_ready.emit(task.frame)
        except Exception:
            logger.error("PipelineBridge.on_task_captured failed", exc_info=True)

    def on_task_processed(self, task: DetectionTask, run_id: int | None = None) -> None:
        """Hook injected into StorageWorker.

        Emits the entire ``DetectionTask`` so the GUI can extract
        ``task.result`` (the raw dict) and build a ``DetectionResult``
        for UI display.
        """
        if not self._accepts_run(run_id):
            return
        if task.result is None:
            return
        try:
            self.result_ready.emit(task)
        except Exception:
            logger.error("PipelineBridge.on_task_processed failed", exc_info=True)

    def on_camera_lost(self, run_id: int | None = None) -> None:
        """Hook invoked by AcquisitionWorker when camera appears disconnected.

        Called on the AcquisitionWorker thread — must only emit signals.
        """
        if not self._accepts_run(run_id):
            return
        try:
            self.camera_disconnected.emit()
        except Exception:
            logger.error("PipelineBridge.on_camera_lost failed", exc_info=True)


# ======================================================================
# Shutdown Worker (blocking stop_pipeline → QThread)
# ======================================================================

class ShutdownWorker(QThread):
    """Executes ``stop_pipeline()`` on a background thread.

    ``stop_pipeline()`` uses a bounded total timeout. Backends such as ONNX
    Runtime can occasionally block inside one inference call; the pipeline
    must still release the GUI and stop camera acquisition promptly.

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
        timeout: float = 3.0,
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
        mode: str = "single",
        bridge: PipelineBridge | None = None,
        run_id: int | None = None,
    ) -> None:
        super().__init__()
        self._system = detection_system
        self._product = product
        self._area = area
        self._inference_type = inference_type
        self._interval = capture_interval
        self._mode = mode
        self._bridge = bridge
        self._run_id = run_id
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation before the pipeline is fully started."""
        self._cancel_event.set()

    def cancel_requested(self) -> bool:
        """Return True when the GUI has requested startup cancellation."""
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            if self.cancel_requested():
                return
            self._system.start_pipeline(
                self._product,
                self._area,
                self._inference_type,
                capture_interval=self._interval,
                mode=self._mode,
                on_task_captured=(
                    (lambda task: self._bridge.on_task_captured(task, self._run_id))
                    if self._bridge else None
                ),
                on_task_processed=(
                    (lambda task: self._bridge.on_task_processed(task, self._run_id))
                    if self._bridge else None
                ),
                on_camera_lost=(
                    (lambda: self._bridge.on_camera_lost(self._run_id))
                    if self._bridge else None
                ),
                cancel_cb=self.cancel_requested,
            )
            if self.cancel_requested():
                try:
                    self._system.stop_pipeline(timeout=1.0)
                except Exception:
                    logger.error("Failed to stop pipeline after startup cancel", exc_info=True)
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
