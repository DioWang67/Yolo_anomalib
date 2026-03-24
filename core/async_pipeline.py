"""Async Producer-Consumer pipeline manager.

Extracted from DetectionSystem to separate the async pipeline lifecycle
(start/stop/stats) from the synchronous detection logic.

Usage::

    manager = AsyncPipelineManager(system)
    manager.start(product, area, "yolo", capture_interval=0.033)
    stats = manager.stats()
    manager.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional

from core.queues import OverwriteQueue
from core.types import DetectionTask
from core.workers import AcquisitionWorker, InferenceWorker, StorageWorker

if TYPE_CHECKING:
    from camera.camera_controller import CameraController


logger = logging.getLogger(__name__)


class AsyncPipelineManager:
    """Manages the three-stage async detection pipeline.

    Lifecycle::

        AcquisitionWorker ─(OverwriteQueue)─> InferenceWorker ─(Queue)─> StorageWorker

    All worker management, queue creation, and graceful shutdown are
    encapsulated here so that ``DetectionSystem`` stays focused on
    inference logic.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: bool = False

        self._inference_queue: Optional[OverwriteQueue[DetectionTask]] = None
        self._io_queue: Optional[queue.Queue[DetectionTask]] = None
        self._acq_worker: Optional[AcquisitionWorker] = None
        self._inf_worker: Optional[InferenceWorker] = None
        self._sto_worker: Optional[StorageWorker] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether the pipeline is currently active."""
        return self._active

    def start(
        self,
        *,
        camera: CameraController,
        detection_system: Any,
        product: str,
        area: str,
        inference_type: str = "yolo",
        buffer_limit: int = 10,
        capture_interval: float = 0.0,
        on_task_captured: Optional[Callable[[DetectionTask], None]] = None,
        on_task_processed: Optional[Callable[[DetectionTask], None]] = None,
    ) -> None:
        """Start the three-stage async pipeline.

        Args:
            camera: Initialised camera controller.
            detection_system: The DetectionSystem instance (provides
                ``_run_inference``, ``_execute_pipeline``, etc.).
            product: Product identifier.
            area: Area identifier.
            inference_type: 'yolo', 'anomalib', or 'fusion'.
            buffer_limit: Max items in the inference queue.
            capture_interval: Seconds between captures (0 = max FPS).
            on_task_captured: Optional GUI callback per captured frame.
            on_task_processed: Optional GUI callback per stored result.

        Raises:
            RuntimeError: If the pipeline is already running.
        """
        with self._lock:
            if self._active:
                raise RuntimeError("Pipeline is already running")

            self._inference_queue = OverwriteQueue(
                maxlen=buffer_limit, name="inference_queue"
            )
            self._io_queue = queue.Queue(maxsize=50)

            self._acq_worker = AcquisitionWorker(
                camera=camera,
                out_queue=self._inference_queue,
                product=product,
                area=area,
                inference_type=inference_type,
                capture_interval=capture_interval,
                on_task_captured=on_task_captured,
            )
            self._inf_worker = InferenceWorker(
                in_queue=self._inference_queue,
                out_queue=self._io_queue,
                detection_system=detection_system,
            )
            self._sto_worker = StorageWorker(
                in_queue=self._io_queue,
                detection_system=detection_system,
                on_task_processed=on_task_processed,
            )

            # Start consumer-first to drain immediately
            self._sto_worker.start()
            self._inf_worker.start()
            self._acq_worker.start()
            self._active = True

            logger.info(
                "Pipeline started: %s/%s/%s (buffer=%d, interval=%.3fs)",
                product, area, inference_type, buffer_limit, capture_interval,
            )

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully shut down all workers via poison-pill propagation.

        Args:
            timeout: Max seconds to wait for each worker to finish.
        """
        with self._lock:
            if not self._active:
                return

            logger.info("Stopping pipeline...")

            if self._acq_worker and self._acq_worker.is_alive():
                self._acq_worker.stop()

            for worker in (self._acq_worker, self._inf_worker, self._sto_worker):
                if worker and worker.is_alive():
                    worker.join(timeout=timeout)
                    if worker.is_alive():
                        logger.warning(
                            "Worker %s did not stop within %.1fs",
                            worker.name, timeout,
                        )

            stats = self.stats()
            logger.info(
                "Pipeline stopped — captured: %d, dropped: %d, saved: %d",
                stats.get("frames_captured", 0),
                stats.get("frames_dropped", 0),
                stats.get("tasks_saved", 0),
            )

            self._active = False
            self._acq_worker = None
            self._inf_worker = None
            self._sto_worker = None
            self._inference_queue = None
            self._io_queue = None

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of pipeline counters for monitoring."""
        return {
            "pipeline_running": self._active,
            "frames_captured": (
                self._acq_worker.frame_count if self._acq_worker else 0
            ),
            "frames_dropped": (
                self._inference_queue.dropped_count
                if self._inference_queue else 0
            ),
            "inference_queue_size": (
                self._inference_queue.qsize() if self._inference_queue else 0
            ),
            "io_queue_size": (
                self._io_queue.qsize() if self._io_queue else 0
            ),
            "tasks_saved": (
                self._sto_worker.saved_count if self._sto_worker else 0
            ),
        }
