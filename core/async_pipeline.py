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
import time
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
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def running(self) -> bool:
        """Whether the pipeline is currently active."""
        return self._active and not self._stop_event.is_set()

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
        mode: str = "continuous",
        on_task_captured: Optional[Callable[[DetectionTask], None]] = None,
        on_task_processed: Optional[Callable[[DetectionTask], None]] = None,
        on_camera_lost: Optional[Callable[[], None]] = None,
        camera_lost_threshold: int = 5,
        camera_reconnect_attempts: int = 0,
        camera_reconnect_backoff: float = 2.0,
    ) -> None:
        """Start the three-stage async pipeline.

        Args:
            camera: Initialised camera controller.
            detection_system: The pipeline host (see
                ``core.workers.DetectionPipelineHost``; provides
                ``run_inference`` and ``persist_detection``).
            product: Product identifier.
            area: Area identifier.
            inference_type: 'yolo', 'anomalib', or 'fusion'.
            buffer_limit: Max items in the inference queue.
            capture_interval: Seconds between captures (0 = max FPS).
            mode: ``single`` captures and stores one terminal result, while
                ``continuous`` runs until stopped manually.
            on_task_captured: Optional GUI callback per captured frame.
            on_task_processed: Optional GUI callback per stored result.
            on_camera_lost: Optional callback when camera disconnects
                (consecutive capture failures exceed threshold).
            camera_lost_threshold: Consecutive capture failures before the
                camera counts as lost.
            camera_reconnect_attempts: Automatic reconnect attempts before
                declaring the camera lost (0 = disabled).
            camera_reconnect_backoff: Base seconds between reconnect
                attempts (doubles per attempt).

        Raises:
            RuntimeError: If the pipeline is already running.
        """
        with self._lock:
            if (
                self._active
                and self._stop_event.is_set()
                and not self._any_worker_alive()
            ):
                self._clear_worker_refs_locked()

            if self._active:
                raise RuntimeError("Pipeline is already running")

            run_mode = str(mode or "continuous").lower()
            if run_mode not in {"single", "continuous"}:
                raise ValueError(f"Unsupported pipeline mode: {mode}")

            self._stop_event.clear()
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
                on_camera_lost=on_camera_lost,
                stop_event=self._stop_event,
                mode=run_mode,
                lost_threshold=camera_lost_threshold,
                reconnect_attempts=camera_reconnect_attempts,
                reconnect_backoff=camera_reconnect_backoff,
            )
            self._inf_worker = InferenceWorker(
                in_queue=self._inference_queue,
                out_queue=self._io_queue,
                detection_system=detection_system,
                stop_event=self._stop_event,
            )
            self._sto_worker = StorageWorker(
                in_queue=self._io_queue,
                detection_system=detection_system,
                on_task_processed=on_task_processed,
                stop_event=self._stop_event,
                mode=run_mode,
                drop_queue=self._inference_queue,
            )

            # Start consumer-first to drain immediately
            self._sto_worker.start()
            self._inf_worker.start()
            self._acq_worker.start()
            self._active = True

            logger.info(
                "Pipeline started: %s/%s/%s (mode=%s, buffer=%d, interval=%.3fs)",
                product, area, inference_type, run_mode, buffer_limit, capture_interval,
            )

    def stop(self, timeout: float = 10.0) -> None:
        """Stop workers without letting one blocked backend freeze the GUI.

        Args:
            timeout: Total seconds to wait for all workers to finish.
        """
        with self._lock:
            if not self._active:
                return

            logger.info("Stopping pipeline...")
            self._stop_event.set()

            if self._acq_worker and self._acq_worker.is_alive():
                self._acq_worker.stop()

            self._wake_workers_for_shutdown()

            workers = (self._acq_worker, self._inf_worker, self._sto_worker)
            deadline = time.monotonic() + max(0.0, timeout)
            for worker in workers:
                if not worker or not worker.is_alive():
                    continue
                remaining = max(0.0, deadline - time.monotonic())
                if remaining > 0:
                    worker.join(timeout=remaining)
                if worker.is_alive():
                    logger.warning(
                        "Worker %s still running after %.1fs stop budget; "
                        "detaching so UI can recover",
                        worker.name,
                        timeout,
                    )

            stats = self.stats()
            logger.info(
                "Pipeline stopped — captured: %d, dropped: %d, saved: %d",
                stats.get("frames_captured", 0),
                stats.get("frames_dropped", 0),
                stats.get("tasks_saved", 0),
            )

            self._active = False
            self._clear_worker_refs_locked()

    def _any_worker_alive(self) -> bool:
        """Return whether any managed worker thread is still alive."""
        return any(
            worker is not None and worker.is_alive()
            for worker in (self._acq_worker, self._inf_worker, self._sto_worker)
        )

    def _clear_worker_refs_locked(self) -> None:
        """Clear worker and queue references while the manager lock is held."""
        self._active = False
        self._acq_worker = None
        self._inf_worker = None
        self._sto_worker = None
        self._inference_queue = None
        self._io_queue = None

    def _wake_workers_for_shutdown(self) -> None:
        """Best-effort wake-up for workers blocked on queues during shutdown."""
        if self._inference_queue is not None:
            try:
                dropped = self._inference_queue.clear()
                if dropped:
                    logger.info(
                        "Dropped %d queued inference tasks during stop",
                        dropped,
                    )
                self._inference_queue.put(DetectionTask.poison_pill())
            except Exception:
                logger.exception("Failed to wake inference queue during stop")

        if self._io_queue is not None:
            try:
                dropped = self._drain_io_queue_for_shutdown()
                if dropped:
                    logger.info(
                        "Dropped %d queued storage tasks during stop",
                        dropped,
                    )
                self._io_queue.put_nowait(DetectionTask.poison_pill())
            except queue.Full:
                logger.warning("IO queue full during stop; storage wake-up skipped")
            except Exception:
                logger.exception("Failed to wake IO queue during stop")

    def _drain_io_queue_for_shutdown(self) -> int:
        """Remove queued storage tasks so stop is not blocked by stale work."""
        if self._io_queue is None:
            return 0

        dropped = 0
        while True:
            try:
                self._io_queue.get_nowait()
            except queue.Empty:
                return dropped
            else:
                dropped += 1
                try:
                    self._io_queue.task_done()
                except ValueError:
                    logger.debug("IO queue task_done mismatch during stop", exc_info=True)

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
            "tasks_dropped": (
                self._inf_worker.dropped_count if self._inf_worker else 0
            ),
        }
