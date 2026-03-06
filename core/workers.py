"""Pipeline workers for the Producer-Consumer detection architecture.

This module defines four worker classes that decompose the synchronous
``DetectionSystem.detect()`` flow into three concurrent stages:

    AcquisitionWorker  ──▶  InferenceWorker  ──▶  StorageWorker
         (camera)        inference_queue       io_queue      (disk I/O)

Design Decisions
----------------
* **BaseWorker** enforces a ``try-finally`` contract: even if ``process()``
  throws an unhandled exception, the worker **always** sends a poison pill
  to its downstream queue before dying.  This prevents deadlocks during
  shutdown.

* **AcquisitionWorker** is a standalone ``Thread`` (no ``in_queue``) that
  loops on ``camera.capture_frame()``.  It is stopped via a
  ``threading.Event`` rather than a poison pill because it is the *source*
  of the pipeline.

* Workers catch ``Exception`` (not ``BaseException``) *per-item* inside the
  processing loop.  A single bad frame logs an error but does **not** kill
  the thread — the loop continues with the next task.

Concurrency Safety
------------------
No worker holds a lock across a blocking queue operation.  All shared state
is limited to thread-safe queues and ``threading.Event``.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional

from core.queues import OverwriteQueue
from core.types import DetectionTask

if TYPE_CHECKING:
    from camera.camera_controller import CameraController

logger = logging.getLogger(__name__)


# ======================================================================
# Base Worker
# ======================================================================

class BaseWorker(threading.Thread):
    """Base class for queue-consuming pipeline workers.

    Subclasses must override :meth:`process`.  The ``run()`` loop
    automatically handles poison-pill detection and guaranteed downstream
    propagation.

    Parameters
    ----------
    in_queue : queue.Queue | OverwriteQueue
        Source queue to pull ``DetectionTask`` items from.
    out_queue : queue.Queue | OverwriteQueue | None
        Downstream queue to push processed tasks into.  ``None`` for
        terminal workers (e.g. ``StorageWorker``).
    name : str
        Thread name (used in log messages).
    """

    def __init__(
        self,
        in_queue: queue.Queue[DetectionTask] | OverwriteQueue[DetectionTask],
        out_queue: queue.Queue[DetectionTask] | OverwriteQueue[DetectionTask] | None = None,
        *,
        name: str = "BaseWorker",
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._logger = logging.getLogger(f"{__name__}.{name}")

    # ------------------------------------------------------------------
    # Subclass API
    # ------------------------------------------------------------------

    def process(self, task: DetectionTask) -> None:
        """Handle a single task.  Override in subclasses.

        The implementation may mutate *task* in-place (e.g. attach inference
        results) and/or push it to ``self.out_queue``.

        Raises
        ------
        Exception
            Any exception is caught by :meth:`run`, logged as an error,
            and the loop continues with the next task.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main loop: pull tasks, delegate to ``process()``, propagate pills.

        **Contract**: the ``finally`` block **always** sends a poison pill
        to ``out_queue`` so downstream workers can shut down even if this
        thread crashes.
        """
        self._logger.info("Worker started")
        try:
            while True:
                # ---- Blocking get from upstream queue ----
                try:
                    task = self.in_queue.get(timeout=1.0)
                except (queue.Empty, TimeoutError):
                    # Periodic wake-up; allows the thread to be joined
                    # even if no poison pill ever arrives (defensive).
                    continue

                # ---- Poison pill → exit gracefully ----
                if task.is_poison_pill:
                    self._logger.info("Poison pill received — shutting down")
                    break

                # ---- Process one task ----
                try:
                    self.process(task)
                except Exception:
                    self._logger.error(
                        "Unhandled error processing task %s",
                        task.task_id,
                        exc_info=True,
                    )
                    # Mark the failed task so StorageWorker can persist the
                    # error record rather than silently dropping it.
                    task.error = task.error or "Worker process() raised an exception"
                    if self.out_queue is not None:
                        try:
                            # Use timeout to prevent deadlock if downstream is stuck
                            self.out_queue.put(task, timeout=2.0)
                        except queue.Full:
                            self._logger.error(
                                "out_queue full, dropping error task %s to prevent deadlock",
                                task.task_id
                            )
                        except Exception:
                            self._logger.error(
                                "Failed to forward error task to out_queue",
                                exc_info=True,
                            )

        except Exception:
            # Catastrophic — something outside the per-task try went wrong.
            self._logger.critical(
                "Worker crashed unexpectedly", exc_info=True,
            )
        finally:
            # ---- GUARANTEED poison pill propagation ----
            if self.out_queue is not None:
                try:
                    # Give downstream time to drain, but never block forever
                    self.out_queue.put(DetectionTask.poison_pill(), timeout=5.0)
                    self._logger.info(
                        "Poison pill forwarded to downstream queue"
                    )
                except queue.Full:
                    self._logger.critical(
                        "CRITICAL: out_queue is permanently full, poison pill dropped!"
                    )
                except Exception:
                    self._logger.error(
                        "CRITICAL: failed to send poison pill downstream",
                        exc_info=True,
                    )
            self._logger.info("Worker stopped")


# ======================================================================
# Acquisition Worker (Producer)
# ======================================================================

class AcquisitionWorker(threading.Thread):
    """Continuously captures frames from the camera and enqueues tasks.

    This worker is the *source* of the pipeline — it has no ``in_queue``.
    It is stopped by setting ``stop()`` which signals an internal
    ``threading.Event``.

    Parameters
    ----------
    camera : CameraController
        Initialised camera instance.
    out_queue : OverwriteQueue[DetectionTask]
        The inference queue.  Uses ``OverwriteQueue`` so that slow
        inference automatically drops old frames.
    product : str
        Current product identifier.
    area : str
        Current area identifier.
    inference_type : str
        Inference mode ('yolo', 'anomalib', 'fusion').
    capture_interval : float
        Minimum seconds between captures (0 = as fast as possible).
    """

    def __init__(
        self,
        camera: CameraController,
        out_queue: OverwriteQueue[DetectionTask],
        *,
        product: str,
        area: str,
        inference_type: str = "yolo",
        capture_interval: float = 0.0,
        name: str = "AcquisitionWorker",
        on_task_captured: Optional[Callable[[DetectionTask], None]] = None,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.camera = camera
        self.out_queue = out_queue
        self.product = product
        self.area = area
        self.inference_type = inference_type
        self.capture_interval = capture_interval
        self._on_task_captured = on_task_captured
        self._stop_event = threading.Event()
        self._logger = logging.getLogger(f"{__name__}.{name}")
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public control
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Signal the worker to stop after the current capture cycle."""
        self._stop_event.set()

    @property
    def frame_count(self) -> int:
        """Total frames successfully captured and enqueued."""
        return self._frame_count

    # ------------------------------------------------------------------
    # Thread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self._logger.info("Acquisition started (interval=%.3fs)", self.capture_interval)
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.camera.capture_frame()
                except Exception:
                    self._logger.error("Camera capture raised exception", exc_info=True)
                    # Brief back-off before retrying — avoids busy-spin on
                    # persistent hardware errors.
                    self._stop_event.wait(0.5)
                    continue

                if frame is None:
                    self._logger.warning(
                        "Camera returned None — skipping frame "
                        "(possible cable disconnect)"
                    )
                    self._stop_event.wait(0.1)
                    continue

                task = DetectionTask(
                    task_id=uuid.uuid4().hex[:12],
                    timestamp=time.time(),
                    product=self.product,
                    area=self.area,
                    inference_type=self.inference_type,
                    frame=frame,
                )
                # Put into queue (OverwriteQueue won't block)
                self.out_queue.put(task)
                self._frame_count += 1

                # Trigger GUI callback if provided
                if self._on_task_captured:
                    try:
                        self._on_task_captured(task)
                    except Exception:
                        self._logger.error("Error in on_task_captured callback", exc_info=True)

                if self.capture_interval > 0:
                    self._stop_event.wait(self.capture_interval)

        except Exception:
            self._logger.critical("AcquisitionWorker crashed", exc_info=True)
        finally:
            # Send poison pill so InferenceWorker knows no more frames
            # are coming and can begin its own shutdown sequence.
            try:
                self.out_queue.put(DetectionTask.poison_pill())
                self._logger.info("Poison pill sent to inference queue")
            except Exception:
                self._logger.error(
                    "CRITICAL: failed to send poison pill", exc_info=True,
                )
            self._logger.info("Acquisition stopped (captured %d frames)", self._frame_count)


# ======================================================================
# Inference Worker
# ======================================================================

class InferenceWorker(BaseWorker):
    """Runs model inference on each frame and forwards results downstream.

    Parameters
    ----------
    in_queue : OverwriteQueue[DetectionTask]
        The inference queue (fed by ``AcquisitionWorker``).
    out_queue : queue.Queue[DetectionTask]
        The I/O queue (consumed by ``StorageWorker``).
    detection_system : DetectionSystem
        The existing system instance — provides ``_run_inference``,
        ``_prepare_resources``, ``model_manager``, and ``config``.
    """

    def __init__(
        self,
        in_queue: OverwriteQueue[DetectionTask],
        out_queue: queue.Queue[DetectionTask],
        detection_system: Any,
        *,
        name: str = "InferenceWorker",
    ) -> None:
        super().__init__(in_queue, out_queue, name=name)
        self._system = detection_system

    def process(self, task: DetectionTask) -> None:
        """Execute inference and attach results to the task."""
        t0 = time.time()

        try:
            result = self._system._run_inference(
                task.frame,
                task.product,
                task.area,
                task.inference_type,
                self._logger,
            )
        except Exception as exc:
            self._logger.error(
                "Inference engine raised for task %s: %s",
                task.task_id, exc, exc_info=True,
            )
            result = {
                "status": "ERROR",
                "error": f"Inference exception: {exc}",
                "detections": [],
            }

        task.result = result
        task.error = result.get("error")

        latency = time.time() - t0
        self._logger.debug(
            "Task %s inferred in %.3fs (status=%s)",
            task.task_id, latency, result.get("status"),
        )

        # Forward to StorageWorker
        if self.out_queue is not None:
            try:
                # Use timeout to prevent deadlock if StorageWorker is stuck
                self.out_queue.put(task, timeout=2.0)
            except queue.Full:
                self._logger.error(
                    "IO queue is full. Dropping inference result for task %s!",
                    task.task_id
                )


# ======================================================================
# Storage Worker (Consumer)
# ======================================================================

class StorageWorker(BaseWorker):
    """Persists inference results to disk (images + Excel).

    This is the terminal stage of the pipeline — it has no
    ``out_queue``.  Being I/O-bound, it runs in a dedicated thread so
    that slow disk writes never block camera acquisition or inference.

    Parameters
    ----------
    in_queue : queue.Queue[DetectionTask]
        The I/O queue (fed by ``InferenceWorker``).
    detection_system : DetectionSystem
        Provides ``result_sink``, ``_execute_pipeline``, ``config``,
        and ``_log_summary`` for persisting results.
    """

    def __init__(
        self,
        in_queue: queue.Queue[DetectionTask],
        detection_system: Any,
        *,
        name: str = "StorageWorker",
        on_task_processed: Optional[Callable[[DetectionTask], None]] = None,
    ) -> None:
        # Terminal worker — no downstream queue.
        super().__init__(in_queue, out_queue=None, name=name)
        self._system = detection_system
        self._saved_count: int = 0
        self._on_task_processed = on_task_processed

    @property
    def saved_count(self) -> int:
        """Total tasks successfully persisted to disk."""
        return self._saved_count

    def process(self, task: DetectionTask) -> None:
        """Persist one detection result to disk / Excel."""
        result = task.result
        if result is None:
            self._logger.warning(
                "Task %s has no result — skipping storage", task.task_id,
            )
            return

        if task.frame is None:
            self._logger.warning(
                "Task %s has no frame — skipping storage", task.task_id,
            )
            return

        try:
            from core.pipeline.context import DetectionContext

            ctx = DetectionContext(
                product=task.product,
                area=task.area,
                inference_type=task.inference_type,
                frame=task.frame,
                processed_image=result.get("processed_image", task.frame),
                result=result,
                status=result.get("status", "ERROR"),
                config=self._system.config,
            )

            self._system._execute_pipeline(ctx, self._logger)
            self._system._log_summary(ctx, self._logger)
            self._saved_count += 1

            pipeline_latency = time.time() - task.timestamp
            self._logger.debug(
                "Task %s persisted (total pipeline latency: %.3fs)",
                task.task_id, pipeline_latency,
            )

            # Trigger GUI callback if provided
            if self._on_task_processed:
                try:
                    self._on_task_processed(task)
                except Exception:
                    self._logger.error("Error in on_task_processed callback", exc_info=True)

        except Exception:
            self._logger.error(
                "Failed to persist task %s", task.task_id, exc_info=True,
            )

    def run(self) -> None:
        """Override to flush the sink after all tasks are processed."""
        try:
            super().run()
        finally:
            # Ensure any buffered writes (Excel rows, pending images)
            # are flushed to disk before the thread exits.
            try:
                if self._system.result_sink:
                    self._system.result_sink.flush()
                    self._logger.info("Result sink flushed on shutdown")
            except Exception:
                self._logger.error(
                    "Failed to flush result sink on shutdown", exc_info=True,
                )
            self._logger.info(
                "Storage stopped (saved %d tasks)", self._saved_count,
            )
