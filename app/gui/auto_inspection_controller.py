"""Auto-inspection controller — GUI integration layer.

Wires the AutoTriggerStateMachine to the camera hardware and the existing
detection_system.detect() call, keeping all blocking work off the UI thread.

Architecture
------------
CameraPreviewWorker (QThread)
    Loops on camera.capture_frame() at camera-native rate.
    Runs state-machine update() on each frame.
    Emits frame_ready(np.ndarray) for the ImagePanel live preview.
    Emits trigger_fired(np.ndarray) when CAPTURE_LOCK state is reached.
    Never touches Qt widgets directly.

AutoInspectionController (QObject, lives on main thread)
    Owns one CameraPreviewWorker.
    On trigger_fired: spawns a daemon thread to call detection_system.detect().
    Routes results back via the existing PipelineBridge.result_ready signal so
    the rest of main_window.py is unmodified.
    Tracks each run generation so cancelled work cannot leak into a later run.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot

from core.auto_trigger import (
    AutoTriggerConfig,
    AutoTriggerStateMachine,
    TriggerState,
    draw_debug_overlay,
)

if TYPE_CHECKING:
    from app.gui.workers import PipelineBridge
    from camera.camera_controller import CameraController
    from core.detection_system import DetectionSystem

logger = logging.getLogger(__name__)

# Default config — can be overridden via set_config() or YAML auto_trigger section
_PREVIEW_MAX_WIDTH = 960  # pixels; preview frames are resized to this before emitting
_PREVIEW_STOP_WAIT_MS = 100
_PREVIEW_CAPTURE_TIMEOUT_CAP_MS = 500

DEFAULT_AUTO_TRIGGER_CONFIG: dict = {
    "enabled": True,
    "roi": [0, 0, 0, 0],           # full frame; change to [x, y, w, h]
    "frame_buffer_size": 8,
    "appear_frames": 3,
    "stable_frames": 6,
    "remove_frames": 8,
    "motion_threshold": 3.0,
    "sharpness_threshold": 100.0,
    "product_area_threshold": 5000,
    "inspection_cooldown_ms": 500,
}


# ---------------------------------------------------------------------------
# Camera preview + state machine worker thread
# ---------------------------------------------------------------------------

class CameraPreviewWorker(QThread):
    """Reads frames from the camera and drives the state machine.

    Signals
    -------
    frame_ready : np.ndarray
        BGR frame for the ImagePanel live preview (with optional debug overlay).
    trigger_fired : np.ndarray
        Best frame selected by the state machine, ready for inspection.
    state_changed : str
        Emitted when TriggerState transitions occur.
    error_occurred : str
        Emitted on camera read failures.
    """

    frame_ready = pyqtSignal(object)      # np.ndarray
    trigger_fired = pyqtSignal(object)    # np.ndarray (best frame)
    state_changed = pyqtSignal(str)       # TriggerState.value
    error_occurred = pyqtSignal(str)

    # Linear scale factor applied to frames before CV computations.
    # 0.25 = 1/4 linear → 1/16 area; fast enough for real-time on full-res cameras.
    _COMPUTE_SCALE = 0.25

    def __init__(
        self,
        camera: CameraController,
        config: AutoTriggerConfig,
        show_debug_overlay: bool = True,
    ) -> None:
        super().__init__()
        self._camera = camera
        self._config = config
        self._show_debug_overlay = show_debug_overlay
        self._stop_event = threading.Event()
        camera_config = getattr(camera, "config", None)
        configured_timeout_ms = getattr(
            camera_config,
            "MV_CC_GetImageBuffer_nMsec",
            _PREVIEW_CAPTURE_TIMEOUT_CAP_MS,
        )
        try:
            configured_timeout_ms = int(configured_timeout_ms)
        except (TypeError, ValueError):
            configured_timeout_ms = _PREVIEW_CAPTURE_TIMEOUT_CAP_MS
        self._capture_timeout_ms = max(
            1,
            min(configured_timeout_ms, _PREVIEW_CAPTURE_TIMEOUT_CAP_MS),
        )
        try:
            self._max_consecutive_failures = max(
                1,
                int(getattr(camera_config, "camera_lost_threshold", 5)),
            )
        except (TypeError, ValueError):
            self._max_consecutive_failures = 5
        # Build a scaled config so threshold values match the downsampled frame.
        # - contour area scales by scale^2 (area is proportional to pixel count)
        # - sharpness (Laplacian variance) scales by scale^1 empirically: downsampling
        #   softens edges, reducing variance roughly proportional to linear scale factor
        # - motion (absdiff mean) is scale-independent (it's a mean, not a sum)
        scale = self._COMPUTE_SCALE
        scaled_config = AutoTriggerConfig(
            enabled=config.enabled,
            roi=[int(v * scale) for v in config.roi],
            frame_buffer_size=config.frame_buffer_size,
            appear_frames=config.appear_frames,
            stable_frames=config.stable_frames,
            remove_frames=config.remove_frames,
            motion_threshold=config.motion_threshold,
            sharpness_threshold=config.sharpness_threshold * scale,
            product_area_threshold=int(config.product_area_threshold * scale * scale),
            inspection_cooldown_ms=config.inspection_cooldown_ms,
        )
        self._sm = AutoTriggerStateMachine(scaled_config)
        self._inspection_running = False  # set by controller via mark_inspecting()
        self._lock = threading.Lock()
        self._last_state = TriggerState.WAIT_EMPTY

    # ------------------------------------------------------------------
    # Public control API (called from main thread)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request the worker loop to exit."""
        self._stop_event.set()

    def mark_inspecting(self) -> None:
        """Called by controller after submitting inspection job."""
        with self._lock:
            self._sm.mark_inspecting()
            self._inspection_running = True

    def mark_result_shown(self, result_label: str = "") -> None:
        """Called by controller when the inspection result is ready."""
        with self._lock:
            self._sm.mark_result_shown(result_label)
            self._inspection_running = False

    def reset_state_machine(self) -> None:
        """Full reset — used when auto mode is toggled off/on."""
        with self._lock:
            self._sm.reset()
            self._inspection_running = False

    # ------------------------------------------------------------------
    # QThread.run
    # ------------------------------------------------------------------

    def run(self) -> None:
        consecutive_failures = 0
        _preview_interval = 1.0 / 15  # cap UI at 15fps
        _last_preview_ts = 0.0

        while not self._stop_event.is_set():
            frame = self._camera.capture_frame(
                timeout_ms=self._capture_timeout_ms
            )
            # capture_frame() is a blocking SDK boundary. A normal Stop can be
            # requested while it is waiting, so discard that final return value
            # before interpreting ``None`` as a camera-health failure.
            if self._stop_event.is_set():
                break
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= self._max_consecutive_failures:
                    mark_unhealthy = getattr(self._camera, "mark_unhealthy", None)
                    if callable(mark_unhealthy):
                        mark_unhealthy()
                    self.error_occurred.emit(
                        "Camera returned None for "
                        f"{self._max_consecutive_failures} consecutive frames"
                    )
                    break
                continue
            consecutive_failures = 0

            # Downsample to 1/4 linear (1/16 area) for all CV computations.
            # Full-res frame is still stored in the buffer for the final trigger capture.
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (w // 4, h // 4), interpolation=cv2.INTER_AREA)

            with self._lock:
                state, should_trigger = self._sm.update(small, store_frame=frame)
                # Capture best frame atomically with the trigger decision so the
                # buffer cannot be overwritten by subsequent frames before we read it.
                # sharpness_threshold here is a soft preference (see select_best_frame):
                # it never stalls capture, so best_frame is non-None whenever the
                # buffer is non-empty (always true at CAPTURE_LOCK).
                best_frame: np.ndarray | None = None
                if should_trigger:
                    best_frame = self._sm.get_best_frame(
                        sharpness_threshold=self._config.sharpness_threshold,
                        motion_threshold=self._config.motion_threshold,
                        roi=self._config.roi,
                    )
                debug_info = self._sm.get_debug_info()

            # Emit state changes
            if state != self._last_state:
                self._last_state = state
                self.state_changed.emit(state.value)

            # Throttle UI preview to 15fps — state machine still runs every frame
            now = time.monotonic()
            if now - _last_preview_ts >= _preview_interval:
                _last_preview_ts = now
                # Resize to display resolution BEFORE emitting — avoids sending
                # 18MB arrays across thread boundary and into Qt paint path.
                dh, dw = frame.shape[:2]
                if dw > _PREVIEW_MAX_WIDTH:
                    scale = _PREVIEW_MAX_WIDTH / dw
                    preview = cv2.resize(
                        frame,
                        (int(dw * scale), int(dh * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                else:
                    preview = frame
                if self._show_debug_overlay:
                    compute_scale = self._COMPUTE_SCALE
                    # Convert computed values back to full-res equivalents for display
                    display_info = debug_info._replace(
                        sharpness=debug_info.sharpness / compute_scale,
                        contour_area=debug_info.contour_area / (compute_scale ** 2),
                    )
                    roi_scaled = [
                        int(v * scale) if dw > _PREVIEW_MAX_WIDTH else v
                        for v in self._config.roi
                    ]
                    preview = draw_debug_overlay(
                        preview, display_info, roi_scaled, config=self._config
                    )
                self.frame_ready.emit(preview)

            # Trigger inspection
            if should_trigger and best_frame is not None:
                logger.info(
                    "Emitting trigger_fired: sharpness=%.1f motion=%.2f",
                    debug_info.sharpness, debug_info.motion_score,
                )
                self.trigger_fired.emit(best_frame)


# ---------------------------------------------------------------------------
# Controller (lives on main thread, coordinates everything)
# ---------------------------------------------------------------------------

class AutoInspectionController(QObject):
    """Orchestrates camera preview, state machine, and inspection dispatch.

    Connects CameraPreviewWorker signals to the existing PipelineBridge so
    that main_window.py receives results via the same on_pipeline_result /
    on_detection_complete path as button-triggered inspections.

    Signals
    -------
    auto_state_changed : str
        Current TriggerState name (for status-bar display).
    auto_error : str
        Fatal error from camera preview (e.g. camera disconnected).
    fully_stopped : int
        Emitted after both preview and inference work for a generation finish.
    """

    auto_state_changed = pyqtSignal(str)
    auto_error = pyqtSignal(str)
    fully_stopped = pyqtSignal(int)
    _inspection_result_ready = pyqtSignal(int, object)
    _inspection_failed = pyqtSignal(int, str)
    _inspection_finished = pyqtSignal(int)

    def __init__(
        self,
        detection_system: DetectionSystem,
        bridge: PipelineBridge,
        config: AutoTriggerConfig | None = None,
        show_debug_overlay: bool = True,
    ) -> None:
        super().__init__()
        self._system = detection_system
        self._bridge = bridge
        self._config = config or AutoTriggerConfig.from_dict(DEFAULT_AUTO_TRIGGER_CONFIG)
        self._show_debug_overlay = show_debug_overlay
        self._worker: CameraPreviewWorker | None = None
        self._worker_generation: int | None = None
        self._inspection_thread: threading.Thread | None = None
        self._inspection_generation: int | None = None
        self._finished_inspection_generation: int | None = None
        self._inspection_finish_timer = QTimer(self)
        self._inspection_finish_timer.setSingleShot(True)
        self._inspection_finish_timer.setInterval(10)
        self._inspection_finish_timer.timeout.connect(
            self._poll_finished_inspection_thread
        )
        self._generation = 0
        self._active_generation: int | None = None
        self._stop_requested_generation: int | None = None
        self._product = ""
        self._area = ""
        self._inference_type = ""
        self._cancel_event: threading.Event | None = None
        self._inspection_result_ready.connect(
            self._on_inspection_result_ready,
            Qt.QueuedConnection,
        )
        self._inspection_failed.connect(
            self._on_inspection_failed,
            Qt.QueuedConnection,
        )
        self._inspection_finished.connect(
            self._on_inspection_finished,
            Qt.QueuedConnection,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, product: str, area: str, inference_type: str) -> bool:
        """Start the camera preview loop and state machine.

        Returns False when the camera is not available.
        """
        target = tuple(
            str(value).strip()
            for value in (product, area, inference_type)
        )
        if not all(target):
            logger.error("Cannot start auto mode with an incomplete target")
            return False
        product, area, inference_type = target
        if (
            self._active_generation is not None
            or self._worker is not None
            or self._inspection_thread is not None
        ):
            logger.warning(
                "AutoInspectionController cannot start before the prior "
                "generation is fully stopped"
            )
            return False

        camera = self._system.camera
        if (
            camera is None
            or not camera.is_initialized
            or not getattr(camera, "is_healthy", True)
        ):
            logger.error("Cannot start auto mode: camera not initialized")
            return False

        worker = CameraPreviewWorker(
            camera=camera,
            config=self._config,
            show_debug_overlay=self._show_debug_overlay,
        )
        self._product = product
        self._area = area
        self._inference_type = inference_type
        self._generation += 1
        generation = self._generation
        self._active_generation = generation
        self._stop_requested_generation = None
        self._cancel_event = threading.Event()
        self._worker = worker
        self._worker_generation = generation
        worker.frame_ready.connect(self._on_frame_ready, Qt.QueuedConnection)
        worker.trigger_fired.connect(
            self._on_trigger_fired,
            Qt.QueuedConnection,
        )
        worker.state_changed.connect(
            self._on_worker_state_changed,
            Qt.QueuedConnection,
        )
        worker.error_occurred.connect(
            self._on_camera_error,
            Qt.QueuedConnection,
        )
        worker.finished.connect(
            self._on_preview_worker_finished,
            Qt.QueuedConnection,
        )
        try:
            worker.start()
        except RuntimeError as exc:
            self._worker = None
            self._worker_generation = None
            self._active_generation = None
            self._cancel_event = None
            logger.error(
                "AutoInspectionController generation %d failed to start: %s",
                generation,
                exc,
            )
            return False
        logger.info(
            "AutoInspectionController generation %d started: %s/%s/%s",
            generation,
            product,
            area,
            inference_type,
        )
        return True

    def stop(self) -> bool:
        """Request stop; return True only after preview and inference finish."""
        generation = self._active_generation
        if generation is None:
            return self._worker is None and self._inspection_thread is None

        self._stop_requested_generation = generation
        cancel_event = self._cancel_event
        if cancel_event is not None:
            cancel_event.set()

        worker = self._worker
        if worker is not None and self._worker_generation == generation:
            worker.stop()
        if worker is not None and not worker.wait(msecs=_PREVIEW_STOP_WAIT_MS):
            logger.warning(
                "Camera preview worker for generation %d is still stopping",
                generation,
            )
            return False
        if worker is not None:
            if self._worker is worker:
                self._worker = None
                self._worker_generation = None

        inspection = self._inspection_thread
        if (
            inspection is not None
            and self._inspection_generation == generation
            and inspection.is_alive()
        ):
            logger.info(
                "Waiting asynchronously for inference generation %d to stop",
                generation,
            )
            return False
        if inspection is not None and self._inspection_generation == generation:
            self._inspection_finish_timer.stop()
            self._finished_inspection_generation = None
            self._inspection_thread = None
            self._inspection_generation = None

        self._finalize_stop(generation, notify=False)
        logger.info(
            "AutoInspectionController generation %d fully stopped",
            generation,
        )
        return True

    @pyqtSlot()
    def _on_preview_worker_finished(self) -> None:
        worker = self.sender()
        if worker is not self._worker:
            return
        generation = self._worker_generation
        self._worker = None
        self._worker_generation = None
        if generation is None:
            return
        if self._stop_requested_generation != generation:
            self._stop_requested_generation = generation
            cancel_event = self._cancel_event
            if cancel_event is not None:
                cancel_event.set()
        self._try_finalize_async_stop(generation)

    def is_running(self) -> bool:
        if self._active_generation is None:
            return False
        preview_running = (
            self._worker is not None and self._worker.isRunning()
        )
        inference_running = (
            self._inspection_thread is not None
            and self._inspection_thread.is_alive()
        )
        return preview_running or inference_running

    @property
    def active_generation(self) -> int | None:
        """Return the current lifecycle generation, including while stopping."""
        return self._active_generation

    def get_current_contour_area(self) -> float | None:
        """Return the latest contour area from the running worker (full-res scale).

        Returns None when Auto Mode is not active.
        Used by the calibration UI to sample empty/product readings.
        """
        if self._worker is None:
            return None
        with self._worker._lock:
            info = self._worker._sm.get_debug_info()
        scale = CameraPreviewWorker._COMPUTE_SCALE
        return info.contour_area / (scale * scale)

    def set_config(self, config: AutoTriggerConfig) -> None:
        """Replace config; takes effect on next start()."""
        self._config = config

    def set_debug_overlay(self, enabled: bool) -> None:
        self._show_debug_overlay = enabled
        if self._worker is not None:
            self._worker._show_debug_overlay = enabled

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Forward preview frames to the GUI image panel via bridge."""
        sender = self.sender()
        if sender is not None and sender is not self._worker:
            return
        try:
            self._bridge.image_ready.emit(frame)
        except RuntimeError as exc:
            logger.debug("Preview frame dropped after bridge shutdown: %s", exc)

    @pyqtSlot(object)
    def _on_trigger_fired(self, best_frame: np.ndarray) -> None:
        """Called on main thread when state machine fires trigger.

        Advances state machine to INSPECTING, then submits detect() job
        on a daemon thread to avoid blocking the UI.
        """
        worker = self.sender()
        if worker is None:
            worker = self._worker
        if worker is not self._worker:
            return
        generation = self._active_generation
        cancel_event = self._cancel_event
        if (
            generation is None
            or cancel_event is None
            or cancel_event.is_set()
            or self._worker_generation != generation
            or self._inspection_thread is not None
        ):
            return

        worker.mark_inspecting()
        logger.info(
            "Auto-trigger generation %d: submitting inspection %s/%s/%s",
            generation,
            self._product,
            self._area,
            self._inference_type,
        )

        inspection_thread = threading.Thread(
            target=self._run_inspection,
            args=(
                generation,
                cancel_event,
                best_frame,
                self._product,
                self._area,
                self._inference_type,
            ),
            daemon=True,
            name=f"auto-inspection-{generation}",
        )
        self._inspection_thread = inspection_thread
        self._inspection_generation = generation
        inspection_thread.start()

    @pyqtSlot(str)
    def _on_camera_error(self, msg: str) -> None:
        if self.sender() is not self._worker:
            return
        logger.error("AutoInspectionController camera error: %s", msg)
        self.auto_error.emit(msg)

    @pyqtSlot(str)
    def _on_worker_state_changed(self, state_name: str) -> None:
        """Forward state only from the current preview generation."""
        if self.sender() is self._worker:
            self.auto_state_changed.emit(state_name)

    # ------------------------------------------------------------------
    # Inspection (runs on daemon thread)
    # ------------------------------------------------------------------

    def _run_inspection(
        self,
        generation: int,
        cancel_event: threading.Event,
        frame: np.ndarray,
        product: str,
        area: str,
        inference_type: str,
    ) -> None:
        t_start = time.monotonic()
        try:
            result = self._system.detect(
                product,
                area,
                inference_type,
                frame=frame,
                cancel_cb=cancel_event.is_set,
            )
            elapsed = time.monotonic() - t_start
            logger.info(
                "Auto-trigger generation %d done: status=%s elapsed=%.2fs",
                generation,
                getattr(result, "status", "UNKNOWN"),
                elapsed,
            )

            if cancel_event.is_set():
                return

            self._inspection_result_ready.emit(generation, result)
        except Exception as exc:  # detection providers expose heterogeneous errors
            elapsed = time.monotonic() - t_start
            logger.error(
                "Auto-trigger generation %d failed after %.2fs: %s",
                generation,
                elapsed,
                traceback.format_exc(),
            )
            if not cancel_event.is_set():
                self._inspection_failed.emit(generation, str(exc))
        finally:
            self._inspection_finished.emit(generation)

    @pyqtSlot(int, object)
    def _on_inspection_result_ready(
        self,
        generation: int,
        result: object,
    ) -> None:
        """Publish a result only if it still belongs to the active generation."""
        cancel_event = self._cancel_event
        worker = self._worker
        if (
            generation != self._active_generation
            or generation != self._inspection_generation
            or generation != self._worker_generation
            or cancel_event is None
            or cancel_event.is_set()
            or worker is None
        ):
            return
        self._bridge.result_ready.emit(result)
        status_label = getattr(result, "status", "")
        worker.mark_result_shown(str(status_label))

    @pyqtSlot(int, str)
    def _on_inspection_failed(self, generation: int, message: str) -> None:
        """Route an active-generation failure and unblock its state machine."""
        cancel_event = self._cancel_event
        worker = self._worker
        if (
            generation != self._active_generation
            or generation != self._inspection_generation
            or generation != self._worker_generation
            or cancel_event is None
            or cancel_event.is_set()
            or worker is None
        ):
            return
        self._bridge.error_occurred.emit(message)
        worker.mark_result_shown("ERROR")

    @pyqtSlot(int)
    def _on_inspection_finished(self, generation: int) -> None:
        """Wait until Python confirms the emitting inference thread has exited."""
        if generation != self._inspection_generation:
            return
        self._finished_inspection_generation = generation
        self._poll_finished_inspection_thread()

    @pyqtSlot()
    def _poll_finished_inspection_thread(self) -> None:
        """Finalize inference state after Thread.is_alive() becomes false."""
        generation = self._finished_inspection_generation
        if generation is None or generation != self._inspection_generation:
            self._finished_inspection_generation = None
            return
        inspection = self._inspection_thread
        if inspection is not None and inspection.is_alive():
            self._inspection_finish_timer.start()
            return
        self._finished_inspection_generation = None
        self._inspection_thread = None
        self._inspection_generation = None
        self._try_finalize_async_stop(generation)

    def _try_finalize_async_stop(self, generation: int) -> bool:
        """Notify once a requested generation has no remaining work."""
        if self._stop_requested_generation != generation:
            return False
        if (
            self._worker is not None
            and self._worker_generation == generation
        ):
            return False
        inspection = self._inspection_thread
        if (
            inspection is not None
            and self._inspection_generation == generation
            and inspection.is_alive()
        ):
            return False
        if inspection is not None and self._inspection_generation == generation:
            self._inspection_finish_timer.stop()
            self._finished_inspection_generation = None
            self._inspection_thread = None
            self._inspection_generation = None
        self._finalize_stop(generation, notify=True)
        logger.info(
            "AutoInspectionController generation %d fully stopped asynchronously",
            generation,
        )
        return True

    def _finalize_stop(self, generation: int, *, notify: bool) -> None:
        """Clear one generation; optionally notify an asynchronous waiter."""
        if generation != self._active_generation:
            return
        if self._worker_generation == generation:
            self._worker = None
            self._worker_generation = None
        if self._inspection_generation == generation:
            self._inspection_finish_timer.stop()
            self._finished_inspection_generation = None
            self._inspection_thread = None
            self._inspection_generation = None
        self._active_generation = None
        self._stop_requested_generation = None
        self._cancel_event = None
        if notify:
            self.fully_stopped.emit(generation)
