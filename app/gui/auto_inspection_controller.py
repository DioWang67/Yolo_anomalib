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
    Enforces single-job-at-a-time with _inspection_running flag.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, pyqtSignal

from core.auto_trigger import (
    AutoTriggerConfig,
    AutoTriggerStateMachine,
    TriggerState,
    draw_debug_overlay,
)

if TYPE_CHECKING:
    from camera.camera_controller import CameraController
    from core.detection_system import DetectionSystem
    from app.gui.workers import PipelineBridge

logger = logging.getLogger(__name__)

# Default config — can be overridden via set_config() or YAML auto_trigger section
_PREVIEW_MAX_WIDTH = 960  # pixels; preview frames are resized to this before emitting

DEFAULT_AUTO_TRIGGER_CONFIG: dict = {
    "enabled": True,
    "roi": [0, 0, 0, 0],           # full frame; change to [x, y, w, h]
    "frame_buffer_size": 15,
    "appear_frames": 5,
    "stable_frames": 12,
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
        max_failures = 10
        _preview_interval = 1.0 / 15  # cap UI at 15fps
        _last_preview_ts = 0.0

        while not self._stop_event.is_set():
            frame = self._camera.capture_frame()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    self.error_occurred.emit(
                        f"Camera returned None for {max_failures} consecutive frames"
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
    """

    auto_state_changed = pyqtSignal(str)
    auto_error = pyqtSignal(str)

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
        self._product = ""
        self._area = ""
        self._inference_type = ""
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, product: str, area: str, inference_type: str) -> bool:
        """Start the camera preview loop and state machine.

        Returns False when the camera is not available.
        """
        if self._worker is not None and self._worker.isRunning():
            logger.warning("AutoInspectionController already running")
            return True

        camera = self._system.camera
        if camera is None or not camera.is_initialized:
            logger.error("Cannot start auto mode: camera not initialized")
            return False

        self._product = product
        self._area = area
        self._inference_type = inference_type
        self._cancel_event.clear()

        self._worker = CameraPreviewWorker(
            camera=camera,
            config=self._config,
            show_debug_overlay=self._show_debug_overlay,
        )
        self._worker.frame_ready.connect(self._on_frame_ready)
        self._worker.trigger_fired.connect(self._on_trigger_fired)
        self._worker.state_changed.connect(self.auto_state_changed)
        self._worker.error_occurred.connect(self._on_camera_error)
        self._worker.start()
        logger.info(
            "AutoInspectionController started: %s/%s/%s",
            product, area, inference_type,
        )
        return True

    def stop(self) -> None:
        """Stop the preview loop and reset state machine."""
        self._cancel_event.set()
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(msecs=3000)
            self._worker = None
        logger.info("AutoInspectionController stopped")

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

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

    def _on_frame_ready(self, frame: np.ndarray) -> None:
        """Forward preview frames to the GUI image panel via bridge."""
        try:
            self._bridge.image_ready.emit(frame)
        except Exception:
            pass

    def _on_trigger_fired(self, best_frame: np.ndarray) -> None:
        """Called on main thread when state machine fires trigger.

        Advances state machine to INSPECTING, then submits detect() job
        on a daemon thread to avoid blocking the UI.
        """
        if self._cancel_event.is_set():
            return
        if self._worker is None:
            return

        self._worker.mark_inspecting()
        logger.info(
            "Auto-trigger: submitting inspection %s/%s/%s",
            self._product, self._area, self._inference_type,
        )

        threading.Thread(
            target=self._run_inspection,
            args=(best_frame, self._product, self._area, self._inference_type),
            daemon=True,
            name="auto-inspection",
        ).start()

    def _on_camera_error(self, msg: str) -> None:
        logger.error("AutoInspectionController camera error: %s", msg)
        self.auto_error.emit(msg)

    # ------------------------------------------------------------------
    # Inspection (runs on daemon thread)
    # ------------------------------------------------------------------

    def _run_inspection(
        self,
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
                cancel_cb=self._cancel_event.is_set,
            )
            elapsed = time.monotonic() - t_start
            logger.info(
                "Auto-trigger inspection done: status=%s elapsed=%.2fs",
                getattr(result, "status", "?"), elapsed,
            )

            if self._cancel_event.is_set():
                return

            # Route result back through existing bridge → main_window.on_pipeline_result
            self._bridge.result_ready.emit(result)

            # Advance state machine to SHOW_RESULT → WAIT_REMOVE
            if self._worker is not None:
                status_label = getattr(result, "status", "")
                self._worker.mark_result_shown(str(status_label))

        except Exception as exc:
            elapsed = time.monotonic() - t_start
            logger.error(
                "Auto-trigger inspection error after %.2fs: %s",
                elapsed, traceback.format_exc(),
            )
            if not self._cancel_event.is_set():
                self._bridge.error_occurred.emit(str(exc))
            # Unblock state machine so it can try again next product
            if self._worker is not None:
                self._worker.mark_result_shown("ERROR")
