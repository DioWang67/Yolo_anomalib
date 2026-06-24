"""Auto-trigger state machine for hands-free inspection.

7-state FSM that monitors a camera ROI and fires a single inspection only
after the product is stable (low motion, sharp image, continuous presence).
YOLO / colour-check are never called from this module.

States
------
WAIT_EMPTY     → no product in ROI
PRODUCT_APPEAR → product detected, counting consecutive present frames
WAIT_STABLE    → product confirmed, waiting for motion/sharpness gate
CAPTURE_LOCK   → gate passed, select best frame from buffer
INSPECTING     → inspection job submitted, waiting for result
SHOW_RESULT    → result ready, transitional (one update cycle)
WAIT_REMOVE    → result displayed, waiting for product to leave
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import NamedTuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class TriggerState(str, Enum):
    WAIT_EMPTY = "WAIT_EMPTY"
    PRODUCT_APPEAR = "PRODUCT_APPEAR"
    WAIT_STABLE = "WAIT_STABLE"
    CAPTURE_LOCK = "CAPTURE_LOCK"
    INSPECTING = "INSPECTING"
    SHOW_RESULT = "SHOW_RESULT"
    WAIT_REMOVE = "WAIT_REMOVE"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class AutoTriggerConfig:
    enabled: bool = True
    # ROI as [x, y, w, h]; all-zero means full frame
    roi: list = field(default_factory=lambda: [0, 0, 0, 0])
    frame_buffer_size: int = 15
    # How many consecutive product-present frames before advancing
    appear_frames: int = 5
    # How many consecutive stable frames required before trigger
    stable_frames: int = 12
    # How many consecutive product-absent frames before resetting
    remove_frames: int = 8
    # absdiff mean threshold: lower = stricter motion requirement
    motion_threshold: float = 3.0
    # Laplacian variance threshold: higher = stricter sharpness requirement
    sharpness_threshold: float = 100.0
    # Contour area threshold for product-presence detection
    product_area_threshold: int = 5000
    # Minimum milliseconds between consecutive inspection triggers
    inspection_cooldown_ms: int = 500

    @classmethod
    def from_dict(cls, d: dict) -> AutoTriggerConfig:
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# Debug snapshot
# ---------------------------------------------------------------------------

class DebugInfo(NamedTuple):
    state: TriggerState
    product_present: bool
    stable_count: int
    appear_count: int
    remove_count: int
    sharpness: float
    motion_score: float
    last_result: str
    contour_area: float = 0.0   # raw value from detect_product_presence — use to calibrate threshold


# ---------------------------------------------------------------------------
# Image helper functions
# ---------------------------------------------------------------------------

def compute_sharpness(frame: np.ndarray) -> float:
    """Laplacian variance — higher means sharper image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_score(prev_roi: np.ndarray, curr_roi: np.ndarray) -> float:
    """Grayscale absdiff mean — higher means more motion between frames."""
    prev_g = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY) if prev_roi.ndim == 3 else prev_roi
    curr_g = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2GRAY) if curr_roi.ndim == 3 else curr_roi
    if prev_g.shape != curr_g.shape:
        curr_g = cv2.resize(curr_g, (prev_g.shape[1], prev_g.shape[0]))
    return float(cv2.absdiff(prev_g, curr_g).mean())


def detect_product_presence(
    frame: np.ndarray,
    roi: list[int],
    product_area_threshold: int,
) -> tuple[bool, float]:
    """Cheap contour-area detector — no YOLO required.

    Returns (present: bool, contour_area: float).
    The raw contour_area is exposed so callers can display it for threshold tuning.
    """
    x, y, w, h = roi
    if w > 0 and h > 0:
        crop = frame[y:y + h, x:x + w]
    else:
        crop = frame
    if crop.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = float(sum(cv2.contourArea(c) for c in contours))
    return total_area > product_area_threshold, total_area


def select_best_frame(
    frame_buffer: deque,
    sharpness_threshold: float,
    motion_threshold: float,
    roi: list[int] | None = None,
) -> np.ndarray | None:
    """Return the sharpest, lowest-motion frame from the buffer.

    The caller's state machine has *already* gated stability/sharpness before
    this is invoked, so the sharpness_threshold here is a **soft preference**,
    not a hard reject: among frames clearing it we pick the best, but if none
    clear it (e.g. the threshold was tuned at a different resolution than the
    buffered frames) we still return the sharpest available frame instead of
    stalling the capture indefinitely.

    Laplacian variance is strongly resolution-dependent — a downsampled frame
    yields a much higher variance than the same scene at full resolution — so a
    single threshold cannot be meaningfully compared across resolutions. Hence
    the soft-preference behaviour.

    Args:
        roi: [x, y, w, h] in the same coordinate space as the buffered frames.
             When provided, sharpness is computed on the ROI crop — matching
             the same region the state machine used to gate stability.
             All-zero (or None) means use the full frame.

    Returns None only when the buffer is empty.
    """
    if not frame_buffer:
        return None

    use_roi = roi is not None and len(roi) == 4 and roi[2] > 0 and roi[3] > 0

    scored: list[tuple[float, float, np.ndarray]] = []
    prev: np.ndarray | None = None
    for frame in frame_buffer:
        if use_roi:
            x, y, w, h = roi
            crop = frame[y:y + h, x:x + w]
            sharp_region = crop if crop.size > 0 else frame
        else:
            sharp_region = frame
        sharpness = compute_sharpness(sharp_region)
        motion = compute_motion_score(prev, frame) if prev is not None else 0.0
        prev = frame
        scored.append((sharpness, motion, frame))

    # Prefer frames that clear the gate; fall back to the full set when none do.
    passing = [s for s in scored if s[0] >= sharpness_threshold]
    pool = passing if passing else scored

    # Best = highest sharpness; tie-break on lowest motion
    pool.sort(key=lambda t: (-t[0], t[1]))
    return pool[0][2]


def draw_debug_overlay(
    frame: np.ndarray,
    info: DebugInfo,
    roi: list[int],
    config: "AutoTriggerConfig | None" = None,
) -> np.ndarray:
    """Draw state-machine debug info onto a copy of frame.

    All numeric values in ``info`` should already be scaled to full-res equivalents
    before calling so thresholds from ``config`` (also full-res) are directly comparable.
    """
    out = frame.copy()

    # ROI rectangle
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        cv2.rectangle(out, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

    # Semi-transparent background for text
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (420, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    state_color = {
        TriggerState.WAIT_EMPTY: (180, 180, 180),
        TriggerState.PRODUCT_APPEAR: (0, 200, 255),
        TriggerState.WAIT_STABLE: (0, 255, 128),
        TriggerState.CAPTURE_LOCK: (0, 255, 0),
        TriggerState.INSPECTING: (255, 200, 0),
        TriggerState.SHOW_RESULT: (0, 128, 255),
        TriggerState.WAIT_REMOVE: (100, 100, 255),
    }.get(info.state, (255, 255, 255))

    def _ok(val: float, threshold: float, higher_is_better: bool) -> tuple:
        passing = (val > threshold) if higher_is_better else (val < threshold)
        return (100, 220, 100) if passing else (80, 80, 220)

    sharp_thr = config.sharpness_threshold if config else 0.0
    motion_thr = config.motion_threshold if config else 0.0
    area_thr = config.product_area_threshold if config else 0.0

    lines = [
        (f"State: {info.state.value}", state_color),
        (f"Present: {info.product_present}  Appear: {info.appear_count}", (200, 200, 200)),
        (f"Stable: {info.stable_count}  Remove: {info.remove_count}", (200, 200, 200)),
        (
            f"Sharpness: {info.sharpness:.0f}  (need >{sharp_thr:.0f})",
            _ok(info.sharpness, sharp_thr, True) if config else (200, 200, 200),
        ),
        (
            f"Motion: {info.motion_score:.2f}  (need <{motion_thr:.1f})",
            _ok(info.motion_score, motion_thr, False) if config else (200, 200, 200),
        ),
        (
            f"Area: {info.contour_area:.0f}  (need >{area_thr:.0f})",
            _ok(info.contour_area, area_thr, True) if config else (255, 200, 100),
        ),
        (f"Result: {info.last_result or '--'}", (200, 200, 200)),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (8, 22 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, color, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class AutoTriggerStateMachine:
    """7-state FSM for stable auto-trigger inspection.

    Usage::

        sm = AutoTriggerStateMachine(config)
        while running:
            frame = camera.capture_frame()
            state, should_trigger = sm.update(frame)
            if should_trigger:
                best = sm.get_best_frame()
                sm.mark_inspecting()
                submit_inspection(best)
            # later, when result arrives:
            sm.mark_result_shown("PASS")
    """

    def __init__(self, config: AutoTriggerConfig) -> None:
        self.config = config
        self._state = TriggerState.WAIT_EMPTY
        self._appear_count = 0
        self._stable_count = 0
        self._remove_count = 0
        self._prev_roi: np.ndarray | None = None
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=config.frame_buffer_size)
        self._last_result: str = ""
        self._last_sharpness: float = 0.0
        self._last_motion: float = 0.0
        self._last_product_present: bool = False
        self._last_trigger_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        frame: np.ndarray,
        store_frame: np.ndarray | None = None,
    ) -> tuple[TriggerState, bool]:
        """Process one frame. Returns (current_state, should_trigger).

        Args:
            frame: Frame used for CV computations (may be downsampled for speed).
            store_frame: Full-res frame to keep in the trigger buffer. When None,
                ``frame`` itself is stored (original behaviour).
        """
        self._frame_buffer.append((store_frame if store_frame is not None else frame).copy())
        roi_crop = self._extract_roi(frame)

        product_present, contour_area = detect_product_presence(
            frame, self.config.roi, self.config.product_area_threshold
        )
        self._last_product_present = product_present
        self._last_contour_area = contour_area

        sharpness = compute_sharpness(roi_crop)
        motion = (
            compute_motion_score(self._prev_roi, roi_crop)
            if self._prev_roi is not None
            else 0.0
        )
        self._prev_roi = roi_crop
        self._last_sharpness = sharpness
        self._last_motion = motion

        should_trigger = False

        if self._state == TriggerState.WAIT_EMPTY:
            if product_present:
                self._appear_count += 1
                if self._appear_count >= self.config.appear_frames:
                    self._appear_count = self.config.appear_frames  # cap
                    self._transition(TriggerState.PRODUCT_APPEAR)
            else:
                self._appear_count = 0

        elif self._state == TriggerState.PRODUCT_APPEAR:
            if not product_present:
                self._appear_count = 0
                self._transition(TriggerState.WAIT_EMPTY)
            else:
                # Already at appear_frames; stay here one cycle, then advance
                self._stable_count = 0
                self._transition(TriggerState.WAIT_STABLE)

        elif self._state == TriggerState.WAIT_STABLE:
            if not product_present:
                self._appear_count = 0
                self._stable_count = 0
                self._transition(TriggerState.WAIT_EMPTY)
            else:
                stable = (
                    motion < self.config.motion_threshold
                    and sharpness > self.config.sharpness_threshold
                )
                if stable:
                    self._stable_count += 1
                    logger.debug(
                        "WAIT_STABLE stable_count=%d sharpness=%.1f motion=%.2f",
                        self._stable_count, sharpness, motion,
                    )
                else:
                    if self._stable_count > 0:
                        logger.debug(
                            "WAIT_STABLE reset sharpness=%.1f (need >%.1f) "
                            "motion=%.2f (need <%.2f)",
                            sharpness, self.config.sharpness_threshold,
                            motion, self.config.motion_threshold,
                        )
                    self._stable_count = 0

                if self._stable_count >= self.config.stable_frames:
                    cooldown_ok = (
                        (time.monotonic() - self._last_trigger_ts) * 1000
                        >= self.config.inspection_cooldown_ms
                    )
                    if cooldown_ok:
                        self._last_trigger_ts = time.monotonic()
                        self._transition(TriggerState.CAPTURE_LOCK)
                        should_trigger = True
                        logger.info(
                            "Trigger fired: sharpness=%.1f motion=%.2f stable=%d",
                            sharpness, motion, self._stable_count,
                        )

        elif self._state == TriggerState.CAPTURE_LOCK:
            # External caller must call mark_inspecting() after frame selection.
            pass

        elif self._state == TriggerState.INSPECTING:
            # External caller calls mark_result_shown() when done.
            pass

        elif self._state == TriggerState.SHOW_RESULT:
            # Transitional: advance to WAIT_REMOVE on next update cycle.
            self._remove_count = 0
            self._transition(TriggerState.WAIT_REMOVE)

        elif self._state == TriggerState.WAIT_REMOVE:
            if not product_present:
                self._remove_count += 1
                if self._remove_count >= self.config.remove_frames:
                    self._reset_counters()
                    self._transition(TriggerState.WAIT_EMPTY)
            else:
                self._remove_count = 0

        return self._state, should_trigger

    def mark_inspecting(self) -> None:
        """Call immediately after submitting the inspection job."""
        if self._state == TriggerState.CAPTURE_LOCK:
            self._transition(TriggerState.INSPECTING)

    def mark_result_shown(self, result_label: str = "") -> None:
        """Call when the inspection result is ready for display."""
        self._last_result = result_label
        if self._state == TriggerState.INSPECTING:
            self._transition(TriggerState.SHOW_RESULT)

    def get_best_frame(
        self,
        sharpness_threshold: float | None = None,
        motion_threshold: float | None = None,
        roi: list[int] | None = None,
    ) -> np.ndarray | None:
        """Return the sharpest low-motion frame from the rolling buffer.

        Override thresholds/roi when the buffer holds full-res frames but the SM
        was configured with downsampled thresholds (store_frame=original_frame).
        The roi should be in the same coordinate space as the buffered frames.
        """
        return select_best_frame(
            self._frame_buffer,
            sharpness_threshold if sharpness_threshold is not None else self.config.sharpness_threshold,
            motion_threshold if motion_threshold is not None else self.config.motion_threshold,
            roi=roi,
        )

    def get_debug_info(self) -> DebugInfo:
        return DebugInfo(
            state=self._state,
            product_present=self._last_product_present,
            stable_count=self._stable_count,
            appear_count=self._appear_count,
            remove_count=self._remove_count,
            sharpness=self._last_sharpness,
            motion_score=self._last_motion,
            last_result=self._last_result,
            contour_area=getattr(self, "_last_contour_area", 0.0),
        )

    def reset(self) -> None:
        """Full reset — call when auto mode is disabled."""
        self._reset_counters()
        self._state = TriggerState.WAIT_EMPTY
        self._frame_buffer.clear()
        self._prev_roi = None
        self._last_result = ""

    @property
    def state(self) -> TriggerState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_roi(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.config.roi
        if w > 0 and h > 0:
            crop = frame[y:y + h, x:x + w]
            if crop.size > 0:
                return crop.copy()
        return frame.copy()

    def _reset_counters(self) -> None:
        self._appear_count = 0
        self._stable_count = 0
        self._remove_count = 0

    def _transition(self, new_state: TriggerState) -> None:
        if new_state != self._state:
            logger.info("AutoTrigger %s → %s", self._state.value, new_state.value)
            self._state = new_state
