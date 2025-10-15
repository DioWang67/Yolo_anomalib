from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QLabel,
    QMainWindow,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .metrics import AdaptiveCalibrator, compute_change_metrics
from .source import CameraSource


class CameraPreviewWindow(QMainWindow):
    """Render frames in a QLabel and show change metrics."""

    def __init__(
        self,
        source: CameraSource,
        refresh_ms: int,
        logger: logging.Logger,
        *,
        resize_width: int,
        use_luma: bool,
        ema_alpha: float,
        fixed_threshold: Optional[int],
        auto_k_sigma: float,
        warmup_frames: int,
        roi: Optional[Tuple[int, int, int, int]],
        scale_quality: str,
        target_fps: Optional[float],
        reopen_failures: int,
    ) -> None:
        super().__init__()
        self.source = source
        self.logger = logger

        self.resize_width = max(0, resize_width)
        self.use_luma = use_luma
        self.ema_alpha = float(ema_alpha)
        self.fixed_threshold = fixed_threshold
        self.calibrator = AdaptiveCalibrator(warmup_frames=warmup_frames, k_sigma=auto_k_sigma)
        self.roi = roi
        self.scale_quality = scale_quality
        self.target_fps = target_fps
        self.reopen_failures = max(1, reopen_failures)

        self._last_gray: Optional[np.ndarray] = None
        self._ema_mean: Optional[float] = None
        self._ema_ratio: Optional[float] = None
        self._fps_ema: Optional[float] = None
        self._last_tick = cv2.getTickCount()
        self._consecutive_fail = 0

        self.setWindowTitle("Camera Preview")
        self.resize(960, 720)

        central = QWidget(self)
        layout = QVBoxLayout(central)

        self.image_label = QLabel("Initializing camera...", central)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)

        self.metric_label = QLabel("dMean: N/A  dRatio: N/A  FPS: N/A  Thr: N/A", central)
        self.metric_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.metric_label)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)  # type: ignore[arg-type]
        self._timer.start(max(15, refresh_ms))

        if self.target_fps is not None and self.source._capture is not None:
            self.source._capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.resize_width and w > self.resize_width:
            scale = self.resize_width / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        gray = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            if self.use_luma
            else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        )

        if self.roi:
            x, y, rw, rh = self.roi
            x2, y2 = x + rw, y + rh
            x = max(0, x)
            y = max(0, y)
            x2 = min(gray.shape[1], x2)
            y2 = min(gray.shape[0], y2)
            if x < x2 and y < y2:
                gray = gray[y:y2, x:x2]
        return gray

    def _scale_pixmap(self, qimg: QImage) -> QPixmap:
        pix = QPixmap.fromImage(qimg)
        target_size = self.image_label.size() * self.devicePixelRatio()
        if self.scale_quality.lower() == "smooth":
            return pix.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pix.scaled(target_size, Qt.KeepAspectRatio, Qt.FastTransformation)

    def _estimate_fps(self) -> float:
        now = cv2.getTickCount()
        dt = (now - self._last_tick) / cv2.getTickFrequency()
        self._last_tick = now
        inst_fps = 1.0 / dt if dt > 1e-6 else 0.0
        self._fps_ema = inst_fps if self._fps_ema is None else (0.2 * inst_fps + 0.8 * self._fps_ema)
        return self._fps_ema or 0.0

    def _current_threshold(self) -> int:
        if self.fixed_threshold is not None:
            return int(self.fixed_threshold)
        if self.calibrator.ready:
            return int(self.calibrator.suggested_threshold)
        return 25

    def _update_frame(self) -> None:
        frame = None
        try:
            frame = self.source.read()
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Camera read error: %s", exc)

        if frame is None:
            self._consecutive_fail += 1
            self.status_bar.showMessage("No frame available", 300)
            self.metric_label.setText("dMean:N/A  dRatio:N/A  FPS:N/A  Thr:N/A")
            if self._consecutive_fail >= self.reopen_failures:
                self.logger.warning("Consecutive read failures %d, try reopen()", self._consecutive_fail)
                try:
                    self.source.reopen()
                    self._consecutive_fail = 0
                except Exception as exc:  # pragma: no cover
                    self.logger.error("Reopen failed: %s", exc)
            return
        else:
            self._consecutive_fail = 0

        try:
            fps = self._estimate_fps()
            gray = self._preprocess(frame)
            threshold = self._current_threshold()

            self.calibrator.update(gray, self._last_gray)

            if self._last_gray is not None and self._last_gray.shape == gray.shape:
                md, cr = compute_change_metrics(gray, self._last_gray, threshold=threshold)
                if self._ema_mean is None:
                    self._ema_mean, self._ema_ratio = md, cr
                else:
                    alpha = self.ema_alpha
                    self._ema_mean = alpha * md + (1 - alpha) * self._ema_mean
                    self._ema_ratio = alpha * cr + (1 - alpha) * self._ema_ratio
                md_show = f"{self._ema_mean:.3f}"
                cr_show = f"{self._ema_ratio:.3f}"
            else:
                md_show = "Baseline"
                cr_show = "Baseline"

            self._last_gray = gray

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            qimg.setDevicePixelRatio(self.devicePixelRatio())
            pixmap = self._scale_pixmap(qimg)
            self.image_label.setPixmap(pixmap)

            thr_show = (
                f"{threshold:d}" if self.calibrator.ready or self.fixed_threshold is not None else "Calibrating..."
            )
            self.metric_label.setText(f"dMean:{md_show}  dRatio:{cr_show}  FPS:{fps:.1f}  Thr:{thr_show}")
            self.status_bar.showMessage(f"Frame: {w}x{h}", 500)

        except Exception as exc:  # pragma: no cover
            self.logger.exception("Display error: %s", exc)
            self.metric_label.setText("dMean:N/A  dRatio:N/A  FPS:N/A  Thr:N/A")

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self._timer.stop()
        finally:
            self.source.shutdown()
        event.accept()
