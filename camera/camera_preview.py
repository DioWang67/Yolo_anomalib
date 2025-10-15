# -*- coding: utf-8 -*-
"""
Standalone Qt camera preview tool with robust change metrics.

Enhancements over the original:
- Dual metrics for per-frame change: normalized mean diff (0~1) and change ratio (0~1).
- EMA smoothing for stable readouts.
- Auto-calibration on startup (estimate noise and suggest threshold = mu + k*sigma).
- Live FPS estimation (EMA).
- ROI support for focused change measurement.
- Resilience: automatic reopen on repeated capture failures.
- HiDPI-friendly rendering; configurable scaling quality.
- Structured logging with CLI-configurable level.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QTimer, Qt, QRect
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

# --- Project imports (keep original behavior) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from camera.camera_controller import CameraController  # noqa: E402
from core.config import DetectionConfig  # noqa: E402


# ===============================
# Utilities: metrics & calibration
# ===============================

def compute_change_metrics(
    curr_gray: np.ndarray,
    prev_gray: np.ndarray,
    thr: int,
) -> Tuple[float, float]:
    """
    計算兩張灰階影格的變化量指標。

    參數:
        curr_gray: 當前影格 (H, W) uint8
        prev_gray: 前一影格 (H, W) uint8
        thr: 像素強度差分門檻，用於 change ratio

    回傳:
        (mean_diff_norm, changed_ratio)
        - mean_diff_norm: 平均絕對差 / 255，範圍 0~1
        - changed_ratio: |diff| >= thr 的像素比例，範圍 0~1

    範例:
        md, cr = compute_change_metrics(curr, prev, thr=25)
    """
    if curr_gray.shape != prev_gray.shape:
        raise ValueError("Frame shape mismatch for compute_change_metrics().")
    diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
    mean_diff_norm = float(diff.mean() / 255.0)
    changed_ratio = float((diff >= thr).mean())
    return mean_diff_norm, changed_ratio


@dataclass
class AdaptiveCalibrator:
    """
    啟動期自動標定背景噪聲水位，推導差分門檻:
      建議門檻 ≈ μ + kσ (以像素差值為單位)
    """
    warmup_frames: int = 20
    k_sigma: float = 3.0

    # internal state
    _count: int = 0
    _sum: float = 0.0
    _sqsum: float = 0.0
    _ready: bool = False
    _suggest_thr: int = 25

    def update(self, curr_gray: np.ndarray, prev_gray: Optional[np.ndarray]) -> None:
        if prev_gray is None or prev_gray.shape != curr_gray.shape:
            return
        diff = cv2.absdiff(curr_gray, prev_gray).astype(np.float32)
        dmean = float(diff.mean())  # 0~255
        self._count += 1
        self._sum += dmean
        self._sqsum += dmean * dmean
        if self._count >= self.warmup_frames:
            mu = self._sum / self._count
            var = max(self._sqsum / self._count - mu * mu, 0.0)
            sigma = var ** 0.5
            suggest = int(round(mu + self.k_sigma * sigma))
            # clamp to [2, 255]
            self._suggest_thr = int(min(max(suggest, 2), 255))
            self._ready = True

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def suggested_threshold(self) -> int:
        return self._suggest_thr


# ===============================
# Camera source abstraction
# ===============================

class CameraSource:
    """Unified wrapper around CameraController / OpenCV capture."""

    def __init__(
        self,
        use_opencv: bool,
        config_path: Path,
        camera_index: int,
        logger: logging.Logger,
    ):
        self._controller: Optional[CameraController] = None
        self._capture: Optional[cv2.VideoCapture] = None
        self._use_opencv = use_opencv
        self._camera_index = camera_index
        self._config_path = config_path
        self._logger = logger

        self._open()

    def _open(self) -> None:
        if self._use_opencv:
            self._logger.info("Opening OpenCV camera index %s", self._camera_index)
            cap = cv2.VideoCapture(self._camera_index)
            if not cap or not cap.isOpened():
                raise RuntimeError(f"Failed to open OpenCV camera index {self._camera_index}")
            self._capture = cap
        else:
            self._logger.info("Opening CameraController with config: %s", self._config_path)
            config = DetectionConfig.from_yaml(str(self._config_path))
            controller = CameraController(config)
            controller.initialize()
            self._controller = controller

    def reopen(self) -> None:
        self.shutdown()
        self._open()

    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame in BGR format, or None if unavailable."""
        if self._controller:
            return self._controller.capture_frame()
        if self._capture:
            ok, frame = self._capture.read()
            if not ok:
                return None
            return frame
        return None

    def shutdown(self) -> None:
        if self._controller:
            try:
                self._controller.shutdown()
            except Exception:  # pragma: no cover
                pass
            self._controller = None
        if self._capture:
            try:
                self._capture.release()
            except Exception:  # pragma: no cover
                pass
            self._capture = None


# ===============================
# Main Qt window
# ===============================

class CameraPreviewWindow(QMainWindow):
    """Simple window that renders frames in a QLabel and shows change metrics."""

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
    ):
        super().__init__()
        self.source = source
        self.logger = logger

        self.resize_width = max(0, resize_width)
        self.use_luma = use_luma
        self.ema_alpha = float(ema_alpha)
        self.fixed_threshold = fixed_threshold  # None → 交給自動標定
        self.calibrator = AdaptiveCalibrator(warmup_frames=warmup_frames, k_sigma=auto_k_sigma)
        self.roi = roi  # (x, y, w, h) or None
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

        self.metric_label = QLabel("Δmean: N/A  Δratio: N/A  FPS: N/A  Thr: N/A", central)
        self.metric_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.metric_label)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.setCentralWidget(central)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)
        self._timer.start(max(15, refresh_ms))

        # Optional: try set camera FPS (not guaranteed)
        if self.target_fps is not None and self.source._capture is not None:
            self.source._capture.set(cv2.CAP_PROP_FPS, float(self.target_fps))

    # ---------- helpers ----------

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.resize_width and w > self.resize_width:
            scale = self.resize_width / w
            frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if self.use_luma:
            ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            gray = ycc[:, :, 0]
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # ROI 裁切（若有）
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
        # HiDPI aware
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
        # 未就緒先給一個保守值
        return 25

    # ---------- main update ----------

    def _update_frame(self) -> None:
        frame = None
        try:
            frame = self.source.read()
        except Exception as exc:  # pragma: no cover
            self.logger.exception("Camera read error: %s", exc)

        if frame is None:
            self._consecutive_fail += 1
            self.status_bar.showMessage("No frame available", 300)
            self.metric_label.setText("Δmean:N/A  Δratio:N/A  FPS:N/A  Thr:N/A")
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
            thr = self._current_threshold()

            # 自動標定更新
            self.calibrator.update(gray, self._last_gray)

            # 計算雙指標 + EMA
            if self._last_gray is not None and self._last_gray.shape == gray.shape:
                md, cr = compute_change_metrics(gray, self._last_gray, thr=thr)
                if self._ema_mean is None:
                    self._ema_mean, self._ema_ratio = md, cr
                else:
                    a = self.ema_alpha
                    self._ema_mean = a * md + (1 - a) * self._ema_mean
                    self._ema_ratio = a * cr + (1 - a) * self._ema_ratio
                md_show = f"{self._ema_mean:.3f}"
                cr_show = f"{self._ema_ratio:.3f}"
            else:
                md_show = "Baseline"
                cr_show = "Baseline"

            self._last_gray = gray

            # 畫面渲染（BGR→RGB）
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            qimg.setDevicePixelRatio(self.devicePixelRatio())
            pixmap = self._scale_pixmap(qimg)
            self.image_label.setPixmap(pixmap)

            thr_show = f"{thr:d}" if self.calibrator.ready or self.fixed_threshold is not None else "Calibrating..."
            self.metric_label.setText(f"Δmean:{md_show}  Δratio:{cr_show}  FPS:{fps:.1f}  Thr:{thr_show}")
            self.status_bar.showMessage(f"Frame: {w}x{h}", 500)

        except Exception as exc:  # pragma: no cover
            self.logger.exception("Display error: %s", exc)
            self.metric_label.setText("Δmean:N/A  Δratio:N/A  FPS:N/A  Thr:N/A")

    # ---------- Qt events ----------

    def closeEvent(self, event):  # noqa: N802
        try:
            self._timer.stop()
        finally:
            self.source.shutdown()
        event.accept()


# ===============================
# CLI
# ===============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone Qt camera preview tool with change metrics.")
    p.add_argument("--config", type=Path, default=Path("config.yaml"),
                   help="Path to the CameraController config (YAML).")
    p.add_argument("--refresh-ms", type=int, default=62,
                   help="UI refresh interval in milliseconds (default: 62 ~= 16 fps).")
    p.add_argument("--opencv-backup", action="store_true",
                   help="Use OpenCV VideoCapture instead of the project CameraController.")
    p.add_argument("--camera-index", type=int, default=0,
                   help="OpenCV camera index (default: 0).")
    p.add_argument("--resize-width", type=int, default=320,
                   help="Downscale width for processing speed/stability (0=disable).")
    p.add_argument("--use-luma", action="store_true",
                   help="Use Y (luma) channel from YCrCb for metrics (more stable).")
    p.add_argument("--ema-alpha", type=float, default=0.2,
                   help="EMA smoothing factor for metrics (0~1, higher reacts faster).")
    p.add_argument("--thr", type=int, default=None,
                   help="Fixed pixel diff threshold for change ratio (if unset, auto-calibration decides).")
    p.add_argument("--auto-k-sigma", type=float, default=3.0,
                   help="Auto threshold = mu + k*sigma during warmup (ignored when --thr is set).")
    p.add_argument("--warmup-frames", type=int, default=20,
                   help="Warmup frames for auto-calibration.")
    p.add_argument("--roi", type=str, default=None,
                   help="ROI as 'x,y,w,h' (metrics computed within ROI only).")
    p.add_argument("--scale-quality", choices=["fast", "smooth"], default="fast",
                   help="Pixmap scaling quality for preview.")
    p.add_argument("--target-fps", type=float, default=16.0,
                   help="Attempt to set camera FPS (may not be honored).")
    p.add_argument("--reopen-failures", type=int, default=10,
                   help="Reopen camera after N consecutive read failures.")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARN", "ERROR"], default="INFO",
                   help="Logging level.")
    return p.parse_args()


def _setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("camera_preview")
    logger.setLevel(getattr(logging, level))
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(getattr(logging, level))
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(ch)
    return logger


def _parse_roi(roi_str: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    if not roi_str:
        return None
    try:
        x, y, w, h = [int(v) for v in roi_str.split(",")]
        if w <= 0 or h <= 0:
            raise ValueError
        return (x, y, w, h)
    except Exception:
        raise argparse.ArgumentTypeError("--roi expects 'x,y,w,h' with positive w/h")


def main() -> int:
    args = parse_args()
    logger = _setup_logging(args.log_level)

    app = QApplication(sys.argv)

    # Decide data source
    if not args.opencv_backup and not args.config.exists():
        QMessageBox.critical(
            None,
            "Missing config",
            f"Config file not found: {args.config}\n"
            "Use --opencv-backup to bypass CameraController.",
        )
        return 1

    try:
        source = CameraSource(
            use_opencv=args.opencv_backup,
            config_path=args.config,
            camera_index=args.camera_index,
            logger=logger,
        )
    except Exception as exc:
        logger.error("Primary camera init failed: %s", exc)
        if not args.opencv_backup:
            reply = QMessageBox.question(
                None,
                "Camera init failed",
                f"Unable to initialize CameraController:\n{exc}\n\n"
                "Would you like to try OpenCV fallback instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    source = CameraSource(
                        use_opencv=True,
                        config_path=args.config,
                        camera_index=args.camera_index,
                        logger=logger,
                    )
                except Exception as inner_exc:
                    QMessageBox.critical(
                        None,
                        "Camera init failed",
                        f"OpenCV fallback also failed:\n{inner_exc}",
                    )
                    return 1
            else:
                QMessageBox.critical(None, "Camera init failed", f"{exc}")
                return 1
        else:
            QMessageBox.critical(None, "Camera init failed", f"OpenCV capture failed:\n{exc}")
            return 1

    window = CameraPreviewWindow(
        source=source,
        refresh_ms=args.refresh_ms,
        logger=logger,
        resize_width=args.resize_width,
        use_luma=args.use_luma,
        ema_alpha=args.ema_alpha,
        fixed_threshold=args.thr,
        auto_k_sigma=args.auto_k_sigma,
        warmup_frames=args.warmup_frames,
        roi=_parse_roi(args.roi),
        scale_quality=args.scale_quality,
        target_fps=args.target_fps,
        reopen_failures=args.reopen_failures,
    )
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
