from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox

from .source import CameraSource
from .window import CameraPreviewWindow


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the camera preview utility."""
    parser = argparse.ArgumentParser(
        description="Standalone Qt camera preview tool with change metrics."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to the CameraController config (YAML).",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=62,
        help="UI refresh interval in milliseconds (default: 62 ~= 16 fps).",
    )
    parser.add_argument(
        "--opencv-backup",
        action="store_true",
        help="Use OpenCV VideoCapture instead of the project CameraController.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=320,
        help="Downscale width for processing speed/stability (0=disable).",
    )
    parser.add_argument(
        "--use-luma",
        action="store_true",
        help="Use Y (luma) channel from YCrCb for metrics (more stable).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for metrics (0~1, higher reacts faster).",
    )
    parser.add_argument(
        "--thr",
        type=int,
        default=None,
        help="Fixed pixel diff threshold for change ratio "
        "(if unset, auto-calibration decides).",
    )
    parser.add_argument(
        "--auto-k-sigma",
        type=float,
        default=3.0,
        help="Auto threshold = mu + k*sigma during warmup (ignored when --thr is set).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Warmup frames for auto-calibration.",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI as 'x,y,w,h' (metrics computed within ROI only).",
    )
    parser.add_argument(
        "--scale-quality",
        choices=["fast", "smooth"],
        default="fast",
        help="Pixmap scaling quality for preview.",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=16.0,
        help="Attempt to set camera FPS (may not be honored).",
    )
    parser.add_argument(
        "--reopen-failures",
        type=int,
        default=10,
        help="Reopen camera after N consecutive read failures.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARN", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args(argv)


def _setup_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("camera_preview")
    log_level = getattr(logging, level)
    logger.setLevel(log_level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.handlers.clear()
    logger.addHandler(handler)
    return logger


def _parse_roi(roi_str: str | None) -> tuple[int, int, int, int] | None:
    if not roi_str:
        return None
    try:
        x, y, w, h = [int(val) for val in roi_str.split(",")]
        if w <= 0 or h <= 0:
            raise ValueError
        return (x, y, w, h)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "--roi expects 'x,y,w,h' with positive w/h"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = _setup_logging(args.log_level)

    app = QApplication(sys.argv)

    if not args.opencv_backup and not args.config.exists():
        QMessageBox.critical(
            None,
            "Missing config",
            f"Config file not found: {args.config}\n"
            "Use --opencv-backup to bypass CameraController.",
        )
        return 1

    def _create_source(use_opencv: bool) -> tuple[CameraSource | None, str | None]:
        try:
            return CameraSource(
                use_opencv=use_opencv,
                config_path=args.config,
                camera_index=args.camera_index,
                logger=logger,
            ), None
        except Exception as exc:  # pragma: no cover
            logger.error("Camera init failed: %s", exc)
            return None, str(exc)

    source, error_msg = _create_source(args.opencv_backup)
    if source is None:
        if not args.opencv_backup:
            reply = QMessageBox.question(
                None,
                "Camera init failed",
                "Unable to initialize CameraController.\n"
                "Would you like to try OpenCV fallback instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                source, error_msg = _create_source(True)
        if source is None:
            msg = "Unable to initialize any camera backend."
            if error_msg:
                msg += f"\n\nDetails: {error_msg}"
            QMessageBox.critical(
                None,
                "Camera init failed",
                msg,
            )
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
