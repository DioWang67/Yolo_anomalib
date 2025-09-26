from __future__ import annotations

import logging
from datetime import datetime
from logging.config import dictConfig
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.logging_utils import ContextFilter

_CONFIG_STATE: Dict[str, Optional[str]] = {
    "configured": False,
    "log_dir": None,
    "log_file": None,
}


def _resolve_level(level: int | str) -> str:
    if isinstance(level, str):
        return level.upper()
    resolved = logging.getLevelName(level)
    return resolved if isinstance(resolved, str) else str(level)


def reset_logging() -> None:
    """Reset logging configuration (primarily for tests)."""
    global _CONFIG_STATE
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    for flt in list(root.filters):
        root.removeFilter(flt)
    _CONFIG_STATE = {"configured": False, "log_dir": None, "log_file": None}


def configure_logging(
    *,
    log_dir: str = "logs",
    level: int | str = logging.INFO,
    stream: bool = True,
    force: bool = False,
) -> Path:
    """Configure root logging with file/console handlers and context filter."""
    global _CONFIG_STATE
    if force:
        reset_logging()
    if _CONFIG_STATE["configured"] and not force:
        assert _CONFIG_STATE["log_file"] is not None
        return Path(_CONFIG_STATE["log_file"])

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_file = log_dir_path / f"detection_{datetime.now():%Y%m%d}.log"

    level_name = _resolve_level(level)

    handlers: Dict[str, Dict[str, Any]] = {
        "file": {
            "class": "logging.FileHandler",
            "level": level_name,
            "formatter": "detailed",
            "filename": str(log_file),
            "encoding": "utf-8",
            "delay": True,
            "filters": ["context"],
        }
    }
    if stream:
        handlers["console"] = {
            "class": "logging.StreamHandler",
            "level": level_name,
            "formatter": "detailed",
            "stream": "ext://sys.stdout",
            "filters": ["context"],
        }

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "context": {"()": "core.logging_utils.ContextFilter"},
            },
            "formatters": {
                "detailed": {
                    "format": (
                        "%(asctime)s - %(levelname)s - [%(product)s/%(area)s/%(infer_type)s:%(request_id)s] - "
                        "%(name)s - %(message)s"
                    )
                }
            },
            "handlers": handlers,
            "root": {
                "level": level_name,
                "handlers": list(handlers.keys()),
            },
        }
    )

    root = logging.getLogger()
    context_filter = next(
        (f for f in root.filters if isinstance(f, ContextFilter)), None
    )
    if context_filter is None:
        context_filter = ContextFilter()
        root.addFilter(context_filter)
    for handler in root.handlers:
        if not any(isinstance(f, ContextFilter) for f in handler.filters):
            handler.addFilter(context_filter)

    _CONFIG_STATE = {
        "configured": True,
        "log_dir": str(log_dir_path),
        "log_file": str(log_file),
    }
    return log_file


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger using the shared configuration."""
    return logging.getLogger(name or "core")


class DetectionLogger:
    def __init__(
        self, log_dir: str = "logs", logger_name: str = "core.detection"
    ) -> None:
        configure_logging(log_dir=log_dir)
        self.logger = logging.getLogger(logger_name)

    def log_detection(
        self, status: str, detections: Sequence[Dict[str, Any]] | None
    ) -> None:
        self.logger.info("Detection status: %s", status)
        items = list(detections or [])
        try:
            from collections import defaultdict

            confidences: Dict[str, List[float]] = defaultdict(list)
            for det in items:
                cls = str(det.get("class", "-"))
                try:
                    conf = float(det.get("confidence", 0.0))
                except (TypeError, ValueError):
                    continue
                confidences[cls].append(conf)
            if not confidences:
                return
            for cls, confs in confidences.items():
                count = len(confs)
                maximum = max(confs)
                minimum = min(confs)
                average = sum(confs) / count
                self.logger.info(
                    "Class %s: x%d, max=%.2f, min=%.2f, avg=%.2f",
                    cls,
                    count,
                    maximum,
                    minimum,
                    average,
                )
        except Exception:
            self.logger.exception(
                "Detection aggregation failed; falling back to per-detection logs"
            )
            for det in items:
                cls = det.get("class", "-")
                confidence = det.get("confidence")
                conf_str = (
                    f"{float(confidence):.2f}"
                    if isinstance(confidence, (int, float))
                    else str(confidence)
                )
                self.logger.info("Class: %s, Confidence: %s", cls, conf_str)

    def log_anomaly(self, status: str, anomaly_score: float) -> None:
        self.logger.info("Anomaly detection status: %s", status)
        self.logger.info("Anomaly score: %.4f", anomaly_score)
