import logging
from pathlib import Path

import pytest

from core.logging_config import DetectionLogger, configure_logging, get_logger, reset_logging


@pytest.fixture(autouse=True)
def _reset_logging_state():
    reset_logging()
    yield
    reset_logging()


def test_configure_logging_idempotent(tmp_path):
    log_file = configure_logging(log_dir=str(tmp_path), stream=False)
    root = logging.getLogger()
    logger = get_logger("tests.idempotent")
    logger.info("ping")
    for handler in root.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()
    assert Path(log_file).exists()

    handler_ids = [id(handler) for handler in root.handlers]

    configure_logging(log_dir=str(tmp_path), stream=False)

    assert [id(handler)
            for handler in logging.getLogger().handlers] == handler_ids
    logging.shutdown()


def test_configure_logging_writes_file_with_context_defaults(tmp_path):
    log_file = configure_logging(log_dir=str(tmp_path), stream=False)
    logger = get_logger("tests.logging")
    logger.info("hello world")
    logging.shutdown()

    content = Path(log_file).read_text(encoding="utf-8")
    assert "hello world" in content
    assert "tests.logging" in content
    assert "[-/-/-:-]" in content


def test_detection_logger_logs_statistics(tmp_path):
    log_file = configure_logging(log_dir=str(
        tmp_path), stream=False, force=True)
    det_logger = DetectionLogger(log_dir=str(
        tmp_path), logger_name="tests.detection")
    det_logger.log_detection(
        "PASS",
        [
            {"class": "A", "confidence": 0.9},
            {"class": "A", "confidence": 0.7},
            {"class": "B", "confidence": 0.5},
        ],
    )
    logging.shutdown()

    content = Path(log_file).read_text(encoding="utf-8")
    assert "Detection status: PASS" in content
    assert "Class A: x2" in content
    assert "Class B: x1" in content
