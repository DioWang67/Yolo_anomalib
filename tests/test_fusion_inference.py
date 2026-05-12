from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from core.fusion_inference import FusionInferenceRunner


class FakeModelManager:
    def __init__(self, engines):
        self.engines = engines

    def get_cached_engine(self, product: str, area: str, inference_type: str):
        return self.engines.get((product, area, inference_type))


class FakeEngine:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def infer(self, image, product, area, inference_type, output_path=None, **kwargs):
        self.calls.append(
            {
                "product": product,
                "area": area,
                "inference_type": str(inference_type),
                "output_path": output_path,
                "kwargs": kwargs,
            }
        )
        return dict(self.result)


class FakeSink:
    def get_annotated_path(self, **kwargs):
        return "Result/TEMP/anomalib.png"


def test_fusion_returns_error_when_yolo_engine_is_missing():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    runner = FusionInferenceRunner(
        FakeModelManager({}), SimpleNamespace(timeout=1), FakeSink()
    )

    result = runner.run(frame, "LED", "A", MagicMock())

    assert result["status"] == "INFERENCE_ERROR"
    assert "YOLO model not loaded" in result["error"]


def test_fusion_falls_back_to_yolo_when_anomalib_engine_is_missing():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    yolo_engine = FakeEngine({"status": "PASS", "detections": [{"class": "led"}]})
    runner = FusionInferenceRunner(
        FakeModelManager({("LED", "A", "yolo"): yolo_engine}),
        SimpleNamespace(timeout=1),
        FakeSink(),
    )

    result = runner.run(frame, "LED", "A", MagicMock())

    assert result["status"] == "PASS"
    assert result["detections"] == [{"class": "led"}]
    assert yolo_engine.calls[0]["kwargs"] == {"force": True}


def test_fusion_merges_failure_status_and_calls_anomalib_path_adjuster():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    yolo_engine = FakeEngine(
        {
            "status": "PASS",
            "detections": [{"class": "led", "confidence": 0.9, "bbox": [1, 1, 2, 2]}],
            "inference_time": 0.2,
        }
    )
    anomalib_engine = FakeEngine(
        {
            "status": "FAIL",
            "detections": [{"class": "scratch"}],
            "missing_items": ["cover"],
            "inference_time": 0.3,
            "anomaly_score": "0.75",
        }
    )
    runner = FusionInferenceRunner(
        FakeModelManager(
            {
                ("LED", "A", "yolo"): yolo_engine,
                ("LED", "A", "anomalib"): anomalib_engine,
            }
        ),
        SimpleNamespace(timeout=1),
        FakeSink(),
    )
    adjuster = MagicMock()

    result = runner.run(
        frame,
        "LED",
        "A",
        MagicMock(),
        adjust_anomalib_output_path=adjuster,
    )

    assert result["status"] == "DETECTION_FAIL"
    assert result["detections"] == [
        {"class": "led", "confidence": 0.9, "bbox": [1, 1, 2, 2]},
        {"class": "scratch"},
    ]
    assert result["missing_items"] == ["cover"]
    assert "Anomalib anomaly score failed" in result["error"]
    assert "Anomalib missing items" in result["error"]
    assert result["inference_time"] == 0.5
    assert result["anomaly_score"] == 0.75
    adjuster.assert_called_once()
