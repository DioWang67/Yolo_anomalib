import os

import numpy as np
import pytest

from core.config import DetectionConfig
from core.detection_system import DetectionSystem


class FakeEngine:
    def __init__(self):
        self.infer_calls = 0

    def initialize(self):
        return True

    def infer(self, image, product, area, inference_type_enum, output_path=None):
        self.infer_calls += 1
        status = "FAIL" if getattr(self, "force_fail", False) else "PASS"
        if inference_type_enum.value == "anomalib":
            return {
                "inference_type": "anomalib",
                "status": status,
                "anomaly_score": 0.01,
                "processed_image": np.zeros_like(image),
                "output_path": output_path,
            }
        else:
            return {
                "inference_type": "yolo",
                "status": status,
                "detections": [],
                "missing_items": [] if status == "PASS" else ["missing"],
                "processed_image": np.zeros_like(image),
            }

    def shutdown(self):
        pass


class FakeModelManager:
    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    def switch(self, base_config, product, area, inference_type):
        self.calls.append((product, area, inference_type))
        # Simulate config overrides minimal
        base_config.enable_color_check = False
        return self.engine, base_config


class FakeSink:
    def __init__(self):
        self.saved = []
        self.flushed = 0

    def get_annotated_path(self, **kwargs):
        # Return a dummy path for anomalib TEMP
        return os.path.join(
            "Result", "YYYYMMDD", "TEMP", "annotated", "anomalib", "temp.jpg"
        )

    def save(self, **kwargs):
        self.saved.append(kwargs)
        # Return minimal expected keys
        return {
            "status": "SUCCESS",
            "original_path": "orig.jpg",
            "preprocessed_path": "pre.jpg",
            "annotated_path": "ann.jpg",
            "heatmap_path": "",
            "cropped_paths": [],
        }

    def flush(self):
        self.flushed += 1

    def close(self):
        pass


@pytest.fixture
def tmp_result_dir(tmp_path):
    base = tmp_path / "Result"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


class DummyCam:
    def __init__(self, config):
        pass

    def initialize(self):
        return True

    def shutdown(self):
        pass

    def capture_frame(self):
        return None


def _mk_system(monkeypatch, tmp_result_dir):
    # Prevent real camera and real sink during DetectionSystem.__init__
    import core.detection_system as ds

    monkeypatch.setattr(ds, "CameraController", DummyCam, raising=True)
    created = []

    def _fake_sink_factory(config, base_dir, logger=None):
        s = FakeSink()
        created.append(s)
        return s

    monkeypatch.setattr(ds, "ExcelImageResultSink",
                        _fake_sink_factory, raising=True)

    def fake_load_config(self, _):
        return DetectionConfig(
            weights="dummy.pt",
            device="cpu",
            conf_thres=0.25,
            iou_thres=0.45,
            imgsz=(8, 8),
            timeout=1,
            exposure_time="1000",
            gain="1.0",
            width=8,
            height=8,
            MV_CC_GetImageBuffer_nMsec=1000,
            expected_items={},
            enable_yolo=True,
            enable_anomalib=True,
            enable_color_check=False,
            output_dir=str(tmp_result_dir),
            anomalib_config=None,
            position_config={},
            max_cache_size=1,
            buffer_limit=1,
            flush_interval=None,
            pipeline=["save_results"],
            steps={},
            backends=None,
            disable_internal_cache=True,
            save_original=False,
            save_processed=False,
            save_annotated=False,
            save_crops=False,
            save_fail_only=False,
            jpeg_quality=90,
            png_compression=3,
            max_crops_per_frame=None,
            fail_on_unexpected=False,
        )

    monkeypatch.setattr(
        ds.DetectionSystem, "load_config", fake_load_config, raising=False
    )
    sys = DetectionSystem()
    # Use the created fake sink
    fake_sink = created[0]
    # Replace model manager to return fake engine
    engine = FakeEngine()
    sys.model_manager = FakeModelManager(engine)
    # Avoid camera
    sys.camera = None
    # Force output_dir to tmp (not used by fake sink but kept for consistency)
    sys.config.output_dir = tmp_result_dir
    return sys, fake_sink


def test_detect_calls_flush_yolo(monkeypatch, tmp_result_dir):
    sys, sink = _mk_system(monkeypatch, tmp_result_dir)
    out = sys.detect("P", "A", "yolo")
    assert out["status"] in ("PASS", "FAIL", "ERROR")
    assert sink.flushed == 1
    assert "error" in out
    assert len(sink.saved) == 1


def test_detect_calls_flush_anomalib(monkeypatch, tmp_result_dir):
    # Avoid os.path.exists moving path to simplify test
    monkeypatch.setattr(os.path, "exists", lambda p: False, raising=True)
    sys, sink = _mk_system(monkeypatch, tmp_result_dir)
    out = sys.detect("P", "A", "anomalib")
    assert out["status"] in ("PASS", "FAIL", "ERROR")
    assert sink.flushed == 1
    assert "error" in out
    assert len(sink.saved) == 1


def test_detect_flush_on_failure(monkeypatch, tmp_result_dir):
    sys, sink = _mk_system(monkeypatch, tmp_result_dir)
    sys.model_manager.engine.force_fail = True
    out = sys.detect("P", "A", "yolo")
    assert out["status"] == "FAIL"
    assert sink.flushed == 1
    assert len(sink.saved) == 1
