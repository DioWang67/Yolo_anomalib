import numpy as np
import pytest

from core.config import DetectionConfig
import core.detection_system as ds


def test_yolo_and_anomalib_end_to_end(monkeypatch, tmp_path):
    base_dir = tmp_path / "results"
    base_dir.mkdir()

    class StubEngine:
        def __init__(self):
            self.calls = []
            self.shutdown_calls = 0

        def infer(self, image, product, area, infer_name, output_path=None):
            self.calls.append(
                (product, area, infer_name, output_path is not None))
            frame = np.ones_like(image)
            if infer_name == "yolo":
                return {
                    "status": "PASS",
                    "detections": [
                        {"class": "Widget", "bbox": [
                            1, 1, 4, 4], "confidence": 0.9}
                    ],
                    "missing_items": [],
                    "processed_image": frame,
                    "ckpt_path": "yolo.pt",
                }
            if infer_name == "anomalib":
                return {
                    "status": "PASS",
                    "anomaly_score": 0.2,
                    "output_path": output_path
                    or str(base_dir / "anomalib_heatmap.png"),
                    "processed_image": frame * 2,
                    "ckpt_path": "anomalib.ckpt",
                }
            raise ValueError(infer_name)

        def shutdown(self):
            self.shutdown_calls += 1

    stub_engine = StubEngine()

    class StubModelManager:
        def __init__(self, logger, max_cache_size):
            self.calls = []

        def switch(self, base_config, product, area, inference_type):
            self.calls.append((product, area, inference_type))
            base_config.current_product = product
            base_config.current_area = area
            base_config.expected_items = {product: {area: ["Widget"]}}
            base_config.enable_yolo = True
            base_config.enable_anomalib = True
            base_config.pipeline = ["save_results"]
            return stub_engine, base_config

    class RecordingSink:
        def __init__(self):
            self.saved = []
            self.flush_calls = 0

        def get_annotated_path(self, **_):
            return str(base_dir / "annotated.png")

        def save(self, **kwargs):
            self.saved.append(kwargs)
            detector = kwargs.get("detector", "unknown")
            heatmap_path = kwargs.get("heatmap_path", "")
            if detector == "anomalib":
                heatmap_path = f"{detector}_annotated.png"
            return {
                "status": "SUCCESS",
                "original_path": f"{detector}_original.png",
                "preprocessed_path": f"{detector}_processed.png",
                "annotated_path": f"{detector}_annotated.png",
                "heatmap_path": heatmap_path,
                "cropped_paths": [],
            }

        def flush(self):
            self.flush_calls += 1

        def close(self):
            pass

    created_sink: dict[str, RecordingSink] = {}

    def fake_sink_factory(config, base_dir, logger=None, **_):
        sink = RecordingSink()
        created_sink["instance"] = sink
        return sink

    class DummyCamera:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return True

        def shutdown(self):
            pass

        def capture_frame(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    def fake_load_config(self, _):
        cfg = DetectionConfig(
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
            current_product=None,
            current_area=None,
            expected_items={"Widget": {"A": ["Widget"]}},
            enable_yolo=True,
            enable_anomalib=True,
            enable_color_check=False,
            output_dir=str(base_dir),
            anomalib_config={"threshold": 0.5},
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
        return cfg

    monkeypatch.setattr(
        ds.DetectionSystem, "load_config", fake_load_config, raising=False
    )
    monkeypatch.setattr(ds, "ExcelImageResultSink",
                        fake_sink_factory, raising=True)
    monkeypatch.setattr(ds, "ModelManager", StubModelManager, raising=True)
    monkeypatch.setattr(ds, "CameraController", DummyCamera, raising=True)

    system = ds.DetectionSystem()
    sink = created_sink["instance"]

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    yolo_result = system.detect("Widget", "A", "yolo", frame=frame.copy())
    assert yolo_result["status"] == "PASS"
    assert yolo_result["product"] == "Widget"
    assert yolo_result["area"] == "A"
    assert yolo_result["detections"][0]["class"] == "Widget"

    anomalib_result = system.detect(
        "Widget", "A", "anomalib", frame=frame.copy())
    assert anomalib_result["status"] == "PASS"
    assert pytest.approx(anomalib_result["anomaly_score"], rel=1e-6) == 0.2
    assert anomalib_result["heatmap_path"] == "anomalib_annotated.png"

    assert stub_engine.calls == [
        ("Widget", "A", "yolo", False),
        ("Widget", "A", "anomalib", True),
    ]
    assert sink.flush_calls == 2
    assert [entry["detector"] for entry in sink.saved] == ["yolo", "anomalib"]

    system.shutdown()
    assert stub_engine.shutdown_calls == 1
