# -*- coding: utf-8 -*-
import os
import numpy as np
import pytest

from core.yolo_inference_model import YOLOInferenceModel
from core.exceptions import ModelInitializationError, ModelInferenceError
from core import yolo_inference_model as yim  # 直接 patch 模組內名稱
import cv2

# ------------------ 測試輔助物件 ------------------


class DummyLogger:
    """符合 self.logger.logger.* 介面的極簡 logger。"""

    def __init__(self):
        self.records = []

        class _L:
            def __init__(self, outer):
                self.outer = outer

            def _format(self, msg, args):
                return msg % args if args else msg

            def info(self, msg, *args, **kwargs):
                self.outer.records.append(("INFO", self._format(msg, args)))

            def debug(self, msg, *args, **kwargs):
                self.outer.records.append(("DEBUG", self._format(msg, args)))

            def warning(self, msg, *args, **kwargs):
                self.outer.records.append(("WARN", self._format(msg, args)))

            def error(self, msg, *args, **kwargs):
                self.outer.records.append(("ERROR", self._format(msg, args)))

            def exception(self, msg, *args, **kwargs):
                self.outer.records.append(
                    ("EXCEPTION", self._format(msg, args)))

        self.logger = _L(self)


class DummyConfig:
    """提供 YOLOInferenceModel 所需屬性與方法。"""

    def __init__(
        self, device="cpu", imgsz=256, conf_thres=0.25, iou_thres=0.45, max_cache_size=3
    ):
        self.device = device
        self.imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_cache_size = max_cache_size
        self.weights = "fake.pt"
        self._items = {}  # (product, area) -> list[str]

    def set_items(self, product, area, items):
        self._items[(product, area)] = list(items)

    def get_items_by_area(self, product, area):
        return self._items.get((product, area), [])


# 功能性 YOLO／Detector／Validator stub
class _FakeInnerModel:
    def __init__(self):
        self.half_called = False

    def half(self):
        self.half_called = True


class FakeYOLO:
    def __init__(self, weights):
        self.weights = weights
        self.model = _FakeInnerModel()
        self.to_device = None
        self.fused = False
        self.call_count = 0

    def to(self, device):
        self.to_device = device
        return self

    def fuse(self):
        self.fused = True
        return self

    def __call__(self, img, conf, iou):
        self.call_count += 1
        # 回一個假預測
        return {"pred": "ok", "shape": img.shape, "conf": conf, "iou": iou}


class FakeDetector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.called = 0

    def process_detections(self, pred, processed_image, image, expected_items):
        self.called += 1
        # 產一個固定 bbox；result_frame 用原圖尺寸
        result_frame = np.zeros_like(image)
        detections = [
            {
                "bbox": [10, 20, 50, 60],
                "label": (expected_items[0] if expected_items else "unk"),
            }
        ]
        missing = set()
        return result_frame, detections, missing


class FakeValidator:
    def __init__(self, config, product, area):
        self.config = config
        self.product = product
        self.area = area

    def validate(self, detections):
        return detections

    def evaluate_status(self, detections, missing_items):
        return "PASS" if not missing_items else "FAIL"


class DummyCtx:
    """輕量 context manager，配合 monkeypatch 使用。"""

    def __init__(self, flag=None):
        self.flag = flag

    def __enter__(self):
        if isinstance(self.flag, dict):
            self.flag["used"] = True

    def __exit__(self, exc_type, exc, tb):
        return False


# ------------------ fixtures ------------------


@pytest.fixture
def image_bgr_dark():
    return np.zeros((60, 80, 3), dtype=np.uint8)


@pytest.fixture
def image_bgr_bright():
    return np.full((60, 80, 3), 255, dtype=np.uint8)


@pytest.fixture
def cfg_cpu():
    return DummyConfig(device="cpu", imgsz=256, max_cache_size=3)


@pytest.fixture
def model_cpu(cfg_cpu, monkeypatch):
    # 預設 patch：YOLO / Detector / Validator
    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetector, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)

    m = YOLOInferenceModel(config=cfg_cpu)
    m.logger = DummyLogger()
    # letterbox 不改變像素（保流程）
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    return m


# ------------------ 測試案例 ------------------


def test_initialize_success_and_cache_reuse(model_cpu):
    # 第一次初始化
    ok1 = model_cpu.initialize(product="P1", area="A1")
    assert ok1 is True and model_cpu.is_initialized is True
    assert ("P1", "A1") in model_cpu.model_cache
    yolo_obj, det = model_cpu.model_cache[("P1", "A1")]
    assert isinstance(yolo_obj, FakeYOLO)

    # 第二次同 key 應該直接用 cache，不再建立新 YOLO
    ok2 = model_cpu.initialize(product="P1", area="A1")
    assert ok2 is True and model_cpu.is_initialized is True
    yolo2, _ = model_cpu.model_cache[("P1", "A1")]
    assert yolo2 is yolo_obj  # 同一物件


def test_initialize_failure_raises(cfg_cpu, monkeypatch):
    # 模擬 YOLO 建立失敗的例外
    class BOOM:
        def __init__(self, *_a, **_k):
            raise RuntimeError("boom")

    monkeypatch.setattr(yim, "YOLO", BOOM, raising=True)
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    with pytest.raises(ModelInitializationError):
        m.initialize("P", "A")


def test_cache_eviction_lru(monkeypatch):
    cfg = DummyConfig(device="cpu", max_cache_size=2)
    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetector, raising=True)
    m = YOLOInferenceModel(cfg)
    m.logger = DummyLogger()

    assert m.initialize("P1", "A1") is True
    assert m.initialize("P2", "A2") is True
    # 觸發第 3 個，最舊(P1,A1) 應淘汰
    assert m.initialize("P3", "A3") is True
    assert ("P1", "A1") not in m.model_cache
    assert ("P2", "A2") in m.model_cache and ("P3", "A3") in m.model_cache


def test_preprocess_image_calls_letterbox(model_cpu, image_bgr_dark, monkeypatch):
    called = {"letterbox": 0}

    def fake_letterbox(img, size, fill_color):
        called["letterbox"] += 1
        return img

    monkeypatch.setattr(
        model_cpu.image_utils, "letterbox", fake_letterbox, raising=True
    )
    out = model_cpu.preprocess_image(image_bgr_dark, "P", "A")
    assert out.shape == image_bgr_dark.shape
    assert called["letterbox"] == 1


def test_infer_happy_path(model_cpu, cfg_cpu, image_bgr_dark, monkeypatch):
    # 準備 expected_items
    cfg_cpu.set_items("P", "A", ["bolt"])
    assert model_cpu.initialize("P", "A") is True

    # torch.inference_mode 用輕量 context
    monkeypatch.setattr(yim.torch, "inference_mode",
                        lambda: DummyCtx(), raising=True)

    out = model_cpu.infer(image_bgr_dark, product="P", area="A")
    # 檢查輸出契約
    required = {
        "inference_type",
        "status",
        "detections",
        "missing_items",
        "inference_time",
        "processed_image",
        "result_frame",
        "expected_items",
    }
    assert required.issubset(out.keys())
    assert out["inference_type"] == "yolo"
    assert out["status"] in {"PASS", "FAIL"}
    assert isinstance(out["processed_image"], np.ndarray)
    assert isinstance(out["result_frame"], np.ndarray)
    assert out["expected_items"] == ["bolt"]
    # bbox 注入的中心點與影像尺寸
    det = out["detections"][0]
    assert (
        "cx" in det and "cy" in det and "image_width" in det and "image_height" in det
    )
    assert det["cx"] == (10 + 50) / 2 and det["cy"] == (20 + 60) / 2
    assert (
        det["image_width"] == image_bgr_dark.shape[1]
        and det["image_height"] == image_bgr_dark.shape[0]
    )


def test_infer_invalid_product_area_raises(model_cpu, image_bgr_dark):
    # 未設定 expected_items → get_items_by_area 回空 → 觸發 ValueError
    model_cpu.initialize("P", "A")
    with pytest.raises(ModelInferenceError):
        model_cpu.infer(image_bgr_dark, product="P", area="A")


def test_infer_requires_initialized(cfg_cpu, image_bgr_dark):
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    with pytest.raises(RuntimeError):
        m.infer(image_bgr_dark, product="P", area="A")


def test_infer_uses_autocast_when_cuda(monkeypatch, image_bgr_dark):
    cfg = DummyConfig(device="cuda", max_cache_size=3)
    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetector, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)

    m = YOLOInferenceModel(cfg)
    m.logger = DummyLogger()
    # letterbox no-op
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    assert m.initialize("P", "A") is True

    used = {"used": False}
    monkeypatch.setattr(
        yim, "autocast", lambda *a, **k: DummyCtx(flag=used), raising=True
    )
    monkeypatch.setattr(yim.torch, "inference_mode",
                        lambda: DummyCtx(), raising=True)

    # 設定 expected_items，避免 ValueError
    cfg.set_items("P", "A", ["bolt"])
    out = m.infer(image_bgr_dark, "P", "A")
    assert used["used"] is True
    assert out["status"] in {"PASS", "FAIL"}


def test_infer_fail_status_when_missing_items(monkeypatch, image_bgr_dark, cfg_cpu):
    # 自訂 FakeDetector 讓 missing_items 非空，FakeValidator 將回 FAIL
    class FakeDetectorMissing(FakeDetector):
        def process_detections(self, pred, processed_image, image, expected_items):
            rf, det, _ = super().process_detections(
                pred, processed_image, image, expected_items
            )
            return rf, det, {"screw"}  # 缺一個

    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetectorMissing, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)

    cfg_cpu.set_items("P", "A", ["screw"])
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    assert m.initialize("P", "A") is True
    monkeypatch.setattr(yim.torch, "inference_mode",
                        lambda: DummyCtx(), raising=True)

    out = m.infer(image_bgr_dark, "P", "A")
    assert out["status"] == "FAIL"


# mkdir reports 2>NUL
# set PYTHONPATH=%CD%
# pytest tests\test_yolo_inference_model.py ^
#   --html=reports\yolo_test_report.html --self-contained-html ^
#   --junitxml=reports\yolo_junit.xml ^
#   --cov=core\yolo_inference_model.py --cov-report=term-missing --cov-report=xml:reports\yolo_coverage.xml
