import numpy as np
import pytest

from core import yolo_inference_model as yim  # 直接 patch 模組內名稱
from core.exceptions import ModelInferenceError, ModelInitializationError
from core.yolo_inference_model import YOLOInferenceModel

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
        self.position_config = {}

    def set_items(self, product, area, items):
        self._items[(product, area)] = list(items)

    def get_items_by_area(self, product, area):
        return self._items.get((product, area), [])

    def get_position_config(self, product, area):
        return self.position_config.get(product, {}).get(area, {})

    def is_position_check_enabled(self, product, area):
        cfg = self.get_position_config(product, area)
        return bool(cfg.get("enabled", False))


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


class FakeExportedYOLO(FakeYOLO):
    def __init__(self, weights):
        super().__init__(weights)
        self.call_devices = []

    def __call__(self, img, conf, iou, **kwargs):
        self.call_count += 1
        self.call_devices.append(kwargs.get("device"))
        return {
            "pred": "ok",
            "shape": img.shape,
            "conf": conf,
            "iou": iou,
            "device": kwargs.get("device"),
        }


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


def test_openvino_exported_model_skips_torch_setup_and_passes_intel_device(
    monkeypatch, image_bgr_dark, tmp_path
):
    openvino_dir = tmp_path / "best_openvino_model"
    openvino_dir.mkdir()
    (openvino_dir / "best.xml").write_text("<xml/>", encoding="utf-8")

    cfg = DummyConfig(device="cpu", max_cache_size=3)
    cfg.weights = str(openvino_dir)
    cfg.set_items("P", "A", ["bolt"])
    monkeypatch.setattr(yim, "YOLO", FakeExportedYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetector, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)

    m = YOLOInferenceModel(cfg)
    m.logger = DummyLogger()
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    monkeypatch.setattr(yim.torch, "inference_mode", lambda: DummyCtx(), raising=True)

    assert m.initialize("P", "A") is True
    assert m.runtime_info is not None
    assert m.runtime_info.runtime == "openvino"
    assert m.model.to_device is None
    assert m.model.fused is False

    out = m.infer(image_bgr_dark, "P", "A")

    assert out["status"] in {"PASS", "FAIL"}
    assert m.model.call_devices[-1] == "intel:cpu"


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


def test_infer_missing_slot_checker_disabled_by_default(monkeypatch, image_bgr_dark, cfg_cpu):
    class FakeDetectorMissing(FakeDetector):
        def process_detections(self, pred, processed_image, image, expected_items):
            rf, det, _ = super().process_detections(
                pred, processed_image, image, expected_items
            )
            return rf, det, ["screw"]

    class FakeMissingSlotChecker:
        def __init__(self, options=None):
            raise AssertionError("checker should not run when feature is not enabled")

    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetectorMissing, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)
    monkeypatch.setattr(yim, "MissingSlotChecker", FakeMissingSlotChecker, raising=True)

    cfg_cpu.set_items("P", "A", ["screw"])
    cfg_cpu.position_config = {
        "P": {"A": {"expected_boxes": {"screw": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}}}}
    }
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    assert m.initialize("P", "A") is True
    monkeypatch.setattr(yim.torch, "inference_mode", lambda: DummyCtx(), raising=True)

    out = m.infer(image_bgr_dark, "P", "A")

    assert out["status"] == "FAIL"
    assert out["missing_items"] == ["screw"]
    assert out["slot_check"] is None


def test_infer_missing_slot_checker_requires_position_check_enabled(
    monkeypatch, image_bgr_dark, cfg_cpu
):
    class FakeDetectorMissing(FakeDetector):
        def process_detections(self, pred, processed_image, image, expected_items):
            rf, det, _ = super().process_detections(
                pred, processed_image, image, expected_items
            )
            return rf, det, ["screw"]

    class FakeMissingSlotChecker:
        def __init__(self, options=None):
            raise AssertionError("checker should not run when position check is disabled")

    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetectorMissing, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)
    monkeypatch.setattr(yim, "MissingSlotChecker", FakeMissingSlotChecker, raising=True)

    cfg_cpu.set_items("P", "A", ["screw"])
    cfg_cpu.position_config = {
        "P": {
            "A": {
                "expected_boxes": {"screw": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}},
                "missing_slot_check": {"enabled": True},
            }
        }
    }
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    assert m.initialize("P", "A") is True
    monkeypatch.setattr(yim.torch, "inference_mode", lambda: DummyCtx(), raising=True)

    out = m.infer(image_bgr_dark, "P", "A")

    assert out["status"] == "FAIL"
    assert out["missing_items"] == ["screw"]
    assert out["slot_check"] is None


def test_infer_missing_slot_checker_can_recover_missing_items(
    monkeypatch, image_bgr_dark, cfg_cpu
):
    class FakeDetectorMissing(FakeDetector):
        def process_detections(self, pred, processed_image, image, expected_items):
            rf, det, _ = super().process_detections(
                pred, processed_image, image, expected_items
            )
            return rf, det, ["screw"]

    class FakeMissingSlotChecker:
        def __init__(self, options=None):
            self.options = options or {}

        def refine_missing_items(
            self, processed_image, detections, missing_items, expected_boxes
        ):
            return [], {
                "enabled": True,
                "recovered_items": ["screw"],
                "remaining_missing_items": [],
                "estimated_shift": {"dx": 0.0, "dy": 0.0},
                "decisions": [],
            }

    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetectorMissing, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)
    monkeypatch.setattr(yim, "MissingSlotChecker", FakeMissingSlotChecker, raising=True)

    cfg_cpu.set_items("P", "A", ["screw"])
    cfg_cpu.position_config = {
        "P": {
            "A": {
                "enabled": True,
                "expected_boxes": {"screw": {"x1": 0, "y1": 0, "x2": 20, "y2": 20}},
                "missing_slot_check": {"enabled": True},
            }
        }
    }
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    assert m.initialize("P", "A") is True
    monkeypatch.setattr(yim.torch, "inference_mode", lambda: DummyCtx(), raising=True)

    out = m.infer(image_bgr_dark, "P", "A")

    assert out["status"] == "PASS"
    assert out["missing_items"] == []
    assert out["slot_check"]["recovered_items"] == ["screw"]


def test_infer_slot_mismatch_does_not_report_missing_item(
    monkeypatch, image_bgr_dark, cfg_cpu
):
    class FakeDetectorSlotMismatch(FakeDetector):
        def process_detections(self, pred, processed_image, image, expected_items):
            result_frame = np.zeros_like(image)
            detections = [
                {
                    "bbox": [0, 0, 20, 20],
                    "class": "A",
                    "class_id": 0,
                    "confidence": 0.9,
                },
                {
                    "bbox": [20, 0, 40, 20],
                    "class": "A",
                    "class_id": 0,
                    "confidence": 0.9,
                },
                {
                    "bbox": [40, 0, 60, 20],
                    "class": "B",
                    "class_id": 1,
                    "confidence": 0.9,
                },
                {
                    "bbox": [60, 0, 80, 20],
                    "class": "C",
                    "class_id": 2,
                    "confidence": 0.9,
                },
                {
                    "bbox": [80, 0, 100, 20],
                    "class": "D",
                    "class_id": 3,
                    "confidence": 0.9,
                },
            ]
            return result_frame, detections, ["E"]

    monkeypatch.setattr(yim, "YOLO", FakeYOLO, raising=True)
    monkeypatch.setattr(yim, "YOLODetector", FakeDetectorSlotMismatch, raising=True)
    monkeypatch.setattr(yim, "PositionValidator", FakeValidator, raising=True)

    cfg_cpu.set_items("P", "A", ["A", "B", "C", "D", "E"])
    cfg_cpu.position_config = {
        "P": {
            "A": {
                "enabled": True,
                "expected_boxes": {
                    "A": {"x1": 0, "y1": 0, "x2": 20, "y2": 20},
                    "B": {"x1": 40, "y1": 0, "x2": 60, "y2": 20},
                    "C": {"x1": 60, "y1": 0, "x2": 80, "y2": 20},
                    "D": {"x1": 80, "y1": 0, "x2": 100, "y2": 20},
                    "E": {"x1": 20, "y1": 0, "x2": 40, "y2": 20},
                },
            }
        }
    }
    m = YOLOInferenceModel(cfg_cpu)
    m.logger = DummyLogger()
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    assert m.initialize("P", "A") is True
    monkeypatch.setattr(yim.torch, "inference_mode", lambda: DummyCtx(), raising=True)

    out = m.infer(image_bgr_dark, "P", "A")

    assert out["status"] == "FAIL"
    assert out["missing_items"] == []
    assert len(out["slot_mismatches"]) == 1
    assert out["slot_mismatches"][0]["expected_class"] == "E"
    assert out["slot_mismatches"][0]["detected_class"] == "A"


# mkdir reports 2>NUL
# set PYTHONPATH=%CD%
# pytest tests\test_yolo_inference_model.py ^
#   --html=reports\yolo_test_report.html --self-contained-html ^
#   --junitxml=reports\yolo_junit.xml ^
#   --cov=core\yolo_inference_model.py --cov-report=term-missing --cov-report=xml:reports\yolo_coverage.xml
