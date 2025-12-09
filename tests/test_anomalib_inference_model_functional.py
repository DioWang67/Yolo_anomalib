import os

import cv2
import numpy as np
import pytest

from core import anomalib_inference_model as aim  # 直接 patch 模組內名稱
from core.anomalib_inference_model import AnomalibInferenceModel

# ------------------ 測試用輔助 ------------------


class DummyLogger:
    """符合 self.logger.logger.info(...) 介面的極簡 logger。"""

    def __init__(self):
        self.records = []

        class _L:
            def __init__(self, outer):
                self.outer = outer

            def info(self, msg):
                self.outer.records.append(("INFO", msg))

            def debug(self, msg):
                self.outer.records.append(("DEBUG", msg))

            def warning(self, msg):
                self.outer.records.append(("WARN", msg))

            def error(self, msg):
                self.outer.records.append(("ERROR", msg))

        self.logger = _L(self)


class DummyBaseConfig:
    """提供最小可用設定物件，符合被測類別取用的屬性。"""

    def __init__(self, imgsz=256, threshold=0.5, anomalib_cfg=None):
        self.imgsz = imgsz
        self.anomalib_config = (
            anomalib_cfg if anomalib_cfg is not None else {
                "threshold": threshold}
        )


# ------------------ fixtures ------------------


@pytest.fixture
def dummy_image_dark():
    return np.zeros((50, 50, 3), dtype=np.uint8)  # 亮度低 → 分數低


@pytest.fixture
def dummy_image_bright():
    return np.full((50, 50, 3), 255, dtype=np.uint8)  # 亮度高 → 分數高


@pytest.fixture
def model(monkeypatch):
    cfg = DummyBaseConfig(imgsz=256, threshold=0.5)
    m = AnomalibInferenceModel(config=cfg)
    m.logger = DummyLogger()
    # 為穩定性，避免 letterbox 改變 shape（保留流程但不更動像素）
    monkeypatch.setattr(
        m.image_utils, "letterbox", lambda img, size, fill_color: img, raising=True
    )
    return m


# ------------------ 輸入/輸出契約（I/O Schema） ------------------


def _assert_output_schema(out, product, area):
    required_keys = {
        "inference_type",
        "status",
        "anomaly_score",
        "is_anomaly",
        "inference_time",
        "processed_image",
        "result_frame",
        "output_path",
        "expected_items",
        "product",
        "area",
    }
    assert required_keys.issubset(out.keys())
    assert out["inference_type"] == "anomalib"
    assert out["product"] == product and out["area"] == area
    assert isinstance(out["processed_image"], np.ndarray)
    assert isinstance(out["result_frame"], np.ndarray)
    assert isinstance(out["inference_time"], float)
    assert isinstance(out["anomaly_score"], float)
    assert out["status"] in {"PASS", "FAIL"}
    assert isinstance(out["expected_items"], list)


# ------------------ 測試案例 ------------------


def test_initialize_success(monkeypatch):
    cfg = DummyBaseConfig(anomalib_cfg={"threshold": 0.5})
    called = {"init": False}

    def fake_initialize(config, product, area):
        called["init"] = True

    monkeypatch.setattr(aim, "initialize", fake_initialize, raising=True)

    m = AnomalibInferenceModel(config=cfg)
    m.logger = DummyLogger()
    ok = m.initialize("P1", "A1")
    assert ok is True and m.is_initialized is True and called["init"] is True


def test_initialize_missing_config_returns_false():
    cfg = DummyBaseConfig(anomalib_cfg=None)
    m = AnomalibInferenceModel(config=cfg)
    m.logger = DummyLogger()
    ok = m.initialize("P", "A")
    assert ok is False and m.is_initialized is False


def test_infer_io_and_decision_with_functional_stub(
    monkeypatch, model, dummy_image_dark, dummy_image_bright
):
    """
    功能性 stub：真的讀臨時 JPG，依平均亮度決定 anomaly_score。
    驗證：I/O 路徑、門檻決策、輸出契約。
    """

    def functional_stub(
        image_path, thread_safe, enable_timing, product, area, output_path
    ):
        img = cv2.imread(image_path)
        score = float(img.mean()) / 255.0  # 0.0 ~ 1.0
        return {"anomaly_score": score, "heatmap_path": "/fake/heatmap.png"}

    monkeypatch.setattr(aim, "lightning_inference",
                        functional_stub, raising=True)
    model.is_initialized = True

    # 暗圖 → PASS
    out1 = model.infer(dummy_image_dark, product="P", area="A")
    _assert_output_schema(out1, "P", "A")
    assert out1["is_anomaly"] is False and out1["status"] == "PASS"

    # 亮圖 → FAIL
    out2 = model.infer(dummy_image_bright, product="P", area="A")
    _assert_output_schema(out2, "P", "A")
    assert out2["is_anomaly"] is True and out2["status"] == "FAIL"


def test_infer_auto_initialize_and_cleanup(monkeypatch, model, dummy_image_dark):
    """未初始化時會自動初始化；臨時檔應被刪除。"""
    called = {"init": 0, "infer": 0}

    def fake_initialize(config, product, area):
        called["init"] += 1

    def fake_infer(image_path, *a, **k):
        called["infer"] += 1
        return {"anomaly_score": 0.2, "heatmap_path": "/h.png"}

    monkeypatch.setattr(aim, "initialize", fake_initialize, raising=True)
    monkeypatch.setattr(aim, "lightning_inference", fake_infer, raising=True)

    model.is_initialized = False
    removed = {"path": None}
    real_remove = os.remove

    def fake_remove(p):
        removed["path"] = p
        return real_remove(p)

    monkeypatch.setattr(os, "remove", fake_remove, raising=True)

    out = model.infer(dummy_image_dark, product="P", area="A")
    _assert_output_schema(out, "P", "A")
    assert called["init"] == 1 and called["infer"] == 1 and removed["path"] is not None


def test_infer_error_from_engine(monkeypatch, model, dummy_image_dark):
    """推論引擎回傳 error 時，infer 應 raise，並保持清理。"""
    model.is_initialized = True
    monkeypatch.setattr(
        aim, "lightning_inference", lambda *a, **k: {"error": "boom"}, raising=True
    )
    with pytest.raises(RuntimeError):
        model.infer(dummy_image_dark, product="P", area="A")


def test_infer_threshold_boundary_equals_is_pass(monkeypatch, model, dummy_image_dark):
    """分數==threshold → PASS（只有 > threshold 視為異常）。"""
    model.is_initialized = True
    model.config.anomalib_config["threshold"] = 0.5
    monkeypatch.setattr(
        aim,
        "lightning_inference",
        lambda *a, **k: {"anomaly_score": 0.5, "heatmap_path": "/h.png"},
        raising=True,
    )
    out = model.infer(dummy_image_dark, product="P", area="A")
    _assert_output_schema(out, "P", "A")
    assert (
        out["anomaly_score"] == 0.5
        and out["is_anomaly"] is False
        and out["status"] == "PASS"
    )


def test_infer_no_anomaly_score_defaults_zero(monkeypatch, model, dummy_image_dark):
    """若引擎未回傳 anomaly_score → 預設 0.0，應為 PASS（在 threshold=0.5 下）。"""
    model.is_initialized = True
    monkeypatch.setattr(
        aim,
        "lightning_inference",
        lambda *a, **k: {"heatmap_path": "/h.png"},
        raising=True,
    )
    out = model.infer(dummy_image_dark, product="P", area="A")
    _assert_output_schema(out, "P", "A")
    assert (
        out["anomaly_score"] == 0.0
        and out["is_anomaly"] is False
        and out["status"] == "PASS"
    )


def test_initialize_engine_raises_then_returns_false(monkeypatch):
    """有 anomalib_config，但引擎 initialize() 丟例外 → initialize() 應返回 False"""
    from core import anomalib_inference_model as aim

    cfg = DummyBaseConfig(anomalib_cfg={"threshold": 0.5})
    m = AnomalibInferenceModel(config=cfg)
    m.logger = DummyLogger()

    def boom_initialize(config, product, area):
        raise RuntimeError("init failed")

    monkeypatch.setattr(aim, "initialize", boom_initialize, raising=True)

    ok = m.initialize("P", "A")
    assert ok is False
    assert m.is_initialized is False


def test_infer_cleanup_warning_when_remove_fails(
    monkeypatch, dummy_image_dark, model, capsys
):
    """模擬 os.remove 拋例外 → 走 warning 分支（finally），不影響主流程"""
    from core import anomalib_inference_model as aim

    model.is_initialized = True

    # 簡單的引擎：回低分數，確保 PASS
    monkeypatch.setattr(
        aim,
        "lightning_inference",
        lambda *a, **k: {"anomaly_score": 0.1, "heatmap_path": "/h.png"},
        raising=True,
    )

    real_remove = os.remove

    def remove_boom(path):
        raise OSError("cannot remove")

    monkeypatch.setattr(os, "remove", remove_boom, raising=True)

    out = model.infer(dummy_image_dark, product="P", area="A")
    assert out["status"] == "PASS"

    # （選擇性）查看 stdout/stderr 是否有 warning 訊息（依你的 logger 實作而定）
    # captured = capsys.readouterr()
    # assert "臨時檔案刪除失敗" in (captured.err + captured.out)

    # 還原 os.remove，避免影響其他測試
    monkeypatch.setattr(os, "remove", real_remove, raising=True)


# set PYTHONPATH=%CD%
# pytest tests\test_anomalib_inference_model_functional.py ^
#   --html=reports\test_report.html --self-contained-html ^
#   --junitxml=reports\junit.xml ^
#   --cov=core\anomalib_inference_model.py --cov-report=term-missing --cov-report=xml:reports\coverage.xml
