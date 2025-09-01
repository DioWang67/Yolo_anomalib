# -*- coding: utf-8 -*-
import os
import io
import time
import shutil
import numpy as np
import pandas as pd
import cv2
import pytest

from core import result_handler as rh
from core.result_handler import ResultHandler

# ----------------- 測試用假物件 -----------------
class DummyLogger:
    def __init__(self):
        self.records = []
        class L:
            def __init__(self, outer): self.outer = outer
            def info(self, msg):    self.outer.records.append(("INFO", msg))
            def debug(self, msg):   self.outer.records.append(("DEBUG", msg))
            def warning(self, msg): self.outer.records.append(("WARN", msg))
            def error(self, msg):   self.outer.records.append(("ERROR", msg))
        self.logger = L(self)

class DummyConfig:
    """提供 ResultHandler 取用的最小設定：buffer_limit/flush_interval。"""
    def __init__(self, buffer_limit=10, flush_interval=None):
        self.buffer_limit = buffer_limit
        self.flush_interval = flush_interval

# ImageUtils / DetectionResults 在 save/標註會用到，提供輕量實作
class FakeImageUtils:
    def draw_label(self, frame, text, pos, color, font_scale=1.0, thickness=1):
        # 簡單畫字避免相依真實字型（不驗證像素內容）
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

class FakeDetectionResults:
    def __init__(self, config): pass

# ----------------- 共用 fixture -----------------
@pytest.fixture
def tmp_result_dir(tmp_path):
    base = tmp_path / "Result"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)

@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # 顏色函數避免依賴 ultralytics
    monkeypatch.setattr(rh, "colors", lambda class_id, bgr=True: (0, 255, 0), raising=True)
    # 替換 ImageUtils/DetectionResults
    monkeypatch.setattr(rh, "ImageUtils", FakeImageUtils, raising=True)
    monkeypatch.setattr(rh, "DetectionResults", FakeDetectionResults, raising=True)
    # 加速任何 sleep（例如 _read_excel/flush 重試）
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None, raising=False)

def _mk_img(w=64, h=48, value=128):
    img = np.full((h, w, 3), value, dtype=np.uint8)  # BGR
    return img

# ----------------- 測試案例 -----------------

def test_excel_initialized_with_columns(tmp_result_dir):
    cfg = DummyConfig(buffer_limit=10, flush_interval=None)
    h = ResultHandler(cfg, base_dir=tmp_result_dir, logger=DummyLogger())

    # 檔案存在且欄位正確
    assert os.path.exists(h.excel_path)
    df = pd.read_excel(h.excel_path, engine="openpyxl")
    assert list(df.columns) == h.columns

def test_get_annotated_path_patterns(tmp_result_dir):
    h = ResultHandler(DummyConfig(), base_dir=tmp_result_dir, logger=DummyLogger())

    # YOLO 命名（不含分數）
    p1 = h.get_annotated_path(status="PASS", detector="YOLO", product="P", area="A")
    assert p1.endswith(".jpg")
    assert os.path.dirname(p1).replace("\\", "/").endswith("/annotated/yolo")
    assert "yolo_P_A_" in os.path.basename(p1)

    # Anomalib 命名（含 anomaly_score 4 位小數）
    p2 = h.get_annotated_path(status="FAIL", detector="Anomalib", product="P", area="A", anomaly_score=0.7319)
    assert "_0.7319.jpg" in os.path.basename(p2)
    assert os.path.dirname(p2).replace("\\", "/").endswith("/annotated/anomalib")

def test_save_results_yolo_success_and_flush(tmp_result_dir):
    # buffer_limit=1 立即 flush
    h = ResultHandler(DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger())

    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]  # 模擬 RGB
    detections = [{
        "bbox": [5, 6, 20, 22],
        "class": "screw",
        "class_id": 1,
        "confidence": 0.93,
    }]
    out = h.save_results(
        frame=frame,
        detections=detections,
        status="PASS",
        detector="yolo",
        missing_items=[],
        processed_image=processed,
        product="P",
        area="A",
        anomaly_score=None,
        heatmap_path=None,
        ckpt_path="ckpt.pt",
    )

    # 檔案存在
    assert out["status"] == "SUCCESS"
    for k in ("original_path", "preprocessed_path", "annotated_path"):
        assert os.path.exists(out[k])
    assert len(out["cropped_paths"]) == 1 and os.path.exists(out["cropped_paths"][0])

    # Excel 已寫入（因 buffer_limit=1）
    df = pd.read_excel(h.excel_path, engine="openpyxl")
    assert len(df) >= 1
    assert (df["結果"] == "PASS").any()
    # 信心分數格式 "class:xx.xx"
    assert any(isinstance(v, str) and "screw:0.93" in v for v in df["信心分數"].fillna(""))

def test_save_results_anomalib_with_existing_heatmap(tmp_result_dir, tmp_path):
    h = ResultHandler(DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger())
    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]

    # 建一個假的 heatmap 檔
    heatmap_file = tmp_path / "hm.jpg"
    cv2.imwrite(str(heatmap_file), _mk_img())

    out = h.save_results(
        frame=frame,
        detections=[],
        status="FAIL",
        detector="anomalib",
        missing_items=[],
        processed_image=processed,
        anomaly_score=0.88,
        heatmap_path=str(heatmap_file),
        product="P",
        area="A",
        ckpt_path=None,
    )
    assert out["status"] == "SUCCESS"
    # 依實作：若檔案存在，回傳 annotated_path 作為 heatmap_path（不複製）
    assert out["heatmap_path"] == out["annotated_path"]
    # Anomalib 不裁切
    assert out["cropped_paths"] == []

def test_error_message_when_fail_and_missing_items(tmp_result_dir):
    h = ResultHandler(DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger())
    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]
    # YOLO，缺件 → FAIL
    out = h.save_results(
        frame=frame,
        detections=[{"bbox": [0,0,1,1], "class": "bolt", "class_id": 0, "confidence": 0.5}],
        status="FAIL",
        detector="yolo",
        missing_items=["bolt", "nut"],
        processed_image=processed,
        anomaly_score=None,
        heatmap_path=None,
        product="P",
        area="A",
        ckpt_path=None,
    )
    assert out["status"] == "SUCCESS"
    df = pd.read_excel(h.excel_path, engine="openpyxl")
    row = df.iloc[-1]
    assert "缺少元件" in str(row["錯誤訊息"])
    assert "bolt" in str(row["錯誤訊息"]) and "nut" in str(row["錯誤訊息"])

def test_buffer_and_manual_flush(tmp_result_dir):
    # buffer_limit=3 → 先不寫檔
    h = ResultHandler(DummyConfig(buffer_limit=3), base_dir=tmp_result_dir, logger=DummyLogger())
    frame = _mk_img(); processed = _mk_img()[:, :, ::-1]
    for i in range(2):
        h.save_results(frame, [], "PASS", "yolo", [], processed, product="P", area="A")
    # 尚未 flush → Excel 應該還是空
    df_before = pd.read_excel(h.excel_path, engine="openpyxl")
    assert len(df_before) == 0

    # 手動 flush 後才寫入
    h.flush()
    df_after = pd.read_excel(h.excel_path, engine="openpyxl")
    assert len(df_after) == 2

def test_cv2_imwrite_failure_returns_error(tmp_result_dir, monkeypatch):
    h = ResultHandler(DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger())
    frame = _mk_img(); processed = _mk_img()[:, :, ::-1]

    # 讓第一次 imwrite 失敗以觸發 except
    called = {"n": 0}
    real_imwrite = cv2.imwrite
    def boom(path, img=None):
        called["n"] += 1
        if called["n"] == 1:
            raise RuntimeError("imwrite failed")
        return real_imwrite(path, img)
    monkeypatch.setattr(cv2, "imwrite", boom, raising=True)

    out = h.save_results(
        frame, [], "PASS", "yolo", [], processed,
        anomaly_score=None, heatmap_path=None, product="P", area="A", ckpt_path=None
    )
    assert out["status"] == "ERROR"


# mkdir reports 2>NUL
# set PYTHONPATH=%CD%
# pytest tests\test_result_handler.py ^
#   --html=reports\result_handler_report.html --self-contained-html ^
#   --junitxml=reports\result_handler_junit.xml ^
#   --cov=core\result_handler.py --cov-report=term-missing --cov-report=xml:reports\result_handler_coverage.xml