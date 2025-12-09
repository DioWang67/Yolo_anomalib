import cv2
import numpy as np
import pytest
import torch

from core import detector as det_mod  # 用來 monkeypatch 模組內名稱
from core.detector import YOLODetector


# --------- 測試用替身 ---------
class DummyConfig:
    def __init__(self, imgsz=256):
        self.imgsz = imgsz


class FakeImageUtils:
    def __init__(self):
        self.calls = []

    def letterbox(self, img, size, stride=32, auto=True):
        # 記錄呼叫參數，回傳可辨識內容
        self.calls.append(
            ("letterbox", {"size": size, "stride": stride, "auto": auto}))
        return np.full_like(img, 7)  # 不改變 shape，內容改成常數便於識別

    def draw_label(self, frame, text, pos, color, font_scale=1.0, thickness=1):
        # 真的畫上去，並記錄呼叫
        self.calls.append(
            ("draw_label", {"text": text, "pos": pos, "color": color}))
        cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


class FakeBox:
    """模擬 ultralytics 的單一 box 介面"""

    def __init__(self, cls_id, conf, xyxy):
        self.cls = cls_id  # 會被 int() 取用
        self.conf = conf  # float 可直接格式化
        self.xyxy = torch.tensor(
            [xyxy], dtype=torch.float32
        )  # [1,4]，支援 .cpu().numpy()


class FakePred:
    """模擬 predictions[0].boxes 可迭代"""

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    names = {0: "bolt", 1: "screw"}


# --------- 共用 fixture ---------
@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # colors(class_id, True) -> 固定回 (0,255,0)
    monkeypatch.setattr(
        det_mod, "colors", lambda class_id, bgr=True: (0, 255, 0), raising=True
    )
    # 用假的 ImageUtils 取代
    monkeypatch.setattr(det_mod, "ImageUtils", FakeImageUtils, raising=True)


@pytest.fixture
def detector():
    return YOLODetector(model=FakeModel(), config=DummyConfig(imgsz=256))


# --------- 測試案例 ---------
def test_preprocess_image_calls_letterbox_and_params(detector):
    # 建立一張 BGR 小圖
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    out = detector.preprocess_image(frame)
    # 由 FakeImageUtils 產生，內容應全為 7，shape 與輸入相同（因我們沒真的縮放）
    assert out.shape == frame.shape and np.all(out == 7)
    # 驗證 letterbox 參數
    iu = detector.image_utils
    assert iu.calls and iu.calls[0][0] == "letterbox"
    assert iu.calls[0][1] == {"size": 256, "stride": 32, "auto": True}


def test_iou_overlap_and_no_overlap():
    # 有重疊
    iou1 = YOLODetector.iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert abs(iou1 - (25 / 175)) < 1e-6
    # 無重疊
    iou2 = YOLODetector.iou([0, 0, 10, 10], [20, 20, 30, 30])
    assert iou2 == 0


def test_check_missing_items():
    miss = YOLODetector.check_missing_items(["a", "b", "c"], {"b"})
    assert miss == {"a", "c"}


def test_process_detections_happy_path_and_labels(detector):
    # 兩個框：bolt 與 screw
    preds = [
        FakePred(
            [
                FakeBox(cls_id=0, conf=0.90, xyxy=[10, 20, 50, 60]),
                FakeBox(cls_id=1, conf=0.75, xyxy=[60, 10, 90, 40]),
            ]
        )
    ]
    processed = np.zeros((100, 120, 3), dtype=np.uint8)
    orig = np.zeros_like(processed)
    expected = ["screw", "nut", "bolt"]  # nut 缺少 → missing_items 應為 ["nut"]

    result_frame, detections, missing_items = detector.process_detections(
        preds, processed, orig, expected
    )

    # 輸出尺寸與內容
    assert result_frame.shape == processed.shape
    assert len(detections) == 2
    # 內容合理（class、confidence、bbox 轉成 list[int]）
    assert detections[0]["class"] in {"bolt", "screw"}
    assert isinstance(detections[0]["confidence"], float)
    assert all(isinstance(v, int) for v in detections[0]["bbox"])
    # 缺件順序依 expected（nut）
    assert missing_items == ["nut"]

    # 有呼叫 draw_label 兩次，文字格式正確
    calls = detector.image_utils.calls
    draw_calls = [c for c in calls if c[0] == "draw_label"]
    assert len(draw_calls) == 2
    texts = [c[1]["text"] for c in draw_calls]
    assert any(t.startswith("bolt:") for t in texts) and any(
        t.startswith("screw:") for t in texts
    )


def test_process_detections_wraps_errors(detector):
    # 壞的 predictions（缺 boxes 屬性）→ 需包成 RuntimeError
    with pytest.raises(RuntimeError) as e:
        detector.process_detections(
            [object()],
            np.zeros((10, 10, 3), np.uint8),
            np.zeros((10, 10, 3), np.uint8),
            [],
        )
    assert "處理檢測結果失敗" in str(e.value)


def test_draw_results_changes_image_and_writes_status(detector):
    # 大圖，方便驗證繪製效果
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    detections = [
        {"bbox": [30, 40, 120, 150], "class": "bolt",
            "confidence": 0.88, "class_id": 0}
    ]
    out = detector.draw_results(frame, status="PASS", detections=detections)
    # 圖像應有變化（有方框與文字）
    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)  # 內容確實被修改


# mkdir reports 2>NUL
# set PYTHONPATH=%CD%
# pytest tests\test_detector.py ^
#   --html=reports\detector_report.html --self-contained-html ^
#   --junitxml=reports\detector_junit.xml ^
#   --cov=core\detector.py --cov-report=term-missing --cov-report=xml:reports\detector_coverage.xml
