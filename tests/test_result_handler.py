import os
import json
import time

import cv2
import numpy as np
import pandas as pd
import pytest

from core.exceptions import ResultImageWriteError
from core.security import SecurityError
from core.services.results import handler as rh
from core.services.results.handler import ResultHandler


# ----------------- 測試輔助項目 -----------------
class DummyLogger:
    def __init__(self):
        self.records = []

        class L:
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

        self.logger = L(self)


class DummyConfig:
    """Minimal config stub exposing buffer_limit/flush_interval."""

    def __init__(self, buffer_limit=10, flush_interval=None, position_config=None):
        self.buffer_limit = buffer_limit
        self.flush_interval = flush_interval
        self.position_config = position_config or {}

    def get_position_config(self, product: str, area: str):
        return self.position_config.get(product, {}).get(area, {})


# ImageUtils / DetectionResults ??save/璅酉??堆???頛?撖虫?
class FakeImageUtils:
    def draw_label(self, frame, text, pos, color, font_scale=1.0, thickness=1):
        # 蝪∪?怠??踹??訾??祕摮?嚗?撽????批捆嚗?
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


class FakeDetectionResults:
    def __init__(self, config):
        pass


# ----------------- ?梁 fixture -----------------
@pytest.fixture
def tmp_result_dir(tmp_path):
    base = tmp_path / "Result"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # 憿?賣?踹?靘陷 ultralytics
    monkeypatch.setattr(
        rh, "colors", lambda class_id, bgr=True: (0, 255, 0), raising=True
    )
    # ?踵? ImageUtils/DetectionResults
    monkeypatch.setattr(rh, "ImageUtils", FakeImageUtils, raising=True)
    monkeypatch.setattr(rh, "DetectionResults",
                        FakeDetectionResults, raising=True)
    # ?遙雿?sleep嚗?憒?_read_excel/flush ?岫嚗?
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None, raising=False)


def _mk_img(w=64, h=48, value=128):
    img = np.full((h, w, 3), value, dtype=np.uint8)  # BGR
    return img


# ----------------- 皜祈岫獢? -----------------


def test_excel_initialized_with_columns(tmp_result_dir):
    cfg = DummyConfig(buffer_limit=10, flush_interval=None)
    h = ResultHandler(cfg, base_dir=tmp_result_dir, logger=DummyLogger())

    # 瑼?摮銝?雿迤蝣?
    assert os.path.exists(h.excel_path)
    df = pd.read_excel(h.excel_path, engine="openpyxl")
    assert list(df.columns) == h.columns


def test_get_annotated_path_patterns(tmp_result_dir):
    h = ResultHandler(DummyConfig(), base_dir=tmp_result_dir,
                      logger=DummyLogger())

    # YOLO ?賢?嚗??怠??賂?
    p1 = h.get_annotated_path(
        status="PASS", detector="YOLO", product="P", area="A")
    assert p1.endswith(".jpg")
    assert os.path.dirname(p1).replace("\\", "/").endswith("/annotated/yolo")
    assert "yolo_P_A_" in os.path.basename(p1)

    # Anomalib ?賢?嚗 anomaly_score 4 雿??賂?
    p2 = h.get_annotated_path(
        status="FAIL", detector="Anomalib", product="P", area="A", anomaly_score=0.7319
    )
    assert "_0.7319.jpg" in os.path.basename(p2)
    assert os.path.dirname(p2).replace(
        "\\", "/").endswith("/annotated/anomalib")


def test_result_paths_reject_product_traversal(tmp_result_dir):
    h = ResultHandler(DummyConfig(), base_dir=tmp_result_dir, logger=DummyLogger())

    with pytest.raises(SecurityError):
        h.get_annotated_path(
            status="PASS", detector="YOLO", product="..", area="A"
        )

    h.close()


def test_result_paths_reject_area_path_separator(tmp_result_dir):
    h = ResultHandler(DummyConfig(), base_dir=tmp_result_dir, logger=DummyLogger())

    with pytest.raises(SecurityError):
        h.get_annotated_path(
            status="PASS", detector="YOLO", product="P", area="../A"
        )

    h.close()


def test_save_results_yolo_success_and_flush(tmp_result_dir):
    # buffer_limit=1 triggers immediate flush
    h = ResultHandler(
        DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger()
    )

    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]
    detections = [
        {
            "bbox": [5, 6, 20, 22],
            "class": "screw",
            "class_id": 1,
            "confidence": 0.93,
        }
    ]
    color_res = {"is_ok": True, "items": [{"diff": 0.1}]}
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
        color_result=color_res,
        decision={"status": "PASS", "reasons": []},
        model_info={"weights": "ckpt.pt", "model_version": "1.0.0"},
        inference_time=0.123,
    )

    assert out["status"] == "SUCCESS"
    assert out["decision"] == {"status": "PASS", "reasons": []}
    assert out["model_info"] == {"weights": "ckpt.pt", "model_version": "1.0.0"}
    assert out["inference_time"] == 0.123
    assert os.path.exists(out["config_snapshot_path"])
    with open(out["config_snapshot_path"], "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["product"] == "P"
    assert snapshot["area"] == "A"
    assert snapshot["decision"] == {"status": "PASS", "reasons": []}
    assert snapshot["model_info"]["model_version"] == "1.0.0"
    assert snapshot["inference_time"] == 0.123
    assert snapshot["config"]["buffer_limit"] == 1
    for key in ("original_path", "preprocessed_path", "annotated_path"):
        assert os.path.exists(out[key])
    assert len(out["cropped_paths"]) == 1 and os.path.exists(
        out["cropped_paths"][0])

    df = pd.read_excel(h.excel_path, engine="openpyxl")
    result_col = h.columns[5]
    confidence_col = h.columns[6]
    color_status_col = h.columns[8]
    color_diff_col = h.columns[9]

    assert len(df) >= 1
    assert (df[result_col] == "PASS").any()
    assert any(
        isinstance(v, str) and "screw:0.93" in v for v in df[confidence_col].fillna("")
    )
    assert (df[color_status_col] == "PASS").any()
    diff_values = df[color_diff_col].dropna().tolist()
    assert any(abs(float(str(v)) - 0.10) < 1e-6 for v in diff_values)


def test_save_results_anomalib_with_existing_heatmap(tmp_result_dir, tmp_path):
    h = ResultHandler(
        DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger()
    )
    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]

    # 撱箔?????heatmap 瑼?
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
    # 靘祕雿??交?獢??剁?? annotated_path 雿 heatmap_path嚗?銴ˊ嚗?
    assert out["heatmap_path"] == out["annotated_path"]
    # Anomalib 銝???
    assert out["cropped_paths"] == []


def test_error_message_when_fail_and_missing_items(tmp_result_dir):
    h = ResultHandler(
        DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger()
    )
    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]
    # YOLO嚗撩隞???FAIL
    out = h.save_results(
        frame=frame,
        detections=[
            {"bbox": [0, 0, 1, 1], "class": "bolt",
                "class_id": 0, "confidence": 0.5}
        ],
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
    error_col = h.columns[10]
    assert "bolt" in str(row[error_col]) and "nut" in str(row[error_col])


def test_save_results_draws_missing_expected_box(tmp_result_dir):
    cfg = DummyConfig(buffer_limit=1)
    cfg.position_config = {
        "P": {
            "A": {
                "expected_boxes": {
                    "bolt": {"x1": 86, "y1": 96, "x2": 116, "y2": 126},
                }
            }
        }
    }
    h = ResultHandler(cfg, base_dir=tmp_result_dir, logger=DummyLogger())

    frame = _mk_img(w=160, h=160)
    processed = frame.copy()
    out = h.save_results(
        frame=frame,
        detections=[],
        status="FAIL",
        detector="yolo",
        missing_items=["bolt"],
        processed_image=processed,
        product="P",
        area="A",
        anomaly_score=None,
        heatmap_path=None,
        ckpt_path=None,
    )

    annotated = cv2.imread(out["annotated_path"])
    assert annotated is not None
    blue, green, red = (int(v) for v in annotated[96, 86])
    assert red >= 180
    assert blue <= 80
    assert green <= 80


def test_save_results_shifts_missing_expected_box_using_detected_offsets(tmp_result_dir):
    cfg = DummyConfig(buffer_limit=1)
    cfg.position_config = {
        "P": {
            "A": {
                "expected_boxes": {
                    "part_a": {"x1": 105, "y1": 95, "x2": 135, "y2": 125},
                    "part_b": {"x1": 155, "y1": 95, "x2": 185, "y2": 125},
                    "part_c": {"x1": 105, "y1": 145, "x2": 135, "y2": 175},
                    "part_d": {"x1": 155, "y1": 145, "x2": 185, "y2": 175},
                }
            }
        }
    }
    h = ResultHandler(cfg, base_dir=tmp_result_dir, logger=DummyLogger())

    frame = _mk_img(w=240, h=240)
    processed = frame.copy()
    detections = [
        {
            "bbox": [85, 85, 115, 115],
            "class": "part_a",
            "class_id": 0,
            "confidence": 0.9,
            "position_expected_key": "part_a",
            "position_offset": {"dx": -20.0, "dy": -10.0},
        },
        {
            "bbox": [135, 85, 165, 115],
            "class": "part_b",
            "class_id": 1,
            "confidence": 0.9,
            "position_expected_key": "part_b",
            "position_offset": {"dx": -20.0, "dy": -10.0},
        },
        {
            "bbox": [85, 135, 115, 165],
            "class": "part_c",
            "class_id": 2,
            "confidence": 0.9,
            "position_expected_key": "part_c",
            "position_offset": {"dx": -20.0, "dy": -10.0},
        },
    ]

    out = h.save_results(
        frame=frame,
        detections=detections,
        status="FAIL",
        detector="yolo",
        missing_items=["part_d"],
        processed_image=processed,
        product="P",
        area="A",
        anomaly_score=None,
        heatmap_path=None,
        ckpt_path=None,
    )

    annotated = cv2.imread(out["annotated_path"])
    assert annotated is not None
    blue, green, red = (int(v) for v in annotated[135, 135])
    assert red >= 180
    assert blue <= 80
    assert green <= 80


def test_save_results_yolo_draws_missing_item_location(tmp_result_dir):
    cfg = DummyConfig(
        buffer_limit=1,
        position_config={
            "P": {
                "A": {
                    "expected_boxes": {
                        "nut": {"x1": 10, "y1": 12, "x2": 35, "y2": 32}
                    }
                }
            }
        },
    )
    h = ResultHandler(cfg, base_dir=tmp_result_dir, logger=DummyLogger())
    frame = _mk_img(value=20)
    processed = _mk_img(value=20)

    out = h.save_results(
        frame=frame,
        detections=[],
        status="DETECTION_FAIL",
        detector="yolo",
        missing_items=["nut"],
        processed_image=processed,
        product="P",
        area="A",
    )

    assert out["missing_locations"] == [
        {
            "class": "nut",
            "expected_key": "nut",
            "bbox": [10, 12, 35, 32],
            "reason": "missing",
        }
    ]
    annotated = cv2.imread(out["annotated_path"])
    assert annotated is not None
    # JPEG compression can soften exact colors, so assert red channel dominance.
    b, g, r = annotated[12, 10]
    assert int(r) > 120 and int(r) > int(g) + 40 and int(r) > int(b) + 40
    assert len(out["failure_crop_paths"]) == 1
    assert "_NG_MISSING_nut_" in os.path.basename(out["failure_crop_paths"][0])
    assert os.path.exists(out["failure_crop_paths"][0])


def test_update_config_refreshes_missing_item_locations(tmp_result_dir):
    h = ResultHandler(
        DummyConfig(buffer_limit=1),
        base_dir=tmp_result_dir,
        logger=DummyLogger(),
    )
    frame = _mk_img(value=20)
    processed = _mk_img(value=20)

    before = h.save_results(
        frame=frame,
        detections=[],
        status="DETECTION_FAIL",
        detector="yolo",
        missing_items=["J3"],
        processed_image=processed,
        product="PCBA1",
        area="B",
    )
    assert before["missing_locations"] == []

    h.update_config(
        DummyConfig(
            buffer_limit=1,
            position_config={
                "PCBA1": {
                    "B": {
                        "expected_boxes": {
                            "J3": {"x1": 10, "y1": 12, "x2": 35, "y2": 32}
                        }
                    }
                }
            },
        )
    )

    after = h.save_results(
        frame=frame,
        detections=[],
        status="DETECTION_FAIL",
        detector="yolo",
        missing_items=["J3"],
        processed_image=processed,
        product="PCBA1",
        area="B",
    )

    assert after["missing_locations"] == [
        {
            "class": "J3",
            "expected_key": "J3",
            "bbox": [10, 12, 35, 32],
            "reason": "missing",
        }
    ]


def test_save_results_yolo_saves_wrong_position_and_slot_mismatch_crops(tmp_result_dir):
    h = ResultHandler(
        DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger()
    )
    frame = _mk_img(value=20)
    processed = _mk_img(value=20)

    out = h.save_results(
        frame=frame,
        detections=[
            {
                "bbox": [5, 6, 25, 26],
                "class": "ic",
                "class_id": 1,
                "confidence": 0.8,
                "position_status": "WRONG",
                "position_expected_key": "ic",
            }
        ],
        status="FAIL",
        detector="yolo",
        missing_items=[],
        processed_image=processed,
        product="P",
        area="A",
        slot_mismatches=[
            {
                "expected_key": "cap",
                "expected_class": "cap",
                "detected_class": "resistor",
                "bbox": [30, 30, 50, 50],
            }
        ],
    )

    names = [os.path.basename(path) for path in out["failure_crop_paths"]]
    assert any("_NG_POSITION_SHIFT_ic_" in name for name in names)
    assert any("_NG_WRONG_COMPONENT_cap_" in name for name in names)
    assert all(os.path.exists(path) for path in out["failure_crop_paths"])


def test_save_results_fusion_draws_missing_item_location(tmp_result_dir):
    cfg = DummyConfig(
        buffer_limit=1,
        position_config={
            "P": {
                "A": {
                    "expected_boxes": {
                        "cover": {"x1": 8, "y1": 9, "x2": 28, "y2": 29}
                    }
                }
            }
        },
    )
    h = ResultHandler(cfg, base_dir=tmp_result_dir, logger=DummyLogger())
    frame = _mk_img(value=30)
    processed = _mk_img(value=30)

    out = h.save_results(
        frame=frame,
        detections=[],
        status="DETECTION_FAIL",
        detector="fusion",
        missing_items=["cover"],
        processed_image=processed,
        product="P",
        area="A",
    )

    annotated = cv2.imread(out["annotated_path"])
    assert annotated is not None
    b, g, r = annotated[9, 8]
    assert int(r) > 120 and int(r) > int(g) + 40 and int(r) > int(b) + 40


def test_buffer_and_manual_flush(tmp_result_dir):
    # buffer_limit=3 ????撖急?
    h = ResultHandler(
        DummyConfig(buffer_limit=3), base_dir=tmp_result_dir, logger=DummyLogger()
    )
    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]
    for i in range(2):
        h.save_results(frame, [], "PASS", "yolo", [],
                       processed, product="P", area="A")
    # 撠 flush ??Excel ?府?蝛?
    df_before = pd.read_excel(h.excel_path, engine="openpyxl")
    assert len(df_before) == 0

    # ?? flush 敺?撖怠
    h.flush()
    df_after = pd.read_excel(h.excel_path, engine="openpyxl")
    assert len(df_after) == 2


def test_cv2_imwrite_failure_raises(tmp_result_dir, monkeypatch):
    h = ResultHandler(
        DummyConfig(buffer_limit=1), base_dir=tmp_result_dir, logger=DummyLogger()
    )
    frame = _mk_img()
    processed = _mk_img()[:, :, ::-1]

    called = {"n": 0}
    real_imwrite = cv2.imwrite

    def boom(path, img=None):
        called["n"] += 1
        if called["n"] == 1:
            raise RuntimeError("imwrite failed")
        return real_imwrite(path, img)

    monkeypatch.setattr(cv2, "imwrite", boom, raising=True)

    with pytest.raises(ResultImageWriteError):
        h.save_results(
            frame,
            [],
            "PASS",
            "yolo",
            [],
            processed,
            anomaly_score=None,
            heatmap_path=None,
            product="P",
            area="A",
            ckpt_path=None,
        )
