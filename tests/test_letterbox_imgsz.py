import numpy as np

from core.config import DetectionConfig
from core.utils import ImageUtils
from core.yolo_inference_model import YOLOInferenceModel


def test_yolo_preprocess_respects_imgsz():
    cfg = DetectionConfig(weights="")
    cfg.imgsz = (320, 480)
    model = YOLOInferenceModel(cfg)
    dummy = np.zeros((100, 200, 3), dtype=np.uint8)
    out = model.preprocess_image(dummy, "prod", "area")
    assert out.shape[:2] == cfg.imgsz


def test_anomalib_letterbox_respects_imgsz():
    cfg = DetectionConfig(weights="")
    cfg.imgsz = (256, 256)
    img = np.zeros((50, 100, 3), dtype=np.uint8)
    utils = ImageUtils()
    out = utils.letterbox(img, size=cfg.imgsz, fill_color=(0, 0, 0))
    assert out.shape[:2] == cfg.imgsz
