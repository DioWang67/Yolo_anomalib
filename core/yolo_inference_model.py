import cv2
import time
import torch
import numpy as np
from typing import Dict, Any
from contextlib import nullcontext
from torch.cuda.amp import autocast
from ultralytics import YOLO

from core.base_model import BaseInferenceModel
from core.detector import YOLODetector
from core.utils import ImageUtils, DetectionResults
from core.logger import DetectionLogger
from core.position_validator import PositionValidator


class YOLOInferenceModel(BaseInferenceModel):
    def __init__(self, config):
        super().__init__(config)
        self.models_cache = {}
        self.detector_cache = {}
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(self.config)

    def initialize(self, product: str = None, area: str = None) -> bool:
        key = (product or "default", area or "default")
        if key in self.models_cache:
            self.model = self.models_cache[key]
            self.detector = self.detector_cache[key]
            self.is_initialized = True
            self.logger.logger.info(f"YOLO 模型快取載入成功 (產品: {product}, 區域: {area})")
            return True

        try:
            self.logger.logger.info("正在初始化 YOLO 模型...")
            self.model = YOLO(self.config.weights)
            self.model.to(self.config.device)
            self.model.fuse()
            if self.config.device != 'cpu':
                self.model.model.half()
                torch.backends.cudnn.benchmark = True
            self.detector = YOLODetector(self.model, self.config)

            self.models_cache[key] = self.model
            self.detector_cache[key] = self.detector

            self.is_initialized = True
            self.logger.logger.info(f"YOLO 模型初始化成功，設備: {self.config.device}，產品: {product}, 區域: {area}")
            return True
        except Exception as e:
            self.logger.logger.error(f"YOLO 模型初始化失敗: {str(e)}")
            return False

    def preprocess_image(self, frame: np.ndarray, product: str, area: str) -> np.ndarray:
        target_size = self.config.imgsz
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = self.image_utils.letterbox(frame_rgb, size=target_size, fill_color=(128, 128, 128))
        self.logger.logger.debug(f"圖像預處理：原始尺寸={frame.shape[:2]}，目標尺寸={target_size}")
        return resized_img

    def infer(self, image: np.ndarray, product: str, area: str, output_path: str = None) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("YOLO 模型未初始化")

        try:
            start_time = time.time()
            processed_image = self.preprocess_image(image, product, area)
            expected_items = self.config.get_items_by_area(product, area)
            if not expected_items:
                raise ValueError(f"無效的產品或區域: {product},{area}")

            amp_ctx = autocast if self.config.device != 'cpu' else nullcontext
            with torch.inference_mode():
                with amp_ctx():
                    pred = self.model(processed_image, conf=self.config.conf_thres, iou=self.config.iou_thres)

            result_frame, detections, missing_items = self.detector.process_detections(
                pred, processed_image, image, expected_items
            )

            # ➕ 注入尺寸與中心點資訊供位置驗證用
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["image_height"] = image.shape[0]
                det["image_width"] = image.shape[1]
                det["cx"] = (x1 + x2) / 2
                det["cy"] = (y1 + y2) / 2

            # ➕ 位置校驗（由 PositionValidator 負責）
            validator = PositionValidator(self.config, product, area)
            detections = validator.validate(detections)

            # ✅ 統一檢查結果狀態（含位置誤差 + 缺漏）
            status = validator.evaluate_status(detections, missing_items)
            inference_time = time.time() - start_time

            self.logger.logger.debug(f"YOLO 推理時間: {inference_time:.3f}s, 檢測數量: {len(detections)}")

            return {
                "inference_type": "yolo",
                "status": status,
                "detections": detections,
                "missing_items": list(missing_items),
                "inference_time": inference_time,
                "processed_image": processed_image,
                "result_frame": result_frame,
                "expected_items": expected_items
            }
        except Exception as e:
            self.logger.logger.error(f"YOLO 推理失敗: {str(e)}")
            raise
