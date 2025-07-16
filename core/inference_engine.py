import torch
import cv2
import time
import numpy as np
import os
import tempfile
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from ultralytics import YOLO
from core.detector import YOLODetector
from core.logger import DetectionLogger
from core.config import DetectionConfig
from core.utils import ImageUtils, DetectionResults
from core.anomalib_lightning_inference import initialize, lightning_inference

class InferenceType(Enum):
    YOLO = "yolo"
    ANOMALIB = "anomalib"
    
    @classmethod
    def from_string(cls, value: str) -> 'InferenceType':
        value = value.lower()
        for inference_type in cls:
            if inference_type.value == value:
                return inference_type
        raise ValueError(f"不支援的推理類型: {value}")

class BaseInferenceModel:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.model = None
        self.is_initialized = False

    def initialize(self, product: str = None, area: str = None) -> bool:
        raise NotImplementedError("子類必須實現 initialize 方法")

    def infer(self, image: np.ndarray, product: str, area: str) -> Dict[str, Any]:
        raise NotImplementedError("子類必須實現 infer 方法")

    def shutdown(self):
        if hasattr(self, 'model') and self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.logger.info("記憶體已清理")

class YOLOInferenceModel(BaseInferenceModel):
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self.detector = None
        self.detection_results = None
        self.image_utils = ImageUtils()

    def initialize(self, product: str = None, area: str = None) -> bool:
        try:
            self.logger.logger.info("正在初始化 YOLO 模型...")
            self.model = YOLO(self.config.weights)
            self.model.to(self.config.device)
            self.detector = YOLODetector(self.model, self.config)
            self.detection_results = DetectionResults(self.config)
            self.is_initialized = True
            self.logger.logger.info(f"YOLO 模型初始化成功，使用設備: {self.config.device}")
            return True
        except Exception as e:
            self.logger.logger.error(f"YOLO 模型初始化失敗: {str(e)}")
            return False

    def preprocess_image(self, frame: np.ndarray, product: str, area: str) -> np.ndarray:
        try:
            target_size = self.config.imgsz
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_img = self.image_utils.letterbox(frame_rgb, size=target_size, fill_color=(128, 128, 128))
            self.logger.logger.debug(f"圖像預處理：原始尺寸={frame.shape[:2]}，目標尺寸={target_size}")
            if frame.shape[:2] != (target_size, target_size):
                self.logger.logger.warning(f"輸入圖像尺寸 {frame.shape[:2]} 不符合預期 {target_size}x{target_size}")
            return resized_img
        except Exception as e:
            self.logger.logger.error(f"YOLO 圖像預處理失敗: {str(e)}")
            raise

    def infer(self, image: np.ndarray, product: str, area: str) -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("YOLO 模型未初始化")
        
        try:
            start_time = time.time()
            processed_image = self.preprocess_image(image, product, area)
            expected_items = self.config.get_items_by_area(product, area)
            if not expected_items:
                raise ValueError(f"無效的產品或區域: {product},{area}")
            
            with torch.no_grad():
                pred = self.model(processed_image, conf=self.config.conf_thres, iou=self.config.iou_thres)
            
            result_frame, detections, missing_items = self.detector.process_detections(
                pred, processed_image, image, expected_items
            )
            
            result, missing_items = self.detection_results.evaluate_detection(detections, expected_items)
            status = "PASS" if result == "PASS" else "FAIL"
            
            inference_time = time.time() - start_time
            
            self.logger.logger.debug(f"YOLO 推理時間: {inference_time:.3f}s, 檢測數量: {len(detections)}, 圖像尺寸: {processed_image.shape}")
            
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

class AnomalibInferenceModel(BaseInferenceModel):
    def __init__(self, config: DetectionConfig):
        super().__init__(config)
        self.image_utils = ImageUtils()

    def initialize(self, product: str = None, area: str = None) -> bool:
        try:
            self.logger.logger.info(f"正在初始化 Anomalib 模型 (產品: {product}, 區域: {area})...")
            anomalib_config = self.config.anomalib_config
            if not anomalib_config:
                raise ValueError("Anomalib 配置缺失")
            initialize(config=anomalib_config, product=product, area=area)
            self.is_initialized = True
            self.logger.logger.info(f"Anomalib 模型初始化成功 (產品: {product}, 區域: {area})")
            return True
        except Exception as e:
            self.logger.logger.error(f"Anomalib 模型初始化失敗: {str(e)}")
            return False

    def infer(self, image: np.ndarray, product: str, area: str) -> Dict[str, Any]:
        if not self.is_initialized:
            self.initialize(product, area)
            if not self.is_initialized:
                raise RuntimeError("Anomalib 模型初始化失敗")
        
        try:
            start_time = time.time()
            processed_image = self.image_utils.letterbox(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                size=self.config.imgsz, 
                fill_color=(0, 0, 0)
            )
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            
            result = lightning_inference(tmp_path, thread_safe=True, enable_timing=True, product=product, area=area)
            
            os.unlink(tmp_path)
            
            if 'error' in result:
                raise RuntimeError(result['error'])
            
            anomaly_score = result.get('anomaly_score', 0.0)
            threshold = self.config.anomalib_config.get('threshold', 0.5)
            is_anomaly = anomaly_score > threshold
            status = "FAIL" if is_anomaly else "PASS"
            
            inference_time = time.time() - start_time
            
            self.logger.logger.debug(f"Anomalib 推理時間: {inference_time:.3f}s, 異常分數: {anomaly_score:.4f}, 圖像尺寸: {processed_image.shape}, 產品: {product}, 區域: {area}")
            
            return {
                "inference_type": "anomalib",
                "status": status,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "inference_time": inference_time,
                "processed_image": processed_image,
                "result_frame": processed_image,
                "output_path": result.get('heatmap_path', ''),  # 修改為 heatmap_path
                "expected_items": [],
                "product": product,
                "area": area
            }
            
        except Exception as e:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            self.logger.logger.error(f"Anomalib 推理失敗: {str(e)}")
            raise

class InferenceEngine:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.models = {}
        self.current_model_type = None

    def initialize(self) -> bool:
        try:
            self.logger.logger.info("正在初始化推理引擎...")
            if self.config.enable_yolo:
                yolo_model = YOLOInferenceModel(self.config)
                if yolo_model.initialize():
                    self.models[InferenceType.YOLO] = yolo_model
                    self.logger.logger.info("YOLO 模型已加載")
            
            if self.config.enable_anomalib:
                anomalib_model = AnomalibInferenceModel(self.config)
                self.models[InferenceType.ANOMALIB] = anomalib_model
                self.logger.logger.info("Anomalib 模型已準備")
            
            if not self.models:
                raise RuntimeError("沒有成功初始化任何推理模型")
            
            self.logger.logger.info(f"推理引擎初始化成功，可用模型: {list(self.models.keys())}")
            return True
        except Exception as e:
            self.logger.logger.error(f"推理引擎初始化失敗: {str(e)}")
            return False

    def infer(self, image: np.ndarray, product: str, area: str, inference_type: InferenceType) -> Dict[str, Any]:
        if inference_type not in self.models:
            raise ValueError(f"未初始化的推理類型: {inference_type}")
        
        try:
            model = self.models[inference_type]
            result = model.infer(image, product, area)
            result.update({
                "product": product,
                "area": area,
                "timestamp": time.time()
            })
            return result
        except Exception as e:
            self.logger.logger.error(f"推理失敗: {str(e)}")
            raise

    def get_available_models(self) -> List[str]:
        return [model_type.value for model_type in self.models.keys()]

    def switch_model(self, inference_type: InferenceType) -> bool:
        if inference_type not in self.models:
            self.logger.logger.warning(f"無法切換到未初始化的模型: {inference_type}")
            return False
        self.current_model_type = inference_type
        self.logger.logger.info(f"已切換到模型: {inference_type.value}")
        return True

    def shutdown(self):
        self.logger.logger.info("正在關閉推理引擎...")
        for model_type, model in self.models.items():
            try:
                model.shutdown()
                self.logger.logger.info(f"已關閉 {model_type.value} 模型")
            except Exception as e:
                self.logger.logger.error(f"關閉 {model_type.value} 模型時出錯: {str(e)}")
        self.models.clear()
        self.logger.logger.info("推理引擎已關閉")
        import gc
        gc.collect()
        self.logger.logger.info("記憶體已清理")