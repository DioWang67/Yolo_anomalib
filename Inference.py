
import torch
import cv2
import time
import os
import logging
import numpy as np
from typing import Dict, Optional, Tuple, Any, List
from ultralytics import YOLO
from core.detector import YOLODetector
from core.logger import DetectionLogger
from core.config import DetectionConfig
from core.result_handler import ResultHandler
from core.utils import ImageUtils, DetectionResults
from MVS_camera_control import MVSCamera

class YOLOInference:
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = DetectionLogger()
        self.config = DetectionConfig.from_yaml(config_path)
        self.camera = MVSCamera(self.config)
        self.model = self._load_model()
        self.detector = YOLODetector(self.model, self.config)
        self.result_handler = ResultHandler(self.config, logger=self.logger)
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(self.config)
        self._initialize_camera()

    def _initialize_camera(self) -> None:
        if not self.camera.enum_devices() or not self.camera.connect_to_camera():
            self.logger.logger.error("無法連接 MVS 相機")
            raise IOError("無法連接 MVS 相機")

    def _load_model(self):
        try:
            model = YOLO(self.config.weights)
            model.to(self.config.device)
            self.logger.logger.info(f"模型加載成功，使用設備: {self.config.device}")
            return model
        except Exception as e:
            self.logger.logger.error(f"模型加載失敗: {str(e)}")
            raise


    def letterbox(self,image, size=(640, 640), stride=32, auto=False):
        h, w = image.shape[:2]
        new_h, new_w = size

        if auto:
            scale = min(new_w / w, new_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w % stride != 0:
                new_w = (new_w // stride + 1) * stride
            if new_h % stride != 0:
                new_h = (new_h // stride + 1) * stride

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((size[1], size[0], 3), 114, dtype=np.uint8)  # 灰色填充
        h_offset = (size[1] - new_h) // 2
        w_offset = (size[0] - new_w) // 2
        canvas[h_offset:h_offset + new_h, w_offset:w_offset + new_w] = resized
        return canvas

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        target_size = self.config.imgsz
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = self.letterbox(frame_rgb, size=target_size, stride=32, auto=True)
        return resized_img

    def process_frame(self, frame: np.ndarray, expected_items: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        start_time = time.time()
        im = self.preprocess_image(frame)
        with torch.no_grad():
            pred = self.model(im, conf=self.config.conf_thres, iou=self.config.iou_thres)
        # 傳入副本確保原圖不被修改
        results = self.detector.process_detections(pred, im, frame.copy(), expected_items)
        inference_time = time.time() - start_time
        self.logger.logger.debug(f"推理時間: {inference_time:.3f}s, 檢測數量: {len(results[1])}")
        return results

    def handle_detection(self, frame: np.ndarray, detections: List[Dict[str, Any]], expected_items: List[str]) -> Tuple[str, np.ndarray, str, List[str]]:
        result, missing_items = self.detection_results.evaluate_detection(detections, expected_items)
        status = "PASS" if result == "PASS" else "FAIL"
        error_message = "" if status == "PASS" else f"缺少元件: {', '.join(missing_items)}" if missing_items else "檢測失敗，無預期元件"
        annotated_frame = self.result_handler.save_results(
            frame=frame.copy(),
            detections=detections,
            status=status,
            detector=self.detector,
            missing_items=missing_items,
        )
        result_frame = self.detector.draw_results(frame.copy(), status, detections)
        return status, result_frame, error_message, missing_items
    
    def detect(self, request: str) -> Dict[str, Any]:
        try:
            if not isinstance(request, str) or ',' not in request:
                raise ValueError("請求格式錯誤，應為 '<product>,<area>'")
            product, area = request.split(',', 1)
            product, area = product.strip(), area.strip()

            frame = self.camera.get_frame()
            if frame is None:
                raise RuntimeError("無法獲取相機幀")
            
            os.makedirs("results", exist_ok=True)
            timestamp = int(time.time())
            original_image_path = f"results/{product}_{area}_{timestamp}_original.jpg"
            cv2.imwrite(original_image_path, frame)
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"無法儲存原圖: {original_image_path}")

            frame = self.preprocess_image(frame)  # 使用 preprocess_image 替代 resize_with_padding
            expected_items = self.config.get_items_by_area(product, area)
            if not expected_items:
                raise ValueError(f"無效的產品或區域: {product},{area}")
            result_frame, detections, missing_items = self.process_frame(frame, expected_items)
            status, annotated_frame, error_message, missing_items = self.handle_detection(frame, detections, expected_items)
            
            annotated_image_path = f"results/{product}_{area}_{timestamp}_annotated.jpg"
            cv2.imwrite(annotated_image_path, annotated_frame)
            if not os.path.exists(annotated_image_path):
                raise FileNotFoundError(f"無法儲存標註圖: {annotated_image_path}")

            return {
                "status": status,
                "error_message": error_message,
                "detections": [{"class": det["class"], "confidence": det["confidence"], "bbox": det["bbox"], "class_id": det["class_id"]} for det in detections],
                "original_image_path": original_image_path,
                "annotated_image_path": annotated_image_path,
                "missing_items": list(missing_items)
            }
        except Exception as e:
            self.logger.logger.error(f"檢測失敗: {str(e)}")
            raise

    def shutdown(self):
        self.camera.close()
        torch.cuda.empty_cache()

def main():
    inference = None
    try:
        inference = YOLOInference()
        test_cases = ["PCBA1,E", "PCBA1,B", "PCBA2,A", "PCBA2,C"]
        for request in test_cases:
            product, area = request.split(',')
            print(f"正在檢測產品: {product}, 區域: {area}")
            result = inference.detect(request)
            print(f"結果: {result}")
            original_img = cv2.imread(result["original_image_path"])
            annotated_img = cv2.imread(result["annotated_image_path"])
            # if original_img is not None and annotated_img is not None:
            #     cv2.imshow(f"原圖 - {product}_{area}", original_img)
            #     cv2.imshow(f"檢測結果 - {product}_{area}", annotated_img)
            #     cv2.waitKey(2000)  # 顯示 2 秒
            #     cv2.destroyAllWindows()  # 統一銷毀所有窗口
            # else:
            #     print(f"無法載入圖像: 原圖 {result['original_image_path']}, 標註圖 {result['annotated_image_path']}")
            break # 只測試第一個請求，方便調試
    except Exception as e:
        print(f"程序執行出錯: {str(e)}")
    finally:
        if inference:
            inference.shutdown()

if __name__ == "__main__":
    main()