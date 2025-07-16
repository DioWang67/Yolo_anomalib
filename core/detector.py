# detector.py
# -*- coding: utf-8 -*-
import torch
import cv2
import numpy as np
from typing import Tuple, List, Dict, Set, Any
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from .utils import ImageUtils

class YOLODetector:
    def __init__(self, model: YOLO, config):
        self.model = model
        self.config = config
        self.colors = colors
        self.image_utils = ImageUtils()
        self.Annotator = Annotator

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_img = self.image_utils.letterbox(frame_rgb, size=self.config.imgsz, stride=32, auto=True)
        return resized_img

    @staticmethod
    def iou(box1: List[int], box2: List[int]) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return intersection_area / float(box1_area + box2_area - intersection_area)

    @staticmethod
    def check_missing_items(expected_items: List[str], detected_class_names: Set[str]) -> Set[str]:
        return set(expected_items) - detected_class_names

    def process_detections(self, predictions, processed_image: np.ndarray, 
                          original_image: np.ndarray, expected_items: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str]]:
        try:
            detections = []
            detected_items = set()
            for pred in predictions[0].boxes:
                class_id = int(pred.cls)
                class_name = self.model.names[class_id]
                confidence = float(pred.conf)
                bbox = pred.xyxy[0].cpu().numpy().astype(int)
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox.tolist(),
                    "class_id": class_id
                })
                detected_items.add(class_name)
            result_frame = processed_image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                color = self.colors(det['class_id'], True)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class']}: {det['confidence']:.2f}"
                self.image_utils.draw_label(result_frame, label, (x1, y1 - 10), color)
            missing_items = [item for item in expected_items if item not in detected_items]
            return result_frame, detections, missing_items
        except Exception as e:
            raise RuntimeError(f"處理檢測結果失敗: {str(e)}")

    def draw_results(self, frame: np.ndarray, status: str, detections: List[Dict]) -> np.ndarray:
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            color = self.colors(det['class_id'], True)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(result_frame, f"Status: {status}", (230, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return result_frame