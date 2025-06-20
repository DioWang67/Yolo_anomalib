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
        """
        預處理圖像，將其轉換為模型所需的格式
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb

    @staticmethod
    def iou(box1: List[int], box2: List[int]) -> float:
        """計算兩個框的 IoU（交並比）"""
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
        """
        檢查是否有缺少的項目
        """
        return set(expected_items) - detected_class_names
    def process_detections(self, pred, im: np.ndarray, frame: np.ndarray, expected_items: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]], Set[str]]:
        detections = []
        # 使用副本避免修改輸入影像
        annotator = self.Annotator(frame.copy(), line_width=3, example=str(self.model.names))
        overlapping_threshold = 0.3
        detected_class_names = set()

        results = pred[0]
        boxes = results.boxes
        names = results.names
        orig_h, orig_w = frame.shape[:2]
        target_h, target_w = self.config.imgsz

        for box in boxes:
            cls_id = int(box.cls.item())
            class_name = names[cls_id]
            confidence = float(box.conf.item())
            if confidence < self.config.conf_thres:  # 嚴格應用信心閾值
                continue
            xyxy = box.xyxy.cpu().numpy()[0]

            x1 = int(xyxy[0] * orig_w / target_w)
            y1 = int(xyxy[1] * orig_h / target_h)
            x2 = int(xyxy[2] * orig_w / target_w)
            y2 = int(xyxy[3] * orig_h / target_h)
            box_coords = [x1, y1, x2, y2]

            if any(self.iou(box_coords, det['bbox']) > overlapping_threshold for det in detections):
                continue

            detected_class_names.add(class_name)
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': box_coords,
                'class_id': cls_id
            })

            color = self.colors(cls_id, True)
            annotator.box_label(box.xyxy[0], f"{class_name} {confidence:.2f}", color=color)

        missing_items = self.check_missing_items(expected_items, detected_class_names)
        return annotator.result(), detections, missing_items

    def draw_results(self, frame: np.ndarray, status: str, detections: List[Dict]) -> np.ndarray:
        """
        在圖像上繪製檢測結果和狀態
        """
        result_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            color = self.colors(det['class_id'], True)
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(result_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
        cv2.putText(result_frame, f"Status: {status}", (230, 230), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 3)
        return result_frame
