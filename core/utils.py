
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np


class ImageUtils:
    @staticmethod
    def letterbox(img: np.ndarray, size: Tuple[int, int] = (640, 640), fill_color: Tuple[int, int, int] = (128, 128, 128)) -> np.ndarray:
        """Resize an image to size while keeping aspect ratio via padding."""
        h, w = img.shape[:2]
        target_h, target_w = size
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        padded_img = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        padded_img[top:top + new_h, left:left + new_w] = resized_img
        if padded_img.shape[:2] != (target_h, target_w):
            padded_img = cv2.resize(padded_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return padded_img

    @staticmethod
    def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], color: Tuple[int, int, int],
                   font_scale: float = 0.6, thickness: int = 2) -> None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    @staticmethod
    def create_result_directories(base_dir: str, status: str) -> Tuple[str, str, str, str, str]:
        date_folder = datetime.now().strftime("%Y%m%d")
        time_stamp = datetime.now().strftime("%H%M%S")
        result_dir = os.path.join(base_dir, date_folder, status)
        annotated_dir = os.path.join(result_dir, "annotated")
        original_dir = os.path.join(result_dir, "original")
        anomalib_dir = os.path.join(result_dir, "anomalib")
        preprocessed_dir = os.path.join(result_dir, "preprocessed")
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(anomalib_dir, exist_ok=True)
        os.makedirs(preprocessed_dir, exist_ok=True)
        return result_dir, time_stamp, annotated_dir, original_dir, anomalib_dir


class DetectionResults:
    def __init__(self, config):
        self.config = config

    def evaluate_detection(self, detections: List[Dict[str, Union[str, float, List[int], int]]],
                           expected_items: List[str]) -> Tuple[str, List[str]]:
        detected_classes = {det['class'] for det in detections}
        missing_items = [item for item in expected_items if item not in detected_classes]
        status = "PASS" if not missing_items else "FAIL"
        return status, missing_items

    def format_detection_data(self, detections: List[Dict[str, Union[str, float, List[int], int]]],
                              annotated_path: str, original_path: str,
                              status: str, missing_items: List[str]) -> Dict[str, Any]:
        confidence_scores = ";".join([f"{det['class']}:{det['confidence']:.2f}" for det in detections])
        error_message = ""
        if status != "PASS" and missing_items:
            error_message = "Missing items: " + ", ".join(missing_items)
        return {
            "timestamp": datetime.now(),
            "test_id": None,
            "status": status,
            "confidence_scores": confidence_scores,
            "error_message": error_message,
            "annotated_image_path": annotated_path,
            "original_image_path": original_path,
            "processed_image_path": "",
        }
