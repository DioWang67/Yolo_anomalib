
import cv2
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Set, Union, Any

class ImageUtils:
    @staticmethod
    def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], font_scale: float = 0.6, 
                   thickness: int = 2) -> None:
        """
        在圖像上繪製標籤
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thickness)

    @staticmethod
    def create_result_directories(base_dir: str, status: str) -> Tuple[str, str, str, str, str]:
        """
        創建結果目錄結構
        """
        date_folder = datetime.now().strftime("%Y%m%d")
        time_stamp = datetime.now().strftime("%H%M%S")
        result_dir = os.path.join(base_dir, date_folder, status)
        
        annotated_dir = os.path.join(result_dir, "annotated")
        original_dir = os.path.join(result_dir, "original")
        anomalib_dir = os.path.join(result_dir, "anomalib")
        os.makedirs(annotated_dir, exist_ok=True)
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(anomalib_dir, exist_ok=True)

        return result_dir, time_stamp, annotated_dir, original_dir, anomalib_dir

class DetectionResults:
    def __init__(self, config):
        self.config = config

    def evaluate_detection(self, detections: List[Dict[str, Union[str, float, List[int], int]]], expected_items: List[str]) -> Tuple[str, List[str]]:
        detected_classes = {det['class'] for det in detections}
        missing_items = [item for item in expected_items if item not in detected_classes]
        result = "PASS" if not missing_items else "FAIL"
        return result, missing_items

    def format_detection_data(self, detections: List[Dict[str, Union[str, float, List[int], int]]], 
                            annotated_path: str, original_path: str, 
                            status: str, missing_items: List[str]) -> Dict[str, Any]:
        confidence_scores = {det['class']: det['confidence'] for det in detections}
        error_message = "" if status == "PASS" else f"缺少元件: {', '.join(missing_items)}"
        
        return {
            "時間戳記": datetime.now(),
            "測試編號": None,
            "結果": status,  # 使用傳入的 status
            "信心分數": confidence_scores,
            "錯誤訊息": error_message,
            "標註影像路徑": annotated_path,
            "原始影像路徑": original_path,
        }
