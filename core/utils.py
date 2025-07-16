# utils.py
import os
import cv2
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

class ImageUtils:
    @staticmethod
    def letterbox(img, size=(640, 640), fill_color=(128, 128, 128)):
        """
        等比例縮放圖像並填充到目標尺寸（預設 640x640），保持長寬比。
        
        Args:
            img: 輸入圖像 (numpy array, RGB 格式)
            size: 目標尺寸 (height, width)
            fill_color: 填充顏色 (RGB tuple, 預設灰色 128,128,128)
        
        Returns:
            resized_img: 調整後的圖像 (RGB 格式, 640x640)
        """
        # 獲取原始圖像尺寸
        h, w = img.shape[:2]
        target_h, target_w = size
        
        # 計算縮放比例（保持長寬比）
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # 縮放圖像
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 創建目標尺寸的畫布（使用指定的填充顏色）
        padded_img = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
        
        # 計算填充邊距（居中放置）
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        
        # 將縮放後的圖像複製到畫布中心
        padded_img[top:top + new_h, left:left + new_w] = resized_img
        
        # 確保輸出尺寸為 640x640
        if padded_img.shape[:2] != (640, 640):
            padded_img = cv2.resize(padded_img, (640, 640), interpolation=cv2.INTER_AREA)
        
        return padded_img

    @staticmethod
    def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], font_scale: float = 0.6, 
                   thickness: int = 2) -> None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thickness)

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

    def evaluate_detection(self, detections: List[Dict[str, Union[str, float, List[int], int]]], expected_items: List[str]) -> Tuple[str, List[str]]:
        detected_classes = {det['class'] for det in detections}
        missing_items = [item for item in expected_items if item not in detected_classes]
        result = "PASS" if not missing_items else "FAIL"
        return result, missing_items

    def format_detection_data(self, detections: List[Dict[str, Union[str, float, List[int], int]]], 
                            annotated_path: str, original_path: str, 
                            status: str, missing_items: List[str]) -> Dict[str, Any]:
        confidence_scores = ";".join([f"{det['class']}:{det['confidence']:.2f}" for det in detections])
        error_message = "" if status == "PASS" else f"缺少元件: {', '.join(missing_items)}"
        
        return {
            "時間戳記": datetime.now(),
            "測試編號": None,
            "結果": status,
            "信心分數": confidence_scores,  # 序列化為字串
            "錯誤訊息": error_message,
            "標註影像路徑": annotated_path,
            "原始影像路徑": original_path,
            "預處理圖像路徑": ""  # 在 ResultHandler 中設置
        }