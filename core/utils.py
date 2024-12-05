import cv2
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

class ImageUtils:
    @staticmethod
    def draw_label(frame: np.ndarray, text: str, position: Tuple[int, int], 
                   color: Tuple[int, int, int], font_scale: float = 0.6, 
                   thickness: int = 2) -> None:
        """
        在圖像上繪製標籤
        
        Args:
            frame: 要繪製的圖像
            text: 標籤文字
            position: 標籤位置 (x, y)
            color: 顏色 (B, G, R)
            font_scale: 字體大小
            thickness: 線條粗細
        """
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, color, thickness)

    @staticmethod
    def create_result_directories(base_dir: str, status: str) -> Tuple[str, str, str, str]:
        """
        創建結果目錄結構
        
        Args:
            base_dir: 基礎目錄
            status: 狀態 (PASS/FAIL)
            
        Returns:
            Tuple[str, str, str, str]: 結果目錄, 時間戳, 標註目錄, 原始目錄
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

        return result_dir, time_stamp, annotated_dir, original_dir ,anomalib_dir

class DetectionResults:
    def __init__(self, config):
        self.config = config

    def check_missing_items(self, expected_items: List[str], detected_labels):
        """
        检查缺少的项目。

        Args:
            expected_items: 预期的项目列表。
            detected_labels: 实际检测到的标签集合。

        Returns:
            Set[str]: 缺少的项目集合。
        """
        return set(expected_items) - detected_labels

    def evaluate_detection(self, detections: List[Dict], required_score: float = 0.5) -> Tuple[str, str]:
        """
        评估检测结果

        Args:
            detections: 检测结果列表
            required_score: 最低要求的置信度分数

        Returns:
            Tuple[str, str]: 结果状态和错误信息
        """
        detected_labels = {det['label'] for det in detections if det['confidence'] >= required_score}
        expected_items = self.config.expected_items

        missing_items = self.check_missing_items(expected_items, detected_labels)

        if missing_items:
            result = "FAIL"
            error_message = f"缺少的项目: {', '.join(missing_items)}"
        else:
            result = "PASS"
            error_message = ""

        return result, error_message


    def format_detection_data(self, detections: List[Dict], 
                              annotated_path: str, original_path: str) -> Dict:
        """
        格式化检测数据以供保存

        Args:
            detections: 检测结果列表
            annotated_path: 标注图像路径
            original_path: 原始图像路径

        Returns:
            Dict: 格式化后的数据
        """
        confidence_scores = {det['label']: det['confidence'] for det in detections}
        
        result, error_message = self.evaluate_detection(detections)
        
        return {
            "时间戳记": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "测试编号": None,
            "检测结果": result,
            "信心分数": confidence_scores,
            "错误讯息": error_message,
            "标注影像路径": annotated_path,
            "原始影像路径": original_path,
        }
