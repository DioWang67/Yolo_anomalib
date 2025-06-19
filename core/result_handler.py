
import os
import pandas as pd
from datetime import datetime
import cv2
from typing import List, Dict, Any
from .utils import ImageUtils, DetectionResults
import numpy as np
from ultralytics.utils.plotting import colors
from core.logger import DetectionLogger

class ResultHandler:
    def __init__(self, config, base_dir: str = "Result", logger: DetectionLogger = None):
        self.base_dir = base_dir
        self.config = config
        self.colors = colors
        self.excel_path = os.path.join(self.base_dir, "results.xlsx")
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(config)
        self.logger = logger or DetectionLogger()  # 使用傳入的 logger 或創建新實例
        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.excel_path):
            self._initialize_excel()

    def _initialize_excel(self) -> None:
        columns = [
            "時間戳記", "測試編號", "結果", "信心分數", "錯誤訊息",
            "標註影像路徑", "原始影像路徑"
        ]
        df = pd.DataFrame(columns=columns)
        df.to_excel(self.excel_path, index=False, engine='openpyxl')

    def _read_excel(self) -> pd.DataFrame:
        try:
            if os.path.exists(self.excel_path):
                return pd.read_excel(self.excel_path, engine='openpyxl')
            return pd.DataFrame()
        except Exception as e:
            self.logger.logger.error(f"讀取 Excel 時發生錯誤: {str(e)}")
            return pd.DataFrame()

    def _append_to_excel(self, data: Dict) -> None:
        try:
            df = self._read_excel()
            if isinstance(data.get('時間戳記'), datetime):
                data['時間戳記'] = data['時間戳記'].strftime('%Y-%m-%d %H:%M:%S')
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
            self.logger.logger.info(f"Excel 數據已保存至 {self.excel_path}")
        except PermissionError:
            self.logger.logger.error(f"權限拒絕，無法寫入 {self.excel_path}，請檢查檔案是否被占用或權限設置")
        except Exception as e:
            self.logger.logger.error(f"寫入 Excel 時發生錯誤: {str(e)}")

    def _draw_detection_box(self, frame: np.ndarray, detection: Dict) -> None:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class']} {detection['confidence']:.2f}"
        color = self.colors(detection['class_id'], True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self.image_utils.draw_label(frame, label, (x1, y1 - 10), color)

    def save_results(self, frame: np.ndarray, detections: List[Dict[str, Any]], status: str, detector, missing_items: List[str]) -> np.ndarray:
        try:
            result_dir, time_stamp, annotated_dir, original_dir, anomalib_dir = \
                self.image_utils.create_result_directories(self.base_dir, status)
            
            original_path = os.path.join(original_dir, f"{time_stamp}.jpg")
            cv2.imwrite(original_path, frame)
            
            annotated_frame = frame.copy()
            cropped_images = []
            if detections:
                for det in detections:
                    self._draw_detection_box(annotated_frame, det)
                    x1, y1, x2, y2 = det['bbox']
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_images.append(cropped_img)

            status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
            self.image_utils.draw_label(annotated_frame, f"Status: {status}", (230, 230),
                                        status_color, font_scale=3, thickness=3)
            
            annotated_path = os.path.join(annotated_dir, f"{time_stamp}.jpg")
            cv2.imwrite(annotated_path, annotated_frame)
            
            cropped_dir = os.path.join(result_dir, "cropped")
            os.makedirs(cropped_dir, exist_ok=True)
            for idx, (cropped_img, det) in enumerate(zip(cropped_images, detections)):
                cropped_filename = f"{time_stamp}_{det['class']}_{idx}.png"
                cropped_path = os.path.join(cropped_dir, cropped_filename)
                cv2.imwrite(cropped_path, cropped_img)

            excel_data = self.detection_results.format_detection_data(
                detections, annotated_path, original_path, status, missing_items
            )
            excel_data["測試編號"] = len(self._read_excel()) + 1
            self._append_to_excel(excel_data)

            return annotated_frame

        except Exception as e:
            self.logger.logger.error(f"保存結果時發生錯誤: {str(e)}")
            return frame
