import os
import pandas as pd
from datetime import datetime
import cv2
from typing import List, Dict
from .utils import ImageUtils, DetectionResults
import numpy as np
from yolov5.utils.plots import colors

class ResultHandler:
    def __init__(self, config, base_dir: str = "Result"):
        self.base_dir = base_dir
        self.config = config
        self.colors = colors
        self.excel_path = os.path.join(self.base_dir, "results.xlsx")
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(config)
        
        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.excel_path):
            self._initialize_excel()

    def _initialize_excel(self) -> None:
        columns = [
            "時間戳記", "測試編號", 
            "結果", "信心分數", "錯誤訊息",
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
            print(f"讀取 Excel 時發生錯誤: {str(e)}")
            return pd.DataFrame()

    def _append_to_excel(self, data: Dict) -> None:
        try:
            df = self._read_excel()
            if isinstance(data.get('時間戳記'), datetime):
                data['時間戳記'] = data['時間戳記'].strftime('%Y-%m-%d %H:%M:%S')
            new_row = pd.DataFrame([data])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
        except Exception as e:
            print(f"寫入 Excel 時發生錯誤: {str(e)}")

    def _draw_detection_box(self, frame: np.ndarray, detection: Dict) -> None:
        x1, y1, x2, y2 = detection['box']
        label = f"{detection['label']} {detection['confidence']:.2f}"
        color = self.colors(detection['class_id'], True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self.image_utils.draw_label(frame, label, (x1, y1 - 10), color)

    # def save_results(self, frame: np.ndarray, detections: List[Dict], 
    #                 status: str, detector):
    #     try:
    #         result_dir, time_stamp, annotated_dir, original_dir ,anomalib_dir= \
    #             self.image_utils.create_result_directories(self.base_dir, status)

    #         # 保存原始图像
    #         original_path = os.path.join(original_dir, f"{time_stamp}.jpg")
    #         cv2.imwrite(original_path, frame)

    #         annotated_frame = frame.copy()

    #         # 绘制状态标签（PASS/FAIL）
    #         status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
    #         self.image_utils.draw_label(annotated_frame, status, (230, 230), 
    #                                     status_color, font_scale=3, thickness=3)

    #         # 创建裁剪图像的保存目录
    #         cropped_dir = os.path.join(result_dir, "cropped")
    #         os.makedirs(cropped_dir, exist_ok=True)

    #         cropped_img = None  # 預設裁剪圖像為 None

    #         if detections:
    #             for idx, det in enumerate(detections):
    #                 # 在标注图像上绘制检测框
    #                 self._draw_detection_box(annotated_frame, det)

    #                 # 获取检测框坐标
    #                 x1, y1, x2, y2 = det['box']

    #                 # 确保坐标在图像范围内
    #                 x1 = max(0, x1)
    #                 y1 = max(0, y1)
    #                 x2 = min(frame.shape[1], x2)
    #                 y2 = min(frame.shape[0], y2)

    #                 # 裁剪检测框区域
    #                 cropped_img = frame[y1:y2, x1:x2]

    #                 # 生成裁剪图像的文件名，包含时间戳和检测类别
    #                 cropped_filename = f"{time_stamp}_{det['label']}_{idx}.png"
    #                 cropped_path = os.path.join(cropped_dir, cropped_filename)

    #                 # 保存裁剪后的图像
    #                 cv2.imwrite(cropped_path, cropped_img)

    #         # 保存标注后的图像
    #         annotated_path = os.path.join(annotated_dir, f"{time_stamp}.jpg")
    #         cv2.imwrite(annotated_path, annotated_frame)

    #         image_name = f"{time_stamp}"

    #         # 準備保存到 Excel 的數據
    #         excel_data = self.detection_results.format_detection_data(
    #             detections, annotated_path, original_path)
    #         excel_data["測試編號"] = len(self._read_excel()) + 1
    #         self._append_to_excel(excel_data)
            
    #         return annotated_frame, cropped_img, image_name, anomalib_dir

    #     except Exception as e:
    #         print(f"保存结果时发生错误: {str(e)}")
    #         return frame, None, "error"


    def process_frame(self, frame: np.ndarray, detections: List[Dict], detector):
        """
        處理圖像：標註和裁切
        
        Args:
            frame: 原始圖像
            detections: 檢測結果列表
            status: 狀態 (PASS/FAIL)
            detector: 檢測器實例
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: (標註後的圖像, 裁切圖像列表)
        """
        try:
            annotated_frame = frame.copy()
            cropped_images = []
            
            if detections:
                for det in detections:
                    # 在標註圖像上繪製檢測框
                    self._draw_detection_box(annotated_frame, det)
                    
                    # 獲取檢測框座標
                    x1, y1, x2, y2 = det['box']
                    
                    # 確保座標在圖像範圍內
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # 裁切檢測框區域
                    cropped_img = frame[y1:y2, x1:x2]
                    cropped_images.append(cropped_img)

            return annotated_frame, cropped_images

        except Exception as e:
            print(f"處理圖像時發生錯誤: {str(e)}")
            return frame, []

    def save_result_images(self, frame: np.ndarray, annotated_frame: np.ndarray, 
                        cropped_images: List[np.ndarray], detections: List[Dict], 
                        status: str) -> None:
        """
        保存所有圖像結果
        
        Args:
            frame: 原始圖像
            annotated_frame: 標註後的圖像
            cropped_images: 裁切圖像列表
            detections: 檢測結果列表
            status: 狀態 (PASS/FAIL)
        """
        try:
            result_dir, time_stamp, annotated_dir, original_dir, anomalib_dir = \
                self.image_utils.create_result_directories(self.base_dir, status)
            
            status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
            self.image_utils.draw_label(annotated_frame, status, (230, 230),
                                    status_color, font_scale=3, thickness=3)
            # 创建裁剪图像的保存目录
            cropped_dir = os.path.join(result_dir, "cropped")
            os.makedirs(cropped_dir, exist_ok=True)

            # 保存原始圖像
            original_path = os.path.join(original_dir, f"{time_stamp}.jpg")
            cv2.imwrite(original_path, frame)
            
            # 保存標註圖像
            annotated_path = os.path.join(annotated_dir, f"{time_stamp}.jpg")
            cv2.imwrite(annotated_path, annotated_frame)
            
            # 保存裁切圖像
            for idx, (cropped_img, det) in enumerate(zip(cropped_images, detections)):
                cropped_filename = f"{time_stamp}_{det['label']}_{idx}.png"
                cropped_path = os.path.join(cropped_dir, cropped_filename)
                cv2.imwrite(cropped_path, cropped_img)

            # 處理 Excel 資料
            excel_data = self.detection_results.format_detection_data(
                detections, annotated_path, original_path)
            excel_data["測試編號"] = len(self._read_excel()) + 1
            self._append_to_excel(excel_data)

            image_name = f"{time_stamp}"
            return    anomalib_dir,image_name
        except Exception as e:
            print(f"保存圖像時發生錯誤: {str(e)}")