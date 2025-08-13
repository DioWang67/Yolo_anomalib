import os
import pandas as pd
from datetime import datetime
import cv2
from typing import List, Dict, Any
from .utils import ImageUtils, DetectionResults
import numpy as np
from ultralytics.utils.plotting import colors
from core.logger import DetectionLogger
import shutil
from openpyxl import load_workbook
import time


class ResultHandler:
    def __init__(self, config, base_dir: str = "Result", logger: DetectionLogger = None):
        self.base_dir = base_dir
        self.config = config
        self.logger = logger or DetectionLogger()
        self.colors = colors
        self.excel_path = os.path.join(self.base_dir, "results.xlsx")
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(config)
        self.columns = [
            "時間戳記", "測試編號", "產品", "區域", "模型類型", "結果", "信心分數", "異常分數",
            "錯誤訊息", "標註影像路徑", "原始影像路徑", "預處理圖像路徑",
            "異常熱圖路徑", "裁剪圖像路徑", "檢查點路徑"
        ]
        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.excel_path):
            self._initialize_excel()

    def _initialize_excel(self) -> None:
        df = pd.DataFrame(columns=self.columns)
        df.to_excel(self.excel_path, index=False, engine='openpyxl')

    def _read_excel(self) -> pd.DataFrame:
        backup_path = self.excel_path + ".bak"
        for attempt in range(3):
            try:
                if os.path.exists(self.excel_path):
                    return pd.read_excel(self.excel_path, engine='openpyxl')
                return pd.DataFrame(columns=self.columns)
            except Exception as e:
                self.logger.logger.error(f"讀取 Excel 時發生錯誤（第{attempt + 1}次）: {str(e)}")
                time.sleep(0.5)
                if os.path.exists(backup_path):
                    self.logger.logger.warning(f"嘗試從備份讀取 {backup_path}")
                    try:
                        return pd.read_excel(backup_path, engine='openpyxl')
                    except Exception as be:
                        self.logger.logger.error(f"讀取備份失敗: {str(be)}")
        return pd.DataFrame(columns=self.columns)

    def _get_next_test_id(self) -> int:
        backup_path = self.excel_path + ".bak"
        for attempt in range(3):
            try:
                if os.path.exists(self.excel_path):
                    wb = load_workbook(self.excel_path)
                    sheet = wb.active
                    return sheet.max_row
                return 1
            except Exception as e:
                self.logger.logger.error(f"取得測試編號失敗（第{attempt + 1}次）: {str(e)}")
                time.sleep(0.5)
                if os.path.exists(backup_path):
                    shutil.copy(backup_path, self.excel_path)
        return 1

    def _append_to_excel(self, data: Dict) -> None:
        backup_path = self.excel_path + ".bak"
        row = []
        for col in self.columns:
            value = data.get(col, "")
            if col == "時間戳記" and isinstance(value, datetime):
                value = value.strftime('%Y-%m-%d %H:%M:%S')
            row.append(value)

        for attempt in range(3):
            try:
                if os.path.exists(self.excel_path):
                    shutil.copy(self.excel_path, backup_path)
                wb = load_workbook(self.excel_path)
                ws = wb.active
                ws.append(row)
                wb.save(self.excel_path)
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                self.logger.logger.info(f"Excel 數據已保存至 {self.excel_path}")
                return
            except PermissionError:
                self.logger.logger.error(f"權限拒絕，無法寫入 {self.excel_path}，請檢查檔案是否被占用或權限設置")
            except Exception as e:
                self.logger.logger.error(f"寫入 Excel 時發生錯誤（第{attempt + 1}次）: {str(e)}")
            time.sleep(0.5)

        if os.path.exists(backup_path):
            shutil.copy(backup_path, self.excel_path)
            self.logger.logger.warning(f"已從備份恢復 {self.excel_path}")
            os.remove(backup_path)

    def _draw_detection_box(self, frame: np.ndarray, detection: Dict) -> None:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class']} {detection['confidence']:.2f}"
        color = self.colors(detection['class_id'], True)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self.image_utils.draw_label(frame, label, (x1, y1 - 10), color)

    def get_annotated_path(self, status: str, detector: str, product: str, area: str, anomaly_score: float = None) -> str:
        """生成標註影像的完整路徑"""
        current_date = datetime.now().strftime("%Y%m%d")
        time_stamp = datetime.now().strftime("%H%M%S")
        base_path = os.path.join(self.base_dir, current_date, status, "annotated", detector.lower())

        if detector.lower() == "anomalib" and anomaly_score is not None:
            image_name = f"{detector.lower()}_{product}_{area}_{time_stamp}_{anomaly_score:.4f}.jpg"
        else:
            image_name = f"{detector.lower()}_{product}_{area}_{time_stamp}.jpg" if product and area else f"{detector.lower()}_{time_stamp}.jpg"

        os.makedirs(base_path, exist_ok=True)
        return os.path.join(base_path, image_name)

    def save_results(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        status: str,
        detector: str,
        missing_items: List[str],
        processed_image: np.ndarray,
        anomaly_score: float = None,
        heatmap_path: str = None,
        product: str = None,
        area: str = None,
        ckpt_path: str = None,
    ) -> Dict:
        try:
            current_date = datetime.now().strftime("%Y%m%d")
            time_stamp = datetime.now().strftime("%H%M%S")
            base_path = os.path.join(self.base_dir, current_date, status)
            detector_prefix = detector.lower()

            os.makedirs(os.path.join(base_path, "original", detector_prefix), exist_ok=True)
            os.makedirs(os.path.join(base_path, "preprocessed", detector_prefix), exist_ok=True)
            os.makedirs(os.path.join(base_path, "annotated", detector_prefix), exist_ok=True)
            os.makedirs(os.path.join(base_path, "cropped", detector_prefix), exist_ok=True)

            image_name = f"{detector_prefix}_{product}_{area}_{time_stamp}.jpg" if product and area else f"{detector_prefix}_{time_stamp}.jpg"
            if detector_prefix == "anomalib" and anomaly_score is not None:
                image_name = f"{detector_prefix}_{product}_{area}_{time_stamp}_{anomaly_score:.4f}.jpg"

            original_path = os.path.join(base_path, "original", detector_prefix, image_name)
            cv2.imwrite(original_path, frame)

            preprocessed_path = os.path.join(base_path, "preprocessed", detector_prefix, image_name)
            cv2.imwrite(preprocessed_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))

            # 處理標註影像
            if detector.lower() == "yolo":
                annotated_frame = processed_image.copy()
                crop_source = processed_image

                if detections:
                    for det in detections:
                        self._draw_detection_box(annotated_frame, det)

                status_color = (0, 255, 0) if status == "PASS" else (0, 0, 255)
                text_x = 50
                text_y = min(50, annotated_frame.shape[0] - 20)
                self.image_utils.draw_label(annotated_frame, f"Status: {status}", (text_x, text_y), status_color, font_scale=1.0, thickness=2)

                annotated_path = os.path.join(base_path, "annotated", detector_prefix, image_name)
                cv2.imwrite(annotated_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

            else:  # Anomalib
                annotated_path = os.path.join(base_path, "annotated", detector_prefix, image_name)
                if heatmap_path and os.path.exists(heatmap_path):
                    if heatmap_path != annotated_path:  # 檢查路徑是否相同
                        shutil.copy(heatmap_path, annotated_path)
                        self.logger.logger.info(f"Anomalib 熱圖已複製至標註影像路徑: {annotated_path}")
                    else:
                        self.logger.logger.info(f"熱圖路徑與標註路徑相同，跳過複製: {annotated_path}")
                else:
                    cv2.imwrite(annotated_path, frame)
                    self.logger.logger.warning("未找到 Anomalib 熱圖，使用原始圖像作為標註影像")

                crop_source = frame

            # 處理裁剪圖像（僅對 YOLO 有效）
            cropped_images = []
            cropped_paths = []
            if detector.lower() == "yolo" and detections:
                for idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det['bbox']
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(crop_source.shape[1], x2), min(crop_source.shape[0], y2)
                    cropped_img = crop_source[y1:y2, x1:x2]
                    cropped_filename = f"{detector_prefix}_{product}_{area}_{time_stamp}_{det['class']}_{idx}.png"
                    cropped_path = os.path.join(base_path, "cropped", detector_prefix, cropped_filename)
                    cv2.imwrite(cropped_path, cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
                    cropped_images.append(cropped_img)
                    cropped_paths.append(cropped_path)

            # 處理熱圖路徑（僅對 Anomalib 有效）
            heatmap_dest_path = annotated_path if detector.lower() == "anomalib" and heatmap_path and os.path.exists(heatmap_path) else None

            confidence_scores = ";".join([f"{det['class']}:{det['confidence']:.2f}" for det in detections]) if detections else ""
            error_message = "" if status == "PASS" else f"缺少元件: {', '.join(missing_items)}" if missing_items else "異常分數超出閾值"

            excel_data = {
                "時間戳記": datetime.now(),
                "測試編號": self._get_next_test_id(),
                "產品": product,
                "區域": area,
                "模型類型": detector,
                "結果": status,
                "信心分數": confidence_scores,
                "異常分數": anomaly_score,
                "錯誤訊息": error_message,
                "標註影像路徑": annotated_path,
                "原始影像路徑": original_path,
                "預處理圖像路徑": preprocessed_path,
                "異常熱圖路徑": heatmap_dest_path if heatmap_dest_path else "",
                "裁剪圖像路徑": ";".join(cropped_paths) if cropped_paths else "",
                "檢查點路徑": ckpt_path or "",
            }
            self._append_to_excel(excel_data)

            return {
                "status": "SUCCESS",
                "original_path": original_path,
                "preprocessed_path": preprocessed_path,
                "annotated_path": annotated_path,
                "heatmap_path": heatmap_dest_path,
                "cropped_paths": cropped_paths,
                "product": product,
                "area": area,
            }
        except Exception as e:
            self.logger.logger.error(f"保存結果失敗: {str(e)}")
            return {"status": "ERROR", "error_message": str(e)}

