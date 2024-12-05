# main.py

import torch
import cv2
import time
import os
import pyfiglet
import numpy as np
from typing import Optional, Tuple

from core.detector import YOLODetector  # YOLO檢測器模組
from core.logger import DetectionLogger  # 日誌記錄模組
from core.config import DetectionConfig  # 配置管理模組
from core.result_handler import ResultHandler  # 結果處理模組
from core.utils import ImageUtils, DetectionResults  # 工具類和檢測結果處理模組
from yolov5.models.common import DetectMultiBackend  # YOLOv5模型載入
from yolov5.utils.general import scale_boxes  # 盒子縮放函數
from MvImport.MvCameraControl_class import *  # MVS相機控制類
from MVS_camera_control import MVSCamera  # MVS相機控制模組
from fastflow_detector import detect_anomaly, save_detection_results  # 異常檢測相關函數
import sys
sys.stdout.reconfigure(encoding='utf-8')  # 重新配置標準輸出編碼為UTF-8

class YOLOInference:
    def __init__(self, config_path: str = "test_config.yaml"):
        """
        初始化YOLO推理類

        參數:
            config_path (str): 配置文件路徑，默認為 "test_config.yaml"
        """
        self.logger = DetectionLogger()  # 初始化日誌記錄器
        self.config = DetectionConfig.from_yaml(config_path)  # 從配置文件加載配置
        self.model = self._load_model()  # 加載YOLO模型
        self.detector = YOLODetector(self.model, self.config)  # 初始化檢測器
        self.result_handler = ResultHandler(self.config)  # 初始化結果處理器
        self.camera = MVSCamera()  # 初始化MVS相機
        self.image_utils = ImageUtils()  # 初始化圖像工具類
        self.detection_results = DetectionResults(self.config)  # 初始化檢測結果處理

    def _load_model(self) -> DetectMultiBackend:
        """加載YOLO模型"""
        try:
            model = DetectMultiBackend(self.config.weights, device=self.config.device)  # 加載模型
            self.logger.logger.info("模型加載成功")
            return model
        except Exception as e:
            self.logger.logger.error(f"模型加載失敗: {str(e)}")
            raise

    def print_large_text(self, text: str) -> None:
        """
        打印大型文本，用於顯示狀態信息

        參數:
            text (str): 要打印的文本
        """
        ascii_art = pyfiglet.figlet_format(text)
        print(ascii_art)

    def handle_detection(self, frame: np.ndarray, detections: list, elapsed_time: float) -> Tuple[str, Optional[np.ndarray]]:
        """
        處理檢測結果，判斷是否通過檢測，並進行異常檢測

        參數:
            frame (np.ndarray): 原始圖像幀
            detections (list): 檢測到的物體列表
            elapsed_time (float): 檢測經過的時間

        返回:
            Tuple[str, Optional[np.ndarray]]: 檢測狀態和結果幀
        """
        result, error_message = self.detection_results.evaluate_detection(detections)  # 評估檢測結果
        
        if result == "PASS" or elapsed_time >= self.config.timeout:
            status = "PASS" if result == "PASS" else "FAIL"  # 設置狀態
            # 處理結果幀，獲取標註後的幀和裁剪的圖像
            annotated_frame, cropped_images = self.result_handler.process_frame(
                frame=frame,
                detections=detections,
                detector=self.detector
            )
            # 如果有裁剪後的圖片，進行異常檢測
            anomaly_results = None
            if cropped_images:
                anomaly_results = detect_anomaly(
                    cropped_images,
                    model_path=r"D:\Git\robotlearning\yolo_inference_test\weld_model.ckpt"
                )
                # 如果檢測到異常，將狀態設置為FAIL
                if anomaly_results and any(result['is_anomaly'] for result in anomaly_results):
                    status = "FAIL"

            # 保存結果圖像
            anomalib_dir, image_name = self.result_handler.save_result_images(
                frame=frame,
                annotated_frame=annotated_frame,
                cropped_images=cropped_images,
                detections=detections,
                status=status,
            )
            save_detection_results(anomalib_dir, image_name, anomaly_results)  # 保存檢測結果
            self.print_large_text(status)  # 打印狀態信息
            result_frame = self.detector.draw_results(frame.copy(), status, detections)  # 繪製結果
            return status, result_frame

        return "", None  # 如果未達到條件，返回空字符串和None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        處理單幀圖像，進行目標檢測

        參數:
            frame (np.ndarray): 輸入的圖像幀

        返回:
            Tuple[np.ndarray, list]: 處理後的幀和檢測結果列表
        """
        im = self.detector.preprocess_image(frame)  # 預處理圖像
        with torch.no_grad():
            pred = self.model(im)  # 模型推理
        return self.detector.process_detections(pred, im, frame)  # 處理檢測結果

    def run_inference(self) -> None:
        """運行推理過程，捕獲相機圖像並進行實時檢測"""
        try:
            if not self.camera.enum_devices():
                raise IOError("無法找到MVS相機")  # 檢查相機設備
            if not self.camera.connect_to_camera():
                raise IOError("無法連接MVS相機")  # 連接相機

            detecting = False  # 標誌位，表示是否正在檢測
            wait_for_restart = False  # 標誌位，表示是否等待重新開始
            result_text = ""  # 結果文本
            last_result_frame = None  # 上一次結果幀

            while True:
                frame = self.camera.get_frame()  # 獲取相機幀
                if frame is None:
                    continue
                    
                frame = cv2.resize(frame, (640, 640))  # 調整幀大小
                original_frame = frame.copy()  # 備份原始幀

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break  # 按下 'q' 鍵退出

                if not detecting and not wait_for_restart and key == ord(' '):
                    # 按下空格鍵開始檢測
                    detecting = True
                    wait_for_restart = False
                    start_time = time.time()
                    result_text = ""
                    last_result_frame = None
                    continue

                if detecting:
                    elapsed_time = time.time() - start_time  # 計算經過的時間
                    result_frame, detections = self.process_frame(frame)  # 處理當前幀
                        
                    result_text, new_result_frame = self.handle_detection(
                        original_frame, detections, elapsed_time)  # 處理檢測結果
                    
                    if result_text:
                        last_result_frame = new_result_frame  # 保存結果幀
                        detecting = False
                        wait_for_restart = True  # 等待重新開始
                    
                    # 始終顯示檢測結果
                    display_frame = self.detector.draw_results(
                        frame, result_text, detections)
                    cv2.imshow('YOLOv5 檢測', display_frame)

                elif wait_for_restart:
                    if key == ord(' '):
                        # 按下空格鍵重新開始檢測
                        detecting = True
                        wait_for_restart = False
                        start_time = time.time()
                        result_text = ""
                        last_result_frame = None
                    cv2.imshow('YOLOv5 檢測', 
                             last_result_frame if last_result_frame is not None else frame)

                else:
                    # 顯示當前幀或上一次的結果幀
                    cv2.imshow('YOLOv5 檢測', 
                             last_result_frame if last_result_frame is not None else frame)

        except Exception as e:
            self.logger.logger.error(f"執行過程中發生錯誤: {str(e)}")  # 記錄錯誤信息
            raise

        finally:
            self.camera.close()  # 關閉相機
            cv2.destroyAllWindows()  # 銷毀所有窗口
            torch.cuda.empty_cache()  # 清理CUDA緩存

if __name__ == "__main__":
    try:
        inference = YOLOInference(r"D:\Git\robotlearning\yolo_inference_test\test_config.yaml")  # 初始化推理類
        inference.run_inference()  # 運行推理
    except Exception as e:
        print(f"程序執行出錯: {str(e)}")  # 輸出錯誤信息
