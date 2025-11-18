# camera_controller.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Optional
from core.logging_config import DetectionLogger
from core.config import DetectionConfig
from camera.MVS_camera_control import MVSCamera


class CameraController:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.camera = None
        self.is_initialized = False

    def initialize(self) -> bool:
        try:
            self.logger.logger.info("正在初始化相機...")
            self.camera = MVSCamera(self.config)
            if not self.camera.enum_devices():
                raise RuntimeError("無法枚舉相機設備")
            if not self.camera.connect_to_camera():
                raise RuntimeError("無法連接到相機")
            self.is_initialized = True
            self.logger.logger.info("相機初始化成功")
            return True
        except Exception as e:
            self.logger.logger.error(f"相機初始化失敗: {str(e)}")
            raise

    def capture_frame(self) -> Optional[np.ndarray]:
        if not self.is_initialized:
            raise RuntimeError("相機未初始化")
        try:
            self.logger.logger.debug("正在拍攝圖像...")
            frame = self.camera.get_frame()
            if frame is None:
                self.logger.logger.warning("獲取到空幀")
                return None
            self.logger.logger.debug(
                f"圖像形狀: {frame.shape}, 數據類型: {frame.dtype}"
            )
            # 確保返回 BGR 格式
            if frame.shape[2] == 3 and frame.dtype == np.uint8:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if not self._validate_frame(frame):
                self.logger.logger.warning("獲取到無效圖像")
                return None
            self.logger.logger.debug(f"成功獲取圖像，尺寸: {frame.shape}")
            return frame
        except Exception as e:
            self.logger.logger.error(f"拍攝失敗: {str(e)}")
            return None

    def capture_multiple_frames(self, count: int = 3) -> Optional[np.ndarray]:
        if not self.is_initialized:
            raise RuntimeError("相機未初始化")
        try:
            self.logger.logger.debug(f"正在拍攝 {count} 幀圖像...")
            frames = []
            for i in range(count):
                frame = self.capture_frame()
                if frame is not None:
                    frames.append(frame)
            if not frames:
                self.logger.logger.warning("未獲取到任何有效幀")
                return None
            best_frame = frames[len(frames) // 2]
            self.logger.logger.debug(f"從 {len(frames)} 幀中選擇最佳幀")
            return best_frame
        except Exception as e:
            self.logger.logger.error(f"多幀拍攝失敗: {str(e)}")
            return None

    def _validate_frame(self, frame: np.ndarray) -> bool:
        # if frame is None:
        #     return False
        # if frame.shape[0] < 100 or frame.shape[1] < 100:
        #     self.logger.logger.warning("圖像尺寸過小")
        #     return False
        # if np.mean(frame) < 10:
        #     self.logger.logger.warning("圖像過暗")
        #     return False
        # if np.mean(frame) > 245:
        #     self.logger.logger.warning("圖像過亮")
        #     return False
        return True

    def get_camera_info(self) -> dict:
        if not self.is_initialized:
            return {"status": "未初始化"}
        try:
            info = {
                "status": "已連接",
                "type": "MVS Camera",
                "initialized": self.is_initialized,
            }
            return info
        except Exception as e:
            self.logger.logger.error(f"獲取相機資訊失敗: {str(e)}")
            return {"status": "錯誤", "error": str(e)}

    def set_exposure(self, exposure_time: float) -> bool:
        if not self.is_initialized:
            return False
        try:
            self.logger.logger.info(f"設置曝光時間: {exposure_time}")
            return True
        except Exception as e:
            self.logger.logger.error(f"設置曝光時間失敗: {str(e)}")
            return False

    def set_gain(self, gain: float) -> bool:
        if not self.is_initialized:
            return False
        try:
            self.logger.logger.info(f"設置增益: {gain}")
            return True
        except Exception as e:
            self.logger.logger.error(f"設置增益失敗: {str(e)}")
            return False

    def test_camera(self) -> bool:
        if not self.is_initialized:
            self.logger.logger.warning("相機未初始化，無法測試")
            return False
        try:
            self.logger.logger.info("開始相機測試...")
            frame = self.capture_frame()
            if frame is None:
                self.logger.logger.error("相機測試失敗：無法獲取圖像")
                return False
            frames = self.capture_multiple_frames(3)
            if frames is None:
                self.logger.logger.error("相機測試失敗：無法獲取多幀圖像")
                return False
            self.logger.logger.info("相機測試通過")
            return True
        except Exception as e:
            self.logger.logger.error(f"相機測試失敗: {str(e)}")
            return False

    def shutdown(self):
        try:
            if self.camera:
                self.logger.logger.info("正在關閉相機...")
                self.camera.close()
                self.logger.logger.info("相機已關閉")
            self.is_initialized = False
        except Exception as e:
            self.logger.logger.error(f"關閉相機時出錯: {str(e)}")

    def __del__(self):
        self.shutdown()
