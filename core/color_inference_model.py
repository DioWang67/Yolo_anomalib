from core.base_model import BaseInferenceModel
from core.logger import DetectionLogger


class ColorInferenceModel(BaseInferenceModel):
    """簡易顏色檢測模型範例，回傳原始影像與 PASS 狀態。"""

    def __init__(self, config):
        super().__init__(config)
        self.logger = DetectionLogger()

    def initialize(self, product=None, area=None):
        if not self.config.color_model_path:
            self.logger.logger.error("未提供顏色模型路徑")
            return False
        self.is_initialized = True
        self.logger.logger.info("顏色檢測模型初始化成功")
        return True

    def infer(self, image, product, area, output_path=None):
        if not self.is_initialized:
            raise RuntimeError("顏色檢測模型尚未初始化")
        return {
            "status": "PASS",
            "processed_image": image,
            "detections": [],
            "missing_items": []
        }
