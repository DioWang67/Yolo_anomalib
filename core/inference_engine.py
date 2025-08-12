from enum import Enum
from core.yolo_inference_model import YOLOInferenceModel
from core.anomalib_inference_model import AnomalibInferenceModel
from core.logger import DetectionLogger
from core.config import DetectionConfig


class InferenceType(Enum):
    YOLO = "yolo"
    ANOMALIB = "anomalib"

    @classmethod
    def from_string(cls, value: str) -> 'InferenceType':
        value = value.lower()
        for inference_type in cls:
            if inference_type.value == value:
                return inference_type
        raise ValueError(f"不支援的推理類型: {value}")


class BaseInferenceModel:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.model = None
        self.is_initialized = False

    def initialize(self, product: str = None, area: str = None) -> bool:
        raise NotImplementedError("子類必須實現 initialize 方法")

    def infer(self, image, product: str, area: str):
        raise NotImplementedError("子類必須實現 infer 方法")

    def shutdown(self):
        import torch
        if hasattr(self, 'model') and self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.logger.info("記憶體已清理")


class InferenceEngine:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.models = {}

    def initialize(self) -> bool:
        try:
            self.logger.logger.info("正在初始化推理引擎...")
            if self.config.enable_yolo:
                yolo_model = YOLOInferenceModel(self.config)
                if yolo_model.initialize():
                    self.models[InferenceType.YOLO] = yolo_model
                    self.logger.logger.info("YOLO 模型已加載")

            if self.config.enable_anomalib:
                anomalib_model = AnomalibInferenceModel(self.config)
                self.models[InferenceType.ANOMALIB] = anomalib_model
                self.logger.logger.info("Anomalib 模型已準備")

            if not self.models:
                raise RuntimeError("沒有成功初始化任何推理模型")

            self.logger.logger.info(f"推理引擎初始化成功，可用模型: {list(self.models.keys())}")
            return True
        except Exception as e:
            self.logger.logger.error(f"推理引擎初始化失敗: {str(e)}")
            return False

    def infer(self, image, product: str, area: str, inference_type: InferenceType, output_path: str = None):
        if inference_type not in self.models:
            raise ValueError(f"未初始化的推理類型: {inference_type}")
        return self.models[inference_type].infer(image, product, area, output_path)
