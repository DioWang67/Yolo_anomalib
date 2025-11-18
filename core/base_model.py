# core/base_model.py
from core.logging_config import DetectionLogger
import torch


class BaseInferenceModel:
    def __init__(self, config):
        self.config = config
        self.logger = DetectionLogger()
        self.model = None
        self.is_initialized = False

    def initialize(self, product=None, area=None):
        raise NotImplementedError

    def infer(self, image, product, area, output_path=None):
        raise NotImplementedError

    def shutdown(self):

        if self.model:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.logger.info("記憶體已清理")
