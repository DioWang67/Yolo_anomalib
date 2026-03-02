# core/base_model.py
"""推論模型的抽象基底類別，定義 initialize / infer / shutdown 介面。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.logging_config import DetectionLogger


class BaseInferenceModel(ABC):
    """Abstract base class for all inference model backends.

    Subclasses **must** implement :meth:`initialize` and :meth:`infer`.
    A default :meth:`shutdown` is provided that clears the model reference
    and releases CUDA memory if available.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.logger = DetectionLogger()
        self.model = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self, product=None, area=None):
        """Load model weights and prepare for inference."""
        ...

    @abstractmethod
    def infer(self, image, product, area, output_path=None):
        """Run inference on a single image and return a result dict."""
        ...

    def shutdown(self):
        if self.model:
            del self.model
        # Lazy import: torch is heavy and not needed at class-definition time.
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        self.logger.logger.info("記憶體已清理")
