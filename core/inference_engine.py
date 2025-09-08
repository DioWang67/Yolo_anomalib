from typing import Dict, Any, Type
from importlib import import_module

from core.base_model import BaseInferenceModel
from core.yolo_inference_model import YOLOInferenceModel
try:
    from core.anomalib_inference_model import AnomalibInferenceModel
except Exception:  # pragma: no cover
    AnomalibInferenceModel = None  # type: ignore
from core.logger import DetectionLogger
from core.config import DetectionConfig


class InferenceEngine:
    """Inference engine with pluggable backends.

    Built-ins: 'yolo', 'anomalib' (if available).
    Custom: via config.backends: { name: { class_path: 'pkg.ModClass', enabled: true } }
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.models: Dict[str, BaseInferenceModel] = {}

    def _register_builtin(self) -> None:
        if getattr(self.config, "enable_yolo", False):
            try:
                m = YOLOInferenceModel(self.config)
                if m.initialize():
                    self.models["yolo"] = m
                    self.logger.logger.info("YOLO backend ready")
            except Exception as e:
                self.logger.logger.error(f"YOLO backend init failed: {e}")

        if getattr(self.config, "enable_anomalib", False):
            if AnomalibInferenceModel is None:
                self.logger.logger.error("Anomalib not available; skip anomalib backend")
            else:
                try:
                    m = AnomalibInferenceModel(self.config)
                    if m.initialize():
                        self.models["anomalib"] = m
                        self.logger.logger.info("Anomalib backend ready")
                except Exception as e:
                    self.logger.logger.error(f"Anomalib backend init failed: {e}")

    def _register_custom(self) -> None:
        backends = getattr(self.config, "backends", None) or {}
        for name, spec in backends.items():
            if not spec.get("enabled", True):
                continue
            class_path = spec.get("class_path")
            if not class_path:
                self.logger.logger.error(f"Backend '{name}' missing class_path")
                continue
            try:
                mod, clsname = class_path.rsplit(".", 1)
                cls: Type[BaseInferenceModel] = getattr(import_module(mod), clsname)  # type: ignore
                inst = cls(self.config)
                if getattr(inst, "initialize", None):
                    ok = inst.initialize()
                    if not ok:
                        self.logger.logger.error(f"Backend '{name}' initialize() returned False")
                        continue
                self.models[name.lower()] = inst
                self.logger.logger.info(f"Backend '{name}' ready")
            except Exception as e:
                self.logger.logger.error(f"Backend '{name}' load failed: {e}")

    def initialize(self) -> bool:
        try:
            self.logger.logger.info("Initializing inference engine...")
            self._register_builtin()
            self._register_custom()
            if not self.models:
                raise RuntimeError("No available inference backends")
            self.logger.logger.info(f"Enabled backends: {list(self.models.keys())}")
            return True
        except Exception as e:
            self.logger.logger.error(f"Inference engine init failed: {str(e)}")
            return False

    def infer(self, image, product: str, area: str, inference_type: Any, output_path: str = None):
        name = inference_type.value if hasattr(inference_type, "value") else str(inference_type)
        name = name.lower()
        if name not in self.models:
            raise ValueError(f"Backend not initialized: {name}")
        return self.models[name].infer(image, product, area, output_path)

    def shutdown(self):
        for model in self.models.values():
            try:
                model.shutdown()
            except Exception:
                pass
        self.models.clear()

