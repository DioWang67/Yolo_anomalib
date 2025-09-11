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
        """No-op for lazy mode. Backends are created on first use in infer()."""
        return

    def _register_custom(self) -> None:
        """No-op for lazy mode. Custom backends are created on demand in infer()."""
        return

    def initialize(self) -> bool:
        """Lazy initialization: defer backend model creation to infer()."""
        try:
            self.logger.logger.info("Inference engine ready (lazy backends)")
            return True
        except Exception as e:
            self.logger.logger.error(f"Inference engine init failed: {str(e)}")
            return False

    def infer(self, image, product: str, area: str, inference_type: Any, output_path: str = None):
        name = inference_type.value if hasattr(inference_type, "value") else str(inference_type)
        name = name.lower()

        # Create backend on first use
        if name not in self.models:
            if name == "yolo" and getattr(self.config, "enable_yolo", False):
                try:
                    m = YOLOInferenceModel(self.config)
                    if m.initialize(product=product, area=area):
                        self.models["yolo"] = m
                        self.logger.logger.info("YOLO backend ready (lazy)")
                    else:
                        raise RuntimeError("YOLO initialize() returned False")
                except Exception as e:
                    raise RuntimeError(f"YOLO backend init failed: {e}")
            elif name == "anomalib" and getattr(self.config, "enable_anomalib", False):
                if AnomalibInferenceModel is None:
                    raise RuntimeError("Anomalib not available")
                try:
                    m = AnomalibInferenceModel(self.config)
                    if m.initialize(product=product, area=area):
                        self.models["anomalib"] = m
                        self.logger.logger.info("Anomalib backend ready (lazy)")
                    else:
                        raise RuntimeError("Anomalib initialize() returned False")
                except Exception as e:
                    raise RuntimeError(f"Anomalib backend init failed: {e}")
            else:
                # custom backend
                backends = getattr(self.config, "backends", None) or {}
                spec = backends.get(name)
                if not spec or not spec.get("enabled", True):
                    raise ValueError(f"Backend not initialized or disabled: {name}")
                class_path = spec.get("class_path")
                if not class_path:
                    raise ValueError(f"Backend '{name}' missing class_path")
                try:
                    mod, clsname = class_path.rsplit(".", 1)
                    cls: Type[BaseInferenceModel] = getattr(import_module(mod), clsname)  # type: ignore
                    inst = cls(self.config)
                    if getattr(inst, "initialize", None):
                        ok = inst.initialize(product=product, area=area)
                        if not ok:
                            raise RuntimeError(f"Backend '{name}' initialize() returned False")
                    self.models[name] = inst
                    self.logger.logger.info(f"Backend '{name}' ready (lazy)")
                except Exception as e:
                    raise RuntimeError(f"Backend '{name}' load failed: {e}")

        return self.models[name].infer(image, product, area, output_path)

    def shutdown(self):
        for model in self.models.values():
            try:
                model.shutdown()
            except Exception:
                pass
        self.models.clear()
