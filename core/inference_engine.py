from importlib import import_module
from typing import Any

from core.base_model import BaseInferenceModel
from core.exceptions import (
    BackendInitializationError,
    BackendNotAvailableError,
    ModelInitializationError,
)
from core.yolo_inference_model import YOLOInferenceModel

try:
    from core.anomalib_inference_model import AnomalibInferenceModel
except Exception:  # pragma: no cover
    AnomalibInferenceModel = None  # type: ignore
from core.config import DetectionConfig
from core.logging_config import DetectionLogger


class InferenceEngine:
    """Inference engine with pluggable backends.

    Built-ins: 'yolo', 'anomalib' (if available).
    Custom: via config.backends: { name: { class_path: 'pkg.ModClass', enabled: true } }
    """

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.logger = DetectionLogger()
        self.models: dict[str, BaseInferenceModel] = {}

    def _register_builtin(self) -> None:
        """No-op for lazy mode. Backends are created on first use in infer()."""
        return

    def _register_custom(self) -> None:
        """No-op for lazy mode. Custom backends are created on demand in infer()."""
        return

    def initialize(self) -> bool:
        """Lazy initialization: defer backend model creation to infer()."""
        self.logger.logger.info("Inference engine ready (lazy backends)")
        return True

    def infer(
        self,
        image,
        product: str,
        area: str,
        inference_type: Any,
        output_path: str = None,
    ):
        name = (
            inference_type.value
            if hasattr(inference_type, "value")
            else str(inference_type)
        )
        name = name.lower()

        if name not in self.models:
            if name == "yolo" and getattr(self.config, "enable_yolo", False):
                model = YOLOInferenceModel(self.config)
                try:
                    model.initialize(product=product, area=area)
                except ModelInitializationError as exc:
                    self.logger.logger.error(
                        "YOLO backend init failed: %s", exc)
                    raise BackendInitializationError(
                        f"YOLO backend init failed: {exc}"
                    ) from exc
                except Exception as exc:
                    self.logger.logger.exception(
                        "YOLO backend unexpected failure")
                    raise BackendInitializationError(
                        "YOLO backend init failed"
                    ) from exc
                self.models["yolo"] = model
                self.logger.logger.info("YOLO backend ready (lazy)")
            elif name == "anomalib" and getattr(self.config, "enable_anomalib", False):
                if AnomalibInferenceModel is None:
                    raise BackendNotAvailableError("Anomalib not available")
                model = AnomalibInferenceModel(
                    self.config)  # type: ignore[call-arg]
                try:
                    model.initialize(product=product, area=area)
                except ModelInitializationError as exc:
                    self.logger.logger.error(
                        "Anomalib backend init failed: %s", exc)
                    raise BackendInitializationError(
                        f"Anomalib backend init failed: {exc}"
                    ) from exc
                except Exception as exc:
                    self.logger.logger.exception(
                        "Anomalib backend unexpected failure")
                    raise BackendInitializationError(
                        "Anomalib backend init failed"
                    ) from exc
                self.models["anomalib"] = model
                self.logger.logger.info("Anomalib backend ready (lazy)")
            else:
                backends = getattr(self.config, "backends", None) or {}
                spec = backends.get(name)
                if not spec or not spec.get("enabled", True):
                    raise BackendNotAvailableError(
                        f"Backend not initialized or disabled: {name}"
                    )
                class_path = spec.get("class_path")
                if not class_path:
                    raise BackendInitializationError(
                        f"Backend '{name}' missing class_path"
                    )
                try:
                    mod, clsname = class_path.rsplit(".", 1)
                    cls: type[BaseInferenceModel] = getattr(
                        import_module(mod), clsname)  # type: ignore[attr-defined]
                    inst = cls(self.config)
                    initializer = getattr(inst, "initialize", None)
                    if callable(initializer):
                        result = initializer(product=product, area=area)
                        if result is False:
                            raise BackendInitializationError(
                                f"Backend '{name}' initialize() returned False"
                            )
                except ModelInitializationError as exc:
                    self.logger.logger.error(
                        "Backend '%s' init failed: %s", name, exc)
                    raise BackendInitializationError(
                        f"Backend '{name}' init failed: {exc}"
                    ) from exc
                except Exception as exc:
                    self.logger.logger.exception(
                        "Backend '%s' load failed", name)
                    raise BackendInitializationError(
                        f"Backend '{name}' load failed: {exc}"
                    ) from exc
                self.models[name] = inst
                self.logger.logger.info("Backend '%s' ready (lazy)", name)

        return self.models[name].infer(image, product, area, output_path)

    def shutdown(self):
        for model in self.models.values():
            try:
                model.shutdown()
            except Exception as exc:
                self.logger.logger.warning(
                    "Backend shutdown failed: %s", exc, exc_info=exc
                )
        self.models.clear()
