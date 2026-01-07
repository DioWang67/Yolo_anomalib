from importlib import import_module
from typing import Any

from core.base_model import BaseInferenceModel
from core.exceptions import (
    BackendInitializationError,
    BackendNotAvailableError,
    ConfigurationError,
    HardwareError,
    ModelInitializationError,
    ResourceExhaustionError,
)
from core.yolo_inference_model import YOLOInferenceModel

try:
    from core.anomalib_inference_model import AnomalibInferenceModel
except Exception:  # pragma: no cover
    AnomalibInferenceModel = None  # type: ignore
import numpy as np

from core.config import DetectionConfig
from core.logging_config import DetectionLogger


class InferenceEngine:
    """Manages model backends and provides a unified interface for inference.

    Supports built-in YOLO and Anomalib models as well as custom backends
    defined in the configuration. Backends are initialized lazily on first use.

    Attributes:
        config (DetectionConfig): The project configuration registry.
        logger (DetectionLogger): Logging interface for inference events.
        models (dict): Cache of initialized inference model instances.
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
        """Prepares the engine for operation.

        Currently uses lazy initialization, so this simply logs readiness.

        Returns:
            bool: Always True in the current lazy implementation.
        """
        self.logger.logger.info("Inference engine ready (lazy backends)")
        return True

    def infer(
        self,
        image: np.ndarray,
        product: str,
        area: str,
        inference_type: Any,
        output_path: str = None,
    ):
        """Dispatches an inference request to the appropriate backend.

        If the required backend is not yet initialized, it will be created
        using the lazy-loading mechanism.

        Args:
            image: The input image array (BGR).
            product: Target product category.
            area: Specific area or station identifier.
            inference_type: The type of inference model to use ('yolo', 'anomalib', etc.).
            output_path: Optional path to save visual debug artifacts (e.g., heatmaps).

        Returns:
            dict: Categorized results from the backend inference model.

        Raises:
            BackendNotAvailableError: If the requested backend is disabled or missing.
            BackendInitializationError: If the backend fails to load or initialize.
        """
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
                except (
                    ConfigurationError,
                    HardwareError,
                    ResourceExhaustionError,
                ):
                    raise
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
                except (
                    ConfigurationError,
                    HardwareError,
                    ResourceExhaustionError,
                ):
                    raise
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
                    if "." not in str(class_path):
                        raise BackendInitializationError(
                            f"Invalid class_path for backend '{name}': {class_path}"
                        )
                    mod, clsname = str(class_path).rsplit(".", 1)
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
                except (
                    ConfigurationError,
                    HardwareError,
                    ResourceExhaustionError,
                ):
                    # Propagate these critical errors as is, or wrap if preferred
                    raise
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
