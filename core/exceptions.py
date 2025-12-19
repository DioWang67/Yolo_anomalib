from __future__ import annotations

"""定義模型初始化、推論與結果儲存相關的例外階層。"""


class ModelError(RuntimeError):
    """Base exception for model-related failures."""


class ModelInitializationError(ModelError):
    """Raised when a model fails to load or warm up."""


class ModelInferenceError(ModelError):
    """Raised when inference fails for a given model."""


class BackendInitializationError(ModelError):
    """Raised when the inference engine cannot initialize a backend."""


class BackendNotAvailableError(ModelError):
    """Raised when a requested backend is disabled or missing."""


class ResultPersistenceError(RuntimeError):
    """Base exception for result persistence failures."""


class ResultImageWriteError(ResultPersistenceError):
    """Raised when writing images or crops fails."""


class ResultExcelWriteError(ResultPersistenceError):
    """Raised when persisting Excel rows fails."""


class HardwareError(ModelError):
    """Base exception for hardware-related failures (Camera, GPU)."""


class CameraConnectionError(HardwareError):
    """Raised when failing to connect to the MVS hardware."""


class ResourceExhaustionError(ModelError):
    """Raised when a resource limit is hit (e.g., CUDA OOM)."""


class ConfigurationError(ModelError):
    """Raised when there is a logical or structural error in the configuration."""
