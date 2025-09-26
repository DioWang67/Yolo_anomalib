from __future__ import annotations


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
