from __future__ import annotations

"""Runtime checks that must pass before model inference starts."""

import importlib
import sys
from pathlib import Path

from core.exceptions import BackendInitializationError


def validate_runtime_for_model(model_path: str | Path) -> None:
    """Validate runtime dependencies for a model artifact.

    Args:
        model_path: Model weights path. Only ``.onnx`` currently needs an
            explicit runtime check.

    Raises:
        BackendInitializationError: If ONNX Runtime cannot be imported or does
            not expose ``CPUExecutionProvider``.
    """
    path = Path(str(model_path))
    if path.suffix.lower() != ".onnx":
        return

    try:
        ort = importlib.import_module("onnxruntime")
        providers = list(ort.get_available_providers())
    except Exception as exc:
        raise BackendInitializationError(
            "ONNX Runtime preflight failed: ONNX Runtime cannot be loaded for "
            ".onnx inference. "
            f"model_path={path}, "
            f"python={sys.executable}, "
            f"version={sys.version}, "
            f"onnxruntime_import_error={exc!r}. "
            "Reinstall onnxruntime, install the Microsoft Visual C++ "
            "Redistributable 2015-2022 x64, or switch this model to .pt "
            "weights."
        ) from exc

    if "CPUExecutionProvider" not in providers:
        raise BackendInitializationError(
            "ONNX Runtime preflight failed: CPUExecutionProvider is not "
            "available. "
            f"model_path={path}, "
            f"python={sys.executable}, "
            f"version={sys.version}, "
            f"providers={providers}. "
            "Reinstall onnxruntime or switch this model to .pt weights."
        )
