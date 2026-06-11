from __future__ import annotations

"""Runtime checks that must pass before model inference starts."""

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path

from core.exceptions import BackendInitializationError

_DLL_DIRECTORY_HANDLES: list[object] = []


def _onnxruntime_diagnostics() -> str:
    """Return import-safe ONNX Runtime installation details."""
    try:
        package_version = importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        package_version = "not-installed"

    package_spec = importlib.util.find_spec("onnxruntime")
    package_path = package_spec.origin if package_spec else "not-found"
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    return (
        f"onnxruntime_version={package_version}, "
        f"onnxruntime_path={package_path}, "
        f"path_head={path_entries[:5]}"
    )


def _prepare_packaged_onnxruntime_dll_path() -> None:
    """Register ONNX Runtime's packaged DLL directory for frozen Windows apps.

    PyInstaller keeps onnxruntime native binaries under
    ``_internal/onnxruntime/capi`` in one-dir builds. Windows does not always
    search that package subdirectory when importing ``onnxruntime_pybind11_state``,
    so register it before importing onnxruntime.
    """
    if os.name != "nt" or not getattr(sys, "frozen", False):
        return
    if not hasattr(os, "add_dll_directory"):
        return

    app_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    capi_candidates = [
        app_root / "onnxruntime" / "capi",
        app_root / "_internal" / "onnxruntime" / "capi",
    ]
    capi_dir = next((path for path in capi_candidates if path.is_dir()), None)
    if capi_dir is None:
        return

    search_dirs = [app_root, capi_dir]
    if app_root.name != "_internal":
        search_dirs.append(app_root / "_internal")

    existing_path = os.environ.get("PATH", "")
    prepend_paths = [str(path) for path in search_dirs if path.is_dir()]
    os.environ["PATH"] = os.pathsep.join(prepend_paths + [existing_path])

    for dll_dir in search_dirs:
        if dll_dir.is_dir():
            handle = os.add_dll_directory(str(dll_dir))
            _DLL_DIRECTORY_HANDLES.append(handle)



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
        _prepare_packaged_onnxruntime_dll_path()
        ort = importlib.import_module("onnxruntime")
        providers = list(ort.get_available_providers())
    except Exception as exc:
        raise BackendInitializationError(
            "ONNX Runtime preflight failed: ONNX Runtime cannot be loaded for "
            ".onnx inference. "
            f"model_path={path}, "
            f"python={sys.executable}, "
            f"version={sys.version}, "
            f"{_onnxruntime_diagnostics()}, "
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
