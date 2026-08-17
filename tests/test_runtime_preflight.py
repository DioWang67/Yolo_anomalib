import sys
import types

import pytest

from core.exceptions import BackendInitializationError
from core.runtime_preflight import (
    _DLL_DIRECTORY_HANDLES,
    _prepare_packaged_onnxruntime_dll_path,
    preload_onnxruntime_before_gui,
    validate_runtime_for_model,
)


def test_validate_runtime_for_model_skips_non_onnx(monkeypatch):
    called = False

    def fake_import_module(_name):
        nonlocal called
        called = True

    monkeypatch.setattr("importlib.import_module", fake_import_module)
    validate_runtime_for_model("model.pt")
    assert called is False


def test_validate_runtime_for_model_reports_onnxruntime_import_failure(monkeypatch):
    def fake_import_module(_name):
        raise ImportError("DLL load failed while importing onnxruntime_pybind11_state")

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    with pytest.raises(BackendInitializationError) as exc_info:
        validate_runtime_for_model(r"D:\models\PCBA1_A.onnx")

    message = str(exc_info.value)
    assert "ONNX Runtime preflight failed" in message
    assert r"D:\models\PCBA1_A.onnx" in message
    assert sys.executable in message
    assert "onnxruntime_pybind11_state" in message
    assert "onnxruntime_version=" in message
    assert "onnxruntime_path=" in message
    assert "path_head=" in message
    assert "start_inference.bat" in message
    assert "requirements.txt" in message


def test_validate_runtime_for_model_requires_cpu_provider(monkeypatch):
    fake_ort = types.SimpleNamespace(
        get_available_providers=lambda: ["CUDAExecutionProvider"]
    )
    monkeypatch.setattr("importlib.import_module", lambda _name: fake_ort)

    with pytest.raises(BackendInitializationError) as exc_info:
        validate_runtime_for_model("model.onnx")

    assert "CPUExecutionProvider" in str(exc_info.value)


def test_prepare_packaged_onnxruntime_dll_path_for_frozen_app(monkeypatch, tmp_path):
    capi_dir = tmp_path / "onnxruntime" / "capi"
    capi_dir.mkdir(parents=True)
    added_paths = []
    fake_handle = object()

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    # Patch the production seam, not the global ``os.name``: rebinding that
    # makes pathlib.Path() resolve to WindowsPath, which cannot be
    # instantiated on POSIX, so these tests took down pytest on Linux.
    monkeypatch.setattr("core.runtime_preflight._is_windows", lambda: True)
    monkeypatch.setattr(
        "os.add_dll_directory",
        lambda path: added_paths.append(path) or fake_handle,
        raising=False,
    )
    _DLL_DIRECTORY_HANDLES.clear()

    _prepare_packaged_onnxruntime_dll_path()

    assert added_paths == [str(tmp_path), str(capi_dir)]
    assert _DLL_DIRECTORY_HANDLES == [fake_handle, fake_handle]


def test_preload_onnxruntime_before_gui_on_windows(monkeypatch):
    imported_modules = []
    # Patch the production seam, not the global ``os.name``: rebinding that
    # makes pathlib.Path() resolve to WindowsPath, which cannot be
    # instantiated on POSIX, so these tests took down pytest on Linux.
    monkeypatch.setattr("core.runtime_preflight._is_windows", lambda: True)
    monkeypatch.setattr(
        "core.runtime_preflight._prepare_packaged_onnxruntime_dll_path",
        lambda: imported_modules.append("dll-path"),
    )
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: imported_modules.append(name),
    )

    preload_onnxruntime_before_gui()

    assert imported_modules == ["dll-path", "onnxruntime"]


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_preload_onnxruntime_defers_native_import_errors(monkeypatch, error_type):
    # Patch the production seam, not the global ``os.name``: rebinding that
    # makes pathlib.Path() resolve to WindowsPath, which cannot be
    # instantiated on POSIX, so these tests took down pytest on Linux.
    monkeypatch.setattr("core.runtime_preflight._is_windows", lambda: True)

    def raise_native_import_error(_name):
        raise error_type("native runtime unavailable")

    monkeypatch.setattr("importlib.import_module", raise_native_import_error)

    preload_onnxruntime_before_gui()
