import sys
import types

import pytest

from core.exceptions import BackendInitializationError
from core.runtime_preflight import (
    _DLL_DIRECTORY_HANDLES,
    _prepare_packaged_onnxruntime_dll_path,
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
    assert ".pt weights" in message


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
    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr(
        "os.add_dll_directory",
        lambda path: added_paths.append(path) or fake_handle,
        raising=False,
    )
    _DLL_DIRECTORY_HANDLES.clear()

    _prepare_packaged_onnxruntime_dll_path()

    assert added_paths == [str(tmp_path), str(capi_dir)]
    assert _DLL_DIRECTORY_HANDLES == [fake_handle, fake_handle]
