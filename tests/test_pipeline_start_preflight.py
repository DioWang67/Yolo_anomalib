import sys
import types
from unittest.mock import Mock

import pytest

fake_torch = types.ModuleType("torch")
fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
fake_torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False))
fake_torch.inference_mode = lambda: None
fake_torch_cuda = types.ModuleType("torch.cuda")
fake_torch_cuda.is_available = fake_torch.cuda.is_available
fake_torch_amp = types.ModuleType("torch.cuda.amp")
fake_torch_amp.autocast = lambda *args, **kwargs: None
fake_ultralytics = types.ModuleType("ultralytics")
fake_ultralytics.YOLO = object
fake_ultralytics_utils = types.ModuleType("ultralytics.utils")
fake_ultralytics_plotting = types.ModuleType("ultralytics.utils.plotting")
fake_ultralytics_plotting.Annotator = object
fake_ultralytics_plotting.colors = lambda *args, **kwargs: (0, 255, 0)
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.cuda", fake_torch_cuda)
sys.modules.setdefault("torch.cuda.amp", fake_torch_amp)
sys.modules.setdefault("ultralytics", fake_ultralytics)
sys.modules.setdefault("ultralytics.utils", fake_ultralytics_utils)
sys.modules.setdefault("ultralytics.utils.plotting", fake_ultralytics_plotting)

from core.detection_system import DetectionSystem
from core.exceptions import BackendInitializationError


def test_start_pipeline_aborts_before_acquisition_when_preflight_fails(monkeypatch):
    ds = DetectionSystem.__new__(DetectionSystem)
    ds.camera = object()
    ds.config = Mock(weights="broken.onnx")
    ds._pipeline = Mock()

    monkeypatch.setattr(
        ds,
        "load_model_configs",
        lambda product, area, inference_type: None,
    )
    monkeypatch.setattr(
        "core.detection_system.validate_runtime_for_model",
        lambda _path: (_ for _ in ()).throw(
            BackendInitializationError("DLL load failed")
        ),
    )

    with pytest.raises(BackendInitializationError):
        ds.start_pipeline("PCBA1", "A", "yolo")

    ds._pipeline.start.assert_not_called()
