from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from camera.camera_controller import CameraController
from core.exceptions import BackendNotAvailableError, CameraConnectionError, ConfigurationError, HardwareError
from core.inference_engine import InferenceEngine


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.weights = "dummy.pt"
    config.device = "cpu"
    config.enable_yolo = True
    config.max_cache_size = 3
    config.imgsz = (640, 640)
    config.conf_thres = 0.25
    config.iou_thres = 0.45
    config.backends = {}  # Start with empty backends
    return config

def test_camera_connection_failure(mock_config):
    """測試相機枚舉失敗時是否拋出 CameraConnectionError"""
    with patch("camera.camera_controller.MVSCamera") as mock_mvs:
        mock_inst = mock_mvs.return_value
        mock_inst.enum_devices.return_value = False

        controller = CameraController(mock_config)
        with pytest.raises(CameraConnectionError, match="無法枚舉相機設備"):
            controller.initialize()

def test_camera_capture_without_init(mock_config):
    """測試未初始化時拍攝是否拋出 HardwareError"""
    controller = CameraController(mock_config)
    with pytest.raises(HardwareError, match="相機未初始化"):
        controller.capture_frame()

def test_inference_engine_invalid_backend(mock_config):
    """測試請求不存在或已禁用的後端時是否拋出 BackendNotAvailableError"""
    mock_config.enable_yolo = False
    engine = InferenceEngine(mock_config)

    with pytest.raises(BackendNotAvailableError):
        engine.infer(np.zeros((640, 640, 3), dtype=np.uint8), "prod", "area", "yolo")

def test_yolo_model_missing_weights(mock_config):
    """測試權重檔案不存在時是否拋出 ConfigurationError"""
    from core.yolo_inference_model import YOLOInferenceModel

    mock_config.weights = "non_existent.pt"
    model = YOLOInferenceModel(mock_config)

    # Use the absolute path if needed, or just mock the class used in the module
    with patch("core.yolo_inference_model.YOLO", side_effect=FileNotFoundError("Mocked file not found")):
        with pytest.raises(ConfigurationError):
            model.initialize("prod", "area")

@patch("core.yolo_inference_model.autocast")
@patch("core.yolo_inference_model.YOLO")
def test_inference_with_empty_image(mock_yolo, mock_autocast, mock_config):
    """測試輸入無效圖像時的行為"""
    from core.yolo_inference_model import YOLOInferenceModel

    # Setup mock model
    mock_model_inst = mock_yolo.return_value
    mock_model_inst.return_value = [] # Return empty detections

    model = YOLOInferenceModel(mock_config)
    model.is_initialized = True
    model.model = mock_model_inst
    model.detector = MagicMock()
    model.detector.process_detections.return_value = (None, [], [])

    # Test with empty array (invalid shape for letterbox)
    with pytest.raises(Exception): # letterbox might raise ValueError
        model.infer(np.array([]), "prod", "area")


# ---- Tests for _acquire_frame RuntimeError (no more dummy images) ----

class _FakeCamera:
    """Camera stub that always returns None (simulates hardware failure)."""
    def capture_frame(self):
        return None


class _FakeLogger:
    """Minimal logger stub that accepts any log call."""
    def __getattr__(self, name):
        return lambda *a, **kw: None


def test_acquire_frame_camera_returns_none():
    """When camera.capture_frame() returns None, _acquire_frame must raise RuntimeError."""
    from core.detection_system import DetectionSystem

    ds = object.__new__(DetectionSystem)  # bypass __init__
    ds.camera = _FakeCamera()
    with pytest.raises(RuntimeError, match="Hardware failure"):
        ds._acquire_frame(None, _FakeLogger())


def test_acquire_frame_no_camera_no_frame():
    """When there is no camera and no frame, _acquire_frame must raise RuntimeError."""
    from core.detection_system import DetectionSystem

    ds = object.__new__(DetectionSystem)  # bypass __init__
    ds.camera = None
    with pytest.raises(RuntimeError, match="No camera available"):
        ds._acquire_frame(None, _FakeLogger())
