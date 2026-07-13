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


def test_camera_clear_image_buffer_uses_sdk_when_available(mock_config):
    """The controller should use the SDK buffer-clear operation without reconnecting."""
    controller = CameraController(mock_config)
    controller.camera = MagicMock()
    controller.camera.cam.MV_CC_ClearImageBuffer.return_value = 0
    controller.is_initialized = True

    assert controller.clear_image_buffer() is True
    controller.camera.cam.MV_CC_ClearImageBuffer.assert_called_once()


def test_camera_clear_image_buffer_failure_is_non_fatal(mock_config):
    """An SDK buffer-clear failure must fall back to normal frame acquisition."""
    controller = CameraController(mock_config)
    controller.camera = MagicMock()
    controller.camera.cam.MV_CC_ClearImageBuffer.return_value = 1
    controller.is_initialized = True

    assert controller.clear_image_buffer() is False


def test_camera_shutdown_is_idempotent(mock_config):
    controller = CameraController(mock_config)
    camera = MagicMock()
    controller.camera = camera
    controller.is_initialized = True

    controller.shutdown()
    controller.shutdown()

    camera.close.assert_called_once()
    assert controller.camera is None
    assert controller.is_initialized is False


def test_camera_reconnect_rebuilds_session(mock_config):
    """reconnect() 應關閉舊 session 並重新枚舉+連線"""
    with patch("camera.camera_controller.MVSCamera") as mock_mvs:
        mock_inst = mock_mvs.return_value
        mock_inst.enum_devices.return_value = True
        mock_inst.connect_to_camera.return_value = True

        controller = CameraController(mock_config)
        old_camera = MagicMock()
        controller.camera = old_camera
        controller.is_initialized = True

        assert controller.reconnect() is True
        old_camera.close.assert_called_once()
        assert controller.is_initialized is True
        assert controller.camera is mock_inst


def test_camera_reconnect_returns_false_on_failure(mock_config):
    """重連失敗時回傳 False 而非拋出例外（呼叫端是背景 worker）"""
    with patch("camera.camera_controller.MVSCamera") as mock_mvs:
        mock_inst = mock_mvs.return_value
        mock_inst.enum_devices.return_value = False

        controller = CameraController(mock_config)
        assert controller.reconnect() is False
        assert controller.is_initialized is False


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


def test_set_exposure_delegates_to_camera(mock_config):
    controller = CameraController(mock_config)
    camera = MagicMock()
    camera.set_exposure_time.return_value = True
    controller.camera = camera
    controller.is_initialized = True

    assert controller.set_exposure(51170.0) is True
    camera.set_exposure_time.assert_called_once_with(51170.0)


def test_get_exposure_and_range_delegate(mock_config):
    controller = CameraController(mock_config)
    camera = MagicMock()
    camera.get_exposure_time.return_value = 51170.0
    camera.get_parameter_range.return_value = {"current": 51170.0, "min": 100.0, "max": 1e6}
    controller.camera = camera
    controller.is_initialized = True

    assert controller.get_exposure() == 51170.0
    assert controller.get_exposure_range()["max"] == 1e6
    camera.get_parameter_range.assert_called_once_with("ExposureTime")


def test_set_gain_delegates_and_reports_failure(mock_config):
    controller = CameraController(mock_config)
    camera = MagicMock()
    camera.set_gain.return_value = False
    controller.camera = camera
    controller.is_initialized = True

    assert controller.set_gain(23.0) is False
    camera.set_gain.assert_called_once_with(23.0)


def test_camera_setters_return_false_when_uninitialized(mock_config):
    controller = CameraController(mock_config)
    assert controller.set_exposure(1000.0) is False
    assert controller.set_gain(1.0) is False
    assert controller.get_exposure() is None
    assert controller.get_gain() is None
    assert controller.get_exposure_range() is None
