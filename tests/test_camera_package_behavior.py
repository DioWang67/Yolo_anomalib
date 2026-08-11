"""Behavior contracts for the camera package's hardware-independent seams."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

import camera.camera_controller as controller_module
import camera.camera_preview as compatibility_entrypoint
import camera.MVS_camera_control as mvs_module
import camera.preview.app as preview_app
import camera.preview.source as source_module
from camera.camera_controller import CameraController
from camera.MVS_camera_control import MVSCamera
from camera.preview.metrics import AdaptiveCalibrator, compute_change_metrics
from camera.preview.source import CameraSource
from camera.preview.window import CameraPreviewWindow
from core.exceptions import CameraConnectionError, HardwareError


def _camera_config() -> SimpleNamespace:
    return SimpleNamespace(
        exposure_time=1000.0,
        gain=1.5,
        width=640,
        height=480,
        MV_CC_GetImageBuffer_nMsec=1000,
    )


@pytest.fixture(autouse=True)
def restore_camera_preview_logger():
    """Keep the named preview logger isolated from later tests and pytest capture."""
    logger = logging.getLogger("camera_preview")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    try:
        yield
    finally:
        for handler in logger.handlers:
            if handler not in original_handlers:
                handler.close()
        logger.handlers[:] = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate


def test_compatibility_entrypoint_exports_the_supported_camera_api() -> None:
    assert compatibility_entrypoint.CameraSource is CameraSource
    assert compatibility_entrypoint.AdaptiveCalibrator is AdaptiveCalibrator
    assert compatibility_entrypoint.parse_args is preview_app.parse_args


def test_change_metrics_and_adaptive_threshold_have_deterministic_contracts() -> None:
    previous = np.zeros((2, 2), dtype=np.uint8)
    current = np.array([[0, 10], [20, 30]], dtype=np.uint8)

    mean_diff, changed_ratio = compute_change_metrics(current, previous, threshold=20)

    assert mean_diff == pytest.approx(15 / 255)
    assert changed_ratio == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Frame shape mismatch"):
        compute_change_metrics(current, np.zeros((1, 1), dtype=np.uint8), threshold=20)

    calibrator = AdaptiveCalibrator(warmup_frames=2, k_sigma=0.0)
    calibrator.update(current, None)
    calibrator.update(current, np.zeros((1, 1), dtype=np.uint8))
    assert not calibrator.ready
    assert calibrator.suggested_threshold == 25

    calibrator.update(np.full((2, 2), 1, dtype=np.uint8), previous)
    calibrator.update(np.full((2, 2), 3, dtype=np.uint8), previous)
    assert calibrator.ready
    assert calibrator.suggested_threshold == 2

    high = AdaptiveCalibrator(warmup_frames=1, k_sigma=10.0)
    high.update(np.full((2, 2), 255, dtype=np.uint8), previous)
    assert high.suggested_threshold == 255


class _Capture:
    def __init__(self, *, opened: bool = True) -> None:
        self.opened = opened
        self.released = False
        self.read_result: tuple[bool, np.ndarray | None] = (
            True,
            np.full((2, 2, 3), 7, dtype=np.uint8),
        )

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV API contract
        return self.opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self.read_result

    def release(self) -> None:
        self.released = True


def test_opencv_camera_source_open_read_reopen_and_shutdown(monkeypatch, tmp_path: Path) -> None:
    captures: list[_Capture] = []

    def create_capture(_index: int) -> _Capture:
        capture = _Capture()
        captures.append(capture)
        return capture

    monkeypatch.setattr(source_module.cv2, "VideoCapture", create_capture)
    source = CameraSource(True, tmp_path / "unused.yaml", 3, logging.getLogger("test.camera"))

    assert np.all(source.read() == 7)
    captures[0].read_result = (False, None)
    assert source.read() is None

    source.reopen()
    assert captures[0].released
    assert len(captures) == 2
    source.shutdown()
    assert captures[1].released
    assert source.read() is None

    monkeypatch.setattr(source_module.cv2, "VideoCapture", lambda _index: _Capture(opened=False))
    with pytest.raises(RuntimeError, match="Failed to open OpenCV camera index 3"):
        CameraSource(True, tmp_path / "unused.yaml", 3, logging.getLogger("test.camera"))


def test_controller_camera_source_loads_config_and_delegates_lifecycle(monkeypatch, tmp_path: Path) -> None:
    config = object()
    controller = SimpleNamespace(
        initialize=Mock(),
        capture_frame=Mock(return_value=np.ones((1, 1, 3), dtype=np.uint8)),
        shutdown=Mock(),
    )
    from_yaml = Mock(return_value=config)
    controller_factory = Mock(return_value=controller)
    monkeypatch.setattr(source_module.DetectionConfig, "from_yaml", from_yaml)
    monkeypatch.setattr(source_module, "CameraController", controller_factory)

    config_path = tmp_path / "config.yaml"
    source = CameraSource(False, config_path, 0, logging.getLogger("test.camera"))

    from_yaml.assert_called_once_with(str(config_path))
    controller_factory.assert_called_once_with(config)
    controller.initialize.assert_called_once_with()
    assert source.read().shape == (1, 1, 3)
    source.shutdown()
    controller.shutdown.assert_called_once_with()


def test_preview_argument_logging_and_roi_validation() -> None:
    args = preview_app.parse_args(
        [
            "--opencv-backup",
            "--camera-index",
            "4",
            "--refresh-ms",
            "25",
            "--resize-width",
            "0",
            "--use-luma",
            "--ema-alpha",
            "0.4",
            "--thr",
            "12",
            "--auto-k-sigma",
            "2.5",
            "--warmup-frames",
            "3",
            "--roi",
            "1,2,3,4",
            "--scale-quality",
            "smooth",
            "--target-fps",
            "20",
            "--reopen-failures",
            "2",
            "--log-level",
            "DEBUG",
        ]
    )
    assert (args.camera_index, args.refresh_ms, args.use_luma, args.thr) == (4, 25, True, 12)
    assert preview_app._parse_roi(None) is None
    assert preview_app._parse_roi("1,2,3,4") == (1, 2, 3, 4)
    for invalid in ("1,2,0,4", "1,2,3", "bad"):
        with pytest.raises(argparse.ArgumentTypeError, match="--roi expects"):
            preview_app._parse_roi(invalid)

    logger = preview_app._setup_logging("DEBUG")
    assert logger.level == logging.DEBUG
    assert len(logger.handlers) == 1


class _PreviewApplication:
    def __init__(self, _argv: list[str]) -> None:
        self.exit_code = 17

    def exec_(self) -> int:
        return self.exit_code


@pytest.mark.gui
def test_preview_main_enforces_config_and_builds_window(monkeypatch, tmp_path: Path) -> None:
    critical = Mock()
    monkeypatch.setattr(preview_app, "QApplication", _PreviewApplication)
    monkeypatch.setattr(preview_app.QMessageBox, "critical", critical)

    assert preview_app.main(["--config", str(tmp_path / "missing.yaml")]) == 1
    assert "Config file not found" in critical.call_args.args[2]

    source = object()
    source_factory = Mock(return_value=source)
    window = SimpleNamespace(show=Mock())
    window_factory = Mock(return_value=window)
    monkeypatch.setattr(preview_app, "CameraSource", source_factory)
    monkeypatch.setattr(preview_app, "CameraPreviewWindow", window_factory)

    result = preview_app.main(
        ["--opencv-backup", "--roi", "1,2,3,4", "--refresh-ms", "25"]
    )

    assert result == 17
    source_factory.assert_called_once()
    assert window_factory.call_args.kwargs["source"] is source
    assert window_factory.call_args.kwargs["roi"] == (1, 2, 3, 4)
    window.show.assert_called_once_with()


@pytest.mark.gui
def test_preview_main_offers_opencv_fallback_after_controller_failure(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("camera: {}", encoding="utf-8")
    attempts: list[bool] = []

    def create_source(*, use_opencv: bool, **_kwargs):
        attempts.append(use_opencv)
        if not use_opencv:
            raise CameraConnectionError("controller unavailable")
        return object()

    monkeypatch.setattr(preview_app, "QApplication", _PreviewApplication)
    monkeypatch.setattr(preview_app, "CameraSource", create_source)
    monkeypatch.setattr(preview_app, "CameraPreviewWindow", lambda **_kwargs: SimpleNamespace(show=Mock()))
    monkeypatch.setattr(preview_app.QMessageBox, "question", lambda *_args: preview_app.QMessageBox.Yes)

    assert preview_app.main(["--config", str(config_path)]) == 17
    assert attempts == [False, True]

    attempts.clear()
    critical = Mock()
    monkeypatch.setattr(preview_app.QMessageBox, "question", lambda *_args: preview_app.QMessageBox.No)
    monkeypatch.setattr(preview_app.QMessageBox, "critical", critical)
    assert preview_app.main(["--config", str(config_path)]) == 1
    assert attempts == [False]
    assert "controller unavailable" in critical.call_args.args[2]


class _FrameSource:
    def __init__(self) -> None:
        self.frames: list[np.ndarray | None] = []
        self.reopen = Mock()
        self.shutdown = Mock()
        self._capture = None

    def read(self) -> np.ndarray | None:
        return self.frames.pop(0) if self.frames else None


@pytest.mark.gui
def test_preview_window_processes_frames_reopens_and_closes(qtbot, monkeypatch) -> None:
    source = _FrameSource()
    logger = Mock()
    window = CameraPreviewWindow(
        source,
        1000,
        logger,
        resize_width=2,
        use_luma=False,
        ema_alpha=0.5,
        fixed_threshold=None,
        auto_k_sigma=3.0,
        warmup_frames=1,
        roi=(0, 0, 2, 2),
        scale_quality="fast",
        target_fps=None,
        reopen_failures=2,
    )
    qtbot.addWidget(window)
    window._timer.stop()

    frame = np.full((4, 4, 3), 20, dtype=np.uint8)
    assert window._preprocess(frame).shape == (2, 2)
    window.use_luma = True
    assert window._preprocess(frame).shape == (2, 2)
    window.roi = (100, 100, 2, 2)
    assert window._preprocess(frame).shape == (2, 2)

    source.frames = [None, None]
    window._update_frame()
    window._update_frame()
    source.reopen.assert_called_once_with()
    assert window._consecutive_fail == 0

    window.roi = None
    source.frames = [frame, np.full((4, 4, 3), 40, dtype=np.uint8)]
    ticks = iter((100, 200))
    monkeypatch.setattr("camera.preview.window.cv2.getTickCount", lambda: next(ticks))
    monkeypatch.setattr("camera.preview.window.cv2.getTickFrequency", lambda: 100.0)
    window._last_tick = 0
    window._update_frame()
    assert "Baseline" in window.metric_label.text()
    window._update_frame()
    assert "dMean:" in window.metric_label.text()
    assert window._ema_mean is not None

    window.fixed_threshold = 9
    assert window._current_threshold() == 9
    window.scale_quality = "smooth"
    assert not window._scale_pixmap(window.image_label.pixmap().toImage()).isNull()

    event = Mock()
    window.closeEvent(event)
    source.shutdown.assert_called_once_with()
    event.accept.assert_called_once_with()


def _controller() -> CameraController:
    controller = CameraController(_camera_config())
    controller.logger = SimpleNamespace(logger=Mock())
    return controller


def test_camera_controller_initialization_capture_and_buffer_contracts(monkeypatch) -> None:
    hardware = SimpleNamespace(
        enum_devices=Mock(return_value=True),
        connect_to_camera=Mock(return_value=True),
        get_frame=Mock(return_value=np.array([[[1, 2, 3]]], dtype=np.uint8)),
        cam=SimpleNamespace(MV_CC_ClearImageBuffer=Mock(return_value=0)),
        close=Mock(),
    )
    monkeypatch.setattr(controller_module, "MVSCamera", Mock(return_value=hardware))
    controller = _controller()

    assert controller.initialize()
    assert controller.is_initialized and controller.is_healthy
    frame = controller.capture_frame(timeout_ms=250)
    hardware.get_frame.assert_called_once_with(timeout_ms=250)
    assert frame[0, 0].tolist() == [3, 2, 1]
    assert controller.clear_image_buffer()

    hardware.cam.MV_CC_ClearImageBuffer.return_value = 9
    assert not controller.clear_image_buffer()
    hardware.cam.MV_CC_ClearImageBuffer.side_effect = OSError("SDK busy")
    assert not controller.clear_image_buffer()

    controller.shutdown()
    hardware.close.assert_called_once_with()
    assert not controller.is_initialized and not controller.is_healthy
    controller.shutdown()


def test_camera_controller_fail_closed_branches(monkeypatch) -> None:
    controller = _controller()
    with pytest.raises(HardwareError):
        controller.capture_frame()
    assert not controller.clear_image_buffer()
    assert not controller.set_exposure(1000)
    assert controller.get_exposure() is None
    assert controller.get_exposure_range() is None
    assert not controller.set_gain(2)
    assert controller.get_gain() is None
    assert not controller.test_camera()

    hardware = SimpleNamespace(
        enum_devices=Mock(return_value=False),
        connect_to_camera=Mock(return_value=False),
        get_frame=Mock(return_value=None),
        close=Mock(side_effect=OSError("close failed")),
    )
    monkeypatch.setattr(controller_module, "MVSCamera", Mock(return_value=hardware))
    with pytest.raises(CameraConnectionError):
        controller.initialize()

    controller.camera = hardware
    controller.is_initialized = True
    assert controller.capture_frame() is None
    hardware.get_frame.side_effect = OSError("capture failed")
    assert controller.capture_frame() is None
    controller.shutdown()


def test_camera_controller_parameter_batch_test_and_reconnect_contracts(monkeypatch) -> None:
    hardware = SimpleNamespace(
        set_exposure_time=Mock(return_value=True),
        get_exposure_time=Mock(return_value=1234.0),
        get_parameter_range=Mock(return_value={"current": 1, "min": 0, "max": 2}),
        set_gain=Mock(return_value=True),
        get_gain=Mock(return_value=2.0),
    )
    controller = _controller()
    controller.camera = hardware
    controller.is_initialized = True

    assert controller.set_exposure(1234)
    assert controller.get_exposure() == 1234.0
    assert controller.get_exposure_range() == {"current": 1, "min": 0, "max": 2}
    assert controller.set_gain(2)
    assert controller.get_gain() == 2.0

    controller.capture_frame = Mock(side_effect=[None, np.ones((1, 1, 3)), np.full((1, 1, 3), 2)])
    assert np.all(controller.capture_multiple_frames(3) == 2)
    controller.capture_frame = Mock(return_value=None)
    assert controller.capture_multiple_frames(2) is None
    assert not controller.test_camera()

    controller.capture_frame = Mock(return_value=np.ones((1, 1, 3)))
    controller.capture_multiple_frames = Mock(return_value=np.ones((1, 1, 3)))
    assert controller.test_camera()

    controller.shutdown = Mock()
    controller.initialize = Mock(return_value=True)
    assert controller.reconnect()
    controller.shutdown.assert_called_once_with()
    controller.initialize.side_effect = CameraConnectionError("offline")
    assert not controller.reconnect()


def test_camera_controller_reports_parameter_and_batch_failures() -> None:
    controller = _controller()
    assert "status" in controller.get_camera_info()
    with pytest.raises(RuntimeError):
        controller.capture_multiple_frames()

    hardware = SimpleNamespace(
        set_exposure_time=Mock(side_effect=OSError("exposure unavailable")),
        get_exposure_time=Mock(side_effect=OSError("exposure unavailable")),
        get_parameter_range=Mock(side_effect=OSError("range unavailable")),
        set_gain=Mock(side_effect=OSError("gain unavailable")),
        get_gain=Mock(side_effect=OSError("gain unavailable")),
    )
    controller.camera = hardware
    controller.is_initialized = True
    assert controller.get_camera_info()["initialized"] is True
    assert not controller.set_exposure(1000)
    assert controller.get_exposure() is None
    assert controller.get_exposure_range() is None
    assert not controller.set_gain(2)
    assert controller.get_gain() is None

    controller.capture_frame = Mock(side_effect=OSError("batch failed"))
    assert controller.capture_multiple_frames() is None
    controller.capture_frame = Mock(side_effect=OSError("test failed"))
    assert not controller.test_camera()


def test_mvs_parameter_trigger_and_device_enumeration_contracts(monkeypatch) -> None:
    camera = MVSCamera(_camera_config())
    camera.cam = MagicMock()
    camera.cam.MV_CC_GetEnumValue.return_value = 0
    assert camera._check_feature_support("TriggerMode")
    camera.cam.MV_CC_GetEnumValue.side_effect = OSError("unsupported")
    assert not camera._check_feature_support("TriggerMode")
    camera.cam.MV_CC_GetEnumValue.side_effect = None

    camera.cam.MV_CC_SetIntValue.side_effect = [0, 0]
    assert camera.set_resolution(640, 480)
    camera.cam.MV_CC_SetIntValue.side_effect = [1]
    assert not camera.set_resolution(640, 480)
    camera.cam.MV_CC_SetIntValue.side_effect = [0, 1]
    assert not camera.set_resolution(640, 480)

    camera.supported_features = {"ExposureAuto": False}
    assert not camera.toggle_auto_exposure()
    camera.supported_features = {"ExposureAuto": True}
    camera.cam.MV_CC_SetEnumValue.return_value = 0
    assert camera.toggle_auto_exposure()
    camera.cam.MV_CC_SetEnumValue.return_value = 1
    assert not camera.toggle_auto_exposure()

    assert camera.process_key(ord("s")) and camera.save_image
    camera.toggle_auto_exposure = Mock(return_value=True)
    assert camera.process_key(ord("a"))
    camera.toggle_auto_exposure.assert_called_once_with()
    assert not camera.process_key(ord("q"))
    assert camera.process_key(ord("x"))

    monkeypatch.setattr(mvs_module._mvs_binding, "MVCAM_DLL_LOAD_ERROR", None)
    monkeypatch.setattr(mvs_module.MvCamera, "MV_CC_EnumDevices", Mock(return_value=1))
    assert not camera.enum_devices()
    enum_devices = Mock(return_value=0)
    monkeypatch.setattr(mvs_module.MvCamera, "MV_CC_EnumDevices", enum_devices)
    camera.deviceList.nDeviceNum = 0
    assert not camera.enum_devices()
    camera.deviceList.nDeviceNum = 2
    assert camera.enum_devices()


def test_mvs_feature_trigger_and_saved_frame_contracts(monkeypatch, tmp_path: Path) -> None:
    camera = MVSCamera(_camera_config())
    camera.cam = MagicMock()
    camera._check_feature_support = Mock(side_effect=[True, False])
    camera._initialize_supported_features()
    assert camera.supported_features == {"ExposureAuto": True, "TriggerMode": False}

    def populate_trigger(_name, param) -> int:
        param.nCurValue = mvs_module.MV_TRIGGER_MODE_OFF
        return 0

    camera.cam.MV_CC_GetEnumValue.side_effect = populate_trigger
    assert camera.check_trigger_mode() == mvs_module.MV_TRIGGER_MODE_OFF
    camera.cam.MV_CC_GetEnumValue.side_effect = None
    camera.cam.MV_CC_GetEnumValue.return_value = 1
    assert camera.check_trigger_mode() is None

    camera.cam.MV_CC_SetEnumValue.return_value = 0
    assert camera.disable_trigger_mode()
    camera.cam.MV_CC_SetEnumValue.return_value = 1
    assert not camera.disable_trigger_mode()

    frame = np.full((2, 2, 3), 9, dtype=np.uint8)
    camera._get_frame_internal = Mock(return_value=frame)
    camera.save_path = str(tmp_path)
    camera.save_image = True
    camera.start_time = 0.0
    monkeypatch.setattr(mvs_module.time, "time", lambda: 2.0)
    write_image = Mock(return_value=True)
    monkeypatch.setattr(mvs_module.cv2, "imwrite", write_image)
    assert camera.get_frame(timeout_ms=25) is frame
    assert camera.current_fps == pytest.approx(0.5)
    assert not camera.save_image
    write_image.assert_called_once()

    camera.save_image = True
    write_image.return_value = False
    assert camera.get_frame() is frame
    assert not camera.save_image
    camera.save_image = True
    write_image.side_effect = OSError("disk full")
    assert camera.get_frame() is frame
    assert not camera.save_image


def test_mvs_numeric_parameters_and_initial_setup(monkeypatch) -> None:
    camera = MVSCamera(_camera_config())
    camera.cam = MagicMock()
    camera.cam.MV_CC_SetFloatValue.return_value = 0
    assert camera.set_exposure_time("1000")
    assert camera.set_gain("2.5")
    camera.cam.MV_CC_SetFloatValue.return_value = 1
    assert not camera.set_exposure_time(1000)
    assert not camera.set_gain(2.5)

    def populate_float(_name, param) -> int:
        param.fCurValue = 5.0
        param.fMin = 1.0
        param.fMax = 10.0
        return 0

    camera.cam.MV_CC_GetFloatValue.side_effect = populate_float
    assert camera.get_exposure_time() == 5.0
    assert camera.get_gain() == 5.0
    assert camera.get_parameter_range("ExposureTime") == {"current": 5.0, "min": 1.0, "max": 10.0}
    camera.cam.MV_CC_GetFloatValue.side_effect = None
    camera.cam.MV_CC_GetFloatValue.return_value = 1
    assert camera.get_exposure_time() is None
    assert camera.get_gain() is None
    assert camera.get_parameter_range("ExposureTime") is None

    camera.cam.MV_CC_SetEnumValue.side_effect = [0, 0]

    def populate_int(_name, param) -> int:
        param.nCurValue = 4096
        return 0

    camera.cam.MV_CC_GetIntValue.side_effect = populate_int
    camera.cam.MV_CC_StartGrabbing.return_value = 0
    assert camera._setup_initial_parameters()
    assert camera.nPayloadSize == 4096
    assert camera._grabbing

    camera.cam.MV_CC_SetEnumValue.side_effect = [1]
    assert not camera._setup_initial_parameters()


def test_mvs_connect_and_frame_failure_contracts(monkeypatch) -> None:
    camera = MVSCamera(_camera_config())
    camera.cam = MagicMock()
    camera._basic_connect = Mock(return_value=True)
    camera._initialize_supported_features = Mock()
    camera.supported_features = {"ExposureAuto": True}
    camera.set_exposure_time = Mock(return_value=True)
    camera.set_gain = Mock(return_value=True)
    camera.check_trigger_mode = Mock(return_value=mvs_module.MV_TRIGGER_MODE_OFF)
    camera.set_resolution = Mock(return_value=True)
    camera.cam.MV_CC_StartGrabbing.return_value = 0
    assert camera.connect_to_camera()
    assert camera._grabbing

    camera.check_trigger_mode.return_value = 1
    camera.disable_trigger_mode = Mock(return_value=False)
    camera.close = Mock()
    assert not camera.connect_to_camera()
    camera.close.assert_called_once_with()

    camera.check_trigger_mode.side_effect = OSError("trigger read failed")
    camera.close.reset_mock()
    assert not camera.connect_to_camera()
    camera.close.assert_called_once_with()

    camera.check_trigger_mode.side_effect = None
    camera.check_trigger_mode.return_value = mvs_module.MV_TRIGGER_MODE_OFF
    camera.cam.MV_CC_StartGrabbing.return_value = 1
    camera.close.reset_mock()
    assert not camera.connect_to_camera()
    camera.close.assert_called_once_with()

    camera.cam.MV_CC_StartGrabbing.return_value = 0
    camera.set_exposure_time.side_effect = OSError("configuration failed")
    assert not camera.connect_to_camera()
    camera.set_exposure_time.side_effect = None

    camera._basic_connect.return_value = False
    assert not camera.connect_to_camera()

    camera._get_frame_internal = Mock(return_value=None)
    assert camera.get_frame() is None
    camera._get_frame_internal.side_effect = OSError("capture failed")
    assert camera.get_frame() is None

    camera.cam.MV_CC_GetImageBuffer.return_value = mvs_module.MV_E_NODATA
    camera._get_frame_internal = MVSCamera._get_frame_internal.__get__(camera)
    assert camera._get_frame_internal(timeout_ms=-5) is None
    camera.cam.MV_CC_GetImageBuffer.return_value = 123
    assert camera._get_frame_internal() is None

    camera.supported_features = {"Contrast": True}
    monkeypatch.setattr(mvs_module.cv2, "namedWindow", Mock())
    create_trackbar = Mock()
    monkeypatch.setattr(mvs_module.cv2, "createTrackbar", create_trackbar)
    camera.create_control_window()
    assert create_trackbar.call_count == 3
