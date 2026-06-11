"""MVS camera SDK loading and lifecycle guard tests."""

from ctypes import POINTER, c_ubyte, cast
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import MvImport.MvCameraControl_class as mvs_binding
from camera.MVS_camera_control import MVSCamera
from core.exceptions import CameraConnectionError


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.exposure_time = "1000"
    config.gain = "1.0"
    config.width = 640
    config.height = 640
    config.MV_CC_GetImageBuffer_nMsec = 1000
    return config


def test_dll_path_is_derived_from_repo_layout():
    """dll_path must not depend on the repo living at a hardcoded location."""
    repo_root = Path(__file__).resolve().parents[1]
    expected = repo_root / "Runtime" / "MvCameraControl.dll"
    assert Path(mvs_binding.dll_path) == expected


def test_enum_devices_reports_dll_load_failure(mock_config, monkeypatch):
    """A DLL load failure must not be misreported as 'no devices found'."""
    monkeypatch.setattr(
        mvs_binding, "MVCAM_DLL_LOAD_ERROR", "Failed to load X.dll: boom"
    )
    camera = MVSCamera(mock_config)

    with pytest.raises(CameraConnectionError, match="Failed to load X.dll"):
        camera.enum_devices()


def test_basic_connect_rejects_out_of_range_device_index(mock_config):
    """Without enumerated devices, connect must fail before any ctypes cast."""
    camera = MVSCamera(mock_config)
    camera.cam = MagicMock()
    assert camera.deviceList.nDeviceNum == 0

    assert camera._basic_connect(0) is False
    camera.cam.MV_CC_CreateHandle.assert_not_called()


def test_close_before_connect_is_a_noop(mock_config):
    camera = MVSCamera(mock_config)
    camera.cam = MagicMock()

    camera.close()

    camera.cam.MV_CC_StopGrabbing.assert_not_called()
    camera.cam.MV_CC_CloseDevice.assert_not_called()
    camera.cam.MV_CC_DestroyHandle.assert_not_called()


def test_close_releases_each_reached_stage_once(mock_config):
    camera = MVSCamera(mock_config)
    camera.cam = MagicMock()
    camera.cam.MV_CC_StopGrabbing.return_value = 0
    camera.cam.MV_CC_CloseDevice.return_value = 0
    camera.cam.MV_CC_DestroyHandle.return_value = 0
    camera._handle_created = True
    camera._device_opened = True
    camera._grabbing = True

    camera.close()
    camera.close()  # idempotent

    camera.cam.MV_CC_StopGrabbing.assert_called_once()
    camera.cam.MV_CC_CloseDevice.assert_called_once()
    camera.cam.MV_CC_DestroyHandle.assert_called_once()


def test_get_frame_returns_unmodified_pixels(mock_config):
    """Captured frames feed inference; no overlay may be drawn on them."""
    camera = MVSCamera(mock_config)
    pristine = np.full((64, 64, 3), 37, dtype=np.uint8)
    camera._get_frame_internal = lambda: pristine.copy()

    frame = camera.get_frame()

    assert frame is not None
    assert np.array_equal(frame, pristine)


class _FakeSdkCam:
    """Simulates MV_CC_GetImageBuffer filling an SDK-owned Bayer buffer."""

    def __init__(self, height: int, width: int, value: int = 37) -> None:
        self._height = height
        self._width = width
        self._value = value
        self.freed = 0
        self._buf = None

    def MV_CC_GetImageBuffer(self, st_out, timeout_ms):
        n = self._height * self._width
        self._buf = (c_ubyte * n)(*([self._value] * n))
        st_out.pBufAddr = cast(self._buf, POINTER(c_ubyte))
        st_out.stFrameInfo.nFrameLen = n
        st_out.stFrameInfo.nWidth = self._width
        st_out.stFrameInfo.nHeight = self._height
        return 0

    def MV_CC_FreeImageBuffer(self, st_out):
        self.freed += 1
        return 0


def test_frame_decode_copies_pixels_out_of_sdk_buffer(mock_config):
    """Decoded frame must be a stable BGR copy, valid after buffer release."""
    camera = MVSCamera(mock_config)
    fake = _FakeSdkCam(8, 8, value=37)
    camera.cam = fake

    frame = camera._get_frame_internal()

    assert fake.freed == 1, "SDK buffer must be returned exactly once"
    assert frame is not None
    assert frame.shape == (8, 8, 3)
    assert frame.dtype == np.uint8
    # A constant Bayer plane demosaics to the same constant in every channel.
    assert np.all(frame == 37)
    # Frame must not alias the (already released) SDK buffer.
    fake._buf[0] = 0
    assert np.all(frame == 37)


def test_frame_decode_rejects_short_buffer(mock_config):
    """A truncated SDK buffer must be dropped, not crash reshape."""
    camera = MVSCamera(mock_config)
    fake = _FakeSdkCam(8, 8)

    original = fake.MV_CC_GetImageBuffer

    def short_buffer(st_out, timeout_ms):
        ret = original(st_out, timeout_ms)
        st_out.stFrameInfo.nFrameLen = 8  # claim 8x8 but provide 8 bytes
        return ret

    fake.MV_CC_GetImageBuffer = short_buffer
    camera.cam = fake

    assert camera._get_frame_internal() is None
    assert fake.freed == 1


def test_open_device_failure_releases_orphaned_handle(mock_config):
    """CreateHandle OK + OpenDevice FAIL must destroy the handle, not leak it."""
    camera = MVSCamera(mock_config)
    camera.cam = MagicMock()
    camera.cam.MV_CC_CreateHandle.return_value = 0
    camera.cam.MV_CC_OpenDevice.return_value = 0x80000000
    camera.cam.MV_CC_DestroyHandle.return_value = 0
    camera.deviceList = MagicMock()
    camera.deviceList.nDeviceNum = 1
    camera.deviceList.pDeviceInfo = (MagicMock(),)

    import camera.MVS_camera_control as mvs_module

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mvs_module, "cast", lambda *a, **k: MagicMock())
        assert camera._basic_connect(0) is False

    camera.cam.MV_CC_DestroyHandle.assert_called_once()
    assert camera._handle_created is False
