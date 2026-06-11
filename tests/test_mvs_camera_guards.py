"""MVS camera SDK loading and lifecycle guard tests."""

from pathlib import Path
from unittest.mock import MagicMock

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
