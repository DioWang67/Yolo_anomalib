"""Backward-compatible entrypoint for launching the PyQt GUI."""

from __future__ import annotations

import multiprocessing
import os
import sys
from ctypes import POINTER, WinDLL, cast
from pathlib import Path

# PyInstaller freeze support must be called before any other code runs.
# Without this, torch's use of multiprocessing spawns a second process on
# Windows that re-executes this script, causing a duplicate GUI window.
multiprocessing.freeze_support()

# When running as a PyInstaller bundle, point TORCH_HOME at the bundled
# timm weight cache so Patchcore backbone loads offline (no internet needed).
if getattr(sys, "frozen", False):
    _timm_cache = os.path.join(getattr(sys, "_MEIPASS", ""), "timm_cache")
    if os.path.isdir(_timm_cache):
        os.environ.setdefault("TORCH_HOME", _timm_cache)


def _runtime_root() -> Path:
    """Return the Hikrobot runtime directory for source or packaged execution."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "")) / "Runtime"
    return Path(__file__).resolve().parent / "Runtime"


def _check_hikrobot_runtime() -> int:
    """Validate packaged Hikrobot runtime files and DLL loading."""
    runtime_dir = _runtime_root()
    cl_protocol_dir = runtime_dir / "CLProtocol"
    required_files = [
        "MvCameraControl.dll",
        "MVGigEVisionSDK.dll",
        "MvUsb3vTL.dll",
        "MvProducerGEV.cti",
        "MvProducerU3V.cti",
        "GenApi_MD_VC120_v3_0_MV.dll",
        "GCBase_MD_VC120_v3_0_MV.dll",
        "CLProtocol/Win64_x64/GenCP_MD_VC120_v3_0_MV.dll",
    ]

    print(f"Hikrobot runtime dir: {runtime_dir}")
    if not runtime_dir.is_dir():
        print("Hikrobot runtime preflight failed: Runtime directory missing", file=sys.stderr)
        return 1

    missing = []
    for name in required_files:
        path = runtime_dir / name
        print(f"{name}: {'OK' if path.exists() else 'MISSING'}")
        if not path.exists():
            missing.append(name)

    if missing:
        print(f"Hikrobot runtime preflight failed: missing {missing}", file=sys.stderr)
        return 1

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(runtime_dir))
    os.environ["PATH"] = str(runtime_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["GENICAM_GENTL64_PATH"] = (
        str(runtime_dir)
        + os.pathsep
        + os.environ.get("GENICAM_GENTL64_PATH", "")
    )
    if cl_protocol_dir.is_dir():
        os.environ["MVCAM_GENICAM_CLPROTOCOL"] = str(cl_protocol_dir)

    try:
        WinDLL(str(runtime_dir / "MvCameraControl.dll"))
    except Exception as exc:
        print(f"Hikrobot runtime preflight failed: {exc}", file=sys.stderr)
        return 1

    print("Hikrobot runtime preflight OK")
    return 0


def _ret_text(ret: int) -> str:
    """Format an MVS return code as unsigned hexadecimal."""
    return f"0x{ret & 0xFFFFFFFF:08x}"


def _get_camera_int_value(camera, name: str) -> tuple[int, int | None]:
    """Read an integer node from an opened Hikrobot camera."""
    from MvImport.CameraParams_header import MVCC_INTVALUE

    int_value = MVCC_INTVALUE()
    ret = camera.MV_CC_GetIntValue(name, int_value)
    if ret != 0:
        return ret, None
    return ret, int(int_value.nCurValue)


def _print_camera_int_value(camera, name: str) -> None:
    """Print an integer camera node without failing the whole preflight."""
    ret, value = _get_camera_int_value(camera, name)
    if ret == 0:
        print(f"{name}: {value}")
    else:
        print(f"{name}: unavailable, ret={_ret_text(ret)}")


def _check_camera_grab() -> int:
    """Open the first Hikrobot camera and validate that one frame can be received."""
    runtime_ret = _check_hikrobot_runtime()
    if runtime_ret != 0:
        return runtime_ret

    from MvImport.CameraParams_const import (
        MV_ACCESS_Exclusive,
        MV_GIGE_DEVICE,
        MV_USB_DEVICE,
    )
    from MvImport.CameraParams_header import (
        MV_CC_DEVICE_INFO,
        MV_CC_DEVICE_INFO_LIST,
        MV_FRAME_OUT,
    )
    from MvImport.MvCameraControl_class import MvCamera
    from MvImport.MvErrorDefine_const import MV_E_NODATA

    device_list = MV_CC_DEVICE_INFO_LIST()
    camera = MvCamera()
    handle_created = False
    device_opened = False
    grabbing = False

    try:
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if ret != 0:
            print(f"EnumDevices: FAIL, ret={_ret_text(ret)}")
            return 1

        print(f"EnumDevices: OK, count={device_list.nDeviceNum}")
        if device_list.nDeviceNum == 0:
            print("Camera grab preflight failed: no Hikrobot camera found")
            return 1

        device_info = cast(
            device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
        ).contents

        ret = camera.MV_CC_CreateHandle(device_info)
        if ret != 0:
            print(f"CreateHandle: FAIL, ret={_ret_text(ret)}")
            return 1
        handle_created = True
        print("CreateHandle: OK")

        ret = camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"OpenDevice: FAIL, ret={_ret_text(ret)}")
            return 1
        device_opened = True
        print("OpenDevice: OK")

        ret = camera.MV_CC_SetEnumValueByString("TriggerMode", "Off")
        print(f"TriggerMode: {'Off' if ret == 0 else 'SET_FAIL'}, ret={_ret_text(ret)}")

        ret = camera.MV_CC_SetEnumValueByString("AcquisitionMode", "Continuous")
        print(
            "AcquisitionMode: "
            f"{'Continuous' if ret == 0 else 'SET_FAIL'}, ret={_ret_text(ret)}"
        )

        _print_camera_int_value(camera, "Width")
        _print_camera_int_value(camera, "Height")
        _print_camera_int_value(camera, "PayloadSize")
        _print_camera_int_value(camera, "GevSCPSPacketSize")

        ret = camera.MV_CC_StartGrabbing()
        if ret != 0:
            print(f"StartGrabbing: FAIL, ret={_ret_text(ret)}")
            return 1
        grabbing = True
        print("StartGrabbing: OK")

        frame = MV_FRAME_OUT()
        ret = camera.MV_CC_GetImageBuffer(frame, 10000)
        if ret == 0:
            try:
                print(
                    "GetImageBuffer: OK, "
                    f"width={frame.stFrameInfo.nWidth}, "
                    f"height={frame.stFrameInfo.nHeight}, "
                    f"frame_len={frame.stFrameInfo.nFrameLen}"
                )
            finally:
                camera.MV_CC_FreeImageBuffer(frame)
            return 0

        if ret == MV_E_NODATA:
            print(f"GetImageBuffer: MV_E_NODATA, ret={_ret_text(ret)}")
        else:
            print(f"GetImageBuffer: FAIL, ret={_ret_text(ret)}")
        return 1
    except Exception as exc:
        print(f"Camera grab preflight failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if grabbing:
            ret = camera.MV_CC_StopGrabbing()
            print(f"StopGrabbing: ret={_ret_text(ret)}")
        if device_opened:
            ret = camera.MV_CC_CloseDevice()
            print(f"CloseDevice: ret={_ret_text(ret)}")
        if handle_created:
            ret = camera.MV_CC_DestroyHandle()
            print(f"DestroyHandle: ret={_ret_text(ret)}")


if "--check-onnxruntime" in sys.argv:
    from core.runtime_preflight import validate_runtime_for_model

    try:
        validate_runtime_for_model("preflight.onnx")
    except Exception as exc:
        print(f"ONNX Runtime preflight failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("ONNX Runtime preflight OK")
    sys.exit(0)

if "--check-hikrobot-runtime" in sys.argv:
    sys.exit(_check_hikrobot_runtime())

if "--check-camera-grab" in sys.argv:
    sys.exit(_check_camera_grab())

if "--help" in sys.argv:
    print(
        "Usage: yolo11_inference.exe "
        "[--check-onnxruntime] [--check-hikrobot-runtime] [--check-camera-grab]"
    )
    sys.exit(0)

from app.gui import DetectionSystemGUI, main  # noqa: E402

# Expose DetectionSystem for tests that patch GUI.DetectionSystem
from core.detection_system import DetectionSystem  # noqa: E402

__all__ = ["DetectionSystemGUI", "DetectionSystem", "main"]


if __name__ == "__main__":
    main()
