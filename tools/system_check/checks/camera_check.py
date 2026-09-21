"""Camera and serial-light checks.

Deliberately limited to enumeration. Enumerating Hikrobot devices is passive;
*opening* one takes exclusive access and would disturb a station that is mid
shift. The deeper test — open the camera and pull one frame — already exists as
``yolo11_inference.exe --check-camera-grab`` and is named in the remedy text so
an operator can run it when the line is idle.

Enumeration needs the ``MvImport`` bindings, which ship inside the application.
When the checker runs as a standalone executable it cannot import them, so it
reports what it *can* confirm (the SDK files and DLL load, covered by
``runtime_check``) and points at the application's own command for the rest.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status

#: Bit flags from MvImport/CameraParams_const.py, inlined so this module does
#: not depend on the application package being importable.
_MV_GIGE_DEVICE = 0x00000001
_MV_USB_DEVICE = 0x00000004


def check(context: AppContext) -> list[CheckResult]:
    """Report attached cameras and serial ports, without claiming either."""
    return [_camera_result(context), _serial_result()]


def _camera_result(context: AppContext) -> CheckResult:
    """Enumerate Hikrobot devices when the bindings are importable."""
    if os.name != "nt":
        return CheckResult(
            check_id="camera.device",
            title="Camera",
            status=Status.SKIP,
            detail="Camera enumeration is Windows-only.",
            requirement="Hikrobot GigE or USB3 Vision camera",
            measured="n/a",
        )

    if os.environ.get("YOLO11_CAMERA_MOCK") == "1":
        return CheckResult(
            check_id="camera.device",
            title="Camera",
            status=Status.WARNING,
            detail=(
                "YOLO11_CAMERA_MOCK=1 is set in this environment. The SDK is "
                "mocked and answers MV_OK to every call, so no real camera "
                "result can be trusted while it is set."
            ),
            requirement="Hikrobot GigE or USB3 Vision camera",
            measured="mock SDK active",
            source="MvImport/MvCameraControl_class.py:61-63",
            remedy="Clear YOLO11_CAMERA_MOCK before running a production preflight.",
        )

    enumerated = _enumerate_devices(context)
    if enumerated.get("status") == "unavailable":
        return CheckResult(
            check_id="camera.device",
            title="Camera",
            status=Status.UNKNOWN,
            detail=(
                "Camera enumeration was not possible from this process: "
                f"{enumerated.get('reason')}. The SDK files and DLL load are "
                "checked separately under Native runtime."
            ),
            requirement="Hikrobot GigE or USB3 Vision camera",
            measured="not enumerated",
            confidence=Confidence.UNKNOWN,
            remedy=(
                "With the line idle, run: yolo11_inference.exe "
                "--check-camera-grab"
            ),
            data=enumerated,
        )

    if enumerated.get("status") == "error":
        return CheckResult(
            check_id="camera.device",
            title="Camera",
            status=Status.FAIL,
            detail=(
                "MV_CC_EnumDevices returned an error: "
                f"{enumerated.get('reason')}."
            ),
            requirement="Hikrobot GigE or USB3 Vision camera",
            measured="enumeration failed",
            source="GUI.py:134-141",
            remedy=(
                "Check the Hikrobot driver installation and, for GigE, the "
                "network adapter configuration."
            ),
            data=enumerated,
        )

    count = int(enumerated.get("count", 0))
    if count == 0:
        return CheckResult(
            check_id="camera.device",
            title="Camera",
            status=Status.FAIL,
            detail=(
                "No Hikrobot camera found. Image-file inference still works; "
                "live inspection does not."
            ),
            requirement="Hikrobot GigE or USB3 Vision camera",
            measured="0 devices",
            source="GUI.py:134-141",
            remedy=(
                "Connect the camera and install the Hikrobot MVS driver. For "
                "GigE, confirm the adapter is on the camera's subnet and jumbo "
                "frames are enabled."
            ),
            data=enumerated,
        )

    return CheckResult(
        check_id="camera.device",
        title="Camera",
        status=Status.PASS,
        detail=(
            f"{count} Hikrobot device(s) enumerated. The device was not opened: "
            "opening takes exclusive access and would interrupt a running line."
        ),
        requirement="Hikrobot GigE or USB3 Vision camera",
        measured=f"{count} device(s)",
        source="GUI.py:134-141",
        remedy=(
            "For a full acquisition test with the line idle, run: "
            "yolo11_inference.exe --check-camera-grab"
        ),
        data=enumerated,
    )


def _enumerate_devices(context: AppContext) -> dict[str, Any]:
    """Enumerate devices through the application's MvImport bindings.

    Returns a payload describing the outcome rather than raising, so a station
    without the bindings still produces a report line.
    """
    try:
        header = importlib.import_module("MvImport.CameraParams_header")
        binding = importlib.import_module("MvImport.MvCameraControl_class")
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unavailable",
            "reason": (
                f"MvImport is not importable here ({type(exc).__name__}: {exc})"
            ),
        }

    load_error = getattr(binding, "MVCAM_DLL_LOAD_ERROR", None)
    if load_error:
        return {"status": "error", "reason": str(load_error)}

    try:
        device_list = header.MV_CC_DEVICE_INFO_LIST()
        ret = binding.MvCamera.MV_CC_EnumDevices(
            _MV_GIGE_DEVICE | _MV_USB_DEVICE, device_list
        )
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}

    if ret != 0:
        return {"status": "error", "reason": f"ret=0x{ret & 0xFFFFFFFF:08x}"}
    return {"status": "ok", "count": int(device_list.nDeviceNum)}


def _serial_result() -> CheckResult:
    """Report serial ports for the optional LED controller."""
    try:
        from serial.tools import list_ports
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="serial.light",
            title="Serial LED controller",
            status=Status.WARNING,
            detail=(
                "pyserial is not importable, so serial ports could not be "
                f"enumerated ({type(exc).__name__}). Illumination control is "
                "optional; inspection runs without it."
            ),
            requirement="pyserial, optional COM port at 115200 baud",
            measured="pyserial unavailable",
            source="core/services/light_controller.py:56-81",
            remedy="pip install pyserial==3.5 if this station drives the LED bar.",
        )

    try:
        ports = [(port.device, port.description or port.device) for port in list_ports.comports()]
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="serial.light",
            title="Serial LED controller",
            status=Status.UNKNOWN,
            detail=f"Serial enumeration failed: {type(exc).__name__}: {exc}",
            requirement="Optional COM port at 115200 baud",
            measured="unknown",
            confidence=Confidence.UNKNOWN,
        )

    if not ports:
        return CheckResult(
            check_id="serial.light",
            title="Serial LED controller",
            status=Status.WARNING,
            detail=(
                "No serial ports detected. The LED controller is optional, but "
                "illumination calibration cannot run without it."
            ),
            requirement="Optional COM port at 115200 baud",
            measured="0 ports",
            source="core/services/light_controller.py:30",
            remedy="Connect the LED controller if this station calibrates illumination.",
        )

    listed = ", ".join(device for device, _ in ports)
    return CheckResult(
        check_id="serial.light",
        title="Serial LED controller",
        status=Status.PASS,
        detail=(
            f"{len(ports)} serial port(s) available: {listed}. The port was not "
            "opened and no frame was written."
        ),
        requirement="Optional COM port at 115200 baud",
        measured=listed,
        source="core/services/light_controller.py:30",
        data={"ports": [{"device": device, "description": text} for device, text in ports]},
    )
