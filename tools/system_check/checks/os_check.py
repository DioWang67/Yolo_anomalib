"""Operating system checks.

The application is Windows-only in practice: the camera SDK is loaded through
``ctypes.WinDLL``, the DLL search path is extended with
``os.add_dll_directory``, and every launcher is a ``.bat`` file.
"""

from __future__ import annotations

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.sysinfo import os_info


def check(context: AppContext) -> list[CheckResult]:
    """Report Windows version, build and architecture."""
    info = os_info()
    results: list[CheckResult] = []

    description = info.edition or f"{info.system} {info.release}"
    build_text = f" build {info.build}" if info.build else ""
    measured = f"{description}{build_text} ({info.machine})"

    if info.system != "Windows":
        results.append(
            CheckResult(
                check_id="os.platform",
                title="Operating system",
                status=Status.FAIL,
                detail=(
                    f"This installation targets Windows; running on {info.system}. "
                    "The Hikrobot SDK is loaded with ctypes.WinDLL and has no "
                    "non-Windows path."
                ),
                requirement="Windows x64",
                measured=measured,
                source="MvImport/MvCameraControl_class.py:31-44",
                remedy="Run the inspection station on 64-bit Windows.",
                data=_payload(info),
            )
        )
        return results

    results.append(
        CheckResult(
            check_id="os.platform",
            title="Operating system",
            status=Status.PASS,
            detail=f"{measured}.",
            requirement="Windows x64",
            measured=measured,
            source="MvImport/MvCameraControl_class.py:31-44",
            data=_payload(info),
        )
    )

    if not info.is_64bit:
        results.append(
            CheckResult(
                check_id="os.architecture",
                title="OS architecture",
                status=Status.FAIL,
                detail=(
                    f"Architecture is {info.machine}. The pinned torch, "
                    "onnxruntime and Hikrobot binaries are all x64."
                ),
                requirement="x64 (AMD64)",
                measured=info.machine,
                source="requirements.txt, Runtime/*.dll",
                remedy="Use a 64-bit Windows installation.",
            )
        )
    elif not info.python_64bit:
        results.append(
            CheckResult(
                check_id="os.architecture",
                title="Interpreter architecture",
                status=Status.FAIL,
                detail=(
                    "A 32-bit Python is running on 64-bit Windows; the pinned "
                    "wheels and the Hikrobot DLLs are 64-bit only."
                ),
                requirement="64-bit interpreter",
                measured="32-bit",
                remedy="Reinstall the 64-bit interpreter, or run the 64-bit build.",
            )
        )
    else:
        results.append(
            CheckResult(
                check_id="os.architecture",
                title="Architecture",
                status=Status.PASS,
                detail="64-bit Windows with a 64-bit interpreter.",
                requirement="x64 (AMD64)",
                measured=f"{info.machine}, 64-bit interpreter",
            )
        )

    results.append(_build_result(info))
    return results


def _build_result(info) -> CheckResult:
    """Report the Windows build, without asserting a minimum.

    No file in the repository names a minimum Windows build, so this check
    records the build and never fails on it.
    """
    return CheckResult(
        check_id="os.build",
        title="Windows build",
        status=Status.PASS if info.build else Status.UNKNOWN,
        detail=(
            f"Windows build {info.build}."
            if info.build
            else "Windows build could not be read from the registry."
        ),
        requirement="No minimum build is stated in the repository",
        measured=info.build or "unknown",
        confidence=Confidence.UNKNOWN,
    )


def _payload(info) -> dict[str, object]:
    """Structured OS facts for the JSON report and baseline comparison."""
    return {
        "system": info.system,
        "release": info.release,
        "version": info.version,
        "build": info.build,
        "edition": info.edition,
        "machine": info.machine,
        "is_64bit": info.is_64bit,
        "python_64bit": info.python_64bit,
    }
