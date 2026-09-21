"""Native runtime checks: ONNX Runtime, Visual C++, Hikrobot SDK, model files.

These are the failures that actually strand a deployment on a fresh Windows
machine. They are checked the same way the application checks them at startup,
so a PASS here means the application's own preflight will pass too:

* ONNX Runtime must import and expose ``CPUExecutionProvider``
  (``core/runtime_preflight.py``).
* The Hikrobot ``Runtime`` directory must hold the eight files
  ``GUI.py --check-hikrobot-runtime`` requires, and ``MvCameraControl.dll``
  must actually load.
* The configured model weights must exist on disk.

DLL probing uses ``os.add_dll_directory``, whose effect is confined to this
process and released afterwards. Nothing here writes to the registry, the
environment, or any file.
"""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path

from tools.system_check.context import AppContext, discover_model_targets
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.spec import (
    HIKROBOT_RUNTIME_FILES,
    REQUIRED_ORT_PROVIDER,
    VC120_RUNTIME_DLLS,
    VCRUNTIME_DLLS,
)
from tools.system_check.sysinfo import BYTES_PER_MB


def _windll(name: str) -> object:
    """Load a DLL by name or path, Windows only.

    ``ctypes.WinDLL`` is absent from the POSIX stubs, so it is resolved
    dynamically; every caller here is already behind an ``os.name == "nt"``
    guard.

    Raises:
        OSError: If the library or one of its dependencies cannot be loaded.
    """
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:  # pragma: no cover - Windows-only code path
        raise OSError("ctypes.WinDLL is unavailable on this platform")
    return loader(name)


def check(context: AppContext) -> list[CheckResult]:
    """Run every native-runtime check."""
    return [
        _onnx_provider_result(),
        *_vcredist_results(context),
        *_hikrobot_results(context),
        *_model_results(context),
    ]


# --------------------------------------------------------------------------
# ONNX Runtime
# --------------------------------------------------------------------------


def _onnx_provider_result() -> CheckResult:
    """Reproduce the gate in ``core/runtime_preflight.validate_runtime_for_model``."""
    try:
        ort = importlib.import_module("onnxruntime")
        providers = list(ort.get_available_providers())
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            check_id="runtime.onnx_provider",
            title="ONNX Runtime",
            status=Status.FAIL,
            detail=(
                f"ONNX Runtime cannot be loaded ({type(exc).__name__}: {exc}). "
                "Every .onnx model in this installation will refuse to "
                "initialise with exactly this error."
            ),
            requirement=f"onnxruntime importable with {REQUIRED_ORT_PROVIDER}",
            measured="import failed",
            source="core/runtime_preflight.py:118-134",
            remedy=(
                "Install the Microsoft Visual C++ Redistributable 2015-2022 "
                "x64, then onnxruntime==1.23.2."
            ),
        )

    if REQUIRED_ORT_PROVIDER not in providers:
        return CheckResult(
            check_id="runtime.onnx_provider",
            title="ONNX Runtime",
            status=Status.FAIL,
            detail=(
                f"{REQUIRED_ORT_PROVIDER} is not registered; available "
                f"providers are {providers}."
            ),
            requirement=f"{REQUIRED_ORT_PROVIDER} available",
            measured=", ".join(providers) or "none",
            source="core/runtime_preflight.py:136-145",
            remedy="Reinstall the onnxruntime version pinned in requirements.txt.",
            data={"providers": providers},
        )

    return CheckResult(
        check_id="runtime.onnx_provider",
        title="ONNX Runtime",
        status=Status.PASS,
        detail=(
            f"onnxruntime {getattr(ort, '__version__', 'unknown')} loaded with "
            f"{REQUIRED_ORT_PROVIDER}."
        ),
        requirement=f"{REQUIRED_ORT_PROVIDER} available",
        measured=", ".join(providers),
        source="core/runtime_preflight.py:136-145",
        data={"providers": providers, "version": getattr(ort, "__version__", None)},
    )


# --------------------------------------------------------------------------
# Visual C++ runtime
# --------------------------------------------------------------------------


def _vcredist_results(context: AppContext) -> list[CheckResult]:
    """Probe the Visual C++ runtimes this installation depends on.

    Two different situations, and conflating them produces a false alarm:

    * ONNX Runtime imports the 2015-2022 CRT and does **not** ship it, so it
      has to be installed system-wide.
    * The Hikrobot SDK imports the 2013 CRT and **does** ship it inside
      ``Runtime``. Windows resolves a dependent DLL from the loading module's
      own directory, so a station with no VC++ 2013 redistributable is still
      perfectly healthy. Checking only the system search path reports a
      missing runtime on every such machine.
    """
    if os.name != "nt":
        return [
            CheckResult(
                check_id="runtime.vcredist",
                title="Visual C++ runtime",
                status=Status.SKIP,
                detail="Not applicable outside Windows.",
                requirement="MSVC 2015-2022 x64 redistributable",
                measured="n/a",
            )
        ]

    return [
        _dll_group_result(
            "runtime.vcredist",
            "Visual C++ 2015-2022 runtime",
            VCRUNTIME_DLLS,
            shipped_dir=None,
            purpose=(
                "onnxruntime.dll and onnxruntime_pybind11_state.pyd import "
                "these directly and do not ship them."
            ),
            remedy=(
                "Install Microsoft Visual C++ Redistributable 2015-2022 x64 "
                "(vc_redist.x64.exe)."
            ),
            source="PE imports of onnxruntime.dll, core/runtime_preflight.py:131-133",
        ),
        _dll_group_result(
            "runtime.vc120",
            "Visual C++ 2013 runtime",
            VC120_RUNTIME_DLLS,
            shipped_dir=context.runtime_dir,
            purpose=(
                "14 binaries in Runtime import these, including "
                "GenApi_MD_VC120_v3_0_MV.dll and GCBase_MD_VC120_v3_0_MV.dll, "
                "which the camera path needs."
            ),
            remedy=(
                "Restore the missing CRT files to Runtime, or install "
                "Microsoft Visual C++ 2013 Redistributable x64."
            ),
            source="PE imports of Runtime/GenApi_MD_VC120_v3_0_MV.dll",
        ),
    ]


def _dll_group_result(
    check_id: str,
    title: str,
    dll_names: tuple[str, ...],
    *,
    shipped_dir: Path | None,
    purpose: str,
    remedy: str,
    source: str,
) -> CheckResult:
    """Report whether each CRT DLL is resolvable, and from where.

    Args:
        shipped_dir: Directory the application bundles its own copies in, or
            ``None`` when the runtime must come from the system. A DLL found
            there counts as satisfied, because Windows resolves dependencies
            from the loading module's directory before the system path.
    """
    from_system: list[str] = []
    from_bundle: list[str] = []
    missing: list[str] = []

    for name in dll_names:
        if shipped_dir is not None and (shipped_dir / name).is_file():
            from_bundle.append(name)
            continue
        try:
            _windll(name)
            from_system.append(name)
        except OSError:
            missing.append(name)

    payload = {
        "from_system": from_system,
        "from_bundle": from_bundle,
        "missing": missing,
        "shipped_dir": str(shipped_dir) if shipped_dir else None,
    }

    if not missing:
        if from_bundle and not from_system:
            measured = f"shipped in {shipped_dir.name if shipped_dir else '?'}"
            detail = (
                f"{', '.join(from_bundle)} ship with the SDK in {shipped_dir}, "
                f"so no separate redistributable is needed. {purpose}"
            )
        elif from_bundle:
            measured = "partly shipped, partly system"
            detail = (
                f"Shipped in {shipped_dir}: {', '.join(from_bundle)}. "
                f"From the system: {', '.join(from_system)}. {purpose}"
            )
        else:
            measured = "installed system-wide"
            detail = f"{', '.join(from_system)} present on the system. {purpose}"
        return CheckResult(
            check_id=check_id,
            title=title,
            status=Status.PASS,
            detail=detail,
            requirement=", ".join(dll_names),
            measured=measured,
            source=source,
            data=payload,
        )

    return CheckResult(
        check_id=check_id,
        title=title,
        status=Status.FAIL,
        detail=(
            f"Not resolvable: {', '.join(missing)}. {purpose} Neither the "
            "system search path nor "
            + (f"{shipped_dir} " if shipped_dir else "the application ")
            + "provides them."
        ),
        requirement=", ".join(dll_names),
        measured=f"missing {', '.join(missing)}",
        source=source,
        remedy=remedy,
        data=payload,
    )


# --------------------------------------------------------------------------
# Hikrobot SDK
# --------------------------------------------------------------------------


def _hikrobot_results(context: AppContext) -> list[CheckResult]:
    """Check the packaged Hikrobot runtime files, then try to load the SDK."""
    runtime_dir = context.runtime_dir
    if not runtime_dir.is_dir():
        return [
            CheckResult(
                check_id="runtime.hikrobot_files",
                title="Hikrobot SDK files",
                status=Status.FAIL,
                detail=(
                    f"Runtime directory not found at {runtime_dir}. Camera "
                    "acquisition cannot start; image-file inference still can."
                ),
                requirement="Runtime/ with the Hikrobot MVS SDK",
                measured="directory missing",
                source="GUI.py:31-60",
                remedy=(
                    "Copy the Runtime directory from the build output; it is "
                    "bundled by yolo11_inference.spec."
                ),
            )
        ]

    missing = [name for name in HIKROBOT_RUNTIME_FILES if not (runtime_dir / name).exists()]
    results: list[CheckResult] = []

    if missing:
        results.append(
            CheckResult(
                check_id="runtime.hikrobot_files",
                title="Hikrobot SDK files",
                status=Status.FAIL,
                detail=(
                    f"{len(missing)} of {len(HIKROBOT_RUNTIME_FILES)} required "
                    f"files are missing from {runtime_dir}: {', '.join(missing)}."
                ),
                requirement=f"{len(HIKROBOT_RUNTIME_FILES)} SDK files in Runtime/",
                measured=f"{len(missing)} missing",
                source="GUI.py:35-44",
                remedy="Re-copy Runtime/ from the build output.",
                data={"runtime_dir": str(runtime_dir), "missing": missing},
            )
        )
    else:
        results.append(
            CheckResult(
                check_id="runtime.hikrobot_files",
                title="Hikrobot SDK files",
                status=Status.PASS,
                detail=(
                    f"All {len(HIKROBOT_RUNTIME_FILES)} required SDK files "
                    f"present in {runtime_dir}."
                ),
                requirement=f"{len(HIKROBOT_RUNTIME_FILES)} SDK files in Runtime/",
                measured="all present",
                source="GUI.py:35-44",
                data={"runtime_dir": str(runtime_dir)},
            )
        )

    results.append(_hikrobot_load_result(runtime_dir))
    return results


def _hikrobot_load_result(runtime_dir: Path) -> CheckResult:
    """Attempt a process-local load of ``MvCameraControl.dll``."""
    dll_path = runtime_dir / "MvCameraControl.dll"
    if os.name != "nt":
        return CheckResult(
            check_id="runtime.hikrobot_load",
            title="Hikrobot SDK load",
            status=Status.SKIP,
            detail="DLL loading is Windows-only.",
            requirement="MvCameraControl.dll loads",
            measured="n/a",
        )
    if not dll_path.exists():
        return CheckResult(
            check_id="runtime.hikrobot_load",
            title="Hikrobot SDK load",
            status=Status.FAIL,
            detail=f"{dll_path} does not exist.",
            requirement="MvCameraControl.dll loads",
            measured="file missing",
            source="MvImport/MvCameraControl_class.py:66",
            remedy="Re-copy Runtime/ from the build output.",
        )

    handle = None
    try:
        if hasattr(os, "add_dll_directory"):
            handle = os.add_dll_directory(str(runtime_dir))
        _windll(str(dll_path))
    except OSError as exc:
        return CheckResult(
            check_id="runtime.hikrobot_load",
            title="Hikrobot SDK load",
            status=Status.FAIL,
            detail=(
                f"MvCameraControl.dll failed to load: {exc}. The application "
                "falls back to a mock SDK that answers MV_OK to every call, so "
                "this would surface in production as 'no camera found' rather "
                "than as a load error."
            ),
            requirement="MvCameraControl.dll loads",
            measured="load failed",
            source="MvImport/MvCameraControl_class.py:58-78",
            remedy=(
                "Install the Visual C++ 2013 x64 redistributable and confirm "
                "every dependent DLL is present in Runtime/."
            ),
        )
    finally:
        if handle is not None:
            handle.close()

    return CheckResult(
        check_id="runtime.hikrobot_load",
        title="Hikrobot SDK load",
        status=Status.PASS,
        detail=f"MvCameraControl.dll loaded from {runtime_dir}.",
        requirement="MvCameraControl.dll loads",
        measured="loaded",
        source="MvImport/MvCameraControl_class.py:66",
    )


# --------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------


def _model_results(context: AppContext) -> list[CheckResult]:
    """Confirm the configured model weights exist and are non-trivial."""
    targets = discover_model_targets(context)
    if not targets:
        return [
            CheckResult(
                check_id="runtime.models",
                title="Model bundles",
                status=Status.FAIL,
                detail=(
                    f"No usable YOLO model bundle found under "
                    f"{context.models_dir}. A bundle qualifies when its "
                    "config.yaml names weights that exist on disk. Models are "
                    "external data and are not bundled into the executable, so "
                    "they must be copied to the target machine separately."
                ),
                requirement="At least one models/<product>/<area>/yolo bundle",
                measured="none found",
                source="core/config.py weights, models/",
                remedy="Copy the models/ directory alongside the executable.",
                data={"models_dir": str(context.models_dir)},
            )
        ]

    total_bytes = 0
    described: list[dict[str, object]] = []
    for target in targets:
        try:
            size = Path(target.weights).stat().st_size
        except OSError:
            size = 0
        total_bytes += size
        described.append(
            {
                "product": target.product,
                "area": target.area,
                "weights": target.weights,
                "size_mb": round(size / BYTES_PER_MB, 1),
                "runtime": Path(target.weights).suffix.lstrip(".").lower(),
                "device": target.device,
                "timeout_s": target.timeout_s,
            }
        )

    first = targets[0]
    return [
        CheckResult(
            check_id="runtime.models",
            title="Model bundles",
            status=Status.PASS,
            detail=(
                f"{len(targets)} bundle(s) ready; the benchmark will exercise "
                f"{first.product}/{first.area} "
                f"({Path(first.weights).name}, device={first.device}, "
                f"timeout={first.timeout_s:g}s)."
            ),
            requirement="At least one models/<product>/<area>/yolo bundle",
            measured=f"{len(targets)} bundle(s), {total_bytes / BYTES_PER_MB:.0f} MB",
            source="models/<product>/<area>/yolo/config.yaml",
            confidence=Confidence.CONFIRMED,
            data={"bundles": described},
        )
    ]
