"""Graphics checks.

This project does not require a GPU, and the report says so rather than
implying a missing card is a defect. ``requirements.txt`` pins the CPU torch
wheel and the CPU ``onnxruntime`` build, and the active model bundle pins
``device: cpu``.

The check still gathers GPU facts, because a station that *does* have a card
changes the baseline comparison and because the code in ``core/yolo_runtime.py``
would use CUDA if the wheels ever changed. Crucially it asks the runtime, not
just ``nvidia-smi``: a driver can be present while ``torch.cuda.is_available()``
is false, and only the second answer decides what the application does.

CUDA detection and GPU inventory live together here rather than in a separate
``cuda_check`` module, because with CPU-only wheels a standalone CUDA module
would have nothing to assert.
"""

from __future__ import annotations

import importlib
from typing import Any

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.sysinfo import run_command

_SMI_QUERY = "name,driver_version,memory.total,memory.used"


def check(context: AppContext) -> list[CheckResult]:
    """Report GPU inventory, driver, CUDA availability and ORT providers."""
    smi = _nvidia_smi()
    torch_facts = _torch_cuda()
    providers = _ort_providers()

    return [
        _inventory_result(smi),
        _cuda_result(torch_facts, smi),
        _provider_result(providers),
    ]


def _nvidia_smi() -> dict[str, Any]:
    """Query ``nvidia-smi``, tolerating its absence."""
    code, stdout, stderr = run_command(
        ["nvidia-smi", f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"],
        timeout=15.0,
    )
    if code != 0:
        return {"available": False, "reason": stderr.strip() or stdout.strip() or "no output"}

    gpus = []
    for line in stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpus.append(
            {
                "name": parts[0],
                "driver_version": parts[1],
                "vram_total_mb": _as_int(parts[2]),
                "vram_used_mb": _as_int(parts[3]),
            }
        )
    return {"available": bool(gpus), "gpus": gpus, "reason": "" if gpus else "no GPUs listed"}


def _as_int(value: str) -> int | None:
    """Parse an ``nvidia-smi`` numeric field."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _torch_cuda() -> dict[str, Any]:
    """Ask torch itself whether CUDA is usable.

    ``nvidia-smi`` reporting a healthy card says nothing about whether the
    installed torch wheel was built with CUDA. Importing torch can be slow and
    can raise on a broken install, so both are handled.
    """
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}

    facts: dict[str, Any] = {
        "importable": True,
        "version": getattr(torch, "__version__", "unknown"),
        "built_with_cuda": getattr(getattr(torch, "version", None), "cuda", None),
    }
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        facts["is_available"] = False
        facts["error"] = f"torch.cuda.is_available() raised {type(exc).__name__}: {exc}"
        return facts

    facts["is_available"] = available
    if not available:
        return facts
    try:
        facts["device_count"] = int(torch.cuda.device_count())
        facts["device_name"] = str(torch.cuda.get_device_name(0))
        total, _ = torch.cuda.mem_get_info()
        facts["vram_free_bytes"] = int(total)
    except Exception as exc:  # noqa: BLE001
        facts["error"] = f"CUDA device query failed: {type(exc).__name__}: {exc}"
    return facts


def _ort_providers() -> dict[str, Any]:
    """List the ONNX Runtime execution providers actually registered."""
    try:
        ort = importlib.import_module("onnxruntime")
        return {
            "importable": True,
            "version": getattr(ort, "__version__", "unknown"),
            "providers": list(ort.get_available_providers()),
        }
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}


def _inventory_result(smi: dict[str, Any]) -> CheckResult:
    """Report discovered GPUs. Absence is never a failure for this project."""
    if not smi.get("available"):
        return CheckResult(
            check_id="gpu.present",
            title="GPU",
            status=Status.PASS,
            detail=(
                "No NVIDIA GPU detected. This project does not need one: the "
                "pinned torch wheel and onnxruntime build are CPU-only and the "
                f"model bundle pins device: cpu. ({smi.get('reason')})"
            ),
            requirement="No GPU required",
            measured="none detected",
            source="requirements.txt:158,279, models/Cable1/A/yolo/config.yaml:2",
            data=dict(smi),
        )

    gpus = smi.get("gpus") or []
    names = ", ".join(str(gpu.get("name")) for gpu in gpus)
    first = gpus[0] if gpus else {}
    vram = first.get("vram_total_mb")
    return CheckResult(
        check_id="gpu.present",
        title="GPU",
        status=Status.PASS,
        detail=(
            f"{names} (driver {first.get('driver_version')}"
            f"{f', {vram} MB VRAM' if vram else ''}). Informational: inference "
            "runs on CPU with the pinned wheels."
        ),
        requirement="No GPU required",
        measured=names,
        source="requirements.txt:158,279",
        data=dict(smi),
    )


def _cuda_result(torch_facts: dict[str, Any], smi: dict[str, Any]) -> CheckResult:
    """Report what torch says about CUDA, contrasted with the driver."""
    if not torch_facts.get("importable"):
        return CheckResult(
            check_id="gpu.cuda",
            title="CUDA availability",
            status=Status.SKIP,
            detail=(
                "torch is not importable in this process, so CUDA availability "
                "was not queried. The standalone checker does not bundle torch; "
                "run the checker from the application environment if you need "
                f"this answer. ({torch_facts.get('error')})"
            ),
            requirement="No CUDA required",
            measured="not queried",
            confidence=Confidence.CONFIRMED,
            data=dict(torch_facts),
        )

    built = torch_facts.get("built_with_cuda")
    available = bool(torch_facts.get("is_available"))
    driver_present = bool(smi.get("available"))

    if available:
        detail = (
            f"torch {torch_facts.get('version')} reports CUDA "
            f"{built} available on {torch_facts.get('device_name')}. The model "
            "bundle still pins device: cpu, so this capacity is unused."
        )
    elif built is None and driver_present:
        detail = (
            f"A GPU and driver are present, but torch {torch_facts.get('version')} "
            "is the CPU-only build, so torch.cuda.is_available() is False. This "
            "matches the pinned requirements and is the expected state."
        )
    elif built is None:
        detail = (
            f"torch {torch_facts.get('version')} is the CPU-only build "
            "(torch.version.cuda is None), matching requirements.txt."
        )
    else:
        detail = (
            f"torch was built against CUDA {built} but "
            "torch.cuda.is_available() is False: driver or runtime mismatch. "
            "Harmless while the model bundle pins device: cpu."
        )

    return CheckResult(
        check_id="gpu.cuda",
        title="CUDA availability",
        status=Status.PASS,
        detail=detail,
        requirement="No CUDA required",
        measured=f"torch.cuda.is_available()={available}, torch.version.cuda={built}",
        source="requirements.txt:279, core/yolo_runtime.py:144-147",
        data=dict(torch_facts),
    )


def _provider_result(providers: dict[str, Any]) -> CheckResult:
    """Report whether ONNX Runtime offers any GPU execution provider.

    Whether ``CPUExecutionProvider`` is present — the gate the application
    enforces — is checked in :mod:`tools.system_check.checks.runtime_check`.
    This result answers only the GPU half of the question, so a missing ONNX
    Runtime is reported once, as a runtime failure, not twice.
    """
    if not providers.get("importable"):
        return CheckResult(
            check_id="gpu.ort_providers",
            title="GPU execution providers",
            status=Status.SKIP,
            detail=(
                "ONNX Runtime could not be imported, so its providers were not "
                "listed. See the Native runtime section for that failure."
            ),
            requirement="GPU providers optional",
            measured="not queried",
            data=dict(providers),
        )

    listed = [str(name) for name in providers.get("providers") or []]
    gpu_providers = [name for name in listed if "CUDA" in name or "Tensorrt" in name]
    return CheckResult(
        check_id="gpu.ort_providers",
        title="GPU execution providers",
        status=Status.PASS,
        detail=(
            f"GPU providers registered: {', '.join(gpu_providers)}."
            if gpu_providers
            else (
                "No GPU execution provider registered, as expected for the "
                f"pinned CPU onnxruntime {providers.get('version')} build. "
                f"Registered: {', '.join(listed)}."
            )
        ),
        requirement="GPU providers optional",
        measured=", ".join(gpu_providers) or "none",
        source="requirements.txt:158",
        data=dict(providers),
    )
