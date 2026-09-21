"""Processor checks.

The repository states no minimum core count or instruction set, so neither is
reported as a hard requirement. What it does state is the concurrency the
pipeline creates: ``core/async_pipeline.py`` starts acquisition, inference and
storage workers that run alongside the Qt GUI thread. That is the basis for the
advisory core-count floor, and it is labelled SUGGESTED everywhere it appears.
"""

from __future__ import annotations

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.spec import PIPELINE_THREADS, SUGGESTED_LOGICAL_CORES
from tools.system_check.sysinfo import CpuInfo, cpu_info

#: Instruction sets that change ONNX Runtime and torch kernel selection.
_PERFORMANCE_FLAGS = ("avx2", "avx512f", "fma", "sse4_2")


def check(context: AppContext) -> list[CheckResult]:
    """Report the processor and judge core count against the advisory floor."""
    info = cpu_info()
    payload = _payload(info)

    results = [
        CheckResult(
            check_id="cpu.model",
            title="Processor",
            status=Status.PASS if info.model else Status.UNKNOWN,
            detail=_model_detail(info),
            measured=info.model or "unknown",
            requirement="No specific processor is required",
            confidence=Confidence.UNKNOWN,
            data=payload,
        ),
        _core_result(info),
        _instruction_result(info),
    ]
    return results


def _model_detail(info: CpuInfo) -> str:
    """Compose the processor description line."""
    if not info.model:
        return "Processor model could not be determined."
    clock = f" @ {info.max_clock_mhz / 1000:.2f} GHz" if info.max_clock_mhz else ""
    return f"{info.model}{clock} ({info.architecture})."


def _core_result(info: CpuInfo) -> CheckResult:
    """Compare logical cores against the pipeline's thread count."""
    logical = info.logical_cores
    physical = info.physical_cores
    measured = (
        f"{physical if physical is not None else '?'} physical / "
        f"{logical if logical is not None else '?'} logical cores"
    )
    requirement = (
        f"{SUGGESTED_LOGICAL_CORES} logical cores (SUGGESTED: "
        f"{PIPELINE_THREADS} pipeline workers + GUI thread)"
    )

    if logical is None:
        return CheckResult(
            check_id="cpu.cores",
            title="CPU cores",
            status=Status.UNKNOWN,
            detail="Core count could not be determined.",
            requirement=requirement,
            measured=measured,
            confidence=Confidence.SUGGESTED,
            source="core/async_pipeline.py:175-197",
        )

    if logical >= SUGGESTED_LOGICAL_CORES:
        status = Status.PASS
        detail = (
            f"{logical} logical cores cover the {PIPELINE_THREADS} pipeline "
            "worker threads and the GUI thread without oversubscription."
        )
        remedy = None
    else:
        status = Status.WARNING
        detail = (
            f"{logical} logical cores is below the suggested "
            f"{SUGGESTED_LOGICAL_CORES}. The acquisition, inference and storage "
            "workers will contend with the GUI thread, which shows up as "
            "latency jitter rather than as an error. This is an advisory floor "
            "derived from the pipeline's thread count, not a stated minimum."
        )
        remedy = (
            "Confirm measured P99 latency still fits the configured inference "
            "timeout before accepting this machine."
        )

    return CheckResult(
        check_id="cpu.cores",
        title="CPU cores",
        status=status,
        detail=detail,
        requirement=requirement,
        measured=measured,
        confidence=Confidence.SUGGESTED,
        source="core/async_pipeline.py:175-197",
        remedy=remedy,
        data={"physical_cores": physical, "logical_cores": logical},
    )


def _instruction_result(info: CpuInfo) -> CheckResult:
    """Report instruction-set support without asserting a requirement."""
    if not info.flags:
        return CheckResult(
            check_id="cpu.instruction_set",
            title="CPU instruction set",
            status=Status.UNKNOWN,
            detail=(
                "CPU feature flags are unavailable in this build (py-cpuinfo is "
                "not bundled with the standalone checker). ONNX Runtime selects "
                "AVX2/AVX-512 kernels at runtime, so this affects speed, not "
                "correctness. The benchmark below measures the real effect."
            ),
            requirement="No instruction set is required by the repository",
            measured="unknown",
            confidence=Confidence.UNKNOWN,
        )

    present = [flag for flag in _PERFORMANCE_FLAGS if flag in info.flags]
    missing = [flag for flag in _PERFORMANCE_FLAGS if flag not in info.flags]
    status = Status.PASS if "avx2" in info.flags else Status.WARNING
    detail = (
        f"Accelerated kernels available: {', '.join(present) or 'none'}."
        if status is Status.PASS
        else (
            "AVX2 is not advertised by this CPU. ONNX Runtime falls back to "
            "slower kernels; this is a performance finding, not a blocker."
        )
    )
    return CheckResult(
        check_id="cpu.instruction_set",
        title="CPU instruction set",
        status=status,
        detail=detail,
        requirement="No instruction set is required by the repository",
        measured=", ".join(present) or "none of avx2/avx512f/fma/sse4_2",
        confidence=Confidence.UNKNOWN,
        data={"present": present, "missing": missing},
    )


def _payload(info: CpuInfo) -> dict[str, object]:
    """Structured CPU facts for the JSON report and baseline comparison."""
    return {
        "model": info.model,
        "physical_cores": info.physical_cores,
        "logical_cores": info.logical_cores,
        "max_clock_mhz": info.max_clock_mhz,
        "architecture": info.architecture,
        "accelerated_flags": [flag for flag in _PERFORMANCE_FLAGS if flag in info.flags],
    }
