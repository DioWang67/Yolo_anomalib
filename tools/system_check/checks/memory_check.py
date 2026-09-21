"""Memory checks.

No file in the repository states a minimum total RAM, so this check does not
invent one. What it reports is: the measured installed and available memory,
and the memory the application is *configured* to hold — the async pipeline's
``image_queue_max_mb`` ceiling and the ``max_cache_size`` model LRU.

The only memory verdict this tool is willing to fail on comes from measurement,
not from a guess: if the benchmark's own peak working set does not fit in
available RAM, that is a fact about this machine. That comparison is made in
:mod:`tools.system_check.report.reporter` once the benchmark has run.
"""

from __future__ import annotations

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.spec import IMAGE_QUEUE_MAX_MB, MODEL_CACHE_SIZE
from tools.system_check.sysinfo import BYTES_PER_GB, BYTES_PER_MB, memory_info


def check(context: AppContext) -> list[CheckResult]:
    """Report installed, available and configured-in-use memory."""
    info = memory_info()

    if info.total is None:
        return [
            CheckResult(
                check_id="memory.total",
                title="Installed memory",
                status=Status.UNKNOWN,
                detail="Physical memory size could not be read from this machine.",
                requirement="Minimum total RAM is not stated in the repository",
                measured="unknown",
                confidence=Confidence.UNKNOWN,
            )
        ]

    total_gb = info.total / BYTES_PER_GB
    available_gb = (info.available or 0) / BYTES_PER_GB
    configured_mb = _configured_footprint_mb(context)

    results = [
        CheckResult(
            check_id="memory.total",
            title="Installed memory",
            status=Status.PASS,
            detail=(
                f"{total_gb:.1f} GB installed, {available_gb:.1f} GB available "
                f"({info.percent_used:.0f}% in use)."
                if info.percent_used is not None
                else f"{total_gb:.1f} GB installed."
            ),
            requirement="Minimum total RAM is not stated in the repository",
            measured=f"{total_gb:.1f} GB",
            confidence=Confidence.UNKNOWN,
            data={
                "total_bytes": info.total,
                "available_bytes": info.available,
                "percent_used": info.percent_used,
            },
        ),
        _available_result(info, configured_mb),
    ]
    return results


def _configured_footprint_mb(context: AppContext) -> int:
    """Return the memory the pipeline is configured to hold, in MB.

    This is the in-flight image ceiling only. It excludes interpreter, torch
    and ONNX Runtime working sets, which the benchmark measures directly.
    """
    raw = context.config.get("image_queue_max_mb", IMAGE_QUEUE_MAX_MB)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return IMAGE_QUEUE_MAX_MB


def _available_result(info, configured_mb: int) -> CheckResult:
    """Judge available memory against the configured in-flight image ceiling."""
    available = info.available
    cache_size = MODEL_CACHE_SIZE
    requirement = (
        f"{configured_mb} MB of in-flight image buffers "
        f"(image_queue_max_mb) plus {cache_size} resident models, "
        "plus interpreter and runtime working set"
    )
    if available is None:
        return CheckResult(
            check_id="memory.available",
            title="Available memory",
            status=Status.UNKNOWN,
            detail="Available memory could not be read.",
            requirement=requirement,
            measured="unknown",
            confidence=Confidence.SUGGESTED,
            source="core/config.py:197",
        )

    available_mb = available / BYTES_PER_MB
    if available_mb < configured_mb:
        status = Status.FAIL
        detail = (
            f"Only {available_mb:.0f} MB is available, which is less than the "
            f"{configured_mb} MB of image buffers the pipeline is configured to "
            "hold before it even loads a model. This is measured against the "
            "station's own config, not an assumed minimum."
        )
        remedy = (
            "Close other applications, or lower image_queue_max_mb in "
            "config.yaml and re-verify throughput."
        )
    elif available_mb < configured_mb * 4:
        status = Status.WARNING
        detail = (
            f"{available_mb:.0f} MB available leaves little headroom above the "
            f"{configured_mb} MB image-buffer ceiling once the interpreter, "
            "ONNX Runtime and the model cache are resident. Compare the peak "
            "working set measured by the benchmark below."
        )
        remedy = "Check Peak RAM in the benchmark section against this figure."
    else:
        status = Status.PASS
        detail = (
            f"{available_mb / 1024:.1f} GB available against a "
            f"{configured_mb} MB in-flight image ceiling."
        )
        remedy = None

    return CheckResult(
        check_id="memory.available",
        title="Available memory",
        status=status,
        detail=detail,
        requirement=requirement,
        measured=f"{available_mb / 1024:.1f} GB",
        confidence=Confidence.SUGGESTED,
        source="core/config.py:197-198",
        remedy=remedy,
        data={"available_bytes": available, "configured_queue_mb": configured_mb},
    )
