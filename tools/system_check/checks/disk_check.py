"""Storage checks.

``min_free_disk_mb`` (default 1024) is a real, stated requirement: below it the
inference pipeline stops writing inspection evidence. That makes it the one
storage threshold this tool is willing to FAIL on.

The volume that matters is the one holding ``output_dir``, which is not
necessarily the volume holding the application — the workspace deliberately
puts inspection results in a separate ``Result`` tree.
"""

from __future__ import annotations

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.spec import MIN_FREE_DISK_MB
from tools.system_check.sysinfo import BYTES_PER_GB, BYTES_PER_MB, disk_info


def check(context: AppContext) -> list[CheckResult]:
    """Report free space on the result volume and the application volume."""
    threshold_mb = _configured_threshold_mb(context)
    results = [_volume_result(context.result_dir, threshold_mb, "result")]

    result_volume = disk_info(context.result_dir).volume
    app_volume = disk_info(context.app_root).volume
    if app_volume.lower() != result_volume.lower():
        results.append(_volume_result(context.app_root, threshold_mb, "application"))
    return results


def _configured_threshold_mb(context: AppContext) -> int:
    """Return this station's ``min_free_disk_mb``, falling back to the default."""
    raw = context.config.get("min_free_disk_mb", MIN_FREE_DISK_MB)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MIN_FREE_DISK_MB
    return value if value >= 0 else MIN_FREE_DISK_MB


def _volume_result(path, threshold_mb: int, role: str) -> CheckResult:
    """Judge one volume's free space against ``min_free_disk_mb``."""
    info = disk_info(path)
    check_id = f"disk.{role}"
    title = f"Free space ({role} volume)"
    requirement = f"{threshold_mb} MB free on {info.volume}"

    if info.free is None or info.total is None:
        return CheckResult(
            check_id=check_id,
            title=title,
            status=Status.UNKNOWN,
            detail=(
                f"Free space on {info.volume} could not be read "
                f"(path {info.path}). A missing or unmapped drive reads the "
                "same way as a permission failure here."
            ),
            requirement=requirement,
            measured="unknown",
            source="core/config.py:199",
            remedy=f"Confirm {info.volume} exists and is reachable by this account.",
            data={"path": info.path, "volume": info.volume},
        )

    free_mb = info.free / BYTES_PER_MB
    free_gb = info.free / BYTES_PER_GB
    total_gb = info.total / BYTES_PER_GB
    payload = {
        "path": info.path,
        "volume": info.volume,
        "total_bytes": info.total,
        "free_bytes": info.free,
        "threshold_mb": threshold_mb,
    }

    if free_mb < threshold_mb:
        return CheckResult(
            check_id=check_id,
            title=title,
            status=Status.FAIL,
            detail=(
                f"{free_gb:.1f} GB free on {info.volume} is below the "
                f"{threshold_mb} MB the pipeline requires before it will write "
                "inspection evidence."
            ),
            requirement=requirement,
            measured=f"{free_gb:.1f} GB free of {total_gb:.0f} GB",
            source="core/config.py:199, core/config_schema.py:157",
            remedy=(
                "Free space on this volume, or point output_dir at a volume "
                "with capacity. Inspection images are the bulk of the growth."
            ),
            data=payload,
        )

    if free_mb < threshold_mb * 5:
        return CheckResult(
            check_id=check_id,
            title=title,
            status=Status.WARNING,
            detail=(
                f"{free_gb:.1f} GB free on {info.volume} clears the "
                f"{threshold_mb} MB floor but leaves under five times the "
                "margin. Inspection images accumulate per shift."
            ),
            requirement=requirement,
            measured=f"{free_gb:.1f} GB free of {total_gb:.0f} GB",
            confidence=Confidence.SUGGESTED,
            source="core/config.py:199",
            remedy=(
                "Enable inspection_retention_cleanup_enabled after a dry run, "
                "or plan capacity for this station's shift volume."
            ),
            data=payload,
        )

    return CheckResult(
        check_id=check_id,
        title=title,
        status=Status.PASS,
        detail=f"{free_gb:.1f} GB free of {total_gb:.0f} GB on {info.volume}.",
        requirement=requirement,
        measured=f"{free_gb:.1f} GB free of {total_gb:.0f} GB",
        source="core/config.py:199",
        data=payload,
    )
