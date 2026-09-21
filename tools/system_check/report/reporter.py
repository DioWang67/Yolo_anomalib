"""Build the JSON payload and render the human-readable report.

Two audiences, one run. The console and ``system_report.txt`` are written for
whoever is standing at the station: every line says what was checked, what was
found, and what to do about it. ``system_report.json`` is written for tooling
and for ``--baseline`` comparisons on the next machine.

The report deliberately carries no host name, user name, serial number or
network address beyond the sync host the config already names. A preflight
report is normally e-mailed between sites, and it should not be the thing that
carries identifying detail out of the plant.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.system_check import REPORT_SCHEMA_VERSION
from tools.system_check.benchmark.cpu_benchmark import CpuBenchmark
from tools.system_check.benchmark.project_benchmark import BenchmarkResult
from tools.system_check.context import AppContext
from tools.system_check.report.comparison import Comparison
from tools.system_check.results import CheckResult, Status, overall_label, overall_status
from tools.system_check.spec import REQUIREMENTS
from tools.system_check.sysinfo import (
    BYTES_PER_GB,
    cpu_info,
    disk_info,
    memory_info,
    os_info,
)

_WIDTH = 78
_STATUS_ORDER = (Status.FAIL, Status.WARNING, Status.UNKNOWN, Status.PASS, Status.SKIP)


# --------------------------------------------------------------------------
# JSON payload
# --------------------------------------------------------------------------


def build_system_summary(
    context: AppContext, results: Sequence[CheckResult]
) -> dict[str, Any]:
    """Collect the machine facts the JSON report and comparisons rely on."""
    operating_system = os_info()
    processor = cpu_info()
    memory = memory_info()
    result_disk = disk_info(context.result_dir)

    gpu_payload = _result_data(results, "gpu.present")
    gpus = gpu_payload.get("gpus") or []
    first_gpu = gpus[0] if gpus else {}

    return {
        "os": {
            "system": operating_system.system,
            "release": operating_system.release,
            "build": operating_system.build,
            "edition": operating_system.edition
            or f"{operating_system.system} {operating_system.release}",
            "machine": operating_system.machine,
            "is_64bit": operating_system.is_64bit,
        },
        "cpu": {
            "model": processor.model,
            "physical_cores": processor.physical_cores,
            "logical_cores": processor.logical_cores,
            "max_clock_mhz": processor.max_clock_mhz,
            "architecture": processor.architecture,
        },
        "memory": {
            "total_gb": round(memory.total / BYTES_PER_GB, 2) if memory.total else None,
            "available_gb": round(memory.available / BYTES_PER_GB, 2)
            if memory.available
            else None,
            "percent_used": memory.percent_used,
        },
        "disk": {
            "path": result_disk.path,
            "volume": result_disk.volume,
            "total_gb": round(result_disk.total / BYTES_PER_GB, 2)
            if result_disk.total
            else None,
            "free_gb": round(result_disk.free / BYTES_PER_GB, 2)
            if result_disk.free
            else None,
        },
        "gpu": {
            "name": first_gpu.get("name"),
            "driver_version": first_gpu.get("driver_version"),
            "vram_total_mb": first_gpu.get("vram_total_mb"),
            "count": len(gpus),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "frozen": context.frozen,
        },
        "installation": {
            "app_root": str(context.app_root),
            "data_root": str(context.data_root),
            "config_path": str(context.config_path) if context.config_path else None,
            "result_dir": str(context.result_dir),
        },
    }


def _result_data(results: Sequence[CheckResult], check_id: str) -> dict[str, Any]:
    """Return one check's structured payload, or ``{}``."""
    for result in results:
        if result.check_id == check_id:
            return dict(result.data)
    return {}


def build_report(
    context: AppContext,
    results: Sequence[CheckResult],
    benchmark: BenchmarkResult | None,
    cpu_benchmark: CpuBenchmark | None,
    comparison: Comparison | None,
    *,
    tool_version: str,
) -> dict[str, Any]:
    """Assemble the machine-readable report."""
    status = overall_status(results)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "tool_version": tool_version,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "overall_status": status.value,
        "overall_label": overall_label(status),
        "summary": {
            state.value: sum(1 for result in results if result.status is state)
            for state in _STATUS_ORDER
        },
        "system": build_system_summary(context, results),
        "checks": [result.to_dict() for result in results],
        "benchmark": benchmark.to_dict() if benchmark else None,
        "cpu_benchmark": cpu_benchmark.to_dict() if cpu_benchmark else None,
        "comparison": comparison.to_dict() if comparison else None,
        "requirements": [
            {
                "key": requirement.key,
                "category": requirement.category,
                "statement": requirement.statement,
                "confidence": requirement.confidence.value,
                "source": requirement.source,
                "notes": requirement.notes,
            }
            for requirement in REQUIREMENTS
        ],
    }


def write_json(report: dict[str, Any], path: Path) -> None:
    """Write the JSON report, creating the directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------


def render_text(
    report: dict[str, Any],
    results: Sequence[CheckResult],
    benchmark: BenchmarkResult | None,
    cpu_benchmark: CpuBenchmark | None,
    comparison: Comparison | None,
    *,
    verbose: bool = False,
) -> str:
    """Render the operator-facing report."""
    lines: list[str] = []
    lines.extend(_header(report))
    lines.extend(_system_block(report))
    lines.extend(_checks_block(results, verbose=verbose))
    lines.extend(_benchmark_block(benchmark, cpu_benchmark))
    if comparison is not None:
        lines.extend(_comparison_block(comparison))
    lines.extend(_unknowns_block())
    lines.extend(_verdict_block(report, results))
    return "\n".join(lines) + "\n"


def _rule(char: str = "-") -> str:
    return char * _WIDTH


def _header(report: dict[str, Any]) -> list[str]:
    return [
        _rule("="),
        "  yolo11_inference - Deployment Preflight Report",
        _rule("="),
        f"  Recorded         : {report['recorded_at']}",
        f"  Checker version  : {report['tool_version']}",
        f"  Installation     : {report['system']['installation']['app_root']}",
        "",
    ]


def _system_block(report: dict[str, Any]) -> list[str]:
    system = report["system"]
    cpu = system["cpu"]
    memory = system["memory"]
    disk = system["disk"]
    gpu = system["gpu"]

    clock = f" @ {cpu['max_clock_mhz'] / 1000:.2f} GHz" if cpu.get("max_clock_mhz") else ""
    cores = (
        f"{cpu.get('physical_cores') or '?'} physical / "
        f"{cpu.get('logical_cores') or '?'} logical"
    )
    return [
        "THIS MACHINE",
        _rule(),
        f"  OS      : {system['os']['edition']} (build {system['os']['build'] or '?'})",
        f"  CPU     : {cpu.get('model') or 'unknown'}{clock}",
        f"            {cores} cores",
        f"  RAM     : {_gb(memory.get('total_gb'))} total, "
        f"{_gb(memory.get('available_gb'))} available",
        f"  Disk    : {_gb(disk.get('free_gb'))} free of {_gb(disk.get('total_gb'))} "
        f"on {disk.get('volume')}",
        f"  GPU     : {gpu.get('name') or 'none detected'}"
        + (f" ({gpu['vram_total_mb']} MB VRAM)" if gpu.get("vram_total_mb") else ""),
        f"  Python  : {system['python']['version']}"
        + (" (packaged build)" if system["python"].get("frozen") else ""),
        "",
    ]


def _gb(value: float | None) -> str:
    """Format a gigabyte figure, or a dash when undetermined."""
    return f"{value:.1f} GB" if isinstance(value, (int, float)) else "-"


def _checks_block(results: Sequence[CheckResult], *, verbose: bool) -> list[str]:
    lines = ["COMPATIBILITY CHECKS", _rule()]
    for result in results:
        lines.append(f"  [{result.status.value:<7}] {result.title}: {result.detail}")
        if verbose and result.requirement:
            lines.append(f"            requirement: {result.requirement}")
            if result.source:
                lines.append(f"            source     : {result.source}")
        if result.remedy and result.status in (Status.FAIL, Status.WARNING, Status.UNKNOWN):
            lines.append(f"            -> {result.remedy}")
    lines.append("")
    return lines


def _benchmark_block(
    benchmark: BenchmarkResult | None, cpu_benchmark: CpuBenchmark | None
) -> list[str]:
    lines = ["MEASURED PERFORMANCE", _rule()]
    if benchmark is None:
        lines.append("  Benchmark did not run. See the checks above for the reason.")
        lines.append("")
        return lines

    lines.extend(
        [
            f"  Model        : {benchmark.product}/{benchmark.area} "
            f"({Path(benchmark.weights).name})",
            f"  Backend      : {benchmark.backend}, device={benchmark.device}, "
            f"imgsz={benchmark.imgsz[0]}x{benchmark.imgsz[1]}",
            f"  Input frame  : {benchmark.source_frame} "
            f"({benchmark.frame_shape[1]}x{benchmark.frame_shape[0]})",
            f"  Runs         : {benchmark.warmup_runs} warm-up (discarded) + "
            f"{benchmark.timed_runs} timed",
            "",
            f"  Model load   : {benchmark.model_load_ms:>9.1f} ms",
            f"  Warm-up      : {benchmark.warmup_ms:>9.1f} ms total",
            "",
            f"  {'Stage':<14}{'mean':>9}{'median':>9}{'P95':>9}{'P99':>9}"
            f"{'min':>9}{'max':>9}",
        ]
    )
    for label, stats in (
        ("Preprocess", benchmark.preprocess),
        ("Inference", benchmark.inference),
        ("Postprocess", benchmark.postprocess),
        ("Total", benchmark.total),
    ):
        lines.append(
            f"  {label:<14}{stats.mean_ms:>9.2f}{stats.median_ms:>9.2f}"
            f"{stats.p95_ms:>9.2f}{stats.p99_ms:>9.2f}"
            f"{stats.min_ms:>9.2f}{stats.max_ms:>9.2f}"
        )

    resources = benchmark.resources
    expected = benchmark.expected_item_count
    detections = (
        f"{benchmark.detections_per_frame:.1f}"
        + (f" (product expects {expected})" if expected else "")
    )
    lines.extend(
        [
            "",
            f"  Detections   : {detections} per frame - drives postprocess cost",
            f"  Timeout      : {benchmark.timeout_s:g} s per inference (enforced; "
            "checked above)",
            f"  Throughput   : {benchmark.throughput_fps:.2f} inferences/s "
            "(measured only - no cycle-time requirement exists)",
            f"  Peak RAM     : {_mb(resources.get('peak_rss_mb'))}",
            f"  Mean CPU     : {_pct(resources.get('mean_cpu_percent'))}"
            f"   Peak CPU: {_pct(resources.get('peak_cpu_percent'))}",
            f"  Peak GPU     : {_pct(resources.get('peak_gpu_percent'))}"
            f"   Peak VRAM: {_mb(resources.get('peak_vram_mb'))}",
        ]
    )
    if cpu_benchmark is not None:
        lines.append(
            f"  CPU probe    : {cpu_benchmark.matmul_gflops:.2f} GFLOPS matmul, "
            f"{cpu_benchmark.scalar_mops:.1f} Mops/s scalar"
        )
    for note in benchmark.notes:
        lines.append(f"  Note: {note}")
    lines.append("")
    return lines


def _mb(value: Any) -> str:
    return f"{value:.0f} MB" if isinstance(value, (int, float)) else "not measured"


def _pct(value: Any) -> str:
    return f"{value:.0f}%" if isinstance(value, (int, float)) else "n/a"


def _comparison_block(comparison: Comparison) -> list[str]:
    lines = [
        "COMPARED WITH BASELINE",
        _rule(),
        f"  Baseline file    : {comparison.baseline_path}",
        f"  Baseline recorded: {comparison.baseline_recorded_at or 'unknown'}",
        "",
        "  A slower machine is not automatically unacceptable. Whether this one",
        "  passes is decided by the checks above against the configured",
        "  inference timeout, not by the differences in this table.",
        "",
        f"  {'':<22}{'Baseline':>14}{'This machine':>16}{'Difference':>16}",
    ]

    for field in comparison.hardware:
        lines.append(
            f"  {field.label:<22}{_text(field.baseline):>14}"
            f"{_text(field.target):>16}{'changed' if field.changed else 'same':>16}"
        )
    for metric in comparison.hardware_metrics:
        lines.append(_metric_row(metric))

    if any(metric.target is not None for metric in comparison.benchmark):
        lines.append("")
        for metric in comparison.benchmark:
            if metric.baseline is None and metric.target is None:
                continue
            lines.append(_metric_row(metric))

    if comparison.warnings:
        lines.append("")
        for warning in comparison.warnings:
            lines.append(f"  [WARNING] {warning}")
    lines.append("")
    return lines


def _text(value: str | None) -> str:
    """Trim a text field to the comparison column width."""
    if not value:
        return "-"
    return value if len(value) <= 14 else value[:13] + "~"


#: Wording for a change, keyed by unit, as ``(went_down, went_up)``. Chosen by
#: the *sign* of the difference, never by the verdict: "+26%" must never read
#: "lower". Whether the change is good or bad is what ``verdict`` carries in
#: the JSON, and for latency the wording happens to say both — a longer time
#: is "slower" and is also worse. A quantity reads higher/lower, because
#: describing a memory figure as "slower" is how a reader starts distrusting
#: the whole table.
_DIRECTION_WORDS: dict[str, tuple[str, str]] = {
    "ms": ("faster", "slower"),
    "s": ("faster", "slower"),
}
_DEFAULT_DIRECTION_WORDS = ("lower", "higher")


def _value_cell(value: float | None, unit: str, width: int) -> str:
    """Right-align one value with its unit inside ``width`` characters."""
    if value is None:
        return f"{'-':>{width}}"
    return f"{f'{value:g} {unit}':>{width}}"


def _metric_row(metric) -> str:
    """Render one comparison row, with the direction spelled out.

    The verdict comes from ``metric.verdict``, which the comparison derived
    from its declared ``higher_is_better`` flag. This function only chooses
    wording; it never re-derives the direction from the sign.
    """
    baseline_cell = _value_cell(metric.baseline, metric.unit, 14)
    target_cell = _value_cell(metric.target, metric.unit, 16)

    if metric.baseline is None or metric.target is None:
        return f"  {metric.label:<22}{baseline_cell}{target_cell}{'not comparable':>16}"

    if metric.verdict == "same":
        difference = "same"
    else:
        went_down, went_up = _DIRECTION_WORDS.get(metric.unit, _DEFAULT_DIRECTION_WORDS)
        word = went_up if (metric.difference or 0) > 0 else went_down
        percent = f"{metric.percent:+.0f}%" if metric.percent is not None else "n/a"
        difference = f"{percent} {word}"

    return f"  {metric.label:<22}{baseline_cell}{target_cell}{difference:>16}"


def _unknowns_block() -> list[str]:
    """List requirements the repository does not state."""
    unknown = [item for item in REQUIREMENTS if item.confidence.value == "UNKNOWN"]
    suggested = [item for item in REQUIREMENTS if item.confidence.value == "SUGGESTED"]
    lines = ["REQUIREMENTS THIS REPORT DOES NOT ASSERT", _rule()]
    lines.append("  Not stated anywhere in the repository, so not judged here:")
    for item in unknown:
        lines.append(f"    - {item.category}: {item.statement}")
    lines.append("")
    lines.append("  Advisory thresholds derived by this tool, not contractual:")
    for item in suggested:
        lines.append(f"    - {item.category}: {item.statement}")
        if item.source:
            lines.append(f"      derived from {item.source}")
    lines.append("")
    return lines


def _verdict_block(report: dict[str, Any], results: Sequence[CheckResult]) -> list[str]:
    summary = report["summary"]
    lines = [
        _rule("="),
        f"  OVERALL COMPATIBILITY: {report['overall_label']}",
        _rule("="),
        f"  {summary.get('PASS', 0)} passed, {summary.get('WARNING', 0)} warnings, "
        f"{summary.get('FAIL', 0)} failed, {summary.get('UNKNOWN', 0)} undetermined, "
        f"{summary.get('SKIP', 0)} not applicable.",
    ]

    blocking = [result for result in results if result.status is Status.FAIL]
    if blocking:
        lines.append("")
        lines.append("  Must be fixed before this machine runs production:")
        for result in blocking:
            lines.append(f"    - {result.title}: {result.detail}")
            if result.remedy:
                lines.append(f"      -> {result.remedy}")
    lines.append(_rule("="))
    return lines
