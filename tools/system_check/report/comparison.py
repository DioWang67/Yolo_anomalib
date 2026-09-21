"""Compare this machine against a saved baseline.

Direction matters and is declared per metric, not inferred from the sign of a
subtraction. Latency going *up* is a regression; throughput going up is an
improvement. Getting that backwards would turn a slower station into a green
report, so every metric carries an explicit ``higher_is_better`` flag and the
verdict is derived from it.

A comparison never decides PASS or FAIL on its own. "Slower than the
development machine" is not a defect — whether this machine is acceptable is
decided against the operational requirement (the configured inference timeout)
in :mod:`tools.system_check.verdicts`. The comparison exists to explain *why*
a machine behaves differently, and it says so in the report.

Every accessor tolerates a baseline that is missing fields, carries nulls, or
was written by an older version of this tool.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class BaselineError(ValueError):
    """Raised when a baseline file cannot be used."""


@dataclass(frozen=True)
class MetricSpec:
    """One comparable metric.

    Args:
        key: Stable identifier for the JSON report.
        label: Report column label.
        path: Dotted path into the report payload.
        unit: Display unit.
        higher_is_better: Direction of improvement. ``False`` for latencies.
        precision: Decimal places for display.
    """

    key: str
    label: str
    path: str
    unit: str
    higher_is_better: bool
    precision: int = 1


#: Hardware facts compared as text, not arithmetic.
HARDWARE_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("os", "OS", "system.os.edition"),
    ("cpu", "CPU", "system.cpu.model"),
    ("cpu_cores", "CPU cores", "system.cpu.logical_cores"),
    ("gpu", "GPU", "system.gpu.name"),
)

#: Numeric hardware facts, where a percentage difference is meaningful.
HARDWARE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("ram_gb", "RAM", "system.memory.total_gb", "GB", True),
    MetricSpec("vram_mb", "VRAM", "system.gpu.vram_total_mb", "MB", True, precision=0),
    MetricSpec("disk_free_gb", "Free disk", "system.disk.free_gb", "GB", True),
)

#: Performance metrics. Latencies are ``higher_is_better=False``.
BENCHMARK_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("model_load_ms", "Model load", "benchmark.model_load_ms", "ms", False),
    MetricSpec("warmup_ms", "Warm-up", "benchmark.warmup_ms", "ms", False),
    MetricSpec("preprocess_mean_ms", "Preprocess (mean)", "benchmark.preprocess.mean_ms", "ms", False, 2),
    MetricSpec("inference_mean_ms", "Inference (mean)", "benchmark.inference.mean_ms", "ms", False, 2),
    MetricSpec("postprocess_mean_ms", "Postprocess (mean)", "benchmark.postprocess.mean_ms", "ms", False, 2),
    MetricSpec("total_mean_ms", "Total (mean)", "benchmark.total.mean_ms", "ms", False, 2),
    MetricSpec("total_median_ms", "Total (median)", "benchmark.total.median_ms", "ms", False, 2),
    MetricSpec("total_p95_ms", "Total (P95)", "benchmark.total.p95_ms", "ms", False, 2),
    MetricSpec("total_p99_ms", "Total (P99)", "benchmark.total.p99_ms", "ms", False, 2),
    MetricSpec("total_max_ms", "Total (max)", "benchmark.total.max_ms", "ms", False, 2),
    MetricSpec("throughput_fps", "Throughput", "benchmark.throughput_fps", "FPS", True, 2),
    MetricSpec("peak_rss_mb", "Peak RAM", "benchmark.resources.peak_rss_mb", "MB", False, 0),
    MetricSpec("peak_vram_mb", "Peak VRAM", "benchmark.resources.peak_vram_mb", "MB", False, 0),
    MetricSpec("mean_cpu_percent", "Mean CPU", "benchmark.resources.mean_cpu_percent", "%", False, 1),
    MetricSpec("cpu_matmul_gflops", "CPU matmul", "cpu_benchmark.matmul_gflops", "GFLOPS", True, 2),
)


@dataclass(frozen=True)
class MetricComparison:
    """One compared metric.

    ``verdict`` is ``better``, ``worse``, ``same`` or ``unknown`` — derived
    from ``higher_is_better``, never from the raw sign of ``difference``.
    """

    key: str
    label: str
    unit: str
    baseline: float | None
    target: float | None
    difference: float | None
    percent: float | None
    higher_is_better: bool
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "key": self.key,
            "label": self.label,
            "unit": self.unit,
            "baseline": self.baseline,
            "target": self.target,
            "difference": self.difference,
            "percent": self.percent,
            "higher_is_better": self.higher_is_better,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class FieldComparison:
    """One compared text field."""

    key: str
    label: str
    baseline: str | None
    target: str | None

    @property
    def changed(self) -> bool:
        """Whether the two sides differ."""
        return (self.baseline or "") != (self.target or "")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "key": self.key,
            "label": self.label,
            "baseline": self.baseline,
            "target": self.target,
            "changed": self.changed,
        }


@dataclass(frozen=True)
class Comparison:
    """A full baseline-versus-target comparison."""

    baseline_path: str
    baseline_recorded_at: str | None
    hardware: tuple[FieldComparison, ...]
    hardware_metrics: tuple[MetricComparison, ...]
    benchmark: tuple[MetricComparison, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "baseline_path": self.baseline_path,
            "baseline_recorded_at": self.baseline_recorded_at,
            "hardware": [item.to_dict() for item in self.hardware],
            "hardware_metrics": [item.to_dict() for item in self.hardware_metrics],
            "benchmark": [item.to_dict() for item in self.benchmark],
            "warnings": list(self.warnings),
        }


def load_baseline(path: str | Path) -> dict[str, Any]:
    """Read a baseline report.

    Args:
        path: Baseline JSON written by ``--save-baseline``.

    Raises:
        BaselineError: If the file is missing, unreadable, not JSON, or not a
            JSON object.
    """
    baseline_path = Path(path)
    try:
        text = baseline_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BaselineError(f"Baseline file not found: {baseline_path}") from exc
    except OSError as exc:
        raise BaselineError(f"Baseline file could not be read: {exc}") from exc

    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BaselineError(
            f"Baseline file is not valid JSON ({exc.msg} at line {exc.lineno})."
        ) from exc

    if not isinstance(loaded, dict):
        raise BaselineError("Baseline file must contain a JSON object at the top level.")
    return loaded


def dig(payload: Mapping[str, Any] | None, dotted: str) -> Any:
    """Read a dotted path out of a nested mapping, tolerating absence."""
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _as_float(value: Any) -> float | None:
    """Coerce a payload value to float, or ``None`` if it is not numeric."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_metric(spec: MetricSpec, baseline: Mapping[str, Any], target: Mapping[str, Any]) -> MetricComparison:
    """Compare one metric between two reports.

    A missing value on either side yields ``verdict="unknown"`` and no
    arithmetic, so an older or partial baseline degrades instead of raising.
    """
    base_value = _as_float(dig(baseline, spec.path))
    target_value = _as_float(dig(target, spec.path))

    if base_value is None or target_value is None:
        return MetricComparison(
            key=spec.key,
            label=spec.label,
            unit=spec.unit,
            baseline=base_value,
            target=target_value,
            difference=None,
            percent=None,
            higher_is_better=spec.higher_is_better,
            verdict="unknown",
        )

    difference = target_value - base_value
    percent = (difference / base_value * 100.0) if base_value else None

    # Direction is declared, never inferred: for latency an increase is worse.
    if abs(difference) < 1e-9 or (percent is not None and abs(percent) < 1.0):
        verdict = "same"
    elif (difference > 0) == spec.higher_is_better:
        verdict = "better"
    else:
        verdict = "worse"

    return MetricComparison(
        key=spec.key,
        label=spec.label,
        unit=spec.unit,
        baseline=round(base_value, spec.precision),
        target=round(target_value, spec.precision),
        difference=round(difference, spec.precision),
        percent=round(percent, 1) if percent is not None else None,
        higher_is_better=spec.higher_is_better,
        verdict=verdict,
    )


def _comparability_warnings(baseline: Mapping[str, Any], target: Mapping[str, Any]) -> list[str]:
    """Flag differences that make the performance numbers non-comparable."""
    warnings: list[str] = []

    pairs = (
        ("benchmark.backend", "benchmark backend"),
        ("benchmark.weights_sha_prefix", "model weights"),
        ("benchmark.source_frame", "benchmark input frame"),
        ("benchmark.imgsz", "model input size"),
    )
    for path, label in pairs:
        base_value = dig(baseline, path)
        target_value = dig(target, path)
        if base_value is None or target_value is None:
            continue
        if base_value != target_value:
            warnings.append(
                f"The {label} differs between baseline ({base_value}) and this "
                f"machine ({target_value}); the performance rows below are not "
                "a like-for-like comparison."
            )

    base_runs = _as_float(dig(baseline, "benchmark.timed_runs"))
    target_runs = _as_float(dig(target, "benchmark.timed_runs"))
    if base_runs and target_runs and base_runs != target_runs:
        warnings.append(
            f"Run counts differ (baseline {base_runs:.0f}, this machine "
            f"{target_runs:.0f}); tail percentiles are the most affected."
        )

    if dig(target, "benchmark") is None:
        warnings.append("This run has no benchmark section, so no performance rows exist.")
    elif dig(baseline, "benchmark") is None:
        warnings.append("The baseline has no benchmark section, so no performance rows exist.")

    return warnings


def build_comparison(
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
    baseline_path: str,
    text_of: Callable[[Any], str | None] | None = None,
) -> Comparison:
    """Compare a target report against a baseline report.

    Args:
        baseline: Parsed baseline report.
        target: The report just produced.
        baseline_path: Path shown in the report.
        text_of: Optional formatter for text fields.

    Returns:
        A populated :class:`Comparison`. Never raises on missing fields.
    """
    stringify = text_of or (lambda value: None if value is None else str(value))

    hardware = tuple(
        FieldComparison(
            key=key,
            label=label,
            baseline=stringify(dig(baseline, path)),
            target=stringify(dig(target, path)),
        )
        for key, label, path in HARDWARE_FIELDS
    )
    hardware_metrics = tuple(
        compare_metric(spec, baseline, target) for spec in HARDWARE_METRICS
    )
    benchmark = tuple(compare_metric(spec, baseline, target) for spec in BENCHMARK_METRICS)

    return Comparison(
        baseline_path=baseline_path,
        baseline_recorded_at=stringify(dig(baseline, "recorded_at")),
        hardware=hardware,
        hardware_metrics=hardware_metrics,
        benchmark=benchmark,
        warnings=tuple(_comparability_warnings(baseline, target)),
    )
