"""Turn benchmark measurements into pass/fail verdicts.

Kept apart from :mod:`tools.system_check.checks`, which inspects the
environment. This module judges *measurements*, and it only judges them against
thresholds the repository actually states.

Two performance concepts, deliberately kept separate, because merging them
would manufacture a requirement that does not exist:

**Inference timeout** — a hard, enforced constraint. ``core/config.py`` sets
``timeout`` and ``core/fusion_inference.py`` raises when a backend exceeds it,
failing the inspection. A machine that breaches it produces real defects, so
this is judged PASS / WARNING / FAIL.

**Cycle time and throughput** — a *line* requirement, and the repository does
not state one. There is no FPS target, no takt time, no throughput figure
anywhere in it. ``auto_trigger.inspection_cooldown_ms`` bounds how often a
trigger may fire; it is not a target. Throughput is therefore measured and
reported, and never judged. It carries no verdict at all rather than a PASS
that would imply a threshold was met, or an UNKNOWN that would nag on every
run about a requirement nobody has written down. It is listed instead under
the report's "requirements this report does not assert" section.

Memory headroom is judged, but only measurement against measurement: the
benchmark's own peak working set against this machine's available memory.
"""

from __future__ import annotations

from tools.system_check.benchmark.project_benchmark import BenchmarkResult
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.spec import SUGGESTED_P99_TIMEOUT_BUDGET
from tools.system_check.sysinfo import BYTES_PER_MB, memory_info


def benchmark_checks(result: BenchmarkResult) -> list[CheckResult]:
    """Evaluate one benchmark run against stated requirements.

    Throughput is absent by design; see the module docstring.
    """
    return [
        _latency_check(result),
        _memory_headroom_check(result),
        *_postprocess_representativeness(result),
    ]


def _latency_check(result: BenchmarkResult) -> CheckResult:
    """Compare measured latency against the configured inference timeout."""
    timeout_ms = result.timeout_s * 1000.0
    p99 = result.total.p99_ms
    p95 = result.total.p95_ms
    mean = result.total.mean_ms
    budget_ms = timeout_ms * SUGGESTED_P99_TIMEOUT_BUDGET
    measured = f"mean {mean:.1f} ms, P95 {p95:.1f} ms, P99 {p99:.1f} ms"
    requirement = f"P99 below the configured timeout of {timeout_ms:.0f} ms"
    payload = {
        "timeout_ms": timeout_ms,
        "mean_ms": mean,
        "p95_ms": p95,
        "p99_ms": p99,
        "budget_ms": round(budget_ms, 1),
    }

    if p99 >= timeout_ms:
        return CheckResult(
            check_id="benchmark.latency",
            title="Inference latency",
            status=Status.FAIL,
            detail=(
                f"P99 latency {p99:.1f} ms meets or exceeds the {timeout_ms:.0f} ms "
                "timeout this station enforces. core/fusion_inference.py raises "
                "and fails the inspection at that point, so this machine would "
                "produce intermittent inspection errors under load."
            ),
            requirement=requirement,
            measured=measured,
            source="core/fusion_inference.py:95-175, model config timeout",
            remedy=(
                "Use a faster machine, or raise the timeout in the model "
                "bundle config after confirming the line's cycle time allows it."
            ),
            data=payload,
        )

    if p99 > budget_ms:
        return CheckResult(
            check_id="benchmark.latency",
            title="Inference latency",
            status=Status.WARNING,
            detail=(
                f"P99 latency {p99:.1f} ms fits inside the {timeout_ms:.0f} ms "
                f"timeout but uses more than "
                f"{SUGGESTED_P99_TIMEOUT_BUDGET:.0%} of it. The margin is this "
                "tool's suggestion, not a stated requirement, but a benchmark "
                "runs on an otherwise idle machine, and production does not."
            ),
            requirement=requirement,
            measured=measured,
            confidence=Confidence.SUGGESTED,
            source="core/fusion_inference.py:95-175",
            remedy=(
                "Re-measure with the GUI and camera running before accepting "
                "this machine."
            ),
            data=payload,
        )

    return CheckResult(
        check_id="benchmark.latency",
        title="Inference latency",
        status=Status.PASS,
        detail=(
            f"P99 latency {p99:.1f} ms against a {timeout_ms:.0f} ms timeout "
            f"({p99 / timeout_ms:.0%} of budget)."
        ),
        requirement=requirement,
        measured=measured,
        source="core/fusion_inference.py:95-175",
        data=payload,
    )


def _postprocess_representativeness(result: BenchmarkResult) -> list[CheckResult]:
    """Warn when the benchmark frame under-exercises postprocessing.

    Postprocess cost is driven by how many candidates clear the confidence
    threshold, because NMS runs over the survivors. Measured on this model:
    1 detection costs 0.07 ms, 19 cost 0.29 ms, 57 cost 0.76 ms. The
    deterministic synthetic frame is built for comparability, not realism, and
    produces a single detection — so its postprocess figure is a *lower bound*
    on what a real board costs.

    In absolute terms the gap is under a millisecond against ~33 ms of
    inference, so it does not change a timeout verdict. It is still reported,
    because a number presented without that caveat invites someone to size a
    line on it.
    """
    expected = result.expected_item_count
    measured = result.detections_per_frame
    payload = {
        "detections_per_frame": measured,
        "expected_item_count": expected,
        "postprocess_mean_ms": result.postprocess.mean_ms,
        "source_frame": result.source_frame,
    }

    if expected is None:
        # No expected_items in the bundle config, so there is nothing to
        # compare against and no claim to make.
        return []

    if measured >= expected:
        return [
            CheckResult(
                check_id="benchmark.postprocess_load",
                title="Postprocess representativeness",
                status=Status.PASS,
                detail=(
                    f"The benchmark frame yielded {measured:.1f} detections "
                    f"against {expected} expected items for this product, so "
                    "NMS was exercised at production scale."
                ),
                requirement="Benchmark frame should produce production-like detections",
                measured=f"{measured:.1f} detections vs {expected} expected",
                confidence=Confidence.SUGGESTED,
                data=payload,
            )
        ]

    return [
        CheckResult(
            check_id="benchmark.postprocess_load",
            title="Postprocess representativeness",
            status=Status.WARNING,
            detail=(
                f"The benchmark frame yielded {measured:.1f} detections, below "
                f"the {expected} items this product expects, so the "
                f"{result.postprocess.mean_ms:.2f} ms postprocess figure is a "
                "lower bound: NMS scales with the number of surviving "
                "candidates. Inference and preprocess timings are unaffected, "
                "and the shortfall is well under a millisecond, so the timeout "
                "verdict stands."
            ),
            requirement="Benchmark frame should produce production-like detections",
            measured=f"{measured:.1f} detections vs {expected} expected",
            confidence=Confidence.SUGGESTED,
            source="models/<product>/<area>/yolo/config.yaml expected_items",
            remedy=(
                "For a representative postprocess figure, re-run with "
                "--benchmark-image pointing at a known-good board photo, and "
                "use the same file on every machine being compared."
            ),
            data=payload,
        )
    ]


def _memory_headroom_check(result: BenchmarkResult) -> CheckResult:
    """Compare the benchmark's measured peak working set against free RAM."""
    peak_mb = result.resources.get("peak_rss_mb")
    info = memory_info()
    available_mb = (info.available or 0) / BYTES_PER_MB if info.available else None

    if peak_mb is None or available_mb is None:
        return CheckResult(
            check_id="benchmark.memory",
            title="Memory headroom",
            status=Status.UNKNOWN,
            detail=(
                "Peak working set or available memory could not be measured, "
                "so no headroom judgement is possible."
            ),
            requirement="Measured peak working set must fit in available memory",
            measured="unknown",
            confidence=Confidence.UNKNOWN,
        )

    measured = f"peak {peak_mb:.0f} MB against {available_mb:.0f} MB available"
    payload = {"peak_rss_mb": peak_mb, "available_mb": round(available_mb, 1)}
    note = (
        " This benchmark holds one model; production holds up to "
        "max_cache_size (3) and the GUI on top."
    )

    if peak_mb > available_mb:
        return CheckResult(
            check_id="benchmark.memory",
            title="Memory headroom",
            status=Status.FAIL,
            detail=(
                f"The benchmark's own peak working set ({peak_mb:.0f} MB) "
                f"exceeds the memory available on this machine "
                f"({available_mb:.0f} MB)." + note
            ),
            requirement="Measured peak working set must fit in available memory",
            measured=measured,
            remedy="Free memory on this machine, or fit more RAM.",
            data=payload,
        )

    if peak_mb * 3 > available_mb:
        return CheckResult(
            check_id="benchmark.memory",
            title="Memory headroom",
            status=Status.WARNING,
            detail=(
                f"Peak working set {peak_mb:.0f} MB leaves under 3x headroom in "
                f"{available_mb:.0f} MB of available memory." + note
            ),
            requirement="Measured peak working set must fit in available memory",
            measured=measured,
            confidence=Confidence.SUGGESTED,
            remedy="Check memory use with the GUI and camera running.",
            data=payload,
        )

    return CheckResult(
        check_id="benchmark.memory",
        title="Memory headroom",
        status=Status.PASS,
        detail=f"Peak working set {peak_mb:.0f} MB of {available_mb:.0f} MB available.",
        requirement="Measured peak working set must fit in available memory",
        measured=measured,
        data=payload,
    )
