"""A small synthetic CPU probe.

This exists only to separate two explanations of a slow machine: a genuinely
slower processor, versus a processor that is fine but is being starved by
something else on the box. It is *not* the performance verdict — the project
benchmark is. Where the two disagree, believe the project benchmark.

The workload is a fixed-size float32 matrix multiply plus a single-threaded
integer loop, so it exercises both the vectorised path ONNX Runtime depends on
and the scalar path Python preprocessing runs on.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

_MATRIX_SIZE = 512
_MATRIX_REPEATS = 12
_SCALAR_ITERATIONS = 2_000_000


@dataclass(frozen=True)
class CpuBenchmark:
    """Synthetic CPU timings in milliseconds and derived rates."""

    matmul_ms: float
    matmul_gflops: float
    scalar_ms: float
    scalar_mops: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable view."""
        return asdict(self)


def run(seed: int = 20260921) -> CpuBenchmark:
    """Run the synthetic probe.

    Args:
        seed: Fixed seed so both machines multiply identical matrices.

    Returns:
        Timings for the vectorised and scalar workloads.
    """
    rng = np.random.default_rng(seed)
    left = rng.random((_MATRIX_SIZE, _MATRIX_SIZE), dtype=np.float32)
    right = rng.random((_MATRIX_SIZE, _MATRIX_SIZE), dtype=np.float32)

    # One untimed pass so BLAS thread pools are already spun up.
    left @ right

    started = time.perf_counter()
    for _ in range(_MATRIX_REPEATS):
        left @ right
    matmul_ms = (time.perf_counter() - started) * 1000.0
    operations = 2.0 * _MATRIX_SIZE**3 * _MATRIX_REPEATS
    matmul_gflops = operations / (matmul_ms / 1000.0) / 1e9 if matmul_ms > 0 else 0.0

    started = time.perf_counter()
    total = 0
    for index in range(_SCALAR_ITERATIONS):
        total += index & 0xFF
    scalar_ms = (time.perf_counter() - started) * 1000.0
    scalar_mops = _SCALAR_ITERATIONS / (scalar_ms / 1000.0) / 1e6 if scalar_ms > 0 else 0.0

    # ``total`` is deliberately unused beyond keeping the loop from being
    # optimised away by a future interpreter.
    assert total >= 0

    return CpuBenchmark(
        matmul_ms=round(matmul_ms, 3),
        matmul_gflops=round(matmul_gflops, 2),
        scalar_ms=round(scalar_ms, 3),
        scalar_mops=round(scalar_mops, 2),
    )
