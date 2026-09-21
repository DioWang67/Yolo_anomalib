"""Benchmark correctness, and the guard against it drifting from production.

The checker keeps its own copy of the production letterbox so it can be
packaged without the application. The first test here is what makes that copy
safe: if ``core.utils.ImageUtils.letterbox`` ever changes, this fails rather
than the benchmark quietly measuring a transform the line no longer performs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.utils import ImageUtils
from tools.system_check.benchmark import project_benchmark
from tools.system_check.benchmark.preprocess import letterbox, synthetic_frame, to_model_input
from tools.system_check.benchmark.project_benchmark import (
    BenchmarkError,
    ResourceSampler,
    run_benchmark,
    select_backend,
    summarize,
)
from tools.system_check.results import Confidence, Status
from tools.system_check.spec import ModelTarget
from tools.system_check.verdicts import benchmark_checks

REAL_MODEL = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "Cable1"
    / "A"
    / "yolo"
    / "weights"
    / "Cable1_A_v1.0.6_20260727.onnx"
)


# --------------------------------------------------------------------------
# Preprocessing fidelity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2048, 3072, 3), (480, 640, 3), (700, 500, 3)])
def test_letterbox_matches_production_exactly(shape: tuple[int, int, int]) -> None:
    """The checker's copy must be byte-identical to the production transform."""
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=shape, dtype=np.uint8)

    np.testing.assert_array_equal(
        letterbox(frame, size=(640, 640)),
        ImageUtils.letterbox(frame, size=(640, 640)),
    )


def test_to_model_input_shape_and_range() -> None:
    """The ONNX tensor must be NCHW float32 in 0..1."""
    boxed = np.full((640, 640, 3), 200, dtype=np.uint8)
    tensor = to_model_input(boxed)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_synthetic_frame_is_deterministic() -> None:
    """Two machines must letterbox identical pixels, or the comparison lies."""
    first = synthetic_frame(320, 240)
    second = synthetic_frame(320, 240)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (240, 320, 3)
    assert first.dtype == np.uint8


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_summarize_percentiles() -> None:
    stats = summarize([float(value) for value in range(1, 101)])

    assert stats.min_ms == 1.0
    assert stats.max_ms == 100.0
    assert stats.median_ms == pytest.approx(50.5)
    assert stats.p95_ms == 95.0
    assert stats.p99_ms == 99.0


def test_summarize_rejects_empty_samples() -> None:
    with pytest.raises(BenchmarkError):
        summarize([])


def test_summarize_single_sample_has_zero_stdev() -> None:
    assert summarize([12.0]).stdev_ms == 0.0


# --------------------------------------------------------------------------
# Postprocessing
# --------------------------------------------------------------------------


def test_nms_suppresses_overlapping_boxes() -> None:
    boxes = np.array(
        [[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)

    kept = project_benchmark._numpy_nms(boxes, scores, 0.5)

    assert sorted(kept) == [0, 2]


def test_nms_on_empty_input() -> None:
    assert project_benchmark._numpy_nms(np.empty((0, 4)), np.empty((0,)), 0.5) == []


def test_postprocess_filters_below_confidence() -> None:
    """One strong box survives; the low-confidence anchors are dropped."""
    anchors = 5
    raw = np.zeros((1, 4 + 3, anchors), dtype=np.float32)
    raw[0, 0:4, 0] = [50, 50, 20, 20]
    raw[0, 4, 0] = 0.9
    raw[0, 4:, 1:] = 0.05

    assert project_benchmark._postprocess_onnx([raw], 0.4, 0.45) == 1
    assert project_benchmark._postprocess_onnx([raw], 0.95, 0.45) == 0


def test_postprocess_tolerates_unexpected_tensor_shape() -> None:
    """A model with an output this decoder does not understand returns zero."""
    assert project_benchmark._postprocess_onnx([np.zeros((1, 8400))], 0.4, 0.45) == 0


# --------------------------------------------------------------------------
# Backend selection and failure handling
# --------------------------------------------------------------------------


def _target(weights: str, **kwargs) -> ModelTarget:
    return ModelTarget(product="P", area="A", weights=weights, **kwargs)


def test_auto_backend_follows_the_artifact() -> None:
    assert select_backend(_target("m.onnx"), "auto") == "onnx"
    assert select_backend(_target("m.pt"), "auto") == "ultralytics"


def test_onnx_backend_refuses_a_torch_checkpoint() -> None:
    with pytest.raises(BenchmarkError, match="ultralytics"):
        select_backend(_target("m.pt"), "onnx")


def test_missing_weights_raise_benchmark_error(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkError, match="not found"):
        run_benchmark(_target(str(tmp_path / "absent.onnx")))


def test_unreadable_model_raises_benchmark_error(tmp_path: Path) -> None:
    """A corrupt .onnx must surface as a BenchmarkError, not an ORT traceback."""
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"definitely not a protobuf")

    with pytest.raises(BenchmarkError, match="ONNX session"):
        run_benchmark(_target(str(broken)), timed_runs=1, warmup_runs=0)


def test_zero_runs_rejected(tmp_path: Path) -> None:
    weights = tmp_path / "m.onnx"
    weights.write_bytes(b"x")
    with pytest.raises(BenchmarkError, match="positive"):
        run_benchmark(_target(str(weights)), timed_runs=0)


def test_missing_benchmark_image_raises(tmp_path: Path) -> None:
    weights = tmp_path / "m.onnx"
    weights.write_bytes(b"x")
    with pytest.raises(BenchmarkError, match="could not be read"):
        run_benchmark(
            _target(str(weights)),
            timed_runs=1,
            warmup_runs=0,
            image_path=str(tmp_path / "no-such-image.png"),
        )


# --------------------------------------------------------------------------
# Resource sampling
# --------------------------------------------------------------------------


def test_resource_sampler_starts_and_stops_cleanly() -> None:
    with ResourceSampler(sample_gpu=False, interval_s=0.01) as sampler:
        sum(index * index for index in range(200_000))
    snapshot = sampler.snapshot()

    assert set(snapshot) == {
        "peak_rss_mb",
        "mean_cpu_percent",
        "peak_cpu_percent",
        "peak_gpu_percent",
        "peak_vram_mb",
    }


def test_resource_sampler_survives_a_broken_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        project_benchmark, "run_command", lambda *a, **k: (0, "garbage output", "")
    )
    assert ResourceSampler._read_gpu() is None


# --------------------------------------------------------------------------
# End-to-end against the real production artifact
# --------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_MODEL.is_file(), reason="production model bundle not present")
def test_benchmark_runs_against_the_real_model() -> None:
    """The full path: load, warm up, time, split stages, and judge latency."""
    target = ModelTarget(
        product="Cable1",
        area="A",
        weights=str(REAL_MODEL),
        imgsz=(640, 640),
        conf_thres=0.4,
        iou_thres=0.45,
        timeout_s=1.0,
        device="cpu",
    )
    result = run_benchmark(target, warmup_runs=1, timed_runs=3, frame_width=640, frame_height=480)

    assert result.backend == "onnx"
    assert result.timed_runs == 3
    assert result.total.mean_ms > 0
    assert result.inference.mean_ms > 0
    assert result.throughput_fps > 0
    assert len(result.weights_sha_prefix) == 12
    # Total must account for every stage.
    assert result.total.mean_ms >= result.inference.mean_ms

    latency = next(item for item in benchmark_checks(result) if item.check_id == "benchmark.latency")
    assert latency.status in (Status.PASS, Status.WARNING, Status.FAIL)


# --------------------------------------------------------------------------
# Verdicts
# --------------------------------------------------------------------------


def _fake_result(
    total_p99_ms: float,
    timeout_s: float,
    peak_rss_mb: float | None = 200.0,
    detections: float = 6.0,
    expected_items: int | None = 6,
):
    """Build a BenchmarkResult with just the fields the verdicts read."""
    stats = summarize([total_p99_ms])
    return project_benchmark.BenchmarkResult(
        backend="onnx",
        product="P",
        area="A",
        weights="m.onnx",
        weights_sha_prefix="abc123abc123",
        device="cpu",
        imgsz=(640, 640),
        conf_thres=0.4,
        iou_thres=0.45,
        source_frame="synthetic:640x480",
        frame_shape=(480, 640),
        model_load_ms=100.0,
        warmup_ms=200.0,
        warmup_runs=1,
        timed_runs=1,
        preprocess=stats,
        inference=stats,
        postprocess=stats,
        total=stats,
        throughput_fps=10.0,
        detections_per_frame=detections,
        expected_item_count=expected_items,
        timeout_s=timeout_s,
        resources={"peak_rss_mb": peak_rss_mb},
    )


def test_latency_over_the_timeout_fails() -> None:
    result = _fake_result(total_p99_ms=1200.0, timeout_s=1.0)
    latency = benchmark_checks(result)[0]

    assert latency.status is Status.FAIL
    assert "fusion_inference" in (latency.source or "")


def test_latency_inside_the_timeout_but_over_budget_warns() -> None:
    result = _fake_result(total_p99_ms=700.0, timeout_s=1.0)
    assert benchmark_checks(result)[0].status is Status.WARNING


def test_comfortable_latency_passes() -> None:
    result = _fake_result(total_p99_ms=50.0, timeout_s=1.0)
    assert benchmark_checks(result)[0].status is Status.PASS


def test_throughput_produces_no_verdict_at_all() -> None:
    """Cycle time is a line requirement the repository does not state.

    It must not appear as a check: a PASS would imply a threshold was met, and
    a permanent UNKNOWN would nag on every run about a requirement nobody has
    written down. It belongs in the measurements and in the "not asserted"
    list, both of which the report carries.
    """
    ids = {item.check_id for item in benchmark_checks(_fake_result(2000.0, 1.0))}

    assert "benchmark.throughput" not in ids
    assert not any("throughput" in check_id for check_id in ids)
    # The enforced timeout is still judged, and here it is breached.
    latency = next(
        item for item in benchmark_checks(_fake_result(2000.0, 1.0))
        if item.check_id == "benchmark.latency"
    )
    assert latency.status is Status.FAIL


def test_timeout_and_cycle_time_are_separate_concepts() -> None:
    """The timeout is judged; cycle time is declared undetermined."""
    from tools.system_check.spec import REQUIREMENTS

    by_key = {item.key: item for item in REQUIREMENTS}

    assert by_key["benchmark.latency"].confidence is Confidence.CONFIRMED
    assert by_key["benchmark.latency"].source
    assert by_key["throughput.fps"].confidence is Confidence.UNKNOWN
    assert by_key["throughput.fps"].source == ""


def test_memory_headroom_unknown_when_peak_not_measured() -> None:
    memory = next(
        item
        for item in benchmark_checks(_fake_result(50.0, 1.0, peak_rss_mb=None))
        if item.check_id == "benchmark.memory"
    )
    assert memory.status is Status.UNKNOWN


# --------------------------------------------------------------------------
# Postprocess representativeness
# --------------------------------------------------------------------------


def test_too_few_detections_flags_postprocess_as_a_lower_bound() -> None:
    """The synthetic frame yields one detection; a real board yields six.

    NMS cost scales with surviving candidates, so the report must not present
    the synthetic figure as production postprocess cost.
    """
    result = _fake_result(50.0, 1.0, detections=1.0, expected_items=6)
    check = next(
        item for item in benchmark_checks(result)
        if item.check_id == "benchmark.postprocess_load"
    )

    assert check.status is Status.WARNING
    assert "lower bound" in check.detail
    assert "--benchmark-image" in (check.remedy or "")


def test_production_like_detection_count_passes() -> None:
    result = _fake_result(50.0, 1.0, detections=6.0, expected_items=6)
    check = next(
        item for item in benchmark_checks(result)
        if item.check_id == "benchmark.postprocess_load"
    )
    assert check.status is Status.PASS


def test_no_expected_items_makes_no_claim() -> None:
    """A bundle that declares no expected_items gets no representativeness row."""
    result = _fake_result(50.0, 1.0, detections=1.0, expected_items=None)
    ids = {item.check_id for item in benchmark_checks(result)}
    assert "benchmark.postprocess_load" not in ids
