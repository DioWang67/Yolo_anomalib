"""Benchmark the real inference path this project runs in production.

Not a synthetic proxy: this loads the same weights the station's model bundle
names, letterboxes at the same ``imgsz``, runs the same ONNX session, and
applies the same confidence and IoU thresholds. Preprocess, inference and
postprocess are timed separately, because they scale with different things —
preprocessing with camera resolution, inference with the model and the CPU,
postprocessing with how many candidates clear the confidence threshold.

Two backends:

``onnx``
    Drives ``onnxruntime`` directly. This is the default for ``.onnx`` weights,
    which is what production deploys. It excludes the Ultralytics Python
    wrapper's own overhead, and the report says so.

``ultralytics``
    Drives ``ultralytics.YOLO`` exactly as ``core/yolo_inference_model.py``
    does. The truest measurement, but it needs torch, so it is unavailable to
    the standalone executable. Required for ``.pt`` weights.

The benchmark reads model files and writes nothing. It never touches the
inspection database, the result tree, or any model bundle.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tools.system_check.benchmark.preprocess import (
    letterbox,
    synthetic_frame,
    to_model_input,
)
from tools.system_check.spec import ModelTarget
from tools.system_check.sysinfo import BYTES_PER_MB, cpu_percent, process_rss, run_command

#: Runs discarded before timing starts, matching the single warmup the
#: application performs in YOLOInferenceModel.initialize.
DEFAULT_WARMUP_RUNS = 5
DEFAULT_TIMED_RUNS = 50

#: P99 from fewer samples than this is just the maximum wearing a label.
P99_RELIABLE_RUNS = 100


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot run at all."""


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageStats:
    """Latency distribution for one stage, in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    stdev_ms: float

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-serializable view."""
        return asdict(self)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list.

    Uses the standard definition: rank ``ceil(fraction * n)``, one-indexed. For
    100 samples that puts P95 on the 95th value, not the 96th.
    """
    if not sorted_values:
        return 0.0
    count = len(sorted_values)
    rank = math.ceil(fraction * count)
    index = max(0, min(count - 1, rank - 1))
    return sorted_values[index]


def summarize(samples: list[float]) -> StageStats:
    """Summarize one stage's latency samples.

    Args:
        samples: Latencies in milliseconds. Must be non-empty.

    Raises:
        BenchmarkError: If no samples were collected.
    """
    if not samples:
        raise BenchmarkError("No latency samples were collected")
    ordered = sorted(samples)
    return StageStats(
        mean_ms=round(statistics.fmean(samples), 3),
        median_ms=round(statistics.median(samples), 3),
        p95_ms=round(_percentile(ordered, 0.95), 3),
        p99_ms=round(_percentile(ordered, 0.99), 3),
        min_ms=round(ordered[0], 3),
        max_ms=round(ordered[-1], 3),
        stdev_ms=round(statistics.stdev(samples), 3) if len(samples) > 1 else 0.0,
    )


# --------------------------------------------------------------------------
# Resource sampling
# --------------------------------------------------------------------------


class ResourceSampler:
    """Sample process memory, CPU and GPU use on a background thread.

    Args:
        sample_gpu: Whether to poll ``nvidia-smi``. Left off unless a GPU was
            actually discovered, because each poll is a subprocess.
        interval_s: Seconds between process samples.
    """

    _GPU_INTERVAL_S = 1.0

    def __init__(self, *, sample_gpu: bool = False, interval_s: float = 0.2) -> None:
        self._sample_gpu = sample_gpu
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Guards every field below; the sampler thread writes, the caller reads
        # only after join(), but the lock keeps that contract explicit and safe
        # if a future caller polls mid-run.
        self._lock = threading.Lock()
        self._peak_rss = 0
        self._cpu_samples: list[float] = []
        self._peak_vram_mb = 0
        self._peak_gpu_util = 0

    def __enter__(self) -> ResourceSampler:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def start(self) -> None:
        """Begin sampling."""
        baseline = process_rss()
        if baseline:
            with self._lock:
                self._peak_rss = baseline
        # Prime the psutil CPU counter so the first real sample is meaningful.
        cpu_percent(interval=0.0)
        self._thread = threading.Thread(
            target=self._run, name="system-check-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the thread to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        last_gpu = 0.0
        while not self._stop.is_set():
            rss = process_rss()
            usage = cpu_percent(interval=0.0)
            now = time.monotonic()
            gpu = None
            if self._sample_gpu and now - last_gpu >= self._GPU_INTERVAL_S:
                gpu = self._read_gpu()
                last_gpu = now

            with self._lock:
                if rss and rss > self._peak_rss:
                    self._peak_rss = rss
                if usage is not None and usage > 0:
                    self._cpu_samples.append(usage)
                if gpu is not None:
                    util, vram = gpu
                    self._peak_gpu_util = max(self._peak_gpu_util, util)
                    self._peak_vram_mb = max(self._peak_vram_mb, vram)

            self._stop.wait(self._interval_s)

    @staticmethod
    def _read_gpu() -> tuple[int, int] | None:
        """Return ``(utilisation_percent, vram_used_mb)`` from ``nvidia-smi``."""
        code, stdout, _ = run_command(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=5.0,
        )
        if code != 0 or not stdout.strip():
            return None
        parts = [part.strip() for part in stdout.splitlines()[0].split(",")]
        try:
            return int(float(parts[0])), int(float(parts[1]))
        except (IndexError, ValueError):
            return None

    def snapshot(self) -> dict[str, Any]:
        """Return the sampled peaks and averages."""
        with self._lock:
            cpu_samples = list(self._cpu_samples)
            return {
                "peak_rss_mb": round(self._peak_rss / BYTES_PER_MB, 1) if self._peak_rss else None,
                "mean_cpu_percent": (
                    round(statistics.fmean(cpu_samples), 1) if cpu_samples else None
                ),
                "peak_cpu_percent": round(max(cpu_samples), 1) if cpu_samples else None,
                "peak_gpu_percent": self._peak_gpu_util or None,
                "peak_vram_mb": self._peak_vram_mb or None,
            }


# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkResult:
    """Everything one benchmark run produced."""

    backend: str
    product: str
    area: str
    weights: str
    weights_sha_prefix: str
    device: str
    imgsz: tuple[int, int]
    conf_thres: float
    iou_thres: float
    source_frame: str
    frame_shape: tuple[int, int]
    model_load_ms: float
    warmup_ms: float
    warmup_runs: int
    timed_runs: int
    preprocess: StageStats
    inference: StageStats
    postprocess: StageStats
    total: StageStats
    throughput_fps: float
    detections_per_frame: float
    expected_item_count: int | None
    timeout_s: float
    resources: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        payload = asdict(self)
        payload["imgsz"] = list(self.imgsz)
        payload["frame_shape"] = list(self.frame_shape)
        payload["preprocess"] = self.preprocess.to_dict()
        payload["inference"] = self.inference.to_dict()
        payload["postprocess"] = self.postprocess.to_dict()
        payload["total"] = self.total.to_dict()
        return payload


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


def select_backend(target: ModelTarget, requested: str) -> str:
    """Choose a benchmark backend for ``target``.

    ``auto`` picks the runtime the artifact actually needs: ``onnx`` for
    ``.onnx`` weights (what production deploys and what the standalone checker
    can run), ``ultralytics`` for PyTorch checkpoints.

    Raises:
        BenchmarkError: If the requested backend cannot run this artifact.
    """
    suffix = Path(target.weights).suffix.lower()
    if requested == "auto":
        return "onnx" if suffix == ".onnx" else "ultralytics"
    if requested == "onnx" and suffix != ".onnx":
        raise BenchmarkError(
            f"The onnx backend cannot run {suffix or 'this artifact'}; "
            "use --benchmark-backend ultralytics."
        )
    return requested


def _load_frame(
    context_width: int, context_height: int, image_path: str | None
) -> tuple[np.ndarray, str]:
    """Return ``(frame, description)`` for the benchmark input.

    Production-image runs are identified by content hash, not by filename.
    Two sites can easily hold different images called ``golden.bmp``, and a
    baseline comparison that silently paired them would present two unrelated
    measurements as a like-for-like difference.
    """
    if image_path:
        import cv2

        path = Path(image_path)
        # Stock cv2.imread returns None for an unreadable file, but Ultralytics
        # replaces it with an np.fromfile version that raises instead. Whether
        # that patch is active depends on whether anything imported ultralytics
        # first, so both outcomes have to be handled or the failure escapes as
        # a bare FileNotFoundError from inside a benchmark.
        try:
            loaded = cv2.imread(str(path))
        except Exception as exc:  # noqa: BLE001 - normalised into BenchmarkError
            raise BenchmarkError(
                f"Benchmark image could not be read: {image_path} ({exc})"
            ) from exc
        if loaded is None:
            raise BenchmarkError(f"Benchmark image could not be read: {image_path}")
        return np.asarray(loaded), f"file:{path.name}@{_file_sha_prefix(path)}"
    frame = synthetic_frame(context_width, context_height)
    return frame, f"synthetic:{context_width}x{context_height}"


def _file_sha_prefix(path: Path) -> str:
    """Return a short content hash, so two machines can prove identical inputs."""
    import hashlib

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()[:12]


def _numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression over ``xyxy`` boxes.

    Kept in numpy rather than delegating to ``cv2.dnn.NMSBoxes`` so the
    postprocess timing does not depend on which OpenCV build is installed.
    """
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        inter_x1 = np.maximum(x1[current], x1[rest])
        inter_y1 = np.maximum(y1[current], y1[rest])
        inter_x2 = np.minimum(x2[current], x2[rest])
        inter_y2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
        union = areas[current] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def _postprocess_onnx(
    outputs: list[np.ndarray], conf_thres: float, iou_thres: float
) -> int:
    """Decode a YOLO11 ONNX output into a detection count.

    The tensor is ``(1, 4 + num_classes, num_anchors)``: box centre and size
    followed by per-class scores. Decoding, thresholding and NMS is the work
    the production postprocess does, and is what this stage times.
    """
    raw = outputs[0]
    if raw.ndim != 3:
        return 0
    predictions = np.squeeze(raw, axis=0).T  # (anchors, 4 + classes)
    if predictions.shape[1] <= 4:
        return 0

    class_scores = predictions[:, 4:]
    best = class_scores.max(axis=1)
    mask = best >= conf_thres
    if not np.any(mask):
        return 0

    kept = predictions[mask]
    scores = best[mask]
    cx, cy, w, h = kept[:, 0], kept[:, 1], kept[:, 2], kept[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return len(_numpy_nms(boxes, scores, iou_thres))


@dataclass
class _Timed:
    """Mutable per-iteration timings, in milliseconds."""

    preprocess: list[float] = field(default_factory=list)
    inference: list[float] = field(default_factory=list)
    postprocess: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    detections: list[int] = field(default_factory=list)


def _run_onnx(
    target: ModelTarget,
    frame: np.ndarray,
    warmup_runs: int,
    timed_runs: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[float, float, _Timed, list[str]]:
    """Time the direct ONNX Runtime path."""
    try:
        import onnxruntime as ort
    except Exception as exc:  # noqa: BLE001
        raise BenchmarkError(f"onnxruntime is not importable: {exc}") from exc

    notes = [
        "Backend onnx: drives onnxruntime directly. Excludes the Ultralytics "
        "Python wrapper overhead that production adds on top."
    ]

    options = ort.SessionOptions()
    started = time.perf_counter()
    try:
        session = ort.InferenceSession(
            str(target.weights), options, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:  # noqa: BLE001
        raise BenchmarkError(f"Could not open the ONNX session: {exc}") from exc
    model_load_ms = (time.perf_counter() - started) * 1000.0

    input_name = session.get_inputs()[0].name

    def one_pass() -> tuple[float, float, float, int]:
        pre_start = time.perf_counter()
        boxed = letterbox(frame, size=target.imgsz)
        tensor = to_model_input(boxed)
        pre_ms = (time.perf_counter() - pre_start) * 1000.0

        inf_start = time.perf_counter()
        outputs = session.run(None, {input_name: tensor})
        inf_ms = (time.perf_counter() - inf_start) * 1000.0

        post_start = time.perf_counter()
        count = _postprocess_onnx(outputs, target.conf_thres, target.iou_thres)
        post_ms = (time.perf_counter() - post_start) * 1000.0
        return pre_ms, inf_ms, post_ms, count

    warmup_started = time.perf_counter()
    for _ in range(max(0, warmup_runs)):
        one_pass()
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    timed = _Timed()
    for index in range(timed_runs):
        pre_ms, inf_ms, post_ms, count = one_pass()
        timed.preprocess.append(pre_ms)
        timed.inference.append(inf_ms)
        timed.postprocess.append(post_ms)
        timed.total.append(pre_ms + inf_ms + post_ms)
        timed.detections.append(count)
        if progress is not None:
            progress(index + 1, timed_runs)

    return model_load_ms, warmup_ms, timed, notes


def _run_ultralytics(
    target: ModelTarget,
    frame: np.ndarray,
    warmup_runs: int,
    timed_runs: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[float, float, _Timed, list[str]]:
    """Time the Ultralytics path the application itself uses."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        raise BenchmarkError(
            f"ultralytics is not importable ({exc}); use --benchmark-backend onnx."
        ) from exc

    notes = [
        "Backend ultralytics: the same call core/yolo_inference_model.py makes, "
        "including wrapper overhead. Stage split comes from Ultralytics' own "
        "speed counters."
    ]

    started = time.perf_counter()
    model = YOLO(str(target.weights))
    model_load_ms = (time.perf_counter() - started) * 1000.0

    device = target.device if target.device and target.device != "auto" else None

    def one_pass() -> tuple[float, float, float, int, float]:
        # Production letterboxes before handing the frame to Ultralytics
        # (core/detector.py preprocess_image), so the same cost is timed here.
        outer_start = time.perf_counter()
        boxed = letterbox(frame, size=target.imgsz)
        own_pre_ms = (time.perf_counter() - outer_start) * 1000.0

        kwargs: dict[str, Any] = {
            "conf": target.conf_thres,
            "iou": target.iou_thres,
            "imgsz": list(target.imgsz),
            "verbose": False,
        }
        if device:
            kwargs["device"] = device
        results = model(boxed, **kwargs)
        wall_ms = (time.perf_counter() - outer_start) * 1000.0

        speed = getattr(results[0], "speed", {}) or {}
        pre_ms = own_pre_ms + float(speed.get("preprocess", 0.0))
        inf_ms = float(speed.get("inference", 0.0))
        post_ms = float(speed.get("postprocess", 0.0))
        boxes = getattr(results[0], "boxes", None)
        count = len(boxes) if boxes is not None else 0
        return pre_ms, inf_ms, post_ms, count, wall_ms

    warmup_started = time.perf_counter()
    for _ in range(max(0, warmup_runs)):
        one_pass()
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    timed = _Timed()
    for index in range(timed_runs):
        pre_ms, inf_ms, post_ms, count, wall_ms = one_pass()
        timed.preprocess.append(pre_ms)
        timed.inference.append(inf_ms)
        timed.postprocess.append(post_ms)
        # Wall time, not the sum of the stages: it includes the wrapper
        # overhead between them, which production pays too.
        timed.total.append(wall_ms)
        timed.detections.append(count)
        if progress is not None:
            progress(index + 1, timed_runs)

    return model_load_ms, warmup_ms, timed, notes


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def run_benchmark(
    target: ModelTarget,
    *,
    backend: str = "auto",
    warmup_runs: int = DEFAULT_WARMUP_RUNS,
    timed_runs: int = DEFAULT_TIMED_RUNS,
    frame_width: int = 3072,
    frame_height: int = 2048,
    image_path: str | None = None,
    sample_gpu: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> BenchmarkResult:
    """Run the production-path benchmark against one model bundle.

    Args:
        target: Model bundle to exercise.
        backend: ``auto``, ``onnx`` or ``ultralytics``.
        warmup_runs: Untimed passes before measurement begins.
        timed_runs: Measured passes.
        frame_width: Synthetic frame width, defaulting to the camera resolution.
        frame_height: Synthetic frame height.
        image_path: Optional real image to benchmark instead of the synthetic
            frame. Read-only.
        sample_gpu: Poll nvidia-smi during the run.
        progress: Optional ``(done, total)`` callback.

    Returns:
        A populated :class:`BenchmarkResult`.

    Raises:
        BenchmarkError: If the model, the runtime or the image is unusable.
    """
    if timed_runs <= 0:
        raise BenchmarkError("timed_runs must be positive")

    weights_path = Path(target.weights)
    if not weights_path.is_file():
        raise BenchmarkError(f"Model weights not found: {weights_path}")

    resolved_backend = select_backend(target, backend)
    frame, frame_description = _load_frame(frame_width, frame_height, image_path)

    runner = _run_onnx if resolved_backend == "onnx" else _run_ultralytics
    with ResourceSampler(sample_gpu=sample_gpu) as sampler:
        model_load_ms, warmup_ms, timed, notes = runner(
            target, frame, warmup_runs, timed_runs, progress
        )
    resources = sampler.snapshot()

    total_stats = summarize(timed.total)
    throughput = 1000.0 / total_stats.mean_ms if total_stats.mean_ms > 0 else 0.0

    if timed_runs < P99_RELIABLE_RUNS:
        notes.append(
            f"P99 is computed from {timed_runs} samples; below {P99_RELIABLE_RUNS} "
            "it is effectively the maximum. Use --benchmark-runs 100 or more "
            "when the P99 figure itself matters."
        )
    if image_path is None:
        notes.append(
            "Measured against a deterministic synthetic frame at the configured "
            "camera resolution, so two machines compare like for like without "
            "shipping inspection images between sites. It is built for "
            "comparability, not realism: it yields few detections, so the "
            "postprocess figure is a lower bound. Use --benchmark-image with a "
            "real board photo for a representative postprocess measurement."
        )
    else:
        notes.append(
            "Measured against a production image. The report records its "
            "content hash; a baseline recorded with a different image is "
            "flagged as not comparable."
        )

    return BenchmarkResult(
        backend=resolved_backend,
        product=target.product,
        area=target.area,
        weights=str(weights_path),
        weights_sha_prefix=_file_sha_prefix(weights_path),
        device=target.device,
        imgsz=target.imgsz,
        conf_thres=target.conf_thres,
        iou_thres=target.iou_thres,
        source_frame=frame_description,
        frame_shape=(int(frame.shape[0]), int(frame.shape[1])),
        model_load_ms=round(model_load_ms, 3),
        warmup_ms=round(warmup_ms, 3),
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        preprocess=summarize(timed.preprocess),
        inference=summarize(timed.inference),
        postprocess=summarize(timed.postprocess),
        total=total_stats,
        throughput_fps=round(throughput, 2),
        detections_per_frame=round(statistics.fmean(timed.detections), 2)
        if timed.detections
        else 0.0,
        expected_item_count=target.expected_item_count,
        timeout_s=target.timeout_s,
        resources=resources,
        notes=notes,
    )
