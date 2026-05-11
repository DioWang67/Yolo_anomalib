"""Benchmark YOLO and current Anomalib runtime artifacts on local images."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class BenchmarkSummary:
    """Latency summary in milliseconds."""

    backend: str
    model: str
    image_count: int
    warmup_runs: int
    timed_runs: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


def collect_image_paths(path: Path) -> list[Path]:
    """Collect image files from a file or directory.

    Args:
        path: Image file or directory.

    Returns:
        Sorted image paths.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If no supported images are found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Image input not found: {path}")
    if path.is_file():
        images = [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    else:
        images = [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ]
    images = sorted(images)
    if not images:
        raise ValueError(f"No supported images found under: {path}")
    return images


def summarize_latencies(
    backend: str,
    model: str,
    image_count: int,
    warmup_runs: int,
    latencies_ms: list[float],
) -> BenchmarkSummary:
    """Summarize measured latency values."""
    if not latencies_ms:
        raise ValueError("No timed latency samples were collected")
    sorted_values = sorted(latencies_ms)
    p95_index = min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))
    return BenchmarkSummary(
        backend=backend,
        model=model,
        image_count=image_count,
        warmup_runs=warmup_runs,
        timed_runs=len(latencies_ms),
        mean_ms=round(statistics.fmean(latencies_ms), 3),
        median_ms=round(statistics.median(latencies_ms), 3),
        p95_ms=round(sorted_values[p95_index], 3),
        min_ms=round(min(latencies_ms), 3),
        max_ms=round(max(latencies_ms), 3),
    )


def parse_imgsz(value: str) -> int | tuple[int, int]:
    """Parse image size as ``640`` or ``640,480``."""
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        if size <= 0:
            raise ValueError("imgsz must be positive")
        return size
    if len(parts) == 2:
        height, width = int(parts[0]), int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("imgsz dimensions must be positive")
        return height, width
    raise ValueError("imgsz must be like 640 or 640,480")


def run_timed_loop(
    images: list[Path],
    warmup_runs: int,
    timed_runs: int,
    infer_once: Callable[[Path], Any],
) -> list[float]:
    """Run warmup and timed inference loops."""
    if timed_runs <= 0:
        raise ValueError("timed_runs must be positive")
    for index in range(max(0, warmup_runs)):
        infer_once(images[index % len(images)])

    latencies_ms: list[float] = []
    for index in range(timed_runs):
        image_path = images[index % len(images)]
        started = time.perf_counter()
        infer_once(image_path)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    return latencies_ms


def benchmark_yolo(
    model_path: Path,
    images: list[Path],
    *,
    device: str | None,
    imgsz: int | tuple[int, int] | None,
    conf: float,
    iou: float,
    warmup_runs: int,
    timed_runs: int,
) -> BenchmarkSummary:
    """Benchmark an Ultralytics-compatible YOLO artifact."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for YOLO benchmarking") from exc

    model = YOLO(str(model_path))

    def infer_once(image_path: Path) -> Any:
        kwargs: dict[str, Any] = {
            "conf": conf,
            "iou": iou,
            "verbose": False,
        }
        if imgsz is not None:
            kwargs["imgsz"] = imgsz
        if device:
            kwargs["device"] = device
        return model(str(image_path), **kwargs)

    latencies = run_timed_loop(images, warmup_runs, timed_runs, infer_once)
    return summarize_latencies(
        "yolo",
        str(model_path),
        len(images),
        warmup_runs,
        latencies,
    )


def benchmark_anomalib_lightning(
    config_path: Path,
    images: list[Path],
    *,
    product: str,
    area: str,
    warmup_runs: int,
    timed_runs: int,
) -> BenchmarkSummary:
    """Benchmark the existing Anomalib Lightning inference wrapper."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for anomalib benchmarking") from exc

    from core.anomalib_lightning_inference import initialize, lightning_inference

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    initialize(config=config, product=product, area=area)

    def infer_once(image_path: Path) -> Any:
        return lightning_inference(str(image_path), product=product, area=area)

    latencies = run_timed_loop(images, warmup_runs, timed_runs, infer_once)
    return summarize_latencies(
        "anomalib-lightning",
        str(config_path),
        len(images),
        warmup_runs,
        latencies,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=["yolo", "anomalib-lightning"])
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="YOLO artifact path")
    parser.add_argument("--config", type=Path, help="Anomalib config.yaml path")
    parser.add_argument("--product")
    parser.add_argument("--area")
    parser.add_argument("--device")
    parser.add_argument("--imgsz")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    images = collect_image_paths(args.images)

    if args.backend == "yolo":
        if args.model is None:
            parser.error("--model is required for --backend yolo")
        imgsz = None
        if args.imgsz:
            imgsz = parse_imgsz(args.imgsz)
        summary = benchmark_yolo(
            args.model,
            images,
            device=args.device,
            imgsz=imgsz,
            conf=args.conf,
            iou=args.iou,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )
    else:
        if args.config is None or not args.product or not args.area:
            parser.error("--config, --product and --area are required for anomalib-lightning")
        summary = benchmark_anomalib_lightning(
            args.config,
            images,
            product=args.product,
            area=args.area,
            warmup_runs=args.warmup,
            timed_runs=args.runs,
        )

    payload = json.dumps(summary.__dict__, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
