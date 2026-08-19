"""Repeatable synthetic benchmark for Phase 2C review image loading."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import psutil
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QImage, QPainter
from PyQt5.QtWidgets import QApplication

from app.gui.async_image_service import AsyncImageService
from app.gui.review_selection_gallery import ReviewSelectionGallery
from app.gui.review_workspace import ReviewImageViewer


class PeakRssSampler:
    """Sample native process RSS while Qt performs image work."""

    def __init__(self, *, interval_seconds: float = 0.005) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._process = psutil.Process()
        self.peak_bytes = self._process.memory_info().rss
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> PeakRssSampler:
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    def _sample(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.peak_bytes = max(
                self.peak_bytes,
                self._process.memory_info().rss,
            )


def _create_fixture(root: Path, *, count: int, width: int, height: int) -> list[Path]:
    source = root / "source.jpg"
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("#243447"))
    painter = QPainter(image)
    try:
        for offset in range(0, width, 96):
            painter.fillRect(offset, 0, 48, height, QColor("#5b8fc9"))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(image.rect(), Qt.AlignCenter, "Phase 2C deterministic fixture")
    finally:
        painter.end()
    if not image.save(str(source), "JPG", 88):
        raise RuntimeError(f"Could not create benchmark image: {source}")
    paths = []
    for index in range(count):
        target = root / f"image_{index:04d}.jpg"
        shutil.copyfile(source, target)
        paths.append(target)
    return paths


def _elapsed_ms(action) -> float:
    started = time.perf_counter()
    action()
    QApplication.processEvents()
    return (time.perf_counter() - started) * 1000.0


def _wait_until(predicate, *, timeout_seconds: float = 30.0) -> float:
    started = time.perf_counter()
    deadline = started + timeout_seconds
    while not predicate():
        QApplication.processEvents()
        if time.perf_counter() >= deadline:
            raise TimeoutError("Image benchmark did not reach the expected state")
        time.sleep(0.001)
    QApplication.processEvents()
    return (time.perf_counter() - started) * 1000.0


def run_benchmark(
    *,
    count: int,
    width: int,
    height: int,
    synchronous: bool,
) -> dict[str, Any]:
    app = QApplication.instance() or QApplication([])
    process = psutil.Process()
    initial_rss = process.memory_info().rss
    with tempfile.TemporaryDirectory(prefix="review-image-benchmark-") as raw_root:
        paths = _create_fixture(Path(raw_root), count=count, width=width, height=height)
        entries = [
            (
                index,
                {
                    "status": "FAIL",
                    "timestamp": f"2026-07-20T12:{index % 60:02d}:00",
                    "product": "Synthetic",
                    "area": "A",
                    "annotated_path": str(path),
                    "original_path": str(path),
                },
            )
            for index, path in enumerate(paths)
        ]
        image_service = AsyncImageService(max_workers=2, synchronous=synchronous)
        gallery = ReviewSelectionGallery(language="en", image_service=image_service)
        gallery.resize(1250, 780)
        gallery.show()
        viewer = ReviewImageViewer(language="en", image_service=image_service)
        viewer.resize(900, 700)
        viewer.show()
        QApplication.processEvents()

        with PeakRssSampler() as sampler:
            overview_ms = _elapsed_ms(lambda: gallery.set_entries(entries, set()))
            first_twenty_ms = overview_ms + _wait_until(
                lambda: gallery.loaded_thumbnail_count >= min(20, count)
            )
            all_started = time.perf_counter()
            gallery.request_all_thumbnails()
            _wait_until(lambda: gallery.loaded_thumbnail_count == count)
            all_thumbnails_ms = (
                first_twenty_ms + (time.perf_counter() - all_started) * 1000.0
            )
            filter_ms = _elapsed_ms(
                lambda: gallery.filter_combo.setCurrentIndex(
                    gallery.filter_combo.findData("unreviewed")
                )
            )
            full_started = time.perf_counter()
            viewer.set_images(
                original_path=paths[0],
                overlay_path=paths[0],
            )
            _wait_until(lambda: viewer.image_state == "loaded")
            full_image_ms = (time.perf_counter() - full_started) * 1000.0

            switch_started = time.perf_counter()
            for path in paths[:20]:
                viewer.set_images(original_path=path, overlay_path=path)
            _wait_until(lambda: viewer.image_state == "loaded")
            switch_twenty_ms = (time.perf_counter() - switch_started) * 1000.0
            close_started = time.perf_counter()
            gallery.close()
            viewer.close()
            image_service.shutdown(wait_ms=5000)
            QApplication.processEvents()
            close_ms = (time.perf_counter() - close_started) * 1000.0

        final_rss = process.memory_info().rss
        app.processEvents()
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_logical": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_total_bytes": psutil.virtual_memory().total,
            "synthetic": True,
            "filesystem_cache": "warm",
            "format": "JPEG",
            "image_width": width,
            "image_height": height,
            "records": count,
            "loader_mode": "sync-fallback" if synchronous else "async",
        },
        "metrics_ms": {
            "overview_first_interactive": round(overview_ms, 3),
            "first_20_thumbnails": round(first_twenty_ms, 3),
            "all_thumbnails": round(all_thumbnails_ms, 3),
            "full_image": round(full_image_ms, 3),
            "switch_20_images": round(switch_twenty_ms, 3),
            "filter_switch": round(filter_ms, 3),
            "close_to_workers_stopped": round(close_ms, 3),
        },
        "memory": {
            "initial_rss_bytes": initial_rss,
            "final_rss_bytes": final_rss,
            "peak_rss_bytes": sampler.peak_bytes,
            "peak_delta_bytes": sampler.peak_bytes - initial_rss,
        },
        "notes": [
            "Component-level image-loading benchmark; manifest scan is excluded.",
            "All-thumbnails time is cumulative from overview start; the explicit all-items queue is issued after the first 20 complete.",
        ],
        "loader": {
            "decode_count": image_service.decode_count,
            "thumbnail_cache_entries_after_close": image_service.thumbnail_cache.entry_count,
            "full_cache_entries_after_close": image_service.full_cache.entry_count,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, choices=(100, 1000), required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        count=args.count,
        width=args.width,
        height=args.height,
        synchronous=args.sync,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
