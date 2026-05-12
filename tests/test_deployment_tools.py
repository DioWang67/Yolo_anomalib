from __future__ import annotations

from tools.runtime_benchmark import (
    collect_image_paths,
    parse_imgsz as parse_benchmark_imgsz,
    summarize_latencies,
)


def test_collect_image_paths_returns_sorted_supported_images(tmp_path):
    (tmp_path / "b.jpg").write_bytes(b"")
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    images = collect_image_paths(tmp_path)

    assert [item.name for item in images] == ["a.png", "b.jpg"]


def test_summarize_latencies_reports_expected_statistics():
    summary = summarize_latencies("yolo", "best.pt", 2, 1, [10.0, 20.0, 30.0, 40.0])

    assert summary.mean_ms == 25.0
    assert summary.median_ms == 25.0
    assert summary.p95_ms == 40.0
    assert summary.timed_runs == 4


def test_benchmark_parse_imgsz_matches_export_parser():
    assert parse_benchmark_imgsz("320x640") == (320, 640)
