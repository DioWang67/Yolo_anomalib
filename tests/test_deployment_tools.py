from __future__ import annotations

import pytest

from tools.runtime_benchmark import (
    collect_image_paths,
    parse_imgsz as parse_benchmark_imgsz,
    summarize_latencies,
)
from tools.yolo_export import ExportRequest, build_export_kwargs, parse_imgsz


def test_yolo_export_parse_imgsz_accepts_square_and_rectangular_values():
    assert parse_imgsz("640") == 640
    assert parse_imgsz("640,480") == (640, 480)
    assert parse_imgsz("640x480") == (640, 480)


def test_yolo_export_rejects_conflicting_precision_modes(tmp_path):
    request = ExportRequest(
        weights=tmp_path / "best.pt",
        export_format="openvino",
        imgsz=640,
        half=True,
        int8=True,
    )

    with pytest.raises(ValueError, match="either half precision or int8"):
        build_export_kwargs(request)


def test_yolo_export_builds_openvino_kwargs(tmp_path):
    request = ExportRequest(
        weights=tmp_path / "best.pt",
        export_format="openvino",
        imgsz=(640, 640),
        half=True,
        device="cpu",
        project=tmp_path / "exports",
        name="best_ov",
    )

    kwargs = build_export_kwargs(request)

    assert kwargs == {
        "format": "openvino",
        "imgsz": (640, 640),
        "half": True,
        "device": "cpu",
        "project": str(tmp_path / "exports"),
        "name": "best_ov",
    }


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
