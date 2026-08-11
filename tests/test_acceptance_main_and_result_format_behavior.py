from __future__ import annotations

from types import SimpleNamespace

import app.acceptance.main as acceptance_main
from core.format_result import format_detection_result
from core.types import DetectionItem, DetectionResult


def test_acceptance_main_configures_shows_and_runs_application(monkeypatch) -> None:
    app = SimpleNamespace(setApplicationName=lambda name: names.append(name), exec_=lambda: 7)
    names: list[str] = []

    class _Application:
        @staticmethod
        def instance():
            return app

    window = SimpleNamespace(show=lambda: shown.append(True))
    shown: list[bool] = []
    monkeypatch.setattr(acceptance_main, "QApplication", _Application)
    monkeypatch.setattr(acceptance_main, "project_root", lambda: ".")
    monkeypatch.setattr(acceptance_main, "ModelAcceptanceWindow", lambda **_kwargs: window)

    assert acceptance_main.main() == 7
    assert names == ["YOLO Model Acceptance"]
    assert shown == [True]


def test_result_formatter_distinguishes_errors_color_pass_fail_and_not_run() -> None:
    error = format_detection_result(
        DetectionResult(
            status="ERROR",
            product="Cable1",
            area="A",
            inference_type="yolo",
            error="backend offline",
        )
    )
    assert "backend offline" in error
    assert "檢查點" not in error

    base = DetectionResult(
        status="PASS",
        product="Cable1",
        area="A",
        inference_type="yolo",
        ckpt_path="best.pt",
        anomaly_score=0.1,
        items=[DetectionItem("LED", 0.9, (1, 2, 3, 4))],
        original_image_path="original.png",
        preprocessed_image_path="processed.png",
        heatmap_path="heat.png",
        cropped_paths=["crop.png"],
    )
    without_color = format_detection_result(base)
    assert "顏色檢測: 未執行" in without_color
    base.color_check = {"is_ok": True, "diff": 0.05}
    assert "顏色檢測: PASS" in format_detection_result(base)
    base.color_check = {"is_ok": False, "diff": 0.5}
    assert "顏色檢測: FAIL" in format_detection_result(base)
