"""The overlay ``LR:`` line and ``sequence_check.observed`` must never diverge.

They used to: the panel rendered the color checker's raw ``best_color`` while
the decision used the fail-closed ``verified_class``, so a rejected board showed
``Red`` in the very slot the log reported as ``missing=['Red']``.
"""

import numpy as np
import pytest
from test_annotations import FakeImageUtils

from core.pipeline.context import DetectionContext
from core.pipeline.steps import SequenceCheckStep
from core.services.detection_sequence import effective_class_name, left_right_sequence
from core.services.results.annotations import annotate_yolo_frame


class _StubLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _cable_detections() -> list[dict]:
    """Six wires; #5 is a red wire the detector labelled ``Orange``."""
    return [
        {"bbox": [290, 370, 310, 395], "class": "Yellow", "class_id": 0,
         "verified_class": "Yellow", "source_index": 0},
        {"bbox": [191, 365, 212, 390], "class": "Orange", "class_id": 1,
         "verified_class": "Orange", "source_index": 1},
        {"bbox": [354, 375, 374, 400], "class": "Black", "class_id": 2,
         "verified_class": "Black", "source_index": 2},
        {"bbox": [321, 372, 342, 397], "class": "Black", "class_id": 2,
         "verified_class": "Black", "source_index": 3},
        {"bbox": [222, 363, 242, 388], "class": "Green", "class_id": 3,
         "verified_class": "Green", "source_index": 4},
        # ColorCheckStep rejected the Red match, so verified_class falls back
        # to the detector class rather than promoting an unverified color.
        {"bbox": [253, 368, 273, 393], "class": "Orange", "class_id": 1,
         "verified_class": "Orange", "source_index": 5},
    ]


def _cable_color_result() -> dict:
    def item(index, class_name, best, diff, thr, is_ok):
        return {"index": index, "class_name": class_name, "best_color": best,
                "diff": diff, "threshold": thr, "is_ok": is_ok}

    return {
        "is_ok": False,
        "status": "evaluated",
        "items": [
            item(0, "Yellow", "Yellow", 0.61, 0.80, True),
            item(1, "Orange", "Orange", 0.33, 0.75, True),
            item(2, "Black", "Black", 0.57, 0.60, True),
            item(3, "Black", "Black", 0.41, 0.60, True),
            item(4, "Green", "Green", 0.46, 0.70, True),
            item(5, "Orange", "Red", 0.23, 0.75, False),
        ],
    }


def _panel_lines(monkeypatch, detections, color_result) -> list[str]:
    captured: list[tuple[str, tuple[int, int, int]]] = []
    monkeypatch.setattr(
        "core.services.results.annotations._draw_info_panel",
        lambda _frame, lines, origin: captured.extend(lines),
    )
    annotate_yolo_frame(
        FakeImageUtils(),
        np.zeros((420, 420, 3), dtype=np.uint8),
        detections,
        color_result,
        "DETECTION_FAIL",
    )
    return [line for line, _ in captured]


def _run_sequence_check(detections, expected) -> dict:
    ctx = DetectionContext(
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=np.zeros((420, 420, 3), dtype=np.uint8),
        processed_image=np.zeros((420, 420, 3), dtype=np.uint8),
        result={"detections": detections},
        status="PASS",
    )
    SequenceCheckStep(
        _StubLogger(), "Cable1", "A", options={"expected": expected}
    ).run(ctx)
    return ctx.result["sequence_check"]


def test_overlay_lr_line_matches_sequence_check_observed(monkeypatch):
    detections = _cable_detections()
    expected = ["Red", "Green", "Orange", "Yellow", "Black", "Black"]

    observed = _run_sequence_check(detections, expected)["observed"]
    lr_lines = [
        line for line in _panel_lines(monkeypatch, detections, _cable_color_result())
        if line.startswith("LR: ")
    ]

    assert len(lr_lines) == 1
    assert lr_lines[0] == "LR: " + " -> ".join(observed)
    assert observed == ["Orange", "Green", "Orange", "Yellow", "Black", "Black"]


def test_overlay_still_reports_the_color_checker_opinion(monkeypatch):
    """Collapsing LR onto the decision must not hide the rejected match."""
    lines = _panel_lines(monkeypatch, _cable_detections(), _cable_color_result())

    assert any("#5 Orange -> Red" in line for line in lines)


def test_left_right_sequence_orders_by_box_center():
    detections = [
        {"bbox": [300, 0, 320, 10], "class": "Black"},
        {"bbox": [100, 0, 120, 10], "class": "Orange"},
        {"bbox": [200, 0, 220, 10], "class": "Green"},
    ]

    assert left_right_sequence(detections) == ["Orange", "Green", "Black"]


def test_left_right_sequence_prefers_verified_class():
    detections = [{"bbox": [0, 0, 10, 10], "class": "Orange", "verified_class": "Red"}]

    assert left_right_sequence(detections) == ["Red"]


@pytest.mark.parametrize(
    "detection",
    [
        {"bbox": None, "class": "Red"},
        {"bbox": [0, 0], "class": "Red"},
        {"bbox": ["x", 0, 10, 10], "class": "Red"},
        {"bbox": [0, 0, 10, 10], "class": "   "},
        {"bbox": [0, 0, 10, 10]},
    ],
)
def test_left_right_sequence_skips_unusable_detections(detection):
    assert left_right_sequence([detection]) == []


def test_effective_class_name_falls_back_to_detector_class():
    assert effective_class_name({"class": "Orange", "verified_class": ""}) == "Orange"
    assert effective_class_name({"class": "Orange"}) == "Orange"
    assert effective_class_name({}) == ""
