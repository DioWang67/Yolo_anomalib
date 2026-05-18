import cv2
import numpy as np

from core.services.results.annotations import annotate_yolo_frame


class FakeImageUtils:
    """Minimal label drawer for annotation tests."""

    def draw_label(self, frame, text, pos, color, font_scale=1.0, thickness=1):
        cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def test_annotate_yolo_frame_draws_expected_position_overlay():
    frame = np.zeros((220, 220, 3), dtype=np.uint8)
    detections = [
        {
            "bbox": [140, 144, 170, 174],
            "class": "part_a",
            "class_id": 0,
            "confidence": 0.98,
            "position_status": "WRONG",
            "position_error": 12.5,
            "position_offset": {"dx": 8.0, "dy": 10.0},
            "position_expected_center": {"cx": 142.0, "cy": 144.0},
            "position_expected_box": {"x1": 132.0, "y1": 134.0, "x2": 152.0, "y2": 154.0},
        }
    ]

    annotate_yolo_frame(FakeImageUtils(), frame, detections, None, "FAIL")

    assert tuple(frame[134, 132]) == (170, 170, 170)
    center_patch = frame[138:150, 136:148]
    assert np.any(np.all(center_patch == np.array([255, 255, 0], dtype=np.uint8), axis=2))
    assert np.count_nonzero(frame) > 0


def test_annotate_yolo_frame_uses_position_status_color_for_detection_box():
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    detections = [
        {
            "bbox": [100, 100, 130, 130],
            "class": "part_b",
            "class_id": 1,
            "confidence": 0.88,
            "position_status": "UNEXPECTED",
        }
    ]

    annotate_yolo_frame(FakeImageUtils(), frame, detections, None, "FAIL")

    assert tuple(frame[100, 100]) == (0, 140, 255)


def test_annotate_yolo_frame_draws_missing_expected_box_without_detection():
    frame = np.zeros((180, 180, 3), dtype=np.uint8)

    annotate_yolo_frame(
        FakeImageUtils(),
        frame,
        detections=[],
        color_result=None,
        status="FAIL",
        missing_items=["bolt"],
        expected_boxes={
            "bolt": {"x1": 120.0, "y1": 80.0, "x2": 150.0, "y2": 110.0},
        },
    )

    assert tuple(frame[80, 120]) == (0, 0, 255)


def test_annotate_yolo_frame_shifts_missing_expected_box_by_detected_offset():
    frame = np.zeros((220, 220, 3), dtype=np.uint8)
    detections = [
        {
            "bbox": [100, 100, 130, 130],
            "class": "part_a",
            "class_id": 0,
            "confidence": 0.95,
            "position_expected_key": "part_a",
            "position_offset": {"dx": -20.0, "dy": -10.0},
            "position_expected_center": {"cx": 135.0, "cy": 125.0},
        },
        {
            "bbox": [150, 100, 180, 130],
            "class": "part_b",
            "class_id": 1,
            "confidence": 0.95,
            "position_expected_key": "part_b",
            "position_offset": {"dx": -20.0, "dy": -10.0},
            "position_expected_center": {"cx": 185.0, "cy": 125.0},
        },
        {
            "bbox": [100, 150, 130, 180],
            "class": "part_c",
            "class_id": 2,
            "confidence": 0.95,
            "position_expected_key": "part_c",
            "position_offset": {"dx": -20.0, "dy": -10.0},
            "position_expected_center": {"cx": 135.0, "cy": 175.0},
        },
    ]

    annotate_yolo_frame(
        FakeImageUtils(),
        frame,
        detections=detections,
        color_result=None,
        status="FAIL",
        missing_items=["part_d"],
        expected_boxes={
            "part_a": {"x1": 105.0, "y1": 95.0, "x2": 135.0, "y2": 125.0},
            "part_b": {"x1": 155.0, "y1": 95.0, "x2": 185.0, "y2": 125.0},
            "part_c": {"x1": 105.0, "y1": 145.0, "x2": 135.0, "y2": 175.0},
            "part_d": {"x1": 155.0, "y1": 145.0, "x2": 185.0, "y2": 175.0},
        },
    )

    assert tuple(frame[135, 135]) == (0, 0, 255)


def test_annotate_yolo_frame_prefers_missing_locations_over_expected_boxes():
    frame = np.zeros((180, 180, 3), dtype=np.uint8)

    annotate_yolo_frame(
        FakeImageUtils(),
        frame,
        detections=[],
        color_result=None,
        status="FAIL",
        missing_items=["part_d"],
        expected_boxes={
            "part_d": {"x1": 110.0, "y1": 110.0, "x2": 140.0, "y2": 140.0},
        },
        missing_locations=[
            {
                "class": "part_d",
                "expected_key": "part_d",
                "bbox": [50, 50, 80, 80],
                "reason": "missing",
            }
        ],
    )

    assert tuple(frame[50, 50]) == (0, 0, 255)
    assert tuple(frame[110, 110]) == (0, 0, 0)
