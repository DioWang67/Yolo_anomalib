import numpy as np

from core.services.missing_slot_checker import MissingSlotChecker


def test_missing_slot_checker_recovers_missing_item_from_roi_occupancy():
    image = np.full((180, 180, 3), 20, dtype=np.uint8)
    image[90:120, 110:140] = 220
    image[96:114, 116:134] = 30

    detections = [
        {
            "position_expected_key": "part_a",
            "position_offset": {"dx": -20.0, "dy": -10.0},
        },
        {
            "position_expected_key": "part_b",
            "position_offset": {"dx": -20.0, "dy": -10.0},
        },
    ]
    expected_boxes = {
        "part_a": {"x1": 105, "y1": 95, "x2": 135, "y2": 125},
        "part_b": {"x1": 145, "y1": 95, "x2": 175, "y2": 125},
        "part_c": {"x1": 130, "y1": 100, "x2": 160, "y2": 130},
    }

    checker = MissingSlotChecker({"enabled": True})
    remaining, report = checker.refine_missing_items(
        processed_image=image,
        detections=detections,
        missing_items=["part_c"],
        expected_boxes=expected_boxes,
    )

    assert remaining == []
    assert report["recovered_items"] == ["part_c"]
    assert report["estimated_shift"] == {"dx": -20.0, "dy": -10.0}


def test_missing_slot_checker_keeps_missing_item_when_roi_is_flat():
    image = np.full((180, 180, 3), 30, dtype=np.uint8)
    expected_boxes = {
        "part_c": {"x1": 110, "y1": 90, "x2": 140, "y2": 120},
    }

    checker = MissingSlotChecker({"enabled": True})
    remaining, report = checker.refine_missing_items(
        processed_image=image,
        detections=[],
        missing_items=["part_c"],
        expected_boxes=expected_boxes,
    )

    assert remaining == ["part_c"]
    assert report["recovered_items"] == []


def test_missing_slot_checker_ignores_wrong_class_overlap():
    image = np.full((180, 180, 3), 30, dtype=np.uint8)
    expected_boxes = {
        "part_c": {"x1": 110, "y1": 90, "x2": 140, "y2": 120},
    }
    detections = [
        {
            "bbox": [112, 92, 138, 118],
            "class": "part_x",
            "position_expected_key": "part_x",
        }
    ]

    checker = MissingSlotChecker({"enabled": True})
    remaining, report = checker.refine_missing_items(
        processed_image=image,
        detections=detections,
        missing_items=["part_c"],
        expected_boxes=expected_boxes,
    )

    assert remaining == ["part_c"]
    assert report["recovered_items"] == []
