import numpy as np

from core.services.alignment import ExpectedLayoutAlignment
from core.services.slot_roi import extract_slot_rois


def test_extract_slot_rois_uses_aligned_expected_boxes():
    image = np.zeros((220, 220, 3), dtype=np.uint8)
    image[135:165, 135:165] = 255
    expected_boxes = {
        "part_d": {"x1": 155, "y1": 145, "x2": 185, "y2": 175},
    }
    alignment = ExpectedLayoutAlignment(dx=-20.0, dy=-10.0, source_count=3)

    rois = extract_slot_rois(image, expected_boxes, alignment)

    assert len(rois) == 1
    assert rois[0].expected_key == "part_d"
    assert rois[0].class_name == "part_d"
    assert rois[0].bbox == (135, 135, 165, 165)
    assert rois[0].image.shape == (30, 30, 3)
    assert int(rois[0].image.mean()) == 255


def test_extract_slot_rois_can_add_margin():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    expected_boxes = {
        "A": {"x1": 40, "y1": 30, "x2": 50, "y2": 40},
    }
    alignment = ExpectedLayoutAlignment()

    rois = extract_slot_rois(image, expected_boxes, alignment, margin=5)

    assert rois[0].bbox == (35, 25, 55, 45)
