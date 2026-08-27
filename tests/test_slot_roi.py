import numpy as np
import pytest

from core.services.alignment import ExpectedLayoutAlignment
from core.services.slot_roi import (
    ColorRoiPolicy,
    extract_bbox_roi,
    extract_slot_rois,
)


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


def test_color_roi_policy_insets_only_the_configured_axis():
    image = np.arange(40 * 20 * 3, dtype=np.int32).reshape(40, 20, 3)
    policy = ColorRoiPolicy.from_mapping(
        {"inset_x_ratio": 0.2, "inset_y_ratio": 0.0, "min_size": 8}
    )

    roi = extract_bbox_roi(image, [0, 0, 20, 40], policy=policy)

    assert roi is not None
    assert roi.shape == (40, 12, 3)
    assert np.array_equal(roi, image[:, 4:16])


@pytest.mark.parametrize(
    "value",
    (
        {"inset_x_ratio": -0.1},
        {"inset_y_ratio": 0.5},
        {"min_size": 0},
        {"unknown": 1},
    ),
)
def test_color_roi_policy_rejects_invalid_external_configuration(value):
    with pytest.raises((TypeError, ValueError)):
        ColorRoiPolicy.from_mapping(value)


def test_color_roi_policy_fails_closed_when_inset_is_too_small():
    image = np.zeros((20, 10, 3), dtype=np.uint8)
    policy = ColorRoiPolicy(inset_x_ratio=0.2, min_size=8)

    assert extract_bbox_roi(image, [0, 0, 10, 20], policy=policy) is None
