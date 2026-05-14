from core.services.alignment import (
    ExpectedLayoutAlignment,
    base_class_name,
    build_aligned_expected_boxes,
    extract_layout_alignment,
    resolve_missing_expected_keys,
)


def test_extract_layout_alignment_uses_median_shift():
    detections = [
        {"position_offset": {"dx": -20.0, "dy": -10.0}},
        {"position_offset": {"dx": -22.0, "dy": -12.0}},
        {"position_offset": {"dx": -18.0, "dy": -8.0}},
    ]

    alignment = extract_layout_alignment(detections)

    assert alignment.dx == -20.0
    assert alignment.dy == -10.0
    assert alignment.source_count == 3


def test_alignment_shift_box_applies_translation():
    alignment = ExpectedLayoutAlignment(dx=-20.0, dy=-10.0, source_count=2)

    assert alignment.shift_box({"x1": 155, "y1": 145, "x2": 185, "y2": 175}) == (
        135,
        135,
        165,
        165,
    )


def test_resolve_missing_expected_keys_supports_multi_instance():
    missing_items = ["LED", "LED", "R"]
    expected_boxes = {
        "LED#0": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        "LED#1": {"x1": 20, "y1": 0, "x2": 30, "y2": 10},
        "R": {"x1": 40, "y1": 0, "x2": 50, "y2": 10},
    }

    keys = resolve_missing_expected_keys(missing_items, expected_boxes, used_expected_keys=set())

    assert keys == ["LED#0", "LED#1", "R"]
    assert base_class_name("LED#1") == "LED"


def test_build_aligned_expected_boxes_shifts_box_and_center():
    alignment = ExpectedLayoutAlignment(dx=-20.0, dy=-10.0, source_count=2)
    expected_boxes = {
        "part_d": {"x1": 155, "y1": 145, "x2": 185, "y2": 175, "cx": 170, "cy": 160}
    }

    aligned = build_aligned_expected_boxes(expected_boxes, alignment)

    assert aligned["part_d"] == {
        "x1": 135.0,
        "y1": 135.0,
        "x2": 165.0,
        "y2": 165.0,
        "cx": 150.0,
        "cy": 150.0,
    }
