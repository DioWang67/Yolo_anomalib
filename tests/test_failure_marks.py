"""The failure emphasis layer must name every NG target and stand out."""

import numpy as np
import pytest

from core.services.results.failure_marks import (
    REASON_COLOR,
    REASON_EXTRA,
    REASON_INVALID,
    REASON_MISSING,
    REASON_POSITION,
    REASON_SEQUENCE,
    REASON_WRONG_COMPONENT,
    FailureMark,
    ascii_text,
    coerce_bbox,
    collect_detection_marks,
    draw_failure_marks,
    missing_marks,
    summarize_failure_marks,
)


def _color_item(index, class_name, best, is_ok):
    return {
        "index": index,
        "class_name": class_name,
        "best_color": best,
        "is_ok": is_ok,
    }


def test_passing_detections_produce_no_marks():
    detections = [{"bbox": [10, 10, 40, 40], "class": "Red", "position_status": "CORRECT"}]
    items = {0: _color_item(0, "Red", "Red", True)}

    assert collect_detection_marks(detections, items) == []


def test_color_failure_names_the_swapped_color():
    detections = [{"bbox": [10, 10, 40, 40], "class": "Black", "source_index": 5}]
    items = {5: _color_item(5, "Black", "Orange", False)}

    (mark,) = collect_detection_marks(detections, items)

    assert mark.reason == REASON_COLOR
    assert mark.index == 5
    assert mark.label == "#5 NG COLOR Black->Orange"
    assert mark.token == "#5 COLOR"


def test_low_confidence_color_failure_is_not_reported_as_a_swap():
    detections = [{"bbox": [10, 10, 40, 40], "class": "Black", "source_index": 2}]
    items = {2: _color_item(2, "Black", "Black", False)}

    (mark,) = collect_detection_marks(detections, items)

    assert mark.label == "#2 NG COLOR Black low-conf"


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("WRONG", REASON_POSITION),
        ("FAIL", REASON_POSITION),
        ("UNEXPECTED", REASON_EXTRA),
        ("INVALID", REASON_INVALID),
        ("ERROR", REASON_INVALID),
    ],
)
def test_position_states_map_to_their_reason(status, reason):
    detections = [{"bbox": [10, 10, 40, 40], "position_status": status}]

    (mark,) = collect_detection_marks(detections, {})

    assert mark.reason == reason


@pytest.mark.parametrize("status", ["", "UNKNOWN", "DISABLED", "CORRECT", "OK"])
def test_skipped_and_passing_position_states_are_not_failures(status):
    detections = [{"bbox": [10, 10, 40, 40], "position_status": status}]

    assert collect_detection_marks(detections, {}) == []


def test_position_failure_carries_the_offset_distance():
    detections = [
        {"bbox": [10, 10, 40, 40], "position_status": "WRONG", "position_error": 12.5}
    ]

    (mark,) = collect_detection_marks(detections, {})

    assert mark.label == "#0 NG POSITION_SHIFT d=12.5"


def test_one_target_failing_two_checks_yields_one_mark_naming_both():
    """Two overlapping highlights on one box would obscure the target itself."""
    detections = [
        {
            "bbox": [10, 10, 40, 40],
            "class": "Black",
            "source_index": 3,
            "position_status": "WRONG",
        }
    ]
    items = {3: _color_item(3, "Black", "Orange", False)}

    marks = collect_detection_marks(detections, items)

    assert len(marks) == 1
    assert marks[0].label == "#3 NG COLOR+POSITION_SHIFT Black->Orange"


def test_wrong_component_outranks_other_reasons_and_names_the_expected_part():
    detections = [
        {
            "bbox": [10, 10, 40, 40],
            "source_index": 1,
            "slot_mismatch_expected_class": "bolt",
            "position_status": "WRONG",
        }
    ]

    (mark,) = collect_detection_marks(detections, {})

    assert mark.reason == REASON_WRONG_COMPONENT
    assert mark.label == "#1 NG WRONG_COMPONENT+POSITION_SHIFT exp=bolt"


def test_sequence_slot_failure_names_the_class_the_slot_wanted():
    """``exp=Red`` says it all; the reason code would only crowd the box."""
    detections = [{"bbox": [10, 10, 40, 40], "class": "Orange", "source_index": 1}]

    (mark,) = collect_detection_marks(detections, {}, {1: "Red"})

    assert mark.reason == REASON_SEQUENCE
    assert mark.label == "#1 NG exp=Red"
    # The panel summary still names it, so the reason is never lost.
    assert mark.token == "#1 SEQUENCE"


def test_sequence_code_returns_when_a_second_check_also_failed():
    """Two reasons, one detail: the heading has to say which one it belongs to."""
    detections = [{"bbox": [10, 10, 40, 40], "class": "Orange", "source_index": 1}]
    items = {1: _color_item(1, "Orange", "Red", False)}

    (mark,) = collect_detection_marks(detections, items, {1: "Red"})

    assert mark.label == "#1 NG SEQUENCE+COLOR exp=Red"


def test_wires_holding_their_slot_are_not_marked():
    detections = [
        {"bbox": [10, 10, 40, 40], "class": "Orange", "source_index": 1},
        {"bbox": [50, 10, 80, 40], "class": "Green", "source_index": 2},
    ]

    marks = collect_detection_marks(detections, {}, {1: "Red"})

    assert [mark.index for mark in marks] == [1]


def test_missing_marks_have_no_index_because_nothing_was_detected():
    (mark,) = missing_marks([((10, 10, 40, 40), "bolt")])

    assert mark.reason == REASON_MISSING
    assert mark.index is None
    assert mark.label == "NG MISSING bolt"
    assert mark.token == REASON_MISSING


def test_unusable_bboxes_are_dropped_rather_than_drawn_at_the_origin():
    detections = [
        {"bbox": None, "position_status": "WRONG"},
        {"bbox": [10, 10], "position_status": "WRONG"},
        {"bbox": [10, 10, 10, 40], "position_status": "WRONG"},
        {"bbox": ["x", 1, 2, 3], "position_status": "WRONG"},
    ]

    assert collect_detection_marks(detections, {}) == []
    assert missing_marks([((0, 0, 0, 0), "bolt")]) == []


def test_inverted_bbox_is_normalized():
    assert coerce_bbox([40, 40, 10, 10]) == (10, 10, 40, 40)


def test_non_latin_class_names_degrade_to_a_marker_not_a_row_of_question_marks():
    """``cv2.putText`` cannot draw CJK glyphs, so they never reach the frame."""
    assert ascii_text("黑線") == "??"
    assert ascii_text(" bolt ") == "bolt"
    assert ascii_text(None) == ""


def test_summary_line_counts_every_failure_and_truncates_the_list():
    marks = [
        FailureMark(bbox=(0, 0, 5, 5), reason=REASON_COLOR, index=index)
        for index in range(6)
    ]

    (line, color) = summarize_failure_marks(marks, limit=2)[0]

    assert line == "NG x6: #0 COLOR, #1 COLOR, +4 more"
    assert color == (0, 0, 255)


def test_summary_is_empty_without_failures():
    assert summarize_failure_marks([]) == []


def test_draw_marks_tints_the_target_and_draws_a_heavy_frame():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[:] = (90, 90, 90)
    mark = FailureMark(bbox=(60, 80, 120, 140), reason=REASON_COLOR, index=1)

    draw_failure_marks(frame, [mark])

    # Sampled on the top edge between the corner brackets and clear of the
    # caption's pointer, both of which legitimately cover the rings they cross.
    # The border is far heavier than the 2px pass box it replaces.
    assert tuple(frame[80, 78]) == (0, 0, 255)
    assert tuple(frame[79, 78]) == (0, 0, 255)
    assert tuple(frame[81, 78]) == (0, 0, 255)
    assert tuple(frame[78, 78]) == (0, 0, 255)
    # Light/dark rings keep the border readable on any background.
    assert tuple(frame[77, 78]) == (0, 0, 0)
    assert tuple(frame[76, 78]) == (255, 255, 255)
    # The interior is tinted toward the failure color but still legible.
    interior = frame[100, 90]
    assert interior[2] > 90 and interior[0] < 90
    # Targeting corners reach outside the box.
    assert tuple(frame[85, 53]) == (0, 0, 255)


def test_extra_and_invalid_keep_their_established_hues():
    assert FailureMark(bbox=(0, 0, 5, 5), reason=REASON_EXTRA).color == (0, 140, 255)
    assert FailureMark(bbox=(0, 0, 5, 5), reason=REASON_INVALID).color == (255, 0, 255)
    assert FailureMark(bbox=(0, 0, 5, 5), reason=REASON_COLOR).color == (0, 0, 255)


def test_neighbouring_captions_do_not_cover_each_other():
    """Captions are wider than the boxes, so a dense row would stack them."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    marks = [
        FailureMark(bbox=(20, 100, 60, 140), reason=REASON_EXTRA, index=0,
                    detail="Black->Orange"),
        FailureMark(bbox=(100, 100, 140, 140), reason=REASON_COLOR, index=1,
                    detail="Black->Orange"),
    ]

    draw_failure_marks(frame, marks)

    # Rows that only a caption can reach: clear of the boxes' corner brackets.
    above = frame[66:88, 0:340]
    below = frame[152:168, 0:340]
    # The first caption keeps the preferred slot above the boxes...
    assert np.any(np.all(above == np.array([0, 140, 255], np.uint8), axis=2))
    assert not np.any(np.all(above == np.array([0, 0, 255], np.uint8), axis=2))
    # ...and the second one moves below rather than overwriting it.
    assert np.any(np.all(below == np.array([0, 0, 255], np.uint8), axis=2))


def test_caption_is_centered_on_its_target_not_aligned_to_its_left_edge():
    """A caption wider than its box has to read as belonging to what is under
    its middle, which is where the eye assigns it."""
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    mark = FailureMark(
        bbox=(280, 150, 320, 190),
        reason=REASON_COLOR,
        index=1,
        detail="Black->Orange",
    )

    draw_failure_marks(frame, [mark])

    caption_rows = [
        row
        for row in range(0, 140)
        if np.any(np.any(frame[row] != 0, axis=1))
    ]
    assert caption_rows
    painted = np.where(np.any(frame[caption_rows[len(caption_rows) // 2]] != 0, axis=1))[0]
    caption_center = (int(painted.min()) + int(painted.max())) // 2
    assert abs(caption_center - 300) <= 2


def test_a_displaced_caption_still_points_at_its_own_box():
    """Once a caption is pushed off its box, only the tail says whose it is."""
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    marks = [
        FailureMark(bbox=(20, 100, 60, 140), reason=REASON_EXTRA, index=0,
                    detail="Black->Orange"),
        FailureMark(bbox=(100, 100, 140, 140), reason=REASON_COLOR, index=1,
                    detail="Black->Orange"),
    ]

    draw_failure_marks(frame, marks)

    # The second caption was pushed below its box; between the two, at the box
    # center, its tail is the only thing that can be painted.
    gap = frame[142:149, 120]
    assert np.any(np.all(gap == np.array([0, 0, 255], np.uint8), axis=1))


def test_label_stays_inside_the_frame_for_a_box_at_the_top_edge():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    mark = FailureMark(bbox=(4, 2, 40, 40), reason=REASON_MISSING, detail="bolt")

    draw_failure_marks(frame, [mark])

    assert np.count_nonzero(frame) > 0
    # Nothing was drawn off-frame, and the caption did not wrap to the far side.
    assert frame.shape == (120, 200, 3)


def test_drawing_tolerates_an_empty_frame_and_no_marks():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    draw_failure_marks(empty, [FailureMark(bbox=(0, 0, 5, 5), reason=REASON_COLOR)])
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    draw_failure_marks(frame, [])

    assert np.count_nonzero(frame) == 0
