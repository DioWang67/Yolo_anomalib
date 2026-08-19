from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI display helpers")

from app.gui.widgets import _color_check_failure_text, _color_item_label
from core.models import ColorCheckItemResult
from core.services.results.annotations import _color_item_overlay_text
from core.services.results.customer_message import build_customer_message
from core.types import DetectionResult


def test_color_item_serialization_keeps_explicit_and_legacy_class_keys():
    item = ColorCheckItemResult(
        index=0,
        class_name="Red",
        bbox=[0, 0, 10, 10],
        best_color="Orange",
        diff=0.5,
        threshold=0.4,
        is_ok=False,
    )

    serialized = item.to_dict()

    assert serialized["class_name"] == "Red"
    assert serialized["class"] == "Red"


@pytest.mark.parametrize("key", ["class_name", "class"])
def test_color_item_label_accepts_both_class_keys(key: str):
    assert _color_item_label({key: "Red"}, "zh") == "Red"


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("zh", "未偵測到元件（全畫面檢查）"),
        ("en", "No detected component (full-frame check)"),
    ],
)
def test_color_item_label_localizes_full_frame(language: str, expected: str):
    assert _color_item_label({"index": -1}, language) == expected


@pytest.mark.parametrize(
    ("language", "expected"),
    [("zh", "未知項目"), ("en", "Unknown item")],
)
def test_color_item_label_localizes_unknown_item(language: str, expected: str):
    assert _color_item_label({}, language) == expected


def test_customer_message_describes_color_mismatch_with_both_colors():
    """Red -> Orange NG must name both colors, not just the detected class."""
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check={
            "is_ok": False,
            "items": [
                {
                    "index": 0,
                    "class": "Red",
                    "best_color": "Orange",
                    "is_ok": False,
                }
            ],
        },
    )

    message = build_customer_message(result)

    assert message.details == ["顏色不符: Red → Orange"]
    assert "?" not in " ".join(message.details)


def test_customer_message_describes_low_confidence_without_repeating_the_class():
    """Red -> Red NG (score below threshold) must not read as 'Red -> Red'."""
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check={
            "is_ok": False,
            "items": [
                {
                    "index": 0,
                    "class": "Red",
                    "best_color": "Red",
                    "is_ok": False,
                }
            ],
        },
    )

    message = build_customer_message(result)

    assert message.details == ["Red 顏色信心不足"]
    assert "→" not in " ".join(message.details)


def test_customer_message_labels_full_frame_color_failure():
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check={
            "is_ok": False,
            "items": [
                {
                    "index": -1,
                    "class": None,
                    "best_color": "Red",
                    "is_ok": False,
                }
            ],
        },
    )

    message = build_customer_message(result)

    assert message.details == ["未偵測到元件（全畫面檢查）"]


def test_customer_message_pass_has_no_color_failure_wording():
    result = DetectionResult(
        status="PASS",
        color_check={
            "is_ok": True,
            "items": [
                {"index": 0, "class": "Red", "best_color": "Red", "is_ok": True}
            ],
        },
    )

    message = build_customer_message(result)

    joined = " ".join(message.details)
    assert "顏色不符" not in joined
    assert "顏色信心不足" not in joined
    assert message.severity == "success"


def test_customer_message_joins_multiple_color_failures_distinctly():
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check={
            "is_ok": False,
            "items": [
                {"index": 0, "class": "Red", "best_color": "Orange", "is_ok": False},
                {"index": 1, "class": "Green", "best_color": "Green", "is_ok": False},
            ],
        },
    )

    message = build_customer_message(result)

    assert message.details == ["顏色不符: Red → Orange; Green 顏色信心不足"]


def _duplicate_incident_color_check() -> dict:
    """One orange wire seen twice; the box called ``Red`` failed and was removed.

    Replays Cable1/A 2026-08-19: ``#5`` carried class ``Red`` against a measured
    Orange and was suppressed as a duplicate of ``#6``, while a surviving Black
    box separately missed its own threshold.
    """
    return {
        "is_ok": False,
        "items": [
            {"index": 4, "class": "Black", "best_color": "Black", "is_ok": False},
            {"index": 5, "class": "Red", "best_color": "Orange", "is_ok": False},
            {"index": 6, "class": "Orange", "best_color": "Orange", "is_ok": True},
        ],
    }


def test_customer_message_omits_a_suppressed_duplicates_color_failure():
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check=_duplicate_incident_color_check(),
        metadata={
            "duplicate_filter": {
                "status": "suppressed",
                "suppressions": [{"suppressed_index": 5, "kept_index": 6}],
            }
        },
    )

    message = build_customer_message(result)

    # The surviving Black still needs operator attention ...
    assert message.details == ["Black 顏色信心不足"]
    # ... but the removed box's mismatch reports the detector, not the board.
    assert "Red → Orange" not in " ".join(message.details)


def test_customer_message_keeps_color_failures_that_were_only_proposed():
    """Report-only mode leaves the box on the board, so it still counts."""
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check=_duplicate_incident_color_check(),
        metadata={
            "duplicate_filter": {
                "status": "reported",
                "suppressions": [],
                "proposed_suppressions": [
                    {"suppressed_index": 5, "kept_index": 6}
                ],
            }
        },
    )

    message = build_customer_message(result)

    assert message.details == ["Black 顏色信心不足; 顏色不符: Red → Orange"]


def test_customer_message_keeps_full_frame_failure_when_boxes_were_suppressed():
    """A ``-1`` item is not a box, so suppression must never filter it away."""
    result = DetectionResult(
        status="DETECTION_FAIL",
        color_check={
            "is_ok": False,
            "items": [
                {"index": -1, "class": None, "best_color": "Red", "is_ok": False}
            ],
        },
        metadata={
            "duplicate_filter": {
                "status": "suppressed",
                "suppressions": [{"suppressed_index": 5, "kept_index": 6}],
            }
        },
    )

    message = build_customer_message(result)

    assert message.details == ["未偵測到元件（全畫面檢查）"]


def test_classify_color_check_failure_is_case_insensitive():
    from core.services.results.customer_message import (
        COLOR_FAILURE_LOW_CONFIDENCE,
        COLOR_FAILURE_MISMATCH,
        classify_color_check_failure,
    )

    same_case = classify_color_check_failure({"class": "red", "best_color": "RED"})
    assert same_case.kind == COLOR_FAILURE_LOW_CONFIDENCE

    different = classify_color_check_failure({"class": "Red", "best_color": "Orange"})
    assert different.kind == COLOR_FAILURE_MISMATCH
    assert different.class_name == "Red"
    assert different.predicted_color == "Orange"


# --- widgets.py: detail panel + one-line fail banner share the same wording ---


def test_widgets_color_failure_text_describes_mismatch():
    item = {"class": "Red", "best_color": "Orange", "is_ok": False}
    assert _color_check_failure_text(item, "zh") == "顏色不符: Red → Orange"


def test_widgets_color_failure_text_describes_low_confidence():
    item = {"class": "Red", "best_color": "Red", "is_ok": False}
    text = _color_check_failure_text(item, "zh")
    assert text == "Red 顏色信心不足"
    assert "→" not in text


def test_widgets_color_failure_text_is_localized_for_english():
    item = {"class": "Red", "best_color": "Orange", "is_ok": False}
    assert _color_check_failure_text(item, "en") == "Color mismatch: Red → Orange"

    same = {"class": "Red", "best_color": "Red", "is_ok": False}
    assert _color_check_failure_text(same, "en") == "Red color confidence too low"


# --- annotations.py: image overlay stays ASCII-safe but keeps the distinction --


def test_annotations_overlay_text_describes_mismatch_in_ascii():
    item = {"class_name": "Red", "best_color": "Orange", "is_ok": False}
    text = _color_item_overlay_text(item, is_ok=False)
    assert text == "Red -> Orange (mismatch)"
    assert text.isascii()


def test_annotations_overlay_text_describes_low_confidence_in_ascii():
    item = {"class_name": "Red", "best_color": "Red", "is_ok": False}
    text = _color_item_overlay_text(item, is_ok=False)
    assert text == "Red (low confidence)"
    assert text.isascii()


def test_annotations_overlay_text_unchanged_for_passing_items():
    item = {"class_name": "Red", "best_color": "Red", "is_ok": True}
    assert _color_item_overlay_text(item, is_ok=True) == "Red -> Red"
