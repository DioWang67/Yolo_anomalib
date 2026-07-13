from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI display helpers")

from app.gui.widgets import _color_item_label
from core.models import ColorCheckItemResult
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


def test_customer_message_uses_legacy_class_alias_for_color_failure():
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

    assert message.details == ["異常項目: Red"]
    assert "?" not in " ".join(message.details)


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

    assert message.details == ["異常項目: 未偵測到元件（全畫面檢查）"]
