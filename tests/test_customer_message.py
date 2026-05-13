from core.services.results.customer_message import build_customer_message
from core.types import DetectionItem, DetectionResult


def test_customer_message_pass_is_release_ready():
    result = DetectionResult(
        status="PASS",
        items=[DetectionItem("part_a", 0.99, (0, 0, 1, 1))],
    )

    message = build_customer_message(result)

    assert message.headline == "檢測通過"
    assert "放行" in message.action
    assert message.severity == "success"


def test_customer_message_missing_item_requests_recheck():
    result = DetectionResult(
        status="FAIL",
        missing_items=["bolt"],
    )

    message = build_customer_message(result)

    assert message.headline == "發現缺件"
    assert "補件" in message.action or "重" in message.action
    assert any("bolt" in detail for detail in message.details)


def test_customer_message_fixture_shift_recommends_fixture_calibration():
    result = DetectionResult(
        status="FAIL",
        items=[
            DetectionItem(
                "a",
                0.9,
                (0, 0, 1, 1),
                metadata={
                    "position_status": "WRONG",
                    "position_error": 8.0,
                    "position_offset": {"dx": 7.5, "dy": 4.0},
                },
            ),
            DetectionItem(
                "b",
                0.9,
                (0, 0, 1, 1),
                metadata={
                    "position_status": "WRONG",
                    "position_error": 8.5,
                    "position_offset": {"dx": 7.8, "dy": 4.1},
                },
            ),
        ],
    )

    message = build_customer_message(result)

    assert message.headline == "疑似治具偏移"
    assert "治具" in message.action
    assert message.severity == "danger"


def test_customer_message_surfaces_slot_check_recovery():
    result = DetectionResult(
        status="PASS",
        metadata={
            "slot_check": {
                "recovered_items": ["bolt"],
                "remaining_missing_items": [],
            }
        },
    )

    message = build_customer_message(result)

    assert message.severity == "warning"
    assert any("bolt" in detail for detail in message.details)


def test_customer_message_reports_slot_mismatch():
    result = DetectionResult(
        status="FAIL",
        metadata={
            "slot_mismatches": [
                {
                    "expected_key": "E",
                    "expected_class": "E",
                    "detected_class": "A",
                }
            ]
        },
    )

    message = build_customer_message(result)

    assert message.headline == "元件類別不符"
    assert "錯料" in message.action or "分類" in message.action
    assert any("E" in detail and "A" in detail for detail in message.details)
