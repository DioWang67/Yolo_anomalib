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
    assert "補件" in message.action or "供料" in message.action
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
