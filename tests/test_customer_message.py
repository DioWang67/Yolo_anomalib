from core.services.results.customer_message import build_customer_message
from core.types import DetectionItem, DetectionResult


def test_customer_message_pass_uses_sop_guidance():
    result = DetectionResult(
        status="PASS",
        items=[DetectionItem("part_a", 0.99, (0, 0, 1, 1))],
    )

    message = build_customer_message(result)

    assert message.headline == "檢測通過"
    assert message.action == "等待下一次檢測"
    assert "放行" not in message.action
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
    assert "等待下一次檢測" in message.action
    assert "放行" not in message.action
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


def test_customer_message_reports_board_alignment_failure():
    result = DetectionResult(
        status="DETECTION_FAIL",
        metadata={
            "decision": {"reasons": ["BOARD_ALIGNMENT"]},
            "alignment_quality": {
                "enabled": True,
                "is_ok": False,
                "issues": ["alignment_shift_out_of_range"],
                "dx": 12.0,
                "dy": -3.0,
                "observed_source_count": 2,
                "required_source_count": 2,
            },
        },
    )

    message = build_customer_message(result)

    assert message.headline == "板件對位異常"
    assert "治具" in message.action
    assert message.severity == "danger"
    assert any("整板偏移超出範圍" in detail for detail in message.details)
