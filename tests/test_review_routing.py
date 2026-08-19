import json

import pytest

from tools.review_routing import (
    ReviewDecision,
    action_route,
    color_failure_items,
    color_result,
    color_summary,
    has_color_failure,
    has_non_color_failure,
    has_position_failure,
    has_position_only_failure,
    has_threshold_color_failure,
)


@pytest.mark.parametrize(
    ("label", "route", "detection"),
    [
        ("confirmed_ng", "yolo", "correct"),
        ("confirmed_ok", "none", "correct"),
        ("verified_empty", "yolo", "correct"),
        ("false_positive", "yolo", "false_positive"),
        ("false_negative", "yolo", "missed"),
        ("wrong_box", "yolo", "wrong_box"),
        ("wrong_class", "yolo", "wrong_class"),
        ("image_quality_issue", "none", "unjudgeable"),
        ("position_false_reject", "position", "correct"),
    ],
)
def test_legacy_decisions_map_to_structured_contract(label, route, detection):
    decision = ReviewDecision.from_legacy_label(label)

    assert decision.action_route == route
    assert decision.detection_verdict == detection
    assert decision.to_columns()["review_label"] == label


def test_color_decision_routes_box_correction_to_both_pipelines():
    decision = ReviewDecision.for_color(
        "actually_ok", detection_verdict="wrong_box", has_non_color_failure=True
    )

    assert decision.review_label == "color_false_reject"
    assert decision.product_verdict == "ng"
    assert decision.color_verdict == "actually_ok"
    assert decision.action_route == "both"


def test_color_decision_keeps_correct_boxes_out_of_yolo():
    decision = ReviewDecision.for_color("confirmed_ng")

    assert decision.review_label == "color_confirmed_ng"
    assert decision.product_verdict == "ng"
    assert decision.action_route == "color"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ReviewDecision("x", "ok", "correct", "not_applicable", "bad"),
        lambda: ReviewDecision("x", "ok", "correct", "bad", "none"),
        lambda: ReviewDecision("x", "ok", "bad", "not_applicable", "none"),
        lambda: ReviewDecision("x", "bad", "correct", "not_applicable", "none"),
        lambda: ReviewDecision.from_legacy_label("unknown"),
        lambda: ReviewDecision.for_color("unknown"),
        lambda: ReviewDecision.for_color("actually_ok", detection_verdict="missed"),
    ],
)
def test_invalid_decisions_fail_closed(factory):
    with pytest.raises(ValueError):
        factory()


def test_action_route_validates_new_contract_and_falls_back_for_legacy():
    assert action_route({"action_route": "both"}) == "both"
    assert action_route({"review_label": "wrong_box"}) == "yolo"
    assert action_route(
        {
            "review_label": "color_false_reject",
            "detection_verdict": "wrong_class",
        }
    ) == "both"
    assert action_route({"review_label": "color_false_reject"}) == "color"
    assert action_route({"review_label": "position_false_reject"}) == "position"
    assert action_route({"review_label": "unknown"}) == "none"


def test_position_routing_requires_an_exclusive_position_failure():
    assert has_position_failure({"decision_reasons": "POSITION_SHIFT"}) is True
    assert (
        has_position_only_failure({"decision_reasons": "POSITION_SHIFT"}) is True
    )
    assert (
        has_position_only_failure(
            {"decision_reasons": "POSITION_SHIFT|MISSING"}
        )
        is False
    )


def test_color_failure_parsing_and_operator_summary():
    row = {
        "decision_reasons": "MISSING|COLOR_MISMATCH",
        "color_result_json": json.dumps(
            {
                "is_ok": False,
                "items": [
                    {
                        "class": "Red",
                        "best_color": "Orange",
                        "diff": 0.55,
                        "threshold": 0.4,
                        "is_ok": False,
                    },
                    {"class": "Green", "is_ok": True},
                ],
            }
        ),
    }

    assert has_color_failure(row) is True
    assert has_non_color_failure(row) is True
    assert len(color_failure_items(row)) == 1
    assert has_threshold_color_failure(row) is True
    assert "Red → Orange" in color_summary(row)
    assert "diff 0.550 > 門檻 0.400" in color_summary(row)


def test_color_failure_supports_count_and_result_fallbacks():
    assert has_color_failure({"color_failure_count": "1"}) is True
    assert has_color_failure({"color_failure_count": "invalid"}) is False
    assert has_color_failure({"color_result_json": '{"is_ok": false}'}) is True
    assert has_color_failure({"color_result_json": "not-json"}) is False
    assert color_result({"color_result_json": {"is_ok": True}}) == {"is_ok": True}
    assert color_failure_items({}) == []


def test_color_summary_handles_incomplete_legacy_values():
    row = {
        "color_result_json": json.dumps(
            {"items": [{"best_color": "Red", "is_ok": False}]}
        )
    }

    assert "分數資料不完整" in color_summary(row)


def test_rule_failure_is_not_misreported_as_threshold_failure():
    row = {
        "color_result_json": json.dumps(
            {
                "items": [
                    {
                        "best_color": "White",
                        "diff": 0.2,
                        "threshold": 0.3,
                        "is_ok": False,
                    }
                ]
            }
        )
    }

    assert has_threshold_color_failure(row) is False
    assert "另有顏色規則未過" in color_summary(row)
