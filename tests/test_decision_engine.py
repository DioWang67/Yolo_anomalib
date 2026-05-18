from core.services.decision_engine import (
    InspectionDecisionEngine,
    InspectionReason,
    InspectionStatus,
)


def test_decision_engine_passes_when_no_validation_failures():
    decision = InspectionDecisionEngine().evaluate(
        detections=[{"class": "R101", "position_status": "CORRECT"}],
        missing_items=[],
        unexpected_items=[],
        slot_mismatches=[],
    )

    assert decision.status == InspectionStatus.PASS
    assert decision.reasons == []


def test_decision_engine_reports_missing_and_wrong_component():
    decision = InspectionDecisionEngine().evaluate(
        detections=[],
        missing_items=["R101"],
        unexpected_items=[],
        slot_mismatches=[
            {
                "expected_key": "C205",
                "expected_class": "C205",
                "detected_class": "R999",
            }
        ],
    )

    assert decision.status == InspectionStatus.FAIL
    assert decision.reasons == [
        InspectionReason.MISSING,
        InspectionReason.WRONG_COMPONENT,
    ]
    assert decision.to_dict()["reasons"] == ["MISSING", "WRONG_COMPONENT"]


def test_decision_engine_reports_position_shift():
    decision = InspectionDecisionEngine().evaluate(
        detections=[
            {
                "class": "U3",
                "position_status": "WRONG",
                "position_error": 18.5,
                "position_tolerance_px": 12.0,
                "position_offset": {"dx": 18.0, "dy": 4.0},
            }
        ],
        missing_items=[],
        unexpected_items=[],
        slot_mismatches=[],
    )

    assert decision.status == InspectionStatus.FAIL
    assert decision.reasons == [InspectionReason.POSITION_SHIFT]
    assert decision.details[0]["items"][0]["class"] == "U3"


def test_decision_engine_respects_unexpected_fail_policy():
    strict_decision = InspectionDecisionEngine(fail_on_unexpected=True).evaluate(
        unexpected_items=["UNKNOWN_PART"]
    )
    lenient_decision = InspectionDecisionEngine(fail_on_unexpected=False).evaluate(
        unexpected_items=["UNKNOWN_PART"]
    )

    assert strict_decision.status == InspectionStatus.FAIL
    assert strict_decision.reasons == [InspectionReason.UNEXPECTED_COMPONENT]
    assert lenient_decision.status == InspectionStatus.PASS
    assert lenient_decision.reasons == []
