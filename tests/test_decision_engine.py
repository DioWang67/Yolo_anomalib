from core.services.decision_engine import (
    InspectionDecisionEngine,
    InspectionReason,
    InspectionStatus,
    collect_fail_reasons,
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


def test_decision_engine_reports_board_alignment_failure():
    decision = InspectionDecisionEngine().evaluate(
        detections=[],
        missing_items=[],
        unexpected_items=[],
        alignment_quality={
            "enabled": True,
            "is_ok": False,
            "issues": ["alignment_shift_out_of_range"],
            "dx": 18.0,
            "dy": 0.0,
        },
    )

    assert decision.status == InspectionStatus.FAIL
    assert decision.reasons == [InspectionReason.BOARD_ALIGNMENT]
    assert decision.details[0]["items"]["issues"] == ["alignment_shift_out_of_range"]


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


def test_collect_fail_reasons_empty_for_pass():
    assert collect_fail_reasons(
        status="PASS",
        decision={"status": "PASS", "reasons": []},
        color_result={"is_ok": False},
        detector="yolo",
    ) == []


def test_collect_fail_reasons_merges_decision_color_and_sequence():
    reasons = collect_fail_reasons(
        status="DETECTION_FAIL",
        decision={"status": "FAIL", "reasons": ["MISSING", "MISSING"]},
        color_result={"is_ok": False},
        sequence_check={"is_ok": False, "reason": "order_mismatch"},
        detector="yolo",
    )

    assert reasons == ["MISSING", "COLOR_MISMATCH", "SEQUENCE_MISMATCH"]


def test_collect_fail_reasons_anomalib_fail_maps_to_anomaly_detected():
    reasons = collect_fail_reasons(
        status="DETECTION_FAIL",
        detector="anomalib",
        anomaly_score=0.87,
    )

    assert reasons == [InspectionReason.ANOMALY_DETECTED.value]


def test_collect_fail_reasons_fusion_attributes_anomaly_only_without_other_signals():
    unexplained = collect_fail_reasons(
        status="DETECTION_FAIL",
        detector="fusion",
        anomaly_score=0.9,
    )
    explained = collect_fail_reasons(
        status="DETECTION_FAIL",
        decision={"status": "FAIL", "reasons": ["MISSING"]},
        detector="fusion",
        anomaly_score=0.9,
    )

    assert unexplained == [InspectionReason.ANOMALY_DETECTED.value]
    assert explained == ["MISSING"]


def test_collect_fail_reasons_inference_error_wins_over_anomaly():
    reasons = collect_fail_reasons(
        status="INFERENCE_ERROR",
        detector="anomalib",
        anomaly_score=None,
        error_message="model load failed",
    )

    assert reasons == [InspectionReason.INFERENCE_ERROR.value]
