from core.types import DetectionResult


def test_detection_result_to_dict_includes_unexpected_items_and_metadata():
    result = DetectionResult(
        status="FAIL",
        product="PCBA",
        area="TOP",
        unexpected_items=["UNKNOWN_PART"],
        metadata={"decision": {"status": "FAIL", "reasons": ["UNEXPECTED_COMPONENT"]}},
    )

    payload = result.to_dict()

    assert payload["unexpected_items"] == ["UNKNOWN_PART"]
    assert payload["metadata"]["decision"]["reasons"] == ["UNEXPECTED_COMPONENT"]
