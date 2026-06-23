from types import SimpleNamespace

from core.pipeline.finalize import finalize_status


def _ctx(status, result, color=None):
    return SimpleNamespace(status=status, result=result, color_result=color)


def test_color_corrected_goodboard_clears_stale_fail():
    # YOLO mislabeled green->orange so the YOLO stage set FAIL (missing green);
    # color check corrected the class so count now reports no missing.
    ctx = _ctx(
        "DETECTION_FAIL",
        {"missing_items": [], "unexpected_items": [], "sequence_check": {"is_ok": True}},
        color={"is_ok": True},
    )
    finalize_status(ctx)
    assert ctx.status == "PASS"


def test_sequence_mismatch_fails():
    ctx = _ctx(
        "PASS",
        {"missing_items": [], "sequence_check": {"is_ok": False}},
        color={"is_ok": True},
    )
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


def test_missing_item_fails():
    ctx = _ctx("PASS", {"missing_items": ["Green"]}, color={"is_ok": True})
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


def test_color_failure_fails():
    ctx = _ctx("PASS", {"missing_items": []}, color={"is_ok": False})
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


def test_unexpected_component_fails_when_configured():
    # PCBA-style: no color result, an unexpected component present.
    ctx = _ctx("PASS", {"missing_items": [], "unexpected_items": ["Resistor"]})
    finalize_status(ctx, fail_on_unexpected=True)
    assert ctx.status == "DETECTION_FAIL"


def test_slot_mismatch_fails():
    ctx = _ctx(
        "PASS",
        {"missing_items": [], "slot_mismatches": [{"expected_key": "R1", "detected_class": "C1"}]},
    )
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


def test_position_wrong_fails():
    ctx = _ctx(
        "PASS",
        {"missing_items": [], "detections": [{"class": "R1", "position_status": "WRONG"}]},
    )
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


def test_clean_board_passes():
    ctx = _ctx(
        "PASS",
        {
            "missing_items": [],
            "unexpected_items": [],
            "slot_mismatches": [],
            "sequence_check": {"is_ok": True},
        },
        color={"is_ok": True},
    )
    finalize_status(ctx)
    assert ctx.status == "PASS"


def test_terminal_status_untouched():
    for terminal in ("INFERENCE_ERROR", "ERROR", "CANCELED"):
        ctx = _ctx(terminal, {"missing_items": []})
        finalize_status(ctx)
        assert ctx.status == terminal
