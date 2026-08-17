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


# ---------------------------------------------------------------------------
# Anomaly dimension
#
# An anomalib-only run leaves no detection-side signal behind at all: no
# detections, no missing/unexpected items, no color or sequence result. Before
# the anomaly gate existed, finalize_status re-derived PASS from that empty
# evidence and shipped defective boards as good (observed in production on
# 2026-07-28: PCBA1/B records with anomaly_score 1.0 saved as PASS).


def test_anomaly_fail_is_not_recomputed_into_pass():
    ctx = _ctx(
        "DETECTION_FAIL",
        {
            "status": "DETECTION_FAIL",
            "detections": [],
            "missing_items": [],
            "unexpected_items": [],
            "is_anomaly": True,
            "anomaly_score": 0.97,
        },
    )
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


def test_clean_anomaly_run_still_passes():
    """Guards against the anomaly gate over-failing every anomalib run."""
    ctx = _ctx(
        "PASS",
        {
            "status": "PASS",
            "detections": [],
            "missing_items": [],
            "unexpected_items": [],
            "is_anomaly": False,
            "anomaly_score": 0.12,
        },
    )
    finalize_status(ctx)
    assert ctx.status == "PASS"


def test_high_anomaly_score_without_verdict_does_not_fail():
    """The threshold belongs to the inference layer, not to finalization.

    A payload carrying a score but no ``is_anomaly`` verdict (e.g. a backend
    that never applied a threshold) must not be second-guessed here.
    """
    ctx = _ctx(
        "PASS",
        {"detections": [], "missing_items": [], "anomaly_score": 0.99},
    )
    finalize_status(ctx)
    assert ctx.status == "PASS"


def test_anomaly_fail_survives_a_clean_yolo_side():
    """Fusion shape: YOLO half spotless, anomalib half failed."""
    ctx = _ctx(
        "DETECTION_FAIL",
        {
            "detections": [{"class": "J2-1", "confidence": 0.9}],
            "missing_items": [],
            "unexpected_items": [],
            "slot_mismatches": [],
            "sequence_check": {"is_ok": True},
            "is_anomaly": True,
            "anomaly_score": 0.88,
        },
        color={"is_ok": True},
    )
    finalize_status(ctx)
    assert ctx.status == "DETECTION_FAIL"


# ---------------------------------------------------------------------------
# Count check verdict ownership


def test_count_check_failure_is_honoured_when_unexpected_does_not_fail():
    """A strict count check fails on surplus parts regardless of the flag.

    ``fail_on_unexpected=False`` only tells the decision engine to ignore the
    UNEXPECTED_COMPONENT dimension; it must not silently overturn a count check
    that already declared the board bad.
    """
    ctx = _ctx(
        "DETECTION_FAIL",
        {
            "detections": [],
            "missing_items": [],
            "unexpected_items": ["Red"],
            "over_items": ["Red"],
            "count_check": {
                "expected": {"Red": 1},
                "detected": {"Red": 2},
                "missing": [],
                "over": ["Red"],
                "strict": True,
                "is_ok": False,
            },
        },
    )
    finalize_status(ctx, fail_on_unexpected=False)
    assert ctx.status == "DETECTION_FAIL"


def test_passing_count_check_does_not_block_pass():
    ctx = _ctx(
        "DETECTION_FAIL",
        {
            "detections": [],
            "missing_items": [],
            "unexpected_items": [],
            "count_check": {
                "expected": {"Red": 1},
                "detected": {"Red": 1},
                "missing": [],
                "over": [],
                "strict": True,
                "is_ok": True,
            },
        },
        color={"is_ok": True},
    )
    finalize_status(ctx)
    assert ctx.status == "PASS"


def test_terminal_status_untouched_even_with_anomaly():
    """A terminal abort must stay terminal, never become DETECTION_FAIL."""
    for terminal in ("INFERENCE_ERROR", "ERROR", "CANCELED"):
        ctx = _ctx(
            terminal,
            {"missing_items": [], "is_anomaly": True, "anomaly_score": 1.0},
        )
        finalize_status(ctx)
        assert ctx.status == terminal
        assert "decision" not in ctx.result
