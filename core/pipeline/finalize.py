from __future__ import annotations

"""Final inspection verdict, computed after color correction and all checks.

Why this exists
---------------
``YOLOInferenceModel.infer`` sets an initial status from the *raw* YOLO class
labels (e.g. it may report ``missing=[Green]`` because the model mislabeled a
green wire as orange). The downstream color checker then corrects the class
(``verified_class``) and ``count_check`` recomputes the real missing/over set —
but the pipeline steps only ever *downgrade* status to FAIL, they never clear a
stale FAIL once a correction makes the board good again. That over-kills good
boards whose only problem was a YOLO misclassification the color checker fixed.

``finalize_status`` recomputes the verdict from the *current, corrected*
signals just before results are saved. It is intentionally comprehensive so it
never drops a failure dimension: a board passes only when color, sequence, and
every YOLO-side dimension (missing / unexpected / slot mismatch / board
alignment / position) are all clean. Products that do not populate a given
signal simply skip that gate, so single-stage pipelines (e.g. PCBA's
``count_check`` only) keep their previous behavior.
"""

from typing import Any

from core.services.decision_engine import InspectionDecisionEngine, InspectionStatus

# Statuses that represent a hard failure/abort and must not be reinterpreted.
_TERMINAL = {"INFERENCE_ERROR", "ERROR", "CANCELED"}


def finalize_status(ctx: Any, *, fail_on_unexpected: bool = True) -> None:
    """Recompute and set ``ctx.status`` from color-corrected inspection signals.

    Mutates ``ctx.status`` (to ``"PASS"`` or ``"DETECTION_FAIL"``) and refreshes
    ``ctx.result['decision']``. No-op for terminal statuses.

    Args:
        ctx: Detection context exposing ``status``, ``result``, ``color_result``.
        fail_on_unexpected: Whether unexpected components force a FAIL.
    """
    current = str(getattr(ctx, "status", "") or "").upper()
    if current in _TERMINAL:
        return

    result: dict[str, Any] = getattr(ctx, "result", None) or {}

    color = getattr(ctx, "color_result", None)
    color_failed = isinstance(color, dict) and not color.get("is_ok", True)

    sequence = result.get("sequence_check")
    sequence_failed = isinstance(sequence, dict) and not sequence.get("is_ok", True)

    decision = InspectionDecisionEngine(fail_on_unexpected=fail_on_unexpected).evaluate(
        detections=result.get("detections", []) or [],
        missing_items=result.get("missing_items", []) or [],
        unexpected_items=result.get("unexpected_items", []) or [],
        slot_mismatches=result.get("slot_mismatches", []) or [],
        alignment_quality=result.get("alignment_quality"),
    )
    result["decision"] = decision.to_dict()

    failed = color_failed or sequence_failed or decision.status != InspectionStatus.PASS
    ctx.status = "DETECTION_FAIL" if failed else "PASS"
