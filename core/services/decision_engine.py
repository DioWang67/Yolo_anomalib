from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class InspectionStatus(str, Enum):
    """Final inspection status emitted by the decision layer."""

    PASS = "PASS"
    FAIL = "FAIL"


class InspectionReason(str, Enum):
    """Machine-readable reason codes for PCBA inspection decisions."""

    MISSING = "MISSING"
    WRONG_COMPONENT = "WRONG_COMPONENT"
    POSITION_SHIFT = "POSITION_SHIFT"
    BOARD_ALIGNMENT = "BOARD_ALIGNMENT"
    UNEXPECTED_COMPONENT = "UNEXPECTED_COMPONENT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    COLOR_MISMATCH = "COLOR_MISMATCH"
    SEQUENCE_MISMATCH = "SEQUENCE_MISMATCH"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"
    INFERENCE_ERROR = "INFERENCE_ERROR"


def collect_fail_reasons(
    *,
    status: str,
    decision: dict[str, Any] | None = None,
    color_result: dict[str, Any] | None = None,
    sequence_check: dict[str, Any] | None = None,
    detector: str | None = None,
    anomaly_score: float | None = None,
    error_message: str | None = None,
) -> list[str]:
    """Merge every failure signal into one machine-readable reason list.

    ``InspectionDecisionEngine`` only sees YOLO-side signals (missing, slot,
    position, alignment). Color check, sequence check, anomalib verdicts, and
    inference errors set the FAIL status outside the engine, so a saved result
    could say FAIL with an empty ``decision.reasons``. This helper is the single
    place that folds all of them together for the persisted result record.

    Args:
        status: Final inspection status string (e.g. ``PASS``,
            ``DETECTION_FAIL``, ``INFERENCE_ERROR``).
        decision: ``InspectionDecision.to_dict()`` payload, if available.
        color_result: Color check result dict with an ``is_ok`` flag.
        sequence_check: Sequence check result dict with an ``is_ok`` flag.
        detector: Detector name (``yolo`` / ``anomalib`` / ``fusion``).
        anomaly_score: Anomaly score when the anomalib path ran.
        error_message: Error text when inference itself failed.

    Returns:
        Ordered, de-duplicated reason code strings; empty when status is PASS.
    """
    normalized_status = str(status or "").upper()
    if normalized_status == "PASS":
        return []

    reasons: list[str] = []

    def _add(code: str) -> None:
        if code not in reasons:
            reasons.append(code)

    if normalized_status in {"INFERENCE_ERROR", "ERROR"} or error_message:
        _add(InspectionReason.INFERENCE_ERROR.value)

    for code in (decision or {}).get("reasons", []) or []:
        _add(str(code))

    if isinstance(color_result, dict) and not color_result.get("is_ok", True):
        _add(InspectionReason.COLOR_MISMATCH.value)

    if isinstance(sequence_check, dict) and not sequence_check.get("is_ok", True):
        _add(InspectionReason.SEQUENCE_MISMATCH.value)

    detector_lower = str(detector or "").lower()
    if normalized_status not in {"INFERENCE_ERROR", "ERROR", "CANCELED"}:
        if detector_lower == "anomalib":
            _add(InspectionReason.ANOMALY_DETECTED.value)
        elif (
            detector_lower == "fusion"
            and anomaly_score is not None
            and not reasons
        ):
            # Fusion merges YOLO and anomalib verdicts without keeping the
            # per-branch status, so anomaly is only attributed when no other
            # failure signal explains the FAIL.
            _add(InspectionReason.ANOMALY_DETECTED.value)

    return reasons


@dataclass(frozen=True)
class InspectionDecision:
    """Final decision and traceable failure reasons for one inspection result.

    Args:
        status: Final PASS/FAIL status.
        reasons: Machine-readable reason codes.
        details: Structured records that explain each reason.

    Returns:
        Serializable decision metadata through ``to_dict``.
    """

    status: InspectionStatus
    reasons: list[InspectionReason] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable decision payload."""
        return {
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
            "details": [dict(item) for item in self.details],
        }


class InspectionDecisionEngine:
    """Rule-based decision engine for YOLO PCBA inspection results.

    This engine intentionally stays small: it converts already-computed model,
    position, and slot validation signals into a final PASS/FAIL decision with
    traceable reason codes. It does not run inference or mutate images.
    """

    def __init__(self, *, fail_on_unexpected: bool = True) -> None:
        self.fail_on_unexpected = bool(fail_on_unexpected)

    def evaluate(
        self,
        *,
        detections: list[dict[str, Any]] | None = None,
        missing_items: list[str] | None = None,
        unexpected_items: list[str] | None = None,
        slot_mismatches: list[dict[str, Any]] | None = None,
        alignment_quality: dict[str, Any] | None = None,
    ) -> InspectionDecision:
        """Evaluate final inspection status from normalized validation signals.

        Args:
            detections: Detection dictionaries, optionally annotated with
                ``position_status`` and confidence fields.
            missing_items: Expected classes that were not found.
            unexpected_items: Classes detected but not expected by product config.
            slot_mismatches: Records where a missing expected slot is occupied by
                a different class.
            alignment_quality: Optional board alignment gate result.

        Returns:
            InspectionDecision containing PASS/FAIL and reason metadata.
        """
        reasons: list[InspectionReason] = []
        details: list[dict[str, Any]] = []

        missing = [str(item).strip() for item in (missing_items or []) if str(item).strip()]
        if missing:
            self._add_reason(
                reasons,
                details,
                InspectionReason.MISSING,
                {"items": missing},
            )

        mismatches = [dict(item) for item in (slot_mismatches or []) if isinstance(item, dict)]
        if mismatches:
            self._add_reason(
                reasons,
                details,
                InspectionReason.WRONG_COMPONENT,
                {"items": mismatches},
            )

        shifted = [
            self._position_detail(det)
            for det in (detections or [])
            if det.get("position_status") == "WRONG"
        ]
        shifted = [item for item in shifted if item]
        if shifted:
            self._add_reason(
                reasons,
                details,
                InspectionReason.POSITION_SHIFT,
                {"items": shifted},
            )

        if isinstance(alignment_quality, dict) and not bool(
            alignment_quality.get("is_ok", True)
        ):
            self._add_reason(
                reasons,
                details,
                InspectionReason.BOARD_ALIGNMENT,
                {"items": dict(alignment_quality)},
            )

        unexpected = [
            str(item).strip() for item in (unexpected_items or []) if str(item).strip()
        ]
        if unexpected and self.fail_on_unexpected:
            self._add_reason(
                reasons,
                details,
                InspectionReason.UNEXPECTED_COMPONENT,
                {"items": unexpected},
            )

        status = InspectionStatus.FAIL if reasons else InspectionStatus.PASS
        return InspectionDecision(status=status, reasons=reasons, details=details)

    @staticmethod
    def _add_reason(
        reasons: list[InspectionReason],
        details: list[dict[str, Any]],
        reason: InspectionReason,
        payload: dict[str, Any],
    ) -> None:
        if reason not in reasons:
            reasons.append(reason)
        detail = {"reason": reason.value}
        detail.update(payload)
        details.append(detail)

    @staticmethod
    def _position_detail(detection: dict[str, Any]) -> dict[str, Any]:
        label = detection.get("class") or detection.get("label") or ""
        return {
            "class": str(label),
            "expected_key": detection.get("position_expected_key"),
            "position_error": detection.get("position_error"),
            "position_tolerance_px": detection.get("position_tolerance_px"),
            "position_offset": detection.get("position_offset"),
        }
