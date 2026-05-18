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
