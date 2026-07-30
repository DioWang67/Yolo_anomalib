"""Structured operator decisions and deterministic training-data routing."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

COLOR_REVIEW_LABELS = frozenset({"color_confirmed_ng", "color_false_reject"})
POSITION_REVIEW_LABELS = frozenset({"position_false_reject"})
POSITION_FAILURE_REASONS = frozenset({"POSITION_SHIFT"})
YOLO_REVIEW_LABELS = frozenset(
    {
        "confirmed_ng",
        "verified_empty",
        "false_positive",
        "false_negative",
        "wrong_box",
        "wrong_class",
    }
)
VALID_ACTION_ROUTES = frozenset({"none", "yolo", "color", "position", "both"})
VALID_COLOR_VERDICTS = frozenset(
    {"not_applicable", "confirmed_ng", "actually_ok", "unjudgeable"}
)
VALID_DETECTION_VERDICTS = frozenset(
    {
        "not_applicable",
        "correct",
        "false_positive",
        "missed",
        "wrong_box",
        "wrong_class",
        "unjudgeable",
    }
)


@dataclass(frozen=True)
class ReviewDecision:
    """One operator verdict split by product, detection, color, and route."""

    review_label: str
    product_verdict: str
    detection_verdict: str
    color_verdict: str
    action_route: str

    def __post_init__(self) -> None:
        if self.action_route not in VALID_ACTION_ROUTES:
            raise ValueError(f"Unsupported action route: {self.action_route}")
        if self.color_verdict not in VALID_COLOR_VERDICTS:
            raise ValueError(f"Unsupported color verdict: {self.color_verdict}")
        if self.detection_verdict not in VALID_DETECTION_VERDICTS:
            raise ValueError(
                f"Unsupported detection verdict: {self.detection_verdict}"
            )
        if self.product_verdict not in {"ok", "ng", "unjudgeable"}:
            raise ValueError(
                f"Unsupported product verdict: {self.product_verdict}"
            )

    def to_columns(self) -> dict[str, str]:
        """Return manifest columns for immediate atomic persistence."""
        return {
            "review_label": self.review_label,
            "product_verdict": self.product_verdict,
            "detection_verdict": self.detection_verdict,
            "color_verdict": self.color_verdict,
            "action_route": self.action_route,
        }

    @classmethod
    def from_legacy_label(cls, review_label: str) -> ReviewDecision:
        """Translate the original one-column decisions into the new contract."""
        decisions = {
            "confirmed_ng": cls(
                "confirmed_ng", "ng", "correct", "not_applicable", "yolo"
            ),
            "confirmed_ok": cls(
                "confirmed_ok", "ok", "correct", "not_applicable", "none"
            ),
            "verified_empty": cls(
                "verified_empty", "ok", "correct", "not_applicable", "yolo"
            ),
            "false_positive": cls(
                "false_positive",
                "ok",
                "false_positive",
                "not_applicable",
                "yolo",
            ),
            "false_negative": cls(
                "false_negative", "ng", "missed", "not_applicable", "yolo"
            ),
            "wrong_box": cls(
                "wrong_box", "ng", "wrong_box", "not_applicable", "yolo"
            ),
            "wrong_class": cls(
                "wrong_class", "ng", "wrong_class", "not_applicable", "yolo"
            ),
            "image_quality_issue": cls(
                "image_quality_issue",
                "unjudgeable",
                "unjudgeable",
                "unjudgeable",
                "none",
            ),
            "position_false_reject": cls(
                "position_false_reject",
                "ok",
                "correct",
                "not_applicable",
                "position",
            ),
        }
        try:
            return decisions[review_label]
        except KeyError as exc:
            raise ValueError(f"Unsupported review label: {review_label}") from exc

    @classmethod
    def for_color(
        cls,
        color_verdict: str,
        *,
        detection_verdict: str = "correct",
        has_non_color_failure: bool = False,
    ) -> ReviewDecision:
        """Build a color decision without sending color-only cases to YOLO."""
        if color_verdict == "confirmed_ng":
            review_label = "color_confirmed_ng"
        elif color_verdict == "actually_ok":
            review_label = "color_false_reject"
        else:
            raise ValueError(f"Unsupported color review verdict: {color_verdict}")
        if detection_verdict not in {"correct", "wrong_box", "wrong_class"}:
            raise ValueError(
                f"Unsupported color-review detection verdict: {detection_verdict}"
            )
        needs_yolo = detection_verdict in {"wrong_box", "wrong_class"}
        product_verdict = (
            "ng"
            if color_verdict == "confirmed_ng" or has_non_color_failure
            else "ok"
        )
        return cls(
            review_label=review_label,
            product_verdict=product_verdict,
            detection_verdict=detection_verdict,
            color_verdict=color_verdict,
            action_route="both" if needs_yolo else "color",
        )


def action_route(row: Mapping[str, Any]) -> str:
    """Return a validated route, including a fallback for old manifests."""
    route = str(row.get("action_route") or "").strip().lower()
    if route in VALID_ACTION_ROUTES:
        return route
    label = str(row.get("review_label") or "").strip().lower()
    if label in COLOR_REVIEW_LABELS:
        detection = str(row.get("detection_verdict") or "correct").strip().lower()
        return "both" if detection in {"wrong_box", "wrong_class"} else "color"
    if label in POSITION_REVIEW_LABELS:
        return "position"
    if label in YOLO_REVIEW_LABELS:
        return "yolo"
    return "none"


def has_color_failure(row: Mapping[str, Any]) -> bool:
    """Return whether the saved inference failed at least one color check."""
    reasons = _decision_reasons(row)
    if "COLOR_MISMATCH" in reasons:
        return True
    try:
        if int(str(row.get("color_failure_count") or "0")) > 0:
            return True
    except ValueError:
        pass
    result = color_result(row)
    if result.get("is_ok") is False:
        return True
    items = result.get("items")
    return isinstance(items, list) and any(
        isinstance(item, dict) and item.get("is_ok") is False for item in items
    )


def has_non_color_failure(row: Mapping[str, Any]) -> bool:
    """Return whether a failure reason other than color mismatch is present."""
    return bool(_decision_reasons(row) - {"COLOR_MISMATCH"})


def has_position_failure(row: Mapping[str, Any]) -> bool:
    """Return whether position validation contributed to the inspection failure."""
    return bool(_decision_reasons(row) & POSITION_FAILURE_REASONS)


def has_position_only_failure(row: Mapping[str, Any]) -> bool:
    """Return whether the failure is exclusively from deployable position checks."""
    reasons = _decision_reasons(row)
    return bool(reasons) and reasons <= POSITION_FAILURE_REASONS


def color_result(row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a manifest color result without trusting external CSV input."""
    raw = row.get("color_result_json")
    if isinstance(raw, dict):
        return dict(raw)
    try:
        value = json.loads(str(raw or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def color_failure_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only failed, structurally valid color result items."""
    items = color_result(row).get("items")
    if not isinstance(items, list):
        return []
    return [
        dict(item)
        for item in items
        if isinstance(item, dict) and item.get("is_ok") is False
    ]


def has_threshold_color_failure(row: Mapping[str, Any]) -> bool:
    """Return whether a failed item specifically failed ``diff > threshold``."""
    for item in color_failure_items(row):
        try:
            if float(str(item.get("diff"))) > float(str(item.get("threshold"))):
                return True
        except (TypeError, ValueError):
            continue
    return False


def color_summary(row: Mapping[str, Any]) -> str:
    """Build a compact, operator-facing summary using the persisted scale."""
    summaries: list[str] = []
    for item in color_failure_items(row):
        expected = str(item.get("class_name") or item.get("class") or "-")
        predicted = str(item.get("best_color") or "-")
        try:
            diff = float(str(item.get("diff")))
            threshold = float(str(item.get("threshold")))
            if diff > threshold:
                values = f"diff {diff:.3f} > 門檻 {threshold:.3f}"
            else:
                values = (
                    f"diff {diff:.3f} ≤ 門檻 {threshold:.3f}，另有顏色規則未過"
                )
        except (TypeError, ValueError):
            values = "分數資料不完整"
        summaries.append(f"{expected} → {predicted}（{values}）")
    return "；".join(summaries)


def _decision_reasons(row: Mapping[str, Any]) -> set[str]:
    raw = str(row.get("decision_reasons") or "")
    return {part.strip().upper() for part in raw.split("|") if part.strip()}
