from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .position_summary import format_fixture_shift_hint, summarize_position_records

if TYPE_CHECKING:
    from core.types import DetectionResult

# ``ColorFailureDescription.kind`` values. A failed color-check item can mean
# two very different things that look identical if only a color name is
# shown: the detected class was swapped for a different color (MISMATCH), or
# the predicted color agrees with the detected class but its own score missed
# its own threshold (LOW_CONFIDENCE). Every UI surface that reports color
# check failures (main verdict message, detail panel, annotated image
# overlay) classifies through this shared logic so the two are never
# conflated or described inconsistently across surfaces.
COLOR_FAILURE_MISMATCH = "mismatch"
COLOR_FAILURE_LOW_CONFIDENCE = "low_confidence"
COLOR_FAILURE_UNIDENTIFIED = "unidentified"


@dataclass(frozen=True)
class ColorFailureDescription:
    """Structured classification of why one color-check item failed.

    Carries the raw class/predicted-color pair rather than pre-formatted
    text, so each UI layer can render its own localized or ASCII-safe wording
    (an annotated image overlay cannot render non-Latin glyphs) while still
    sharing the same underlying semantic distinction.
    """

    kind: str
    class_name: str | None
    predicted_color: str | None


def classify_color_check_failure(item: dict[str, Any]) -> ColorFailureDescription:
    """Classify one failed color-check item's failure mode.

    Args:
        item: A serialized ``ColorCheckItemResult`` (``class_name``/``class``,
            ``best_color``).

    Returns:
        ``COLOR_FAILURE_UNIDENTIFIED`` when there is no detector class to
        compare against (e.g. a legacy full-frame check with no detections);
        ``COLOR_FAILURE_MISMATCH`` when the predicted color differs from the
        detected class; otherwise ``COLOR_FAILURE_LOW_CONFIDENCE``.
    """
    class_name = item.get("class_name") or item.get("class")
    if not class_name:
        return ColorFailureDescription(
            kind=COLOR_FAILURE_UNIDENTIFIED,
            class_name=None,
            predicted_color=_clean_str(item.get("best_color")),
        )
    label = str(class_name).strip()
    predicted = _clean_str(item.get("best_color"))
    if predicted and predicted.casefold() != label.casefold():
        return ColorFailureDescription(
            kind=COLOR_FAILURE_MISMATCH, class_name=label, predicted_color=predicted
        )
    return ColorFailureDescription(
        kind=COLOR_FAILURE_LOW_CONFIDENCE, class_name=label, predicted_color=predicted
    )


def _clean_str(value: object) -> str | None:
    text = str(value).strip() if value else ""
    return text or None


@dataclass(frozen=True)
class CustomerMessage:
    """Customer-facing result summary for the operator panel."""

    headline: str
    action: str
    severity: str
    details: list[str]


def build_customer_message(result: DetectionResult) -> CustomerMessage:
    """Return a concise operator-facing message for the current result."""
    status = str(result.status or "").upper()
    slot_check = _get_slot_check(result)
    slot_mismatches = _get_slot_mismatches(result)
    decision_reasons = _get_decision_reasons(result)
    alignment_quality = _get_alignment_quality(result)
    recovered_items = _slot_check_recovered_items(slot_check)
    duplicate_filter = _get_duplicate_filter(result)
    position_summary = summarize_position_records(
        [{"label": item.label, **item.metadata} for item in (result.items or [])]
    )

    if status == "PASS":
        details = [
            f"檢出 {len(result.items or [])} 件",
            f"缺件 {len(result.missing_items or [])} 件",
        ]
        if recovered_items:
            details.append(f"槽位複核補回: {', '.join(_limit_items(recovered_items))}")
        suppressed_count = int(duplicate_filter.get("suppressed_count", 0) or 0)
        if suppressed_count:
            details.append(f"已排除 {suppressed_count} 個高重疊重複框")
        return CustomerMessage(
            headline="檢測通過" if not recovered_items else "檢測通過（已做槽位複核）",
            action=(
                "等待下一次檢測"
                if not recovered_items
                else "等待下一次檢測，建議抽查複核槽位影像"
            ),
            severity="success" if not recovered_items else "warning",
            details=details,
        )

    if status in {"INFERENCE_ERROR", "ERROR"}:
        return CustomerMessage(
            headline="系統異常",
            action="請重新檢測；若持續異常請通知工程人員",
            severity="warning",
            details=_nonempty([result.error or "推論或後端執行失敗"]),
        )

    if result.missing_items:
        return CustomerMessage(
            headline="發現缺件",
            action="請補件後重新檢測",
            severity="danger",
            details=_nonempty(
                [
                    f"缺件: {', '.join(_limit_items(result.missing_items))}",
                    _slot_check_detail(slot_check),
                    _maybe_fixture_detail(position_summary),
                ]
            ),
        )

    if slot_mismatches:
        details = [
            f"槽位 {item['expected_key']} 判成 {item['detected_class']}"
            for item in slot_mismatches[:3]
        ]
        return CustomerMessage(
            headline="元件類別不符",
            action="請確認錯料、模型分類或光源後重測",
            severity="danger",
            details=details,
        )

    if "BOARD_ALIGNMENT" in decision_reasons:
        return CustomerMessage(
            headline="板件對位異常",
            action="請確認治具、相機視野與板件放置位置後重新檢測",
            severity="danger",
            details=_nonempty([_alignment_quality_detail(alignment_quality)]),
        )

    fixture_hint = format_fixture_shift_hint(position_summary)
    if fixture_hint:
        return CustomerMessage(
            headline="疑似治具偏移",
            action="請先確認治具定位後重新檢測",
            severity="danger",
            details=_nonempty([fixture_hint]),
        )

    if (
        duplicate_filter.get("status") == "reported"
        and int(duplicate_filter.get("would_suppress_count", 0) or 0) > 0
    ):
        return CustomerMessage(
            headline="疑似模型重複框",
            action="請查看標註圖並重新檢測；若持續出現請通知工程人員",
            severity="warning",
            details=[
                "目前為僅觀察模式，尚未自動消除，也不代表實物真的多一件"
            ],
        )

    color_check = result.color_check or {}
    if color_check and not color_check.get("is_ok", True):
        bad = [
            _describe_color_check_failure(item)
            for item in reportable_color_failures(result)
        ]
        return CustomerMessage(
            headline="顏色檢查異常",
            action="請確認來料或顏色設定後再檢測",
            severity="danger",
            details=_nonempty(["; ".join(_limit_items(bad)) if bad else None]),
        )

    sequence_check = result.sequence_check or {}
    if sequence_check and not sequence_check.get("is_ok", True):
        return CustomerMessage(
            headline="排列順序異常",
            action="請確認工件擺放順序後重新檢測",
            severity="danger",
            details=_nonempty([str(sequence_check.get("reason") or "順序檢查失敗")]),
        )

    if position_summary.fail_count > 0:
        issue = position_summary.issues[0]
        detail = issue.label
        if issue.dx is not None and issue.dy is not None:
            detail += f" dx={issue.dx:+.1f}, dy={issue.dy:+.1f}"
        return CustomerMessage(
            headline="位置偏移",
            action="請確認治具或上料位置後重測",
            severity="danger",
            details=_nonempty([detail]),
        )

    if status == "DETECTION_FAIL":
        return CustomerMessage(
            headline="檢測失敗",
            action="請重新取像後再檢測",
            severity="warning",
            details=_nonempty([_slot_check_detail(slot_check)]),
        )

    return CustomerMessage(
        headline="檢測異常",
        action="請確認失敗原因後重新檢測",
        severity="danger",
        details=_nonempty([_slot_check_detail(slot_check)]),
    )


def _limit_items(items: list[str], limit: int = 3) -> list[str]:
    return [str(item) for item in items[:limit]]


def _get_duplicate_filter(result: DetectionResult) -> dict[str, Any]:
    metadata = result.metadata or {}
    value = metadata.get("duplicate_filter")
    return value if isinstance(value, dict) else {}


def reportable_color_failures(result: DetectionResult) -> list[dict[str, Any]]:
    """Return the failed color items that describe the board as inspected.

    A box the duplicate filter removed is no longer part of the board, so its
    color failure is not something an operator can act on: it reports the
    detector's error, not the product's condition. Every cross-class duplicate
    produces one such failure by construction -- two boxes over one object carry
    two different classes and the measured color can only match one -- so any
    surface that lists failures per box has to drop them or it will name a box
    the operator cannot find in the image.

    This lives here, beside ``classify_color_check_failure``, because the same
    filtering was needed independently by the operator card, the detail panel,
    the annotated overlay, and the one-line fail-reason banner. The first three
    each grew their own copy; the banner was missed and kept reporting a removed
    box after the others had stopped. Any new surface should call this rather
    than re-deriving it.
    """
    color_check = getattr(result, "color_check", None) or {}
    suppressed = suppressed_source_indices(result)
    failures: list[dict[str, Any]] = []
    for position, item in enumerate(color_check.get("items") or []):
        if not isinstance(item, dict) or item.get("is_ok", True):
            continue
        if _color_item_source_index(item, position) in suppressed:
            continue
        failures.append(item)
    return failures


def suppressed_source_indices(result: DetectionResult) -> set[int]:
    """Return the source indices of boxes the duplicate filter removed.

    Reads ``suppressions`` rather than inferring removal from a box's absence
    among the effective items: absence also covers a legacy full-frame check
    (``index == -1``) and any result that carries no items at all, and silently
    dropping those failures would hide real evidence. ``proposed_suppressions``
    is deliberately not consulted -- in report-only mode those boxes are still
    on the board and still count.
    """
    indices: set[int] = set()
    for record in _get_duplicate_filter(result).get("suppressions") or []:
        if not isinstance(record, dict):
            continue
        try:
            indices.add(int(record["suppressed_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return indices


def _color_item_source_index(item: dict[str, Any], position: int) -> int:
    try:
        return int(item.get("index", position))
    except (TypeError, ValueError):
        return position


def _color_item_label(item: dict[str, Any]) -> str:
    """Return a meaningful Chinese label for one color-check item."""
    label = item.get("class_name") or item.get("class")
    if label:
        return str(label)
    if item.get("index") == -1:
        return "未偵測到元件（全畫面檢查）"
    return "未知項目"


def _describe_color_check_failure(item: dict[str, Any]) -> str:
    """Return an operator-facing Chinese description of one failed item.

    Names both the detected class and the predicted color for a mismatch
    (``顏色不符: Red → Orange``) instead of the detected class alone, and uses
    distinct wording when they agree but the score still missed its own
    threshold (``Red 顏色信心不足``) so the two failure modes are never
    displayed as if they were the same thing.
    """
    description = classify_color_check_failure(item)
    if description.kind == COLOR_FAILURE_MISMATCH:
        return f"顏色不符: {description.class_name} → {description.predicted_color}"
    if description.kind == COLOR_FAILURE_LOW_CONFIDENCE:
        return f"{description.class_name} 顏色信心不足"
    return _color_item_label(item)


def _nonempty(values: list[str | None]) -> list[str]:
    return [value for value in values if value]


def _maybe_fixture_detail(position_summary: Any) -> str | None:
    return format_fixture_shift_hint(position_summary)


def _get_slot_check(result: DetectionResult) -> dict[str, Any] | None:
    metadata = getattr(result, "metadata", {}) or {}
    slot_check = metadata.get("slot_check")
    return slot_check if isinstance(slot_check, dict) else None


def _get_slot_mismatches(result: DetectionResult) -> list[dict[str, Any]]:
    metadata = getattr(result, "metadata", {}) or {}
    values = metadata.get("slot_mismatches") or []
    return [value for value in values if isinstance(value, dict)]


def _get_decision_reasons(result: DetectionResult) -> list[str]:
    metadata = getattr(result, "metadata", {}) or {}
    decision = metadata.get("decision")
    if not isinstance(decision, dict):
        return []
    reasons = decision.get("reasons") or []
    return [str(reason) for reason in reasons if str(reason).strip()]


def _get_alignment_quality(result: DetectionResult) -> dict[str, Any] | None:
    metadata = getattr(result, "metadata", {}) or {}
    value = metadata.get("alignment_quality")
    return value if isinstance(value, dict) else None


def _alignment_quality_detail(alignment_quality: dict[str, Any] | None) -> str | None:
    if not isinstance(alignment_quality, dict):
        return "板件與治具對位未通過"
    issues = alignment_quality.get("issues") or []
    issue_text = "、".join(
        _alignment_issue_label(str(issue))
        for issue in issues
        if str(issue).strip()
    )
    dx = alignment_quality.get("dx")
    dy = alignment_quality.get("dy")
    sources = alignment_quality.get("observed_source_count")
    required = alignment_quality.get("required_source_count")
    parts = []
    if issue_text:
        parts.append(issue_text)
    if dx is not None and dy is not None:
        try:
            parts.append(f"dx={float(dx):+.1f}, dy={float(dy):+.1f}")
        except (TypeError, ValueError):
            pass
    if sources is not None and required is not None:
        parts.append(f"sources={sources}/{required}")
    return " | ".join(parts) if parts else "板件與治具對位未通過"


def _alignment_issue_label(issue: str) -> str:
    labels = {
        "insufficient_alignment_sources": "可用對位特徵不足",
        "insufficient_alignment_inliers": "對位特徵一致性不足",
        "alignment_dx_out_of_range": "水平偏移超出範圍",
        "alignment_dy_out_of_range": "垂直偏移超出範圍",
        "alignment_shift_out_of_range": "整板偏移超出範圍",
    }
    return labels.get(issue, issue)


def _slot_check_recovered_items(slot_check: dict[str, Any] | None) -> list[str]:
    if not isinstance(slot_check, dict):
        return []
    values = slot_check.get("recovered_items") or []
    return [str(value) for value in values if str(value).strip()]


def _slot_check_detail(slot_check: dict[str, Any] | None) -> str | None:
    if not isinstance(slot_check, dict):
        return None

    recovered = _slot_check_recovered_items(slot_check)
    if recovered:
        return f"槽位複核補回: {', '.join(_limit_items(recovered))}"

    remaining = slot_check.get("remaining_missing_items") or []
    values = [str(value) for value in remaining if str(value).strip()]
    if values:
        return f"槽位複核後仍缺件: {', '.join(_limit_items(values))}"
    return None
