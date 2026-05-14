from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .position_summary import format_fixture_shift_hint, summarize_position_records

if TYPE_CHECKING:
    from core.types import DetectionResult


@dataclass(frozen=True)
class CustomerMessage:
    """Customer-facing result summary for the operator panel."""

    headline: str
    action: str
    severity: str
    details: list[str]


def build_customer_message(result: "DetectionResult") -> CustomerMessage:
    """Return a concise operator-facing message for the current result."""
    status = str(result.status or "").upper()
    slot_check = _get_slot_check(result)
    slot_mismatches = _get_slot_mismatches(result)
    recovered_items = _slot_check_recovered_items(slot_check)
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

    fixture_hint = format_fixture_shift_hint(position_summary)
    if fixture_hint:
        return CustomerMessage(
            headline="疑似治具偏移",
            action="請先確認治具定位後重新檢測",
            severity="danger",
            details=_nonempty([fixture_hint]),
        )

    color_check = result.color_check or {}
    if color_check and not color_check.get("is_ok", True):
        bad = [
            str(item.get("class_name") or "?")
            for item in (color_check.get("items") or [])
            if not item.get("is_ok", True)
        ]
        return CustomerMessage(
            headline="顏色檢查異常",
            action="請確認來料或顏色設定後再檢測",
            severity="danger",
            details=_nonempty([f"異常項目: {', '.join(_limit_items(bad))}" if bad else None]),
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


def _nonempty(values: list[str | None]) -> list[str]:
    return [value for value in values if value]


def _maybe_fixture_detail(position_summary: Any) -> str | None:
    return format_fixture_shift_hint(position_summary)


def _get_slot_check(result: "DetectionResult") -> dict[str, Any] | None:
    metadata = getattr(result, "metadata", {}) or {}
    slot_check = metadata.get("slot_check")
    return slot_check if isinstance(slot_check, dict) else None


def _get_slot_mismatches(result: "DetectionResult") -> list[dict[str, Any]]:
    metadata = getattr(result, "metadata", {}) or {}
    values = metadata.get("slot_mismatches") or []
    return [value for value in values if isinstance(value, dict)]


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
