"""Shared presentation helpers for inspection release components."""

from __future__ import annotations

from datetime import datetime, tzinfo

from core.services.inspection_release_models import InspectionRelease

_COLOR_BASELINE_LIFECYCLE_LABELS = {
    "CANDIDATE": "候選基準",
    "DEPLOYED": "正式使用",
    "DEFAULT": "子系統預設",
    "HISTORY": "歷史版本",
}
_COLOR_BASELINE_QUALITY_LABELS = {
    "READY": "品質檢查通過",
    "REVIEW_REQUIRED": "需要人工複核",
    "INCOMPLETE": "基準資料不足",
}


def format_local_timestamp(
    value: str,
    *,
    local_timezone: tzinfo | None = None,
) -> str:
    """Render an ISO-8601 timestamp in the station's local timezone.

    Legacy timestamps without an offset are treated as already-local wall
    clock values.  This avoids silently shifting old records whose timezone
    was never persisted.
    """
    normalized = str(value or "").strip()
    if not normalized:
        return "—"
    try:
        parsed = datetime.fromisoformat(
            normalized[:-1] + "+00:00"
            if normalized.endswith("Z")
            else normalized
        )
    except ValueError:
        return normalized.replace("T", " ")[:19]
    if parsed.tzinfo is not None:
        target_timezone = (
            local_timezone
            if local_timezone is not None
            else datetime.now().astimezone().tzinfo
        )
        if target_timezone is not None:
            parsed = parsed.astimezone(target_timezone)
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def format_color_baseline_summary(
    *,
    color_count: int,
    created_at: str,
    lifecycle_status: str,
    quality_status: str,
) -> str:
    """Return a human-facing baseline label without exposing its content ID."""
    normalized_count = (
        color_count
        if isinstance(color_count, int)
        and not isinstance(color_count, bool)
        and color_count >= 0
        else 0
    )
    values = [
        f"完整 {normalized_count} 色基準"
        if normalized_count
        else "完整顏色基準"
    ]
    local_created_at = format_local_timestamp(created_at)
    if local_created_at != "—":
        values.append(f"基準版本 {local_created_at[:16]}")
    normalized_lifecycle = str(lifecycle_status or "").strip().upper()
    if normalized_lifecycle:
        values.append(
            _COLOR_BASELINE_LIFECYCLE_LABELS.get(
                normalized_lifecycle,
                normalized_lifecycle,
            )
        )
    normalized_quality = str(quality_status or "").strip().upper()
    if normalized_quality:
        values.append(
            _COLOR_BASELINE_QUALITY_LABELS.get(
                normalized_quality,
                normalized_quality,
            )
        )
    return "｜".join(values)


def format_component_summary(release: InspectionRelease) -> str:
    """Describe a full color profile without presenting one patch as a model."""
    values: list[str] = []
    for component in release.components:
        metadata = dict(component.metadata)
        if component.role == "color_check":
            profile = str(metadata.get("profile_summary") or "").strip()
            if profile:
                values.append(f"Stats Color｜{profile}")
                continue
            threshold = str(metadata.get("threshold_key") or "").strip()
            suffix = f"｜{threshold.title()} 單色修訂" if threshold else ""
            values.append(
                f"Stats Color {component.version}{suffix}（舊格式）"
            )
            continue
        values.append(f"{component.kind} {component.version}")
    return " + ".join(values)
