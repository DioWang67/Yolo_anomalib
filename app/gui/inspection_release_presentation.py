"""Shared presentation helpers for inspection release components."""

from __future__ import annotations

from datetime import datetime, tzinfo

from core.services.inspection_release_models import InspectionRelease


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
