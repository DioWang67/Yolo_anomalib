"""Validated human-readable identities for immutable retraining batches."""

from __future__ import annotations

import re

_SEMANTIC_VERSION_PATTERN = re.compile(
    r"^v(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)$",
    re.IGNORECASE,
)


class TrainingBatchVersionError(ValueError):
    """A retraining batch version is missing or does not match its target."""


def training_batch_prefix(product: str, area: str) -> str:
    """Return the stable, filesystem-safe target prefix used in the UI."""
    product_segment = _safe_segment(product, "product")
    area_segment = _safe_segment(area, "area")
    return f"{product_segment}_{area_segment}"


def format_training_batch_version(
    product: str,
    area: str,
    version: tuple[int, int, int],
) -> str:
    """Build a canonical target-scoped batch version."""
    major, minor, patch = version
    if min(major, minor, patch) < 0:
        raise TrainingBatchVersionError("補訓版本號不可小於 0。")
    return f"{training_batch_prefix(product, area)}_v{major}.{minor}.{patch}"


def validate_training_batch_version(
    value: str,
    *,
    product: str,
    area: str,
) -> str:
    """Validate and canonicalize ``<product>_<area>_vMAJOR.MINOR.PATCH``."""
    normalized = str(value or "").strip()
    if not normalized:
        raise TrainingBatchVersionError("請輸入補訓批次版本。")
    expected_prefix = training_batch_prefix(product, area)
    separator = normalized.rfind("_v")
    if separator < 0:
        separator = normalized.rfind("_V")
    if separator < 1:
        raise TrainingBatchVersionError(
            f"補訓批次版本格式必須為 {expected_prefix}_v0.0.1。"
        )
    prefix = normalized[:separator]
    version_text = normalized[separator + 1 :]
    if prefix.casefold() != expected_prefix.casefold():
        raise TrainingBatchVersionError(
            f"補訓批次版本必須以 {expected_prefix}_ 開頭。"
        )
    match = _SEMANTIC_VERSION_PATTERN.fullmatch(version_text)
    if match is None:
        raise TrainingBatchVersionError(
            "版本號必須使用 v主版.次版.修訂版，例如 v0.0.1。"
        )
    return format_training_batch_version(
        product,
        area,
        (
            int(match.group("major")),
            int(match.group("minor")),
            int(match.group("patch")),
        ),
    )


def parse_training_batch_semver(value: str) -> tuple[int, int, int] | None:
    """Return the trailing semantic version for a validated or legacy label."""
    normalized = str(value or "").strip()
    separator = max(normalized.rfind("_v"), normalized.rfind("_V"))
    if separator < 0:
        return None
    match = _SEMANTIC_VERSION_PATTERN.fullmatch(normalized[separator + 1 :])
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _safe_segment(value: str, label: str) -> str:
    normalized = re.sub(r"[^\w-]+", "_", str(value or "").strip()).strip("_")
    if not normalized:
        raise TrainingBatchVersionError(f"{label} 不可為空白。")
    return normalized[:64]
