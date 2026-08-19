"""Extensible failure classifications recorded independently of review routing."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

THRESHOLD_NOT_MET = "threshold_not_met"
MISCLASSIFICATION = "misclassification"
MISSED_DETECTION = "missed_detection"
NEW_DEFECT_TYPE = "new_defect_type"
WRONG_BOX = "wrong_box"
WRONG_CLASS = "wrong_class"
COLOR_ISSUE = "color_issue"
LIGHTING_ISSUE = "lighting_issue"
OTHER_FAILURE = "other"
VALID_FAILURE_CATEGORIES = frozenset(
    {
        THRESHOLD_NOT_MET,
        MISCLASSIFICATION,
        MISSED_DETECTION,
        NEW_DEFECT_TYPE,
        WRONG_BOX,
        WRONG_CLASS,
        COLOR_ISSUE,
        LIGHTING_ISSUE,
        OTHER_FAILURE,
    }
)

_SOURCE_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_BUILTIN_THRESHOLD_SOURCES = ("yolo", "color")


@dataclass(frozen=True)
class ReviewFailureClassification:
    """One operator-selected failure category and the subsystem that produced it."""

    category: str
    source: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.category not in VALID_FAILURE_CATEGORIES:
            raise ValueError(f"Unsupported failure category: {self.category}")
        if not _SOURCE_KEY_PATTERN.fullmatch(self.source):
            raise ValueError(f"Invalid failure source: {self.source}")
        normalized_note = self.note.strip()
        if len(normalized_note) > 500:
            raise ValueError("Failure note must not exceed 500 characters")
        if self.category == OTHER_FAILURE and not normalized_note:
            raise ValueError("A custom failure note is required for other failures")

    def to_columns(self) -> dict[str, str]:
        """Return manifest columns without coupling the category to an action route."""
        return {
            "failure_category": self.category,
            "failure_source": self.source,
            "failure_note": self.note.strip(),
        }


def threshold_source_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return built-in and detector-provided threshold sources in stable order.

    The source column intentionally accepts a generic stable identifier. A future
    detector therefore becomes available without a manifest schema migration.
    """
    sources = list(_BUILTIN_THRESHOLD_SOURCES)
    detector = normalize_source_key(row.get("detector"))
    if detector and detector not in sources:
        sources.append(detector)
    return tuple(sources)


def normalize_source_key(value: Any) -> str:
    """Normalize a subsystem name to a stable manifest identifier when possible."""
    normalized = str(value or "").strip().lower().replace(" ", "_")
    return normalized if _SOURCE_KEY_PATTERN.fullmatch(normalized) else ""
