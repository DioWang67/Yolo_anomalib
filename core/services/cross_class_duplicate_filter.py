from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DuplicateFilterConfigurationError(ValueError):
    """Raised when duplicate-filter policy values are unsafe or inconsistent."""


class DuplicateFilterMode(str, Enum):
    REPORT_ONLY = "report_only"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class DuplicateFilterPolicy:
    """Immutable safety policy for cross-class duplicate analysis."""

    mode: DuplicateFilterMode = DuplicateFilterMode.REPORT_ONLY
    iou_threshold: float = 0.90
    center_distance_ratio_max: float = 0.10
    area_similarity_min: float = 0.80
    require_same_verified_class: bool = True
    require_color_check_pass: bool = True
    require_different_raw_class: bool = True
    require_position_disabled: bool = True

    @classmethod
    def from_options(cls, options: Mapping[str, Any] | None) -> DuplicateFilterPolicy:
        values = dict(options or {})
        raw_mode = str(values.get("mode", DuplicateFilterMode.REPORT_ONLY.value))
        try:
            mode = DuplicateFilterMode(raw_mode.strip().lower())
        except ValueError as exc:
            raise DuplicateFilterConfigurationError(
                "cross_class_duplicate_filter.mode must be report_only or suppress"
            ) from exc

        policy = cls(
            mode=mode,
            iou_threshold=_number(values, "iou_threshold", 0.90),
            center_distance_ratio_max=_number(
                values, "center_distance_ratio_max", 0.10
            ),
            area_similarity_min=_number(values, "area_similarity_min", 0.80),
            require_same_verified_class=_boolean(
                values, "require_same_verified_class", True
            ),
            require_color_check_pass=_boolean(
                values, "require_color_check_pass", True
            ),
            require_different_raw_class=_boolean(
                values, "require_different_raw_class", True
            ),
            require_position_disabled=_boolean(
                values, "require_position_disabled", True
            ),
        )
        if not 0.0 < policy.iou_threshold <= 1.0:
            raise DuplicateFilterConfigurationError(
                "cross_class_duplicate_filter.iou_threshold must be in (0, 1]"
            )
        if not 0.0 <= policy.center_distance_ratio_max <= 1.0:
            raise DuplicateFilterConfigurationError(
                "cross_class_duplicate_filter.center_distance_ratio_max "
                "must be in [0, 1]"
            )
        if not 0.0 < policy.area_similarity_min <= 1.0:
            raise DuplicateFilterConfigurationError(
                "cross_class_duplicate_filter.area_similarity_min must be in (0, 1]"
            )
        return policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "iou_threshold": self.iou_threshold,
            "center_distance_ratio_max": self.center_distance_ratio_max,
            "area_similarity_min": self.area_similarity_min,
            "require_same_verified_class": self.require_same_verified_class,
            "require_color_check_pass": self.require_color_check_pass,
            "require_different_raw_class": self.require_different_raw_class,
            "require_position_disabled": self.require_position_disabled,
        }


@dataclass(frozen=True)
class _DetectionView:
    index: int
    raw_class: str
    verified_class: str
    confidence: float
    bbox: tuple[float, float, float, float]
    color_check_passed: bool


@dataclass(frozen=True)
class _Pair:
    first_index: int
    second_index: int
    iou: float
    center_distance_ratio: float
    area_similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_index": self.first_index,
            "second_index": self.second_index,
            "iou": round(self.iou, 6),
            "center_distance_ratio": round(self.center_distance_ratio, 6),
            "area_similarity": round(self.area_similarity, 6),
        }


def analyze_cross_class_duplicates(
    detections: Sequence[Mapping[str, Any]],
    color_items_by_index: Mapping[int, Mapping[str, Any]],
    policy: DuplicateFilterPolicy,
) -> dict[str, Any]:
    """Return deterministic duplicate candidates without mutating detections."""

    views: list[_DetectionView] = []
    invalid_indices: list[int] = []
    for index, detection in enumerate(detections):
        view = _build_view(index, detection, color_items_by_index.get(index))
        if view is None:
            invalid_indices.append(index)
            continue
        views.append(view)

    pairs: list[_Pair] = []
    for left_pos, left in enumerate(views):
        for right in views[left_pos + 1 :]:
            pair = _qualified_pair(left, right, policy)
            if pair is not None:
                pairs.append(pair)

    rank = {
        view.index: position
        for position, view in enumerate(sorted(views, key=_retention_sort_key))
    }
    pair_by_indices = {
        frozenset((pair.first_index, pair.second_index)): pair for pair in pairs
    }
    proposed: list[dict[str, Any]] = []
    rejected: set[int] = set()
    for winner in sorted(views, key=_retention_sort_key):
        if winner.index in rejected:
            continue
        losers = [
            view
            for view in views
            if view.index not in rejected
            and view.index != winner.index
            and frozenset((winner.index, view.index)) in pair_by_indices
            and rank[winner.index] < rank[view.index]
        ]
        for loser in sorted(losers, key=_retention_sort_key):
            if loser.index in rejected:
                continue
            pair = pair_by_indices[frozenset((winner.index, loser.index))]
            rejected.add(loser.index)
            proposed.append(
                {
                    "reason": "CROSS_CLASS_DUPLICATE",
                    "policy_version": 1,
                    "kept_index": winner.index,
                    "suppressed_index": loser.index,
                    "kept_raw_class": winner.raw_class,
                    "suppressed_raw_class": loser.raw_class,
                    "verified_class": winner.verified_class,
                    "kept_confidence": round(winner.confidence, 6),
                    "suppressed_confidence": round(loser.confidence, 6),
                    "iou": round(pair.iou, 6),
                    "center_distance_ratio": round(
                        pair.center_distance_ratio, 6
                    ),
                    "area_similarity": round(pair.area_similarity, 6),
                }
            )

    candidate_payload = []
    views_by_index = {view.index: view for view in views}
    for pair in sorted(pairs, key=lambda item: (item.first_index, item.second_index)):
        first = views_by_index[pair.first_index]
        second = views_by_index[pair.second_index]
        candidate_payload.append(
            {
                **pair.to_dict(),
                "first_raw_class": first.raw_class,
                "second_raw_class": second.raw_class,
                "verified_class": first.verified_class,
            }
        )

    return {
        "policy_version": 1,
        "policy": policy.to_dict(),
        "raw_count": len(detections),
        "valid_count": len(views),
        "invalid_indices": invalid_indices,
        "candidate_count": len(candidate_payload),
        "candidates": candidate_payload,
        "proposed_suppressions": proposed,
        "proposed_suppressed_indices": [
            item["suppressed_index"] for item in proposed
        ],
    }


def _build_view(
    index: int,
    detection: Mapping[str, Any],
    color_item: Mapping[str, Any] | None,
) -> _DetectionView | None:
    bbox = _bbox(detection.get("bbox"))
    if bbox is None:
        return None
    raw_class = str(detection.get("class") or "").strip()
    verified_class = str(detection.get("verified_class") or "").strip()
    try:
        confidence = float(detection.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence):
        return None
    color_best = str((color_item or {}).get("best_color") or "").strip()
    return _DetectionView(
        index=index,
        raw_class=raw_class,
        verified_class=verified_class,
        confidence=confidence,
        bbox=bbox,
        color_check_passed=bool(
            color_item is not None
            and color_item.get("is_ok") is True
            and verified_class
            and color_best.casefold() == verified_class.casefold()
        ),
    )


def _qualified_pair(
    left: _DetectionView,
    right: _DetectionView,
    policy: DuplicateFilterPolicy,
) -> _Pair | None:
    if (
        policy.require_different_raw_class
        and (
            not left.raw_class
            or not right.raw_class
            or left.raw_class.casefold() == right.raw_class.casefold()
        )
    ):
        return None
    if (
        policy.require_same_verified_class
        and (
            not left.verified_class
            or left.verified_class.casefold() != right.verified_class.casefold()
        )
    ):
        return None
    if policy.require_color_check_pass and not (
        left.color_check_passed and right.color_check_passed
    ):
        return None

    iou = _intersection_over_union(left.bbox, right.bbox)
    if iou < policy.iou_threshold:
        return None
    center_ratio = _center_distance_ratio(left.bbox, right.bbox)
    if center_ratio > policy.center_distance_ratio_max:
        return None
    area_similarity = _area_similarity(left.bbox, right.bbox)
    if area_similarity < policy.area_similarity_min:
        return None
    return _Pair(
        first_index=min(left.index, right.index),
        second_index=max(left.index, right.index),
        iou=iou,
        center_distance_ratio=center_ratio,
        area_similarity=area_similarity,
    )


def _retention_sort_key(view: _DetectionView) -> tuple[Any, ...]:
    # Confidence is authoritative. Geometry and names make equal-confidence
    # outcomes stable even if an inference backend changes list ordering.
    return (
        -view.confidence,
        view.bbox,
        view.raw_class.casefold(),
        view.verified_class.casefold(),
        view.index,
    )


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    inter_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    inter_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = inter_width * inter_height
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance_ratio(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    distance = math.dist(left_center, right_center)
    left_diagonal = math.hypot(left[2] - left[0], left[3] - left[1])
    right_diagonal = math.hypot(right[2] - right[0], right[3] - right[1])
    denominator = min(left_diagonal, right_diagonal)
    return distance / denominator if denominator > 0.0 else math.inf


def _area_similarity(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return min(left_area, right_area) / max(left_area, right_area)


def _number(options: Mapping[str, Any], key: str, default: float) -> float:
    value = options.get(key, default)
    if isinstance(value, bool):
        raise DuplicateFilterConfigurationError(
            f"cross_class_duplicate_filter.{key} must be a number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise DuplicateFilterConfigurationError(
            f"cross_class_duplicate_filter.{key} must be a number"
        ) from exc
    if not math.isfinite(number):
        raise DuplicateFilterConfigurationError(
            f"cross_class_duplicate_filter.{key} must be finite"
        )
    return number


def _boolean(options: Mapping[str, Any], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if isinstance(value, bool):
        return value
    raise DuplicateFilterConfigurationError(
        f"cross_class_duplicate_filter.{key} must be true or false"
    )
