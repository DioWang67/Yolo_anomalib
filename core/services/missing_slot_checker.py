from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.services.alignment import (
    ExpectedLayoutAlignment,
    base_class_name,
    extract_layout_alignment,
    resolve_missing_expected_keys,
)


@dataclass(frozen=True)
class MissingSlotDecision:
    """Evaluation result for one expected-but-missing slot."""

    expected_key: str
    class_name: str
    shifted_box: tuple[int, int, int, int]
    occupied: bool
    reason: str
    score: float


class MissingSlotChecker:
    """Conservative occupancy-based refinement for YOLO missing-item output."""

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        self.options = options or {}
        self.enabled = bool(self.options.get("enabled", False))
        self.min_roi_size = int(self.options.get("min_roi_size", 12))
        self.min_intensity_std = float(self.options.get("min_intensity_std", 10.0))
        self.min_laplacian_var = float(self.options.get("min_laplacian_var", 28.0))
        self.min_edge_ratio = float(self.options.get("min_edge_ratio", 0.035))
        self.min_occupancy_score = float(self.options.get("min_occupancy_score", 1.8))

    def refine_missing_items(
        self,
        processed_image: np.ndarray,
        detections: list[dict[str, Any]],
        missing_items: list[str],
        expected_boxes: dict[str, dict[str, Any]],
    ) -> tuple[list[str], dict[str, Any]]:
        """Refine missing items using slot occupancy evidence."""
        if (
            not self.enabled
            or processed_image is None
            or processed_image.size == 0
            or not missing_items
            or not expected_boxes
        ):
            return list(missing_items), self._empty_report(missing_items)

        alignment = extract_layout_alignment(detections)
        used_expected_keys = {
            str(det.get("position_expected_key"))
            for det in (detections or [])
            if det.get("position_expected_key")
        }

        remaining = list(missing_items)
        decisions: list[MissingSlotDecision] = []
        missing_keys = resolve_missing_expected_keys(
            missing_items,
            expected_boxes,
            used_expected_keys,
        )
        remaining_counter = Counter(str(name).strip() for name in missing_items if str(name).strip())

        for expected_key in missing_keys:
            expected_box = expected_boxes.get(expected_key)
            if not isinstance(expected_box, dict):
                continue
            shifted_box = alignment.shift_box(expected_box)
            occupied, reason, score = self._evaluate_slot(
                processed_image,
                detections,
                shifted_box,
                expected_key=expected_key,
                class_name=base_class_name(expected_key),
            )
            class_name = base_class_name(expected_key)
            decisions.append(
                MissingSlotDecision(
                    expected_key=expected_key,
                    class_name=class_name,
                    shifted_box=shifted_box,
                    occupied=occupied,
                    reason=reason,
                    score=score,
                )
            )
            if occupied and remaining_counter.get(class_name, 0) > 0:
                remaining_counter[class_name] -= 1

        refined_missing: list[str] = []
        for name in missing_items:
            key = str(name).strip()
            if not key:
                continue
            if remaining_counter.get(key, 0) > 0:
                refined_missing.append(key)
                remaining_counter[key] -= 1

        report = {
            "enabled": True,
            "estimated_shift": alignment.to_dict(),
            "recovered_items": [
                decision.class_name for decision in decisions if decision.occupied
            ],
            "remaining_missing_items": list(refined_missing),
            "decisions": [
                {
                    "expected_key": decision.expected_key,
                    "class_name": decision.class_name,
                    "shifted_box": decision.shifted_box,
                    "occupied": decision.occupied,
                    "reason": decision.reason,
                    "score": decision.score,
                }
                for decision in decisions
            ],
        }
        return refined_missing, report

    def _evaluate_slot(
        self,
        processed_image: np.ndarray,
        detections: list[dict[str, Any]],
        shifted_box: tuple[int, int, int, int],
        *,
        expected_key: str,
        class_name: str,
    ) -> tuple[bool, str, float]:
        if self._has_matching_detection_overlap(
            detections,
            shifted_box,
            expected_key=expected_key,
            class_name=class_name,
        ):
            return True, "matching_detection_overlap", 10.0

        roi = self._extract_roi(processed_image, shifted_box)
        if roi is None:
            return False, "invalid_roi", 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        intensity_std = float(gray.std())
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 60, 180)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size or 1)

        score = 0.0
        if intensity_std >= self.min_intensity_std:
            score += 1.0
        if laplacian_var >= self.min_laplacian_var:
            score += 1.0
        if edge_ratio >= self.min_edge_ratio:
            score += 1.0

        occupied = score >= self.min_occupancy_score
        reason = (
            f"roi_stats(std={intensity_std:.1f}, lap={laplacian_var:.1f}, edge={edge_ratio:.3f})"
        )
        return occupied, reason, score

    def _extract_roi(
        self,
        image: np.ndarray,
        shifted_box: tuple[int, int, int, int],
    ) -> np.ndarray | None:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = shifted_box
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 - x1 < self.min_roi_size or y2 - y1 < self.min_roi_size:
            return None
        return image[y1:y2, x1:x2]

    def _has_matching_detection_overlap(
        self,
        detections: list[dict[str, Any]],
        shifted_box: tuple[int, int, int, int],
        *,
        expected_key: str,
        class_name: str,
    ) -> bool:
        sx1, sy1, sx2, sy2 = shifted_box
        for det in detections or []:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            try:
                x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            except (TypeError, ValueError):
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
                continue

            det_expected_key = str(det.get("position_expected_key") or "").strip()
            if det_expected_key:
                if det_expected_key == expected_key:
                    return True
                continue

            det_class = str(det.get("class") or "").strip()
            if det_class == class_name:
                return True
        return False

    @staticmethod
    def _empty_report(missing_items: list[str]) -> dict[str, Any]:
        return {
            "enabled": False,
            "estimated_shift": {"dx": 0.0, "dy": 0.0},
            "recovered_items": [],
            "remaining_missing_items": list(missing_items),
            "decisions": [],
        }
