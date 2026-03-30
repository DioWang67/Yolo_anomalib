"""Position validation utilities for YOLO detections.

Supports three modes:
- **center** (default): Euclidean distance from detection center to expected center.
- **iou**: 1 - IoU between detection bbox and expected bbox.
- **region**: Point-to-rectangle edge distance (existing behaviour).

Legacy mode name ``bbox`` is accepted as an alias for ``center``.

Multi-instance classes use ``ClassName#N`` indexed keys in expected_boxes
and are matched via greedy nearest-neighbor assignment.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


def _base_class_name(key: str) -> str:
    """Strip ``#N`` instance index suffix from a class key."""
    idx = key.rfind("#")
    if idx > 0:
        suffix = key[idx + 1:]
        if suffix.isdigit():
            return key[:idx]
    return key


class PositionValidator:
    """Validate whether detection centers stay within the expected tolerance."""

    _DEPRECATED_MODES = {"bbox"}

    def __init__(self, config, product: str, area: str) -> None:
        self.config = config
        self.product = product
        self.area = area
        self.pos_config = self.config.get_position_config(product, area) or {}
        self.expected_boxes: dict[str, dict[str, Any]] = self.pos_config.get(
            "expected_boxes", {}
        )
        self.mode = str(self.pos_config.get("mode", "center")).lower()

        if self.mode in self._DEPRECATED_MODES:
            logger.warning(
                "Position mode '%s' is deprecated — treating as 'center' "
                "(Euclidean distance). Update config to mode: center.",
                self.mode,
            )
            self.mode = "center"

        self.imgsz = self._get_image_size()
        self.tolerance_px = self._calculate_tolerance_px()
        self._validate_config()
        self.expected_centers = self._precompute_expected_centers()

        # Group expected boxes by base class for multi-instance matching
        self._base_class_groups: dict[str, list[str]] = defaultdict(list)
        for key in self.expected_boxes:
            self._base_class_groups[_base_class_name(key)].append(key)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def _get_image_size(self) -> int:
        area_imgsz = self.pos_config.get("imgsz")
        if area_imgsz:
            return int(area_imgsz if isinstance(area_imgsz, (int, float)) else area_imgsz[0])

        global_imgsz = getattr(self.config, "imgsz", None)
        if global_imgsz:
            return int(global_imgsz if isinstance(global_imgsz, (int, float)) else global_imgsz[0])

        logger.warning("imgsz missing in position config; fallback to 640")
        return 640

    def _calculate_tolerance_px(self) -> float:
        tolerance = float(self.pos_config.get("tolerance", 0.0))
        tolerance_unit = str(self.pos_config.get("tolerance_unit", "percent")).lower()
        if tolerance_unit == "pixel":
            return tolerance
        return self.imgsz * (tolerance / 100.0)

    def _get_class_tolerance_px(self, box: dict[str, Any]) -> float:
        """Return per-class tolerance in pixels, falling back to global."""
        class_tol = box.get("tolerance")
        if class_tol is not None:
            tol = float(class_tol)
            tolerance_unit = str(self.pos_config.get("tolerance_unit", "percent")).lower()
            if tolerance_unit == "pixel":
                return tol
            return self.imgsz * (tol / 100.0)
        return self.tolerance_px

    def _validate_config(self) -> None:
        for class_name, box in self.expected_boxes.items():
            missing = [key for key in ("x1", "y1", "x2", "y2") if key not in box]
            if missing:
                logger.error("Expected box for %s is missing keys: %s", class_name, missing)
                continue

            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            if x1 >= x2 or y1 >= y2:
                logger.error(
                    "Invalid box for %s: x1>=x2 or y1>=y2 (x1=%s, x2=%s, y1=%s, y2=%s)",
                    class_name, x1, x2, y1, y2,
                )
            if x1 < 0 or y1 < 0 or x2 > self.imgsz or y2 > self.imgsz:
                logger.warning(
                    "Expected box for %s is outside image size %s: (%s, %s, %s, %s)",
                    class_name, self.imgsz, x1, y1, x2, y2,
                )

    def _precompute_expected_centers(self) -> dict[str, tuple[float, float]]:
        centers: dict[str, tuple[float, float]] = {}
        for class_name, box in self.expected_boxes.items():
            try:
                # Prefer precomputed cx/cy from statistical config
                if "cx" in box and "cy" in box:
                    centers[class_name] = (float(box["cx"]), float(box["cy"]))
                else:
                    cx = (float(box["x1"]) + float(box["x2"])) / 2.0
                    cy = (float(box["y1"]) + float(box["y2"])) / 2.0
                    centers[class_name] = (cx, cy)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to compute expected center for %s: %s", class_name, exc)
        return centers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Annotate detections with position meta-data.

        For multi-instance classes (``#N`` keys), detections are matched to
        expected positions via greedy nearest-neighbour before annotation.
        """
        # Group detections by class for multi-instance matching
        det_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        unclassified: list[dict[str, Any]] = []
        for det in detections:
            cls = det.get("class", "")
            if cls in self._base_class_groups:
                det_by_class[cls].append(det)
            else:
                unclassified.append(det)

        # Assign multi-instance detections to specific expected keys
        assigned: dict[int, str] = {}  # id(det) -> expected key

        for base_class, keys in self._base_class_groups.items():
            if len(keys) <= 1 and "#" not in keys[0]:
                # Single-instance: assign best detection directly
                candidates = det_by_class.get(base_class, [])
                if candidates:
                    best = max(candidates, key=lambda d: float(d.get("confidence", 0)))
                    assigned[id(best)] = keys[0]
                continue

            # Multi-instance: greedy nearest-neighbour matching
            candidates = det_by_class.get(base_class, [])
            self._greedy_assign(keys, candidates, assigned)

        # Now annotate each detection
        for det in detections:
            if "cx" not in det or "cy" not in det:
                self._mark_unknown(det, reason="missing center coordinates")
                continue

            try:
                cx = float(det["cx"])
                cy = float(det["cy"])
            except (TypeError, ValueError):
                self._mark_error(det, reason="invalid center values", payload=det)
                continue

            if not self._is_valid_coordinate(cx, cy):
                self._mark_invalid(det, cx, cy)
                continue

            expected_key = assigned.get(id(det))
            if expected_key:
                class_name = expected_key
            else:
                class_name = det.get("class", "")

            (
                status,
                error_distance,
                offset,
                expected_center,
                edge_distance,
            ) = self._check_position(class_name, cx, cy, det)

            det["position_status"] = status
            det["position_error"] = error_distance
            det["position_offset"] = {"dx": offset[0], "dy": offset[1]}
            det["position_expected_center"] = (
                {"cx": expected_center[0], "cy": expected_center[1]}
                if expected_center
                else None
            )
            det["position_edge_distance"] = edge_distance
        return detections

    def _greedy_assign(
        self,
        expected_keys: list[str],
        candidates: list[dict[str, Any]],
        assigned: dict[int, str],
    ) -> None:
        """Assign detections to expected keys by nearest Euclidean distance."""
        pairs: list[tuple[float, int, int]] = []
        for ki, key in enumerate(expected_keys):
            ec = self.expected_centers.get(key)
            if not ec:
                continue
            for di, det in enumerate(candidates):
                dcx = det.get("cx")
                dcy = det.get("cy")
                if dcx is None or dcy is None:
                    continue
                dist = ((float(dcx) - ec[0]) ** 2 + (float(dcy) - ec[1]) ** 2) ** 0.5
                pairs.append((dist, ki, di))

        pairs.sort(key=lambda t: t[0])
        used_keys: set[int] = set()
        used_dets: set[int] = set()

        for _, ki, di in pairs:
            if ki in used_keys or di in used_dets:
                continue
            used_keys.add(ki)
            used_dets.add(di)
            assigned[id(candidates[di])] = expected_keys[ki]

    def has_wrong_position(self, detections: list[dict[str, Any]]) -> bool:
        return any(det.get("position_status") == "WRONG" for det in detections)

    def has_unexpected_position(self, detections: list[dict[str, Any]]) -> bool:
        return any(det.get("position_status") == "UNEXPECTED" for det in detections)

    def get_position_errors(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [det for det in detections if det.get("position_status") in {"WRONG", "UNEXPECTED"}]

    def evaluate_status(
        self, detections: list[dict[str, Any]], missing_items: list[str]
    ) -> str:
        if self.config.is_position_check_enabled(self.product, self.area):
            if missing_items or self.has_wrong_position(detections):
                return "FAIL"
        else:
            if missing_items:
                return "FAIL"
        return "PASS"

    def get_summary(self, detections: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "total": len(detections),
            "correct": sum(1 for d in detections if d.get("position_status") == "CORRECT"),
            "wrong": sum(1 for d in detections if d.get("position_status") == "WRONG"),
            "unexpected": sum(1 for d in detections if d.get("position_status") == "UNEXPECTED"),
            "disabled": sum(1 for d in detections if d.get("position_status") in {None, "DISABLED"}),
            "unknown": sum(1 for d in detections if d.get("position_status") == "UNKNOWN"),
            "tolerance_px": self.tolerance_px,
            "imgsz": self.imgsz,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _mark_unknown(self, det: dict[str, Any], *, reason: str) -> None:
        det["position_status"] = "UNKNOWN"
        det["position_error"] = None
        det["position_offset"] = None
        det["position_expected_center"] = None
        det["position_edge_distance"] = None
        logger.warning("Position check skipped (%s): %s", reason, det)

    def _mark_invalid(self, det: dict[str, Any], cx: float, cy: float) -> None:
        det["position_status"] = "INVALID"
        det["position_error"] = None
        det["position_offset"] = None
        det["position_expected_center"] = None
        det["position_edge_distance"] = None
        logger.warning(
            "Detection out of image bounds: class=%s, cx=%.2f, cy=%.2f, size=%s",
            det.get("class", ""), cx, cy, self.imgsz,
        )

    def _mark_error(self, det: dict[str, Any], *, reason: str, payload: Any) -> None:
        det["position_status"] = "ERROR"
        det["position_error"] = None
        det["position_offset"] = None
        det["position_expected_center"] = None
        det["position_edge_distance"] = None
        logger.error("Position validation failed (%s): %s", reason, payload)

    def _is_valid_coordinate(self, cx: float, cy: float) -> bool:
        return 0 <= cx <= self.imgsz and 0 <= cy <= self.imgsz

    def _check_position(
        self,
        class_name: str,
        cx: float,
        cy: float,
        det: dict[str, Any] | None = None,
    ) -> tuple[str, float | None, tuple[float, float], tuple[float, float] | None, float]:
        box = self.expected_boxes.get(class_name)
        if not box:
            return "UNEXPECTED", None, (0.0, 0.0), None, 0.0

        error_distance, dx, dy, expected_center, edge_distance = self._compute_position_error(
            class_name, cx, cy, box, det
        )

        effective_tolerance = self._get_class_tolerance_px(box)

        if self.mode == "iou":
            # For IoU mode, tolerance is minimum acceptable IoU (0-1 scale).
            # Read raw config value directly (not pixel-converted).
            raw_tol = box.get("tolerance")
            if raw_tol is None:
                raw_tol = float(self.pos_config.get("tolerance", 0.0))
            else:
                raw_tol = float(raw_tol)
            # Normalize: if > 1, treat as percentage (e.g. 50 → 0.50)
            min_iou = raw_tol / 100.0 if raw_tol > 1.0 else raw_tol
            threshold = 1.0 - min_iou  # max acceptable error
            if error_distance <= threshold:
                status = "CORRECT"
            else:
                status = "WRONG"
        elif error_distance <= effective_tolerance:
            status = "CORRECT"
        else:
            status = "WRONG"
            logger.debug(
                "Position mismatch: class=%s, center=(%.1f, %.1f), expected=%s, "
                "offset=(%.1f, %.1f), distance=%.1fpx, tolerance=%.1fpx, mode=%s",
                class_name, cx, cy, expected_center, dx, dy,
                error_distance, effective_tolerance, self.mode,
            )

        return status, error_distance, (dx, dy), expected_center, edge_distance

    @staticmethod
    def _point_to_rect_distance(
        px: float, py: float,
        x1: float, y1: float, x2: float, y2: float,
    ) -> float:
        if px < x1:
            dx = x1 - px
        elif px > x2:
            dx = px - x2
        else:
            dx = 0.0

        if py < y1:
            dy = y1 - py
        elif py > y2:
            dy = py - y2
        else:
            dy = 0.0

        return (dx ** 2 + dy ** 2) ** 0.5

    @staticmethod
    def _compute_iou(
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        """Compute Intersection over Union between two (x1, y1, x2, y2) boxes."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def _compute_position_error(
        self,
        class_name: str,
        cx: float,
        cy: float,
        box: dict[str, Any],
        det: dict[str, Any] | None = None,
    ) -> tuple[float, float, float, tuple[float, float] | None, float]:
        edge_distance = self._point_to_rect_distance(
            cx, cy,
            float(box["x1"]), float(box["y1"]),
            float(box["x2"]), float(box["y2"]),
        )

        expected_center = self.expected_centers.get(class_name)
        dx = dy = 0.0
        if expected_center:
            dx = cx - expected_center[0]
            dy = cy - expected_center[1]

        # IoU mode: compare detection bbox against expected bbox
        if self.mode == "iou" and det is not None:
            det_bbox = det.get("bbox") or det.get("bbox_letterbox")
            if det_bbox and len(det_bbox) >= 4:
                iou = self._compute_iou(
                    (float(det_bbox[0]), float(det_bbox[1]),
                     float(det_bbox[2]), float(det_bbox[3])),
                    (float(box["x1"]), float(box["y1"]),
                     float(box["x2"]), float(box["y2"])),
                )
                error = 1.0 - iou  # 0 = perfect overlap, 1 = no overlap
                return error, dx, dy, expected_center, edge_distance

        # Region mode: use edge distance
        if self.mode in {"region", "bbox_region"}:
            return edge_distance, dx, dy, expected_center, edge_distance

        if expected_center is None:
            return edge_distance, dx, dy, expected_center, edge_distance

        # Center mode (default): Euclidean distance
        euclidean_error = (dx ** 2 + dy ** 2) ** 0.5
        return euclidean_error, dx, dy, expected_center, edge_distance
