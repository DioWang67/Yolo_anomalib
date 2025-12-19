"""Position validation utilities for YOLO detections."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PositionValidator:
    """Validate whether detection centers stay within the expected tolerance."""

    def __init__(self, config, product: str, area: str) -> None:
        self.config = config
        self.product = product
        self.area = area
        self.pos_config = self.config.get_position_config(product, area) or {}
        self.expected_boxes: dict[str, dict[str, float]] = self.pos_config.get(
            "expected_boxes", {}
        )
        self.mode = str(self.pos_config.get("mode", "bbox")).lower()

        self.imgsz = self._get_image_size()
        self.tolerance_px = self._calculate_tolerance_px()
        self._validate_config()
        self.expected_centers = self._precompute_expected_centers()

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
                    class_name,
                    x1,
                    x2,
                    y1,
                    y2,
                )
            if x1 < 0 or y1 < 0 or x2 > self.imgsz or y2 > self.imgsz:
                logger.warning(
                    "Expected box for %s is outside image size %s: (%s, %s, %s, %s)",
                    class_name,
                    self.imgsz,
                    x1,
                    y1,
                    x2,
                    y2,
                )

    def _precompute_expected_centers(self) -> dict[str, tuple[float, float]]:
        centers: dict[str, tuple[float, float]] = {}
        for class_name, box in self.expected_boxes.items():
            try:
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
        """Annotate detections with position meta-data."""
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

            class_name = det.get("class", "")
            (
                status,
                error_distance,
                offset,
                expected_center,
                edge_distance,
            ) = self._check_position(class_name, cx, cy)

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
            det.get("class", ""),
            cx,
            cy,
            self.imgsz,
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
        self, class_name: str, cx: float, cy: float
    ) -> tuple[str, float | None, tuple[float, float], tuple[float, float] | None, float]:
        box = self.expected_boxes.get(class_name)
        if not box:
            return "UNEXPECTED", None, (0.0, 0.0), None, 0.0

        error_distance, dx, dy, expected_center, edge_distance = self._compute_position_error(
            class_name, cx, cy, box
        )

        if error_distance <= self.tolerance_px:
            status = "CORRECT"
        else:
            status = "WRONG"
            logger.debug(
                "Position mismatch: class=%s, center=(%.1f, %.1f), expected=%s, offset=(%.1f, %.1f), "
                "distance=%.1fpx, tolerance=%.1fpx, mode=%s",
                class_name,
                cx,
                cy,
                expected_center,
                dx,
                dy,
                error_distance,
                self.tolerance_px,
                self.mode,
            )

        return status, error_distance, (dx, dy), expected_center, edge_distance

    @staticmethod
    def _point_to_rect_distance(
        px: float,
        py: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
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

    def _compute_position_error(
        self,
        class_name: str,
        cx: float,
        cy: float,
        box: dict[str, Any],
    ) -> tuple[float, float, float, tuple[float, float] | None, float]:
        edge_distance = self._point_to_rect_distance(
            cx,
            cy,
            float(box["x1"]),
            float(box["y1"]),
            float(box["x2"]),
            float(box["y2"]),
        )

        expected_center = self.expected_centers.get(class_name)
        dx = dy = 0.0
        if expected_center:
            dx = cx - expected_center[0]
            dy = cy - expected_center[1]

        if self.mode in {"region", "bbox_region"}:
            return edge_distance, dx, dy, expected_center, edge_distance

        if expected_center is None:
            return edge_distance, dx, dy, expected_center, edge_distance

        axial_error = max(abs(dx), abs(dy))
        if edge_distance > 0:
            error = max(axial_error, edge_distance)
        else:
            error = axial_error

        return error, dx, dy, expected_center, edge_distance
