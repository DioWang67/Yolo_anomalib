from typing import Dict, List, Any
import logging


class PositionValidator:
    def __init__(self, config, product: str, area: str):
        self.config = config
        self.product = product
        self.area = area
        self.pos_config = self.config.get_position_config(product, area) or {}
        self.expected_boxes = self.pos_config.get("expected_boxes", {})

    def validate(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for det in detections:
            if "cx" not in det or "cy" not in det:
                det["position_status"] = "UNKNOWN"
                continue
            status = self.is_position_correct(
                det.get("class"), float(det.get("cx", 0.0)), float(det.get("cy", 0.0))
            )
            det["position_status"] = status
        return detections

    def is_position_correct(self, class_name: str, cx: float, cy: float) -> str:
        box = self.expected_boxes.get(class_name)
        if not box:
            return "UNEXPECTED"

        imgsz = self.config.imgsz[0]
        tolerance_ratio = self.config.get_tolerance_ratio(self.product, self.area)
        tolerance_px = imgsz * tolerance_ratio

        x1, y1 = box["x1"] - tolerance_px, box["y1"] - tolerance_px
        x2, y2 = box["x2"] + tolerance_px, box["y2"] + tolerance_px

        dx = min(abs(cx - x1), abs(cx - x2)) if cx < x1 or cx > x2 else 0
        dy = min(abs(cy - y1), abs(cy - y2)) if cy < y1 or cy > y2 else 0

        if dx == 0 and dy == 0:
            return "CORRECT"
        try:
            logging.getLogger(__name__).debug(
                "Position error: class=%s, dx=%.1f, dy=%.1f, tol_px=%.1f, x=[%.1f,%.1f], y=[%.1f,%.1f], center=(%.1f,%.1f)",
                class_name, dx, dy, tolerance_px, x1, x2, y1, y2, cx, cy
            )
        except Exception:
            pass
        return "WRONG"

    def has_wrong_position(self, detections: List[Dict[str, Any]]) -> bool:
        return any(det.get("position_status") == "WRONG" for det in detections)

    def evaluate_status(self, detections: List[Dict[str, Any]], missing_items: List[str]) -> str:
        if self.config.is_position_check_enabled(self.product, self.area):
            if missing_items or self.has_wrong_position(detections):
                return "FAIL"
        else:
            if missing_items:
                return "FAIL"
        return "PASS"

