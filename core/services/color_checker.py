from __future__ import annotations


from typing import List, Optional, Dict, Any

import numpy as np

from core.led_qc_enhanced import LEDQCEnhanced
from core.models import ColorCheckItemResult, ColorCheckResult


class ColorCheckerService:
    """Wrapper around LEDQCEnhanced that manages model lifecycle.

    Responsibilities:
    - Load advanced JSON color model (once per path change)
    - Provide helper to run color check across multiple detections' ROIs
    """

    def __init__(self) -> None:
        self._checker: Optional[LEDQCEnhanced] = None
        self._model_path: Optional[str] = None

    def ensure_loaded(
        self,
        model_path: str,
        overrides: Optional[Dict[str, float]] = None,
        rules_overrides: Optional[Dict[str,
                                       Dict[str, Optional[float]]]] = None,
    ) -> None:
        """Load/Reload the color model if needed and apply overrides if provided."""
        if self._checker is None or self._model_path != model_path:
            self._checker = LEDQCEnhanced.from_json(model_path)
            self._model_path = model_path
        # Apply threshold overrides (case-insensitive) if provided
        if overrides:
            try:
                self._checker.apply_threshold_overrides(overrides)
            except Exception:
                pass
        if rules_overrides:
            try:
                self._checker.apply_color_rules_overrides(rules_overrides)
            except Exception:
                pass

    def is_ready(self) -> bool:
        """Return True if a model is loaded and ready."""
        return self._checker is not None

    def check_items(
        self,
        frame: np.ndarray,
        processed_image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> ColorCheckResult:
        """Run color check on detections.

        If no detections are present, fallback to check on full frame.
        """
        if self._checker is None:
            raise RuntimeError("ColorChecker not loaded")

        items: List[ColorCheckItemResult] = []
        all_ok = True
        if detections:
            proc = processed_image if processed_image is not None else frame
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(proc.shape[1], x2), min(proc.shape[0], y2)
                roi = proc[y1:y2, x1:x2]
                allowed = [det.get("class")] if det.get("class") else None
                c_res = self._checker.check(roi, allowed_colors=allowed)
                items.append(
                    ColorCheckItemResult(
                        index=idx,
                        class_name=det.get("class"),
                        bbox=det.get("bbox"),
                        best_color=c_res.best_color,
                        diff=float(c_res.diff),
                        threshold=float(c_res.threshold),
                        is_ok=bool(c_res.is_ok),
                    )
                )
                if not c_res.is_ok:
                    all_ok = False
        else:
            # No detections: estimate on full frame
            c_res = self._checker.check(frame)
            items.append(
                ColorCheckItemResult(
                    index=-1,
                    class_name=None,
                    bbox=None,
                    best_color=c_res.best_color,
                    diff=float(c_res.diff),
                    threshold=float(c_res.threshold),
                    is_ok=bool(c_res.is_ok),
                )
            )
            all_ok = bool(c_res.is_ok)

        return ColorCheckResult(is_ok=all_ok, items=items)
