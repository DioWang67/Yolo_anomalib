"""負責載入 LED 色彩模型並對偵測結果進行色彩檢查的服務。"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from core.color_qc_enhanced import ColorQCEnhanced
from core.models import ColorCheckItemResult, ColorCheckResult
from core.stats_color_checker import ColorDecisionTuning, StatsColorChecker

logger = logging.getLogger(__name__)

_DEFAULT_GENERIC_DETECTOR_CLASSES = ("LED",)

#: Every detection ROI was measured against the loaded color model.
COLOR_CHECK_EVALUATED_STATUS = "evaluated"
#: There was no detection ROI to measure, so no color evidence exists.
COLOR_CHECK_NO_DETECTIONS_STATUS = "no_detections"


def _is_expected_color_match(
    expected_color: object,
    observed_color: object,
    supported_colors: Iterable[str],
    generic_detector_classes: Iterable[str],
) -> bool:
    """Return whether an observed color satisfies a color-labelled detection.

    Some detectors use generic class names (for example, ``LED``). Those class
    names are not color expectations and must continue to rely on the color
    checker's threshold result alone. Every other detector class is treated as
    a color expectation and therefore fails closed if the checker cannot
    evaluate it.
    """
    expected = str(expected_color or "").strip().casefold()
    if not expected:
        return True

    supported = {
        str(color or "").strip().casefold()
        for color in supported_colors
        if str(color or "").strip()
    }
    generic = {
        str(class_name or "").strip().casefold()
        for class_name in generic_detector_classes
        if str(class_name or "").strip()
    }
    if expected in generic:
        return True
    return (
        expected in supported
        and expected == str(observed_color or "").strip().casefold()
    )


def _configured_colors_are_supported(
    configured_colors: Iterable[str],
    supported_colors: Iterable[str],
) -> bool:
    """Fail closed when explicit color candidates cannot be evaluated."""
    configured = {
        str(color or "").strip().casefold()
        for color in configured_colors
        if str(color or "").strip()
    }
    if not configured:
        return True
    supported = {
        str(color or "").strip().casefold()
        for color in supported_colors
        if str(color or "").strip()
    }
    return bool(configured & supported)


def _supported_color_names(checker: object) -> frozenset[str]:
    """Return the normalized color vocabulary exposed by a loaded checker."""
    return frozenset(
        normalized.casefold()
        for color in getattr(checker, "supported_colors", ())
        if (normalized := str(color or "").strip())
    )


def _normalize_candidates(candidates: Iterable[str] | None) -> tuple[str, ...]:
    if candidates is None:
        return ()
    if isinstance(candidates, str):
        return (candidates,)
    try:
        return tuple(candidates)
    except TypeError:
        return ()


def _supported_candidates(
    candidates: Iterable[object],
    supported_colors: frozenset[str],
) -> list[str] | None:
    filtered = [
        normalized
        for candidate in candidates
        if (normalized := str(candidate or "").strip())
        and normalized.casefold() in supported_colors
    ]
    return filtered or None


class ColorCheckerService:
    """Wrapper around ColorQCEnhanced that manages model lifecycle.

    Responsibilities:
    - Load advanced JSON color model (once per path change)
    - Provide helper to run color check across multiple detections' ROIs
    """

    def __init__(self) -> None:
        self._checker: Any | None = None
        self._model_path: str | None = None
        self._checker_type: str = "color_qc"
        self._decision_tuning: dict[str, Any] | None = None

    def ensure_loaded(
        self,
        model_path: str,
        overrides: dict[str, float] | None = None,
        rules_overrides: dict[str, dict[str, float | None]] | None = None,
        checker_type: str = "color_qc",
        default_threshold: float | None = None,
        decision_tuning: dict[str, Any] | None = None,
    ) -> None:
        """Load/Reload the color model if needed and apply overrides if provided."""
        checker_type = (checker_type or "color_qc").lower()
        if checker_type == "led_qc":
            checker_type = "color_qc"  # backward compatibility alias
        need_reload = (
            self._checker is None
            or self._model_path != model_path
            or self._checker_type != checker_type
            or (checker_type == "stats" and self._decision_tuning != decision_tuning)
        )
        if need_reload and checker_type == "stats":
            try:
                tuning = ColorDecisionTuning.from_dict(decision_tuning)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Invalid color_decision_tuning %s (%s); using defaults",
                    decision_tuning,
                    e,
                )
                tuning = ColorDecisionTuning()
            try:
                # Runtime overrides are deliberately not baked in here: they are
                # applied below through the same reset-then-apply path used when
                # an already-loaded checker is reused, so both paths produce an
                # identical effective configuration.
                self._checker = StatsColorChecker.from_json(model_path, tuning=tuning)
                self._checker_type = checker_type
                self._model_path = model_path
                self._decision_tuning = (
                    dict(decision_tuning) if decision_tuning else None
                )
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                logger.warning("Failed to load StatsColorChecker from %s: %s", model_path, e)
                self._checker = None
                self._model_path = None
                self._decision_tuning = None
                raise RuntimeError(
                    f"Failed to load StatsColorChecker from {model_path}: {e}"
                ) from e
        elif need_reload:
            try:
                self._checker = ColorQCEnhanced.from_json(model_path)
                self._model_path = model_path
                self._checker_type = checker_type
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                logger.warning("Failed to load ColorQCEnhanced from %s: %s", model_path, e)
                self._checker = None
                self._model_path = None
                raise RuntimeError(
                    f"Failed to load ColorQCEnhanced from {model_path}: {e}"
                ) from e

        # Reapply the whole runtime configuration on every invocation, including
        # when nothing was supplied. A checker instance is cached across products
        # that share a model file, so anything not restated here must fall back
        # to the model baseline rather than linger from the previous product.
        if checker_type == "stats":
            self._apply_runtime_configuration(
                default_threshold=default_threshold,
                color_thresholds=overrides,
            )
            return
        self._apply_runtime_configuration(
            color_thresholds=overrides,
            color_rules=rules_overrides,
        )

    def _apply_runtime_configuration(self, **configuration: Any) -> None:
        """Hand this invocation's configuration to the checker, failing loudly.

        Raises:
            RuntimeError: If the checker rejects the configuration. The checker
                resets itself to baseline first, so a rejected configuration can
                never leave the previous product's values in effect.
        """
        try:
            self._checker.apply_runtime_configuration(**configuration)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not apply active color configuration: {exc}"
            ) from exc

    def is_ready(self) -> bool:
        """Return True if a model is loaded and ready."""
        return self._checker is not None

    def check_items(
        self,
        frame: np.ndarray,
        processed_image: np.ndarray,
        detections: list[dict[str, Any]],
        candidates: Iterable[str] | None = None,
        generic_classes: Iterable[str] | None = None,
    ) -> ColorCheckResult:
        """Run color check on detections.

        A frame without detections carries no region to measure. Judging the
        full frame instead would let an unrelated background color decide the
        verdict, so the missing evidence is reported and the result fails
        closed.
        """
        if self._checker is None:
            raise RuntimeError("ColorChecker not loaded")

        if not detections:
            logger.info(
                "Color check has no detection ROI to evaluate; failing closed"
            )
            return ColorCheckResult(
                is_ok=False,
                items=[],
                status=COLOR_CHECK_NO_DETECTIONS_STATUS,
            )

        items: list[ColorCheckItemResult] = []
        all_ok = True
        requested_candidates = _normalize_candidates(candidates)
        supported_colors = _supported_color_names(self._checker)
        generic_detector_classes = {
            normalized.casefold()
            for value in (
                _DEFAULT_GENERIC_DETECTOR_CLASSES
                if generic_classes is None
                else _normalize_candidates(generic_classes)
            )
            if (normalized := str(value or "").strip())
        }
        configured_colors = {
            normalized.casefold()
            for value in requested_candidates
            if (normalized := str(value or "").strip())
            and normalized.casefold() not in generic_detector_classes
        }
        configured_colors_are_supported = _configured_colors_are_supported(
            configured_colors,
            supported_colors,
        )
        proc = processed_image if processed_image is not None else frame
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det.get("bbox", [0, 0, 0, 0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(proc.shape[1], x2), min(proc.shape[0], y2)
            roi = proc[y1:y2, x1:x2]
            # Priority: explicit candidates > YOLO class
            candidate_pool: Iterable[object] = requested_candidates
            if not requested_candidates and det.get("class"):
                candidate_pool = (det.get("class"),)
            allowed = _supported_candidates(candidate_pool, supported_colors)
            c_res = self._checker.check(roi, allowed_colors=allowed)
            # The measurement's own verdict, kept separate from whether it
            # agrees with the detector. An unsupported configured vocabulary
            # folds in here rather than below: it means the measurement was
            # taken against the wrong palette, so the color itself is not
            # trustworthy either.
            measurement_is_ok = bool(c_res.is_ok) and configured_colors_are_supported
            item_is_ok = measurement_is_ok and _is_expected_color_match(
                det.get("class"),
                c_res.best_color,
                supported_colors,
                generic_detector_classes,
            )
            items.append(
                ColorCheckItemResult(
                    index=idx,
                    class_name=det.get("class"),
                    bbox=det.get("bbox"),
                    best_color=c_res.best_color,
                    diff=float(c_res.diff),
                    threshold=float(c_res.threshold),
                    is_ok=item_is_ok,
                    measurement_is_ok=measurement_is_ok,
                )
            )
            if not item_is_ok:
                all_ok = False

        return ColorCheckResult(
            is_ok=all_ok,
            items=items,
            status=COLOR_CHECK_EVALUATED_STATUS,
        )
