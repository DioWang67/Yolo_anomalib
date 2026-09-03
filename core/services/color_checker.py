"""負責載入 LED 色彩模型並對偵測結果進行色彩檢查的服務。"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from core.color_baseline_contract import color_model_compatibility_failure
from core.color_qc_enhanced import ColorQCEnhanced
from core.models import ColorCheckItemResult, ColorCheckResult
from core.services.slot_roi import ColorRoiPolicy, extract_bbox_roi
from core.services.station_color_settings import (
    DEFAULT_GENERIC_DETECTOR_CLASSES,
)
from core.stats_color_checker import ColorDecisionTuning, StatsColorChecker

logger = logging.getLogger(__name__)

_DEFAULT_GENERIC_DETECTOR_CLASSES = DEFAULT_GENERIC_DETECTOR_CLASSES

#: Every detection ROI was measured against the loaded color model.
COLOR_CHECK_EVALUATED_STATUS = "evaluated"
#: There was no detection ROI to measure, so no color evidence exists.
COLOR_CHECK_NO_DETECTIONS_STATUS = "no_detections"
#: At least one detection carried a box that yields no pixels, so that item
#: could not be measured. Distinct from ``evaluated``, which promises every ROI
#: was actually compared against the loaded model.
COLOR_CHECK_UNMEASURABLE_ROI_STATUS = "unmeasurable_roi"

#: Log the mismatch and keep running. The default, because a station whose
#: deployed baseline predates the current algorithm must not be stopped by a
#: code update alone -- it needs a rebuilt baseline first, and until then the
#: mismatch is a recorded fact rather than a silent one.
ALGORITHM_ENFORCEMENT_WARN = "warn"
#: Refuse to load a baseline that cannot be shown to match the current
#: algorithm. Set once a baseline rebuilt by that algorithm is deployed.
ALGORITHM_ENFORCEMENT_STRICT = "strict"

#: Reported for an item whose ROI could not be cropped. The measurement never
#: happened, so it must not read as a small distance.
_UNMEASURED_DIFF = 1.0


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


def _unsupported_configured_colors(
    configured_colors: Iterable[str],
    supported_colors: Iterable[str],
) -> frozenset[str]:
    """Return every explicit color candidate the checker cannot evaluate."""
    configured = {
        str(color or "").strip().casefold()
        for color in configured_colors
        if str(color or "").strip()
    }
    supported = {
        str(color or "").strip().casefold()
        for color in supported_colors
        if str(color or "").strip()
    }
    return frozenset(configured - supported)


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


def _resolve_algorithm_enforcement(value: object) -> str:
    """Normalize the configured enforcement mode.

    An unrecognized value resolves to ``strict`` rather than to the permissive
    default: the key is absent unless somebody set it, so anyone who wrote a
    value here was turning enforcement on, and honouring a typo as "warn" would
    silently grant the opposite of what the station config asked for.
    """
    if value is None:
        return ALGORITHM_ENFORCEMENT_WARN
    mode = str(value).strip().casefold()
    if mode in {ALGORITHM_ENFORCEMENT_WARN, ALGORITHM_ENFORCEMENT_STRICT}:
        return mode
    logger.warning(
        "Unknown color_baseline_algorithm_enforcement %r; enforcing %r. "
        "Accepted values are %r and %r.",
        value,
        ALGORITHM_ENFORCEMENT_STRICT,
        ALGORITHM_ENFORCEMENT_WARN,
        ALGORITHM_ENFORCEMENT_STRICT,
    )
    return ALGORITHM_ENFORCEMENT_STRICT


def _supported_candidates(
    candidates: Iterable[object],
    supported_colors: frozenset[str],
    generic_detector_classes: frozenset[str],
) -> list[str] | None:
    """Narrow an explicit color palette to what the loaded model can score.

    ``None`` means "no restriction" and is reserved for the case where no color
    palette was configured at all -- a station that names only generic detector
    classes included, since those are not colors. A palette that *was*
    configured but that the model can score none of returns an empty list,
    which the checker fails closed on. Collapsing that case to ``None`` widened
    a wrong palette into the full vocabulary and reported a measurement taken
    against a palette nobody asked for.
    """
    requested = [
        normalized
        for candidate in candidates
        if (normalized := str(candidate or "").strip())
        and normalized.casefold() not in generic_detector_classes
    ]
    if not requested:
        return None
    return [
        candidate
        for candidate in requested
        if candidate.casefold() in supported_colors
    ]


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
        self._roi_policy = ColorRoiPolicy()
        self._algorithm_enforcement: str = ALGORITHM_ENFORCEMENT_WARN

    def ensure_loaded(
        self,
        model_path: str,
        overrides: dict[str, float] | None = None,
        rules_overrides: dict[str, dict[str, float | None]] | None = None,
        checker_type: str = "color_qc",
        default_threshold: float | None = None,
        decision_tuning: dict[str, Any] | None = None,
        roi_policy: dict[str, Any] | None = None,
        algorithm_enforcement: str | None = None,
    ) -> None:
        """Load/Reload the color model if needed and apply overrides if provided."""
        try:
            resolved_roi_policy = ColorRoiPolicy.from_mapping(roi_policy)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid color ROI policy: {exc}") from exc
        enforcement = _resolve_algorithm_enforcement(algorithm_enforcement)
        checker_type = (checker_type or "color_qc").lower()
        if checker_type == "led_qc":
            checker_type = "color_qc"  # backward compatibility alias
        resolved_tuning: ColorDecisionTuning | None = None
        resolved_tuning_payload: dict[str, float] | None = None
        if checker_type == "stats":
            try:
                resolved_tuning = ColorDecisionTuning.from_dict(decision_tuning)
            except (TypeError, ValueError) as e:
                logger.warning(
                    "Invalid color_decision_tuning %s (%s); using defaults",
                    decision_tuning,
                    e,
                )
                resolved_tuning = ColorDecisionTuning()
            resolved_tuning_payload = resolved_tuning.to_dict()
        need_reload = (
            self._checker is None
            or self._model_path != model_path
            or self._checker_type != checker_type
            or (
                checker_type == "stats"
                and self._decision_tuning != resolved_tuning_payload
            )
            # Geometry is part of a stats baseline's meaning. Reusing the same
            # model path under a different ROI must re-run compatibility rather
            # than merely changing the crop applied to an already-trusted file.
            or (checker_type == "stats" and self._roi_policy != resolved_roi_policy)
            # A tightened mode must re-examine a baseline that was already
            # loaded under the permissive one, or turning enforcement on would
            # do nothing until the next unrelated reload.
            or (checker_type == "stats" and self._algorithm_enforcement != enforcement)
        )
        if need_reload and checker_type == "stats":
            assert resolved_tuning is not None
            assert resolved_tuning_payload is not None
            try:
                # Runtime overrides are deliberately not baked in here: they are
                # applied below through the same reset-then-apply path used when
                # an already-loaded checker is reused, so both paths produce an
                # identical effective configuration.
                self._checker = StatsColorChecker.from_json(
                    model_path,
                    tuning=resolved_tuning,
                )
                self._checker_type = checker_type
                self._model_path = model_path
                self._decision_tuning = resolved_tuning_payload
            except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
                logger.warning("Failed to load StatsColorChecker from %s: %s", model_path, e)
                self._checker = None
                self._model_path = None
                self._decision_tuning = None
                raise RuntimeError(
                    f"Failed to load StatsColorChecker from {model_path}: {e}"
                ) from e
            # Checked here rather than inside the loader because it is not a
            # format problem: the file parses, and every number in it is
            # well-formed. What cannot be established is whether those numbers
            # were measured on the crop geometry this code measures, and a
            # mismatch shifts every score instead of failing.
            incompatible = color_model_compatibility_failure(
                model_path,
                expected_roi_policy=resolved_roi_policy.to_dict(),
                expected_decision_tuning=resolved_tuning_payload,
            )
            self._algorithm_enforcement = enforcement
            if incompatible:
                if enforcement == ALGORITHM_ENFORCEMENT_STRICT:
                    self._checker = None
                    self._model_path = None
                    self._decision_tuning = None
                    raise RuntimeError(
                        f"拒絕載入顏色基準 {model_path}：{incompatible}"
                    )
                logger.warning(
                    "Color baseline provenance is not current for %s: %s. Scores "
                    "are being compared against statistics measured on another "
                    "crop geometry. Deploy a baseline rebuilt by the current "
                    "algorithm, then set color_baseline_algorithm_enforcement "
                    "to %r.",
                    model_path,
                    incompatible,
                    ALGORITHM_ENFORCEMENT_STRICT,
                )
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
            self._roi_policy = resolved_roi_policy
            return
        # ``default_threshold`` reaches this path too. Dropping it here made
        # an activated global color revision silently inert for every product
        # on the ``color_qc`` checker, while the run log still announced the
        # revision as applied.
        self._apply_runtime_configuration(
            default_threshold=default_threshold,
            color_thresholds=overrides,
            color_rules=rules_overrides,
        )
        self._roi_policy = resolved_roi_policy

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
        unsupported_configured_colors = _unsupported_configured_colors(
            configured_colors,
            supported_colors,
        )
        configured_colors_are_supported = not unsupported_configured_colors
        if unsupported_configured_colors:
            logger.error(
                "Color candidates are absent from the loaded model: %s",
                ", ".join(sorted(unsupported_configured_colors)),
            )
        proc = processed_image if processed_image is not None else frame
        if proc is None or getattr(proc, "size", 0) == 0:
            proc = frame
        unmeasurable = 0
        for idx, det in enumerate(detections):
            # A box can be degenerate or land outside the image; cropping it
            # blind produced an empty array, and OpenCV answers that with an
            # assertion failure that escapes the whole pipeline. One unusable
            # detection turned into a frame-wide ERROR, and in the async
            # pipeline into a line stop. Fail this item closed instead.
            roi = extract_bbox_roi(
                proc,
                det.get("bbox"),
                policy=self._roi_policy,
            )
            if roi is None:
                unmeasurable += 1
                all_ok = False
                logger.warning(
                    "Color check skipped detection %d: bbox %r yields no pixels",
                    idx,
                    det.get("bbox"),
                )
                items.append(
                    ColorCheckItemResult(
                        index=idx,
                        class_name=det.get("class"),
                        bbox=det.get("bbox"),
                        best_color="",
                        diff=_UNMEASURED_DIFF,
                        threshold=0.0,
                        is_ok=False,
                        measurement_is_ok=False,
                    )
                )
                continue
            # Explicit product candidates may narrow the palette. With no
            # candidates, score the checker's full vocabulary: using the YOLO
            # class as the only candidate makes the color check circular and
            # hides exactly the class mismatch it exists to detect.
            allowed = _supported_candidates(
                requested_candidates,
                supported_colors,
                generic_detector_classes,
            )
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
            status=(
                COLOR_CHECK_UNMEASURABLE_ROI_STATUS
                if unmeasurable
                else COLOR_CHECK_EVALUATED_STATUS
            ),
        )
