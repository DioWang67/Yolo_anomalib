from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from core.exceptions import ResultPersistenceError
from core.pipeline.context import DetectionContext
from core.position_validator import PositionValidator
from core.services.color_checker import (
    COLOR_CHECK_NO_DETECTIONS_STATUS,
    ColorCheckerService,
)
from core.services.cross_class_duplicate_filter import (
    DuplicateFilterMode,
    DuplicateFilterPolicy,
    analyze_cross_class_duplicates,
    duplicate_filter_color_block_status,
    duplicate_filter_position_block_status,
)
from core.services.result_sink import ExcelImageResultSink

INFERENCE_ERROR_STATUS = "INFERENCE_ERROR"
DETECTION_FAIL_STATUS = "DETECTION_FAIL"

#: ``count_check.status`` marking a count check that could not be evaluated
#: because its expected-items configuration could not be read. Distinct from a
#: normal count FAIL, which carries missing/over items instead.
EXPECTED_ITEMS_LOOKUP_FAILED_STATUS = "expected_items_lookup_failed"

#: ``position_check.status`` marking a position check that could not be
#: evaluated because its enable flag could not be read. Distinct from a normal
#: position FAIL, which annotates detections with ``position_status``.
POSITION_CONFIG_LOOKUP_FAILED_STATUS = "position_config_lookup_failed"


class Step:
    """Pipeline step interface. Implement run(ctx) in subclasses."""

    def run(self, ctx: DetectionContext) -> None:  # pragma: no cover
        raise NotImplementedError


class ColorCheckStep(Step):
    def __init__(
        self, color_service: ColorCheckerService, logger, options: dict | None = None
    ) -> None:
        self.color_service = color_service
        self.logger = logger
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
        """Run color check on detections and attach ctx.color_result."""
        if str(ctx.status).upper() == INFERENCE_ERROR_STATUS:
            return
        fail_closed = bool(getattr(ctx.config, "color_fail_closed", True))
        if not self.color_service.is_ready():
            self.logger.warning("ColorChecker not ready (possibly missing JSON file); skipping color check")
            ctx.color_result = {"is_ok": False, "items": [], "error": "Not loaded"}
            if fail_closed:
                ctx.status = DETECTION_FAIL_STATUS
                self.logger.info("Color checker unavailable -> overall FAIL")
            return

        detections: list[dict[str, Any]] = ctx.result.get("detections", [])

        # Extract candidates from config to restrict search space. A lookup
        # failure must never degrade silently into an unrestricted vocabulary:
        # that widens the search space and makes a wrong color easier to accept.
        candidates: set[str] = set()
        try:
            expected = ctx.config.get_items_by_area(ctx.product, ctx.area)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self.logger.error(
                "Color candidate lookup failed for product=%s area=%s: %s",
                ctx.product,
                ctx.area,
                exc,
            )
            if fail_closed:
                ctx.color_result = {
                    "is_ok": False,
                    "items": [],
                    "status": "candidate_lookup_failed",
                    "error": f"Color candidate lookup failed: {exc}",
                }
                ctx.status = DETECTION_FAIL_STATUS
                self.logger.info("Color candidates unavailable -> overall FAIL")
                return
            self.logger.warning(
                "Proceeding with the unrestricted color vocabulary because "
                "color_fail_closed is disabled"
            )
            expected = None
        if expected:
            candidates = {str(c).strip() for c in expected if c}

        c_res = self.color_service.check_items(
            frame=ctx.frame,
            processed_image=ctx.processed_image,
            detections=detections,
            candidates=list(candidates) if candidates else None,
            generic_classes=self.options.get("generic_classes"),
        )

        # Only an accepted color result may replace the detector class. A rejected
        # best match is diagnostic evidence, not a trustworthy downstream label.
        for idx, it in enumerate(c_res.items):
            if 0 <= idx < len(detections):
                detections[idx]["verified_class"] = (
                    it.best_color if it.is_ok else detections[idx].get("class")
                )

        ctx.color_result = c_res.to_dict()

        # Compact logging: show up to N items then summary
        max_log = int(self.options.get("max_log_items", 5))
        total = len(c_res.items)
        fail_cnt = 0
        for idx, it in enumerate(c_res.items):
            if not it.is_ok:
                fail_cnt += 1
            if idx < max_log:
                state = "PASS" if it.is_ok else "FAIL"
                message = (
                    f"Color check {state} (idx={it.index}, class={it.class_name}, pred={it.best_color}, "
                    f"diff={it.diff:.2f}, thr={it.threshold:.2f})"
                )
                self.logger.info(message)
        if total > max_log:
            self.logger.info(
                f"Color check logs truncated: {total-max_log} more items..."
            )
        status = str(ctx.color_result.get("status") or "")
        self.logger.info(
            f"Color check summary: total={total}, fail={fail_cnt}, status={status or 'unknown'}"
        )
        # Enforce FAIL when color check is enabled and any item fails. This must
        # stay unguarded: swallowing an error here would turn the only
        # fail-closed enforcement point into a silent pass.
        if not bool(ctx.color_result.get("is_ok", True)):
            ctx.status = DETECTION_FAIL_STATUS
            reason = (
                "no detection ROI to evaluate"
                if status == COLOR_CHECK_NO_DETECTIONS_STATUS
                else "mismatch"
            )
            self.logger.info("Color check %s -> overall FAIL", reason)


class CrossClassDuplicateFilterStep(Step):
    """Analyze or suppress high-confidence cross-class duplicate boxes."""

    def __init__(self, logger, options: dict | None = None) -> None:
        self.logger = logger
        self.options = options or {}
        self.policy = DuplicateFilterPolicy.from_options(self.options)

    def run(self, ctx: DetectionContext) -> None:
        if str(ctx.status).upper() == INFERENCE_ERROR_STATUS:
            return

        detections = ctx.result.get("detections", []) or []
        metadata: dict[str, Any] = {
            "enabled": True,
            "mode": self.policy.mode.value,
            "status": "not_evaluated",
            "policy_version": 1,
            "policy": self.policy.to_dict(),
            "raw_count": len(detections),
            "effective_count": len(detections),
            "candidate_count": 0,
            "suppressed_count": 0,
            "would_suppress_count": 0,
            "candidates": [],
            "suppressions": [],
            "proposed_suppressions": [],
        }
        ctx.result["duplicate_filter"] = metadata
        if not detections:
            metadata["status"] = "no_detections"
            return

        position_state = (
            self._position_check_state(ctx)
            if self.policy.require_position_disabled
            else "unknown"
        )
        block_status = duplicate_filter_position_block_status(
            require_position_disabled=self.policy.require_position_disabled,
            position_state=position_state,
        )
        if block_status is not None:
            metadata["status"] = block_status
            self.logger.warning(
                "Cross-class duplicate filter blocked: position check state=%s",
                position_state,
            )
            return

        color_items = self._color_items_by_index(ctx.color_result)
        color_block_status = duplicate_filter_color_block_status(
            has_color_items=bool(color_items)
        )
        if color_block_status is not None:
            metadata["status"] = color_block_status
            self.logger.warning(
                "Cross-class duplicate filter blocked: color result unavailable"
            )
            return

        for index, detection in enumerate(detections):
            detection.setdefault("source_index", index)

        analysis = analyze_cross_class_duplicates(
            detections=detections,
            color_items_by_index=color_items,
            policy=self.policy,
        )
        proposed = analysis["proposed_suppressions"]
        metadata.update(analysis)
        metadata["effective_count"] = len(detections)
        metadata["would_suppress_count"] = len(proposed)
        metadata["proposed_suppressions"] = proposed
        if not proposed:
            metadata["status"] = "no_candidates"
            return

        if self.policy.mode is DuplicateFilterMode.REPORT_ONLY:
            metadata["status"] = "reported"
            self.logger.info(
                "Cross-class duplicate candidates reported: count=%s",
                len(proposed),
            )
            return

        suppressed_indices = {
            int(item["suppressed_index"]) for item in proposed
        }
        ctx.result["raw_detections"] = deepcopy(detections)
        ctx.result["detections"] = [
            detection
            for index, detection in enumerate(detections)
            if index not in suppressed_indices
        ]
        metadata["status"] = "suppressed"
        metadata["suppressions"] = proposed
        metadata["suppressed_count"] = len(proposed)
        metadata["effective_count"] = len(ctx.result["detections"])
        self.logger.info(
            "Cross-class duplicate boxes suppressed: raw=%s, effective=%s, indices=%s",
            len(detections),
            len(ctx.result["detections"]),
            sorted(suppressed_indices),
        )

    @staticmethod
    def _color_items_by_index(
        color_result: dict[str, Any] | None,
    ) -> dict[int, dict[str, Any]]:
        items = (color_result or {}).get("items", []) or []
        indexed: dict[int, dict[str, Any]] = {}
        for position, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index", position))
            except (TypeError, ValueError):
                continue
            indexed[index] = item
        return indexed

    @staticmethod
    def _position_check_state(ctx: DetectionContext) -> str:
        config = ctx.config
        if config is None:
            return "unknown"
        try:
            enabled = config.is_position_check_enabled(ctx.product, ctx.area)
        except (AttributeError, KeyError, TypeError, ValueError):
            return "unknown"
        return "enabled" if bool(enabled) else "disabled"


class CountCheckStep(Step):
    def __init__(
        self, logger, product: str, area: str, options: dict | None = None
    ) -> None:
        self.logger = logger
        self.product = product
        self.area = area
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
        if str(ctx.status).upper() == INFERENCE_ERROR_STATUS:
            return
        if not self.options.get("enabled", True):
            return

        try:
            expected_items = ctx.config.get_items_by_area(self.product, self.area)
            expected_list = (
                [str(x).strip() for x in expected_items if str(x).strip()]
                if expected_items
                else []
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._fail_closed_on_unreadable_expectation(ctx, exc)
            return

        # An empty result from a *successful* lookup means this area genuinely
        # has no count expectation, which keeps its long-standing skip. That is
        # a different fact from a lookup that failed, handled above.
        if not expected_list:
            self.logger.debug(
                "Count check skipped: no expected items configured for product=%s area=%s",
                self.product,
                self.area,
            )
            return

        strict = bool(self.options.get("strict", False))
        expected_counter = Counter(expected_list)
        detections = ctx.result.get("detections", []) or []
        detected_counter: Counter = Counter()
        expected_set = set(expected_counter)
        # Classes the model reported that this area does not expect at all.
        # They can never appear in ``over_items`` (that only tracks surplus
        # copies of *expected* classes), so they have to be collected here or
        # the strict branch below would publish an ``unexpected_items`` list
        # that silently drops them.
        foreign_items: list[str] = []
        seen_foreign: set[str] = set()
        for det in detections:
            name = str(det.get("verified_class") or det.get("class", "")).strip()
            if not name:
                continue
            if name in expected_set:
                detected_counter[name] += 1
            elif name not in seen_foreign:
                seen_foreign.add(name)
                foreign_items.append(name)

        missing_items: list[str] = []
        over_items: list[str] = []
        for name, need in expected_counter.items():
            have = int(detected_counter.get(name, 0))
            if have < need:
                missing_items.extend([name] * (need - have))
            elif strict and have > need:
                over_items.extend([name] * (have - need))

        ctx.result["missing_items"] = missing_items
        ctx.result["over_items"] = over_items
        if strict:
            # Recomputed from the *effective* detections, so this both drops a
            # stale entry the color checker or duplicate filter has since
            # invalidated and keeps a genuinely foreign class that the previous
            # over_items-only assignment used to erase.
            ctx.result["unexpected_items"] = foreign_items + over_items
        ctx.result["count_check"] = {
            "expected": dict(expected_counter),
            "detected": dict(detected_counter),
            "missing": list(missing_items),
            "over": list(over_items),
            "strict": strict,
            "is_ok": not missing_items and (not strict or not over_items),
        }

        if missing_items or (strict and over_items):
            ctx.status = DETECTION_FAIL_STATUS
            self.logger.info(
                "Count check FAIL: missing=%s, over=%s",
                missing_items,
                over_items,
            )
        else:
            self.logger.info("Count check PASS")

    def _fail_closed_on_unreadable_expectation(
        self, ctx: DetectionContext, exc: Exception
    ) -> None:
        """Record an unevaluable count check and fail the inspection.

        An unreadable expectation means the required item set is *unknown*, not
        empty. Skipping the comparison would accept any detection set at all, so
        the inspection fails instead. There is deliberately no opt-out: an
        expected-items config that cannot be read is a defect to fix, not an
        operating mode.
        """
        self.logger.error(
            "Count check expected-items lookup failed for product=%s area=%s: %s",
            self.product,
            self.area,
            exc,
        )
        ctx.result["count_check"] = {
            "expected": {},
            "detected": {},
            "missing": [],
            "over": [],
            "strict": bool(self.options.get("strict", False)),
            "is_ok": False,
            "status": EXPECTED_ITEMS_LOOKUP_FAILED_STATUS,
            "error": f"Expected items lookup failed: {exc}",
        }
        ctx.status = DETECTION_FAIL_STATUS
        self.logger.info("Count expectations unavailable -> overall FAIL")


class SequenceCheckStep(Step):
    def __init__(
        self, logger, product: str, area: str, options: dict | None = None
    ) -> None:
        self.logger = logger
        self.product = product
        self.area = area
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
        if str(ctx.status).upper() == INFERENCE_ERROR_STATUS:
            return
        if not self.options.get("enabled", True):
            return
        expected = (
            self.options.get("expected")
            or self.options.get("sequence")
            or self.options.get("order")
        )
        if not expected:
            return
        expected_seq = [str(x).strip() for x in expected if str(x).strip()]
        if not expected_seq:
            return

        detections = ctx.result.get("detections", []) or []
        observed = self._left_right_sequence(detections)
        direction = str(self.options.get("direction", "left_to_right")).lower()
        if direction in {"right_to_left", "rtl"}:
            observed = list(reversed(observed))

        is_ok = observed == expected_seq
        reason = ""
        if not is_ok:
            if len(observed) != len(expected_seq):
                reason = "length_mismatch"
            else:
                reason = "order_mismatch"

        ctx.result["sequence_check"] = {
            "expected": list(expected_seq),
            "observed": list(observed),
            "direction": direction,
            "is_ok": is_ok,
            "reason": reason,
        }

        if not is_ok:
            ctx.status = DETECTION_FAIL_STATUS
            self.logger.info(
                "Sequence check FAIL: expected=%s, observed=%s",
                expected_seq,
                observed,
            )
        else:
            self.logger.info("Sequence check PASS")

    @staticmethod
    def _left_right_sequence(detections: list[dict[str, Any]]) -> list[str]:
        seq: list[tuple[float, str]] = []
        for det in detections or []:
            bbox = det.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            x1, _, x2, _ = bbox
            try:
                center = (float(x1) + float(x2)) / 2.0
            except (TypeError, ValueError):
                continue
            # Prioritize verified_class from ColorCheckStep over YOLO class
            name = str(det.get("verified_class") or det.get("class", "")).strip()
            if not name:
                continue
            seq.append((center, name))
        seq.sort(key=lambda item: item[0])
        return [name for _, name in seq]


class SaveResultsStep(Step):
    def __init__(
        self, sink: ExcelImageResultSink, logger, options: dict | None = None
    ) -> None:
        self.sink = sink
        self.logger = logger
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
        """Persist results and flush workbook/images via sink."""
        if not self.options.get("enabled", True):
            self.logger.debug("SaveResultsStep is disabled, skipping.")
            return
        try:
            is_anomalib_only = str(ctx.inference_type).lower() == "anomalib"
            duplicate_kwargs: dict[str, Any] = {}
            if ctx.result.get("duplicate_filter") is not None:
                duplicate_kwargs["duplicate_filter"] = ctx.result.get(
                    "duplicate_filter"
                )
            if ctx.result.get("raw_detections") is not None:
                duplicate_kwargs["raw_detections"] = ctx.result.get(
                    "raw_detections"
                )
            if is_anomalib_only and ctx.result.get("anomaly_score") is not None:
                save_result = self.sink.save(
                    frame=ctx.frame,
                    detections=[],
                    status=ctx.status,
                    detector=ctx.inference_type,
                    missing_items=[],
                    processed_image=ctx.processed_image,
                    anomaly_score=ctx.result.get("anomaly_score"),
                    heatmap_path=ctx.result.get("output_path"),
                    product=ctx.product,
                    area=ctx.area,
                    ckpt_path=ctx.result.get("ckpt_path"),
                    color_result=ctx.color_result,
                    sequence_check=ctx.result.get("sequence_check"),
                    error_message=ctx.result.get("error"),
                    decision=ctx.result.get("decision"),
                    model_info=ctx.result.get("model_info"),
                    inference_time=ctx.result.get("inference_time"),
                    slot_mismatches=ctx.result.get("slot_mismatches", []),
                    **duplicate_kwargs,
                )
            else:
                save_result = self.sink.save(
                    frame=ctx.frame,
                    detections=ctx.result.get("detections", []),
                    status=ctx.status,
                    detector=ctx.inference_type,
                    missing_items=ctx.result.get("missing_items", []),
                    processed_image=ctx.processed_image,
                    anomaly_score=ctx.result.get("anomaly_score"),
                    heatmap_path=ctx.result.get("output_path")
                    or ctx.result.get("heatmap_path"),
                    product=ctx.product,
                    area=ctx.area,
                    ckpt_path=ctx.result.get("ckpt_path"),
                    color_result=ctx.color_result,
                    sequence_check=ctx.result.get("sequence_check"),
                    error_message=ctx.result.get("error"),
                    decision=ctx.result.get("decision"),
                    model_info=ctx.result.get("model_info"),
                    inference_time=ctx.result.get("inference_time"),
                    slot_mismatches=ctx.result.get("slot_mismatches", []),
                    **duplicate_kwargs,
                )
            ctx.save_result = save_result
        except ResultPersistenceError as exc:
            self.logger.error("Save results failed: %s", exc)
            ctx.status = "ERROR"
            ctx.save_result = {"status": "ERROR", "error": str(exc)}
            return
        flush_mode = str(self.options.get("flush", "background")).lower()
        should_flush = flush_mode == "always" or (
            flush_mode == "fail" and str(ctx.status).upper() != "PASS"
        )
        if should_flush:
            try:
                self.sink.flush()
            except Exception as _e:
                self.logger.warning(f"Excel flush failed: {_e}")
        elif flush_mode in {"background", "buffered", "async"}:
            flush_async = getattr(self.sink, "flush_async", None)
            if callable(flush_async):
                flush_async()


class PositionCheckStep(Step):
    def __init__(
        self, logger, product: str, area: str, options: dict | None = None
    ) -> None:
        self.logger = logger
        self.product = product
        self.area = area
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
        """Validate detections against configured expected boxes and update status."""
        if str(ctx.status).upper() == INFERENCE_ERROR_STATUS:
            return
        detections = ctx.result.get("detections", []) or []
        missing = ctx.result.get("missing_items", [])
        if not detections and not missing:
            return
        validator = PositionValidator(
            ctx.config or self.options.get("config"), self.product, self.area
        )

        # If not enabled in config, allow forcing via options. A failed lookup
        # must not be read as "disabled": that skips validate() entirely, so no
        # detection is ever annotated with position_status and finalize_status
        # has nothing left to recompute a FAIL from — a shifted part would PASS.
        try:
            enabled = bool(
                validator.config.is_position_check_enabled(
                    self.product, self.area)
            )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            self._fail_closed_on_unreadable_config(ctx, exc)
            return
        if not enabled and not self.options.get("force", False):
            return

        # Validate and update status
        dets = validator.validate(detections)
        ctx.result["detections"] = dets
        missing = ctx.result.get("missing_items", [])
        try:
            new_status = validator.evaluate_status(dets, missing)
            # Downgrade only. This step sees position and missing items; it
            # knows nothing about color, sequence or count, so publishing its
            # own PASS here would overwrite an earlier step's FAIL. Only
            # finalize_status, which sees every dimension, may declare PASS.
            if new_status in {"FAIL", DETECTION_FAIL_STATUS}:
                ctx.status = DETECTION_FAIL_STATUS
            self.logger.info(f"Position check evaluated status: {new_status}")
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            # Deliberately not fail-closed. validate() above already annotated
            # every detection with position_status, and missing_items is already
            # on ctx.result, so finalize_status recomputes this same FAIL from
            # POSITION_SHIFT / MISSING. Losing the verdict here is recoverable;
            # see test_position_evaluation_error_is_recovered_by_finalize.
            # If that safety net is ever removed, this must become fail-closed.
            self.logger.warning(
                "Position check evaluation failed for product=%s area=%s: %s "
                "(verdict deferred to finalize_status)",
                self.product,
                self.area,
                exc,
                exc_info=True,
            )

    def _fail_closed_on_unreadable_config(
        self, ctx: DetectionContext, exc: Exception
    ) -> None:
        """Record an unevaluable position check and fail the inspection.

        An unreadable enable flag means it is *unknown* whether positions must
        be checked, which is not the same as the check being switched off.
        """
        self.logger.error(
            "Position check config lookup failed for product=%s area=%s: %s",
            self.product,
            self.area,
            exc,
        )
        ctx.result["position_check"] = {
            "is_ok": False,
            "status": POSITION_CONFIG_LOOKUP_FAILED_STATUS,
            "error": f"Position config lookup failed: {exc}",
        }
        ctx.status = DETECTION_FAIL_STATUS
        self.logger.info("Position check configuration unavailable -> overall FAIL")
