from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any

from core.exceptions import ResultPersistenceError
from core.pipeline.context import DetectionContext
from core.position_validator import PositionValidator
from core.services.color_checker import ColorCheckerService
from core.services.cross_class_duplicate_filter import (
    DuplicateFilterMode,
    DuplicateFilterPolicy,
    analyze_cross_class_duplicates,
)
from core.services.decision_engine import InspectionDecisionEngine
from core.services.result_sink import ExcelImageResultSink

INFERENCE_ERROR_STATUS = "INFERENCE_ERROR"
DETECTION_FAIL_STATUS = "DETECTION_FAIL"


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
        if not self.color_service.is_ready():
            self.logger.warning("ColorChecker not ready (possibly missing JSON file); skipping color check")
            ctx.color_result = {"is_ok": False, "items": [], "error": "Not loaded"}
            fail_closed = bool(getattr(ctx.config, "color_fail_closed", True))
            if fail_closed:
                ctx.status = DETECTION_FAIL_STATUS
                self.logger.info("Color checker unavailable -> overall FAIL")
            return

        detections: list[dict[str, Any]] = ctx.result.get("detections", [])

        # Extract candidates from config to restrict search space
        candidates = set()
        try:
            expected = ctx.config.get_items_by_area(ctx.product, ctx.area)
            if expected:
                candidates = {str(c).strip() for c in expected if c}
        except Exception:
            pass

        c_res = self.color_service.check_items(
            frame=ctx.frame,
            processed_image=ctx.processed_image,
            detections=detections,
            candidates=list(candidates) if candidates else None,
        )

        # Attach verified class to detections for downstream steps (e.g. sequence check)
        for idx, it in enumerate(c_res.items):
            if 0 <= idx < len(detections):
                detections[idx]["verified_class"] = it.best_color

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
        self.logger.info(
            f"Color check summary: total={total}, fail={fail_cnt}")
        # Enforce FAIL when color check is enabled and any item fails
        try:
            if not bool(ctx.color_result.get("is_ok", True)):
                ctx.status = DETECTION_FAIL_STATUS
                self.logger.info("Color check mismatch -> overall FAIL")
        except Exception:
            pass


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

        if self.policy.require_position_disabled:
            position_state = self._position_check_state(ctx)
            if position_state != "disabled":
                metadata["status"] = (
                    "blocked_position_enabled"
                    if position_state == "enabled"
                    else "blocked_position_state_unknown"
                )
                self.logger.warning(
                    "Cross-class duplicate filter blocked: position check state=%s",
                    position_state,
                )
                return

        color_items = self._color_items_by_index(ctx.color_result)
        if not color_items:
            metadata["status"] = "blocked_color_result_unavailable"
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
        expected_items = None
        try:
            expected_items = ctx.config.get_items_by_area(self.product, self.area)
        except Exception:
            expected_items = None
        if not expected_items:
            return

        expected_list = [str(x).strip() for x in expected_items if str(x).strip()]
        if not expected_list:
            return

        strict = bool(self.options.get("strict", False))
        expected_counter = Counter(expected_list)
        detections = ctx.result.get("detections", []) or []
        detected_counter: Counter = Counter()
        expected_set = set(expected_counter)
        for det in detections:
            name = str(det.get("verified_class") or det.get("class", "")).strip()
            if name and name in expected_set:
                detected_counter[name] += 1

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
            ctx.result["unexpected_items"] = list(over_items)
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
        self._sync_count_decision(ctx, missing_items, over_items, strict)

    @staticmethod
    def _sync_count_decision(
        ctx: DetectionContext,
        missing_items: list[str],
        over_items: list[str],
        strict: bool,
    ) -> None:
        """Keep decision metadata aligned with post-color count check output."""
        decision = InspectionDecisionEngine(fail_on_unexpected=True).evaluate(
            detections=ctx.result.get("detections", []) or [],
            missing_items=missing_items,
            unexpected_items=list(over_items) if strict else [],
            slot_mismatches=ctx.result.get("slot_mismatches", []) or [],
            alignment_quality=ctx.result.get("alignment_quality"),
        )
        ctx.result["decision"] = decision.to_dict()


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

        # If not enabled in config, allow forcing via options
        enabled = False
        try:
            enabled = bool(
                validator.config.is_position_check_enabled(
                    self.product, self.area)
            )
        except Exception:
            pass
        if not enabled and not self.options.get("force", False):
            return

        # Validate and update status
        dets = validator.validate(detections)
        ctx.result["detections"] = dets
        missing = ctx.result.get("missing_items", [])
        try:
            new_status = validator.evaluate_status(dets, missing)
            if new_status == "FAIL":
                new_status = DETECTION_FAIL_STATUS
            ctx.status = new_status
            self.logger.info(f"Position check evaluated status: {new_status}")
        except Exception as e:
            self.logger.warning(f"Position check failed: {e}")
