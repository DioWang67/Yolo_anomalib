from __future__ import annotations

from collections import Counter
from typing import Any

from core.exceptions import ResultPersistenceError
from core.pipeline.context import DetectionContext
from core.position_validator import PositionValidator
from core.services.color_checker import ColorCheckerService
from core.services.result_sink import ExcelImageResultSink


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
                ctx.status = "FAIL"
                self.logger.info("Color check mismatch -> overall FAIL")
        except Exception:
            pass


class CountCheckStep(Step):
    def __init__(
        self, logger, product: str, area: str, options: dict | None = None
    ) -> None:
        self.logger = logger
        self.product = product
        self.area = area
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
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
        ctx.result["count_check"] = {
            "expected": dict(expected_counter),
            "detected": dict(detected_counter),
            "missing": list(missing_items),
            "over": list(over_items),
            "strict": strict,
            "is_ok": not missing_items and (not strict or not over_items),
        }

        if missing_items or (strict and over_items):
            ctx.status = "FAIL"
            self.logger.info(
                "Count check FAIL: missing=%s, over=%s",
                missing_items,
                over_items,
            )
        else:
            self.logger.info("Count check PASS")


class SequenceCheckStep(Step):
    def __init__(
        self, logger, product: str, area: str, options: dict | None = None
    ) -> None:
        self.logger = logger
        self.product = product
        self.area = area
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
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
            ctx.status = "FAIL"
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
            if ctx.result.get("anomaly_score") is not None:
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
                )
            else:
                save_result = self.sink.save(
                    frame=ctx.frame,
                    detections=ctx.result.get("detections", []),
                    status=ctx.status,
                    detector=ctx.inference_type,
                    missing_items=ctx.result.get("missing_items", []),
                    processed_image=ctx.processed_image,
                    anomaly_score=None,
                    heatmap_path=None,
                    product=ctx.product,
                    area=ctx.area,
                    ckpt_path=ctx.result.get("ckpt_path"),
                    color_result=ctx.color_result,
                )
            ctx.save_result = save_result
        except ResultPersistenceError as exc:
            self.logger.error("Save results failed: %s", exc)
            ctx.status = "ERROR"
            ctx.save_result = {"status": "ERROR", "error": str(exc)}
            return
        flush_mode = str(self.options.get("flush", "always")).lower()
        should_flush = flush_mode == "always" or (
            flush_mode == "fail" and str(ctx.status).upper() != "PASS"
        )
        if should_flush:
            try:
                self.sink.flush()
            except Exception as _e:
                self.logger.warning(f"Excel flush failed: {_e}")


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
            ctx.status = new_status
            self.logger.info(f"Position check evaluated status: {new_status}")
        except Exception as e:
            self.logger.warning(f"Position check failed: {e}")
