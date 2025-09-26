from __future__ import annotations

from typing import Dict, Any, List

from core.pipeline.context import DetectionContext
from core.services.color_checker import ColorCheckerService
from core.services.result_sink import ExcelImageResultSink
from core.position_validator import PositionValidator
from core.exceptions import ResultPersistenceError


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
        detections: List[Dict[str, Any]] = ctx.result.get("detections", [])
        c_res = self.color_service.check_items(
            frame=ctx.frame, processed_image=ctx.processed_image, detections=detections
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
                self.logger.info(
                    f"Color check {state} (idx={it.index}, class={it.class_name}, pred={it.best_color}, diff={it.diff:.2f}, thr={it.threshold:.2f})"
                )
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


class SaveResultsStep(Step):
    def __init__(
        self, sink: ExcelImageResultSink, logger, options: dict | None = None
    ) -> None:
        self.sink = sink
        self.logger = logger
        self.options = options or {}

    def run(self, ctx: DetectionContext) -> None:
        """Persist results and flush workbook/images via sink."""
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
        if not detections:
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
