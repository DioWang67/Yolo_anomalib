"""Fusion inference runner for combined YOLO and Anomalib inspection."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import numpy as np

from core.inference_tokens import InferenceTypeToken
from core.result_adapter import normalize_result


class FusionInferenceRunner:
    """Runs YOLO and Anomalib inference concurrently and merges their results.

    Args:
        model_manager: Model manager that exposes cached backend engines.
        config: Active detection configuration.
        result_sink: Optional sink used to reserve temporary anomalib artifact paths.
    """

    def __init__(self, model_manager: Any, config: Any, result_sink: Any | None) -> None:
        self.model_manager = model_manager
        self.config = config
        self.result_sink = result_sink

    def run(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        run_logger: Any,
        *,
        adjust_anomalib_output_path=None,
    ) -> dict[str, Any]:
        """Run fusion inference for one frame.

        Args:
            frame: Input image in BGR ndarray format.
            product: Product name.
            area: Area/station name.
            run_logger: Logger-compatible object.
            adjust_anomalib_output_path: Optional callback used to move temporary
                anomalib output into its final PASS/FAIL directory.

        Returns:
            A normalized merged inference result dictionary.
        """
        yolo_engine = self.model_manager.get_cached_engine(product, area, "yolo")
        if not yolo_engine:
            return {"status": "ERROR", "error": "YOLO model not loaded for fusion"}

        ano_engine = self.model_manager.get_cached_engine(product, area, "anomalib")
        if not ano_engine:
            run_logger.warning(
                "Anomalib model not loaded for fusion, falling back to YOLO-only"
            )
            raw_yolo = yolo_engine.infer(
                frame, product, area, InferenceTypeToken("yolo"), force=True
            )
            return normalize_result(raw_yolo, "yolo", frame)

        ano_output_path = self._get_anomalib_temp_path(product, area)
        yolo_res, ano_res = self._run_backends(
            frame, product, area, yolo_engine, ano_engine, ano_output_path, run_logger
        )

        if ano_output_path and adjust_anomalib_output_path:
            adjust_anomalib_output_path(ano_res, ano_output_path, run_logger)

        return self._merge_results(frame, yolo_res, ano_res, run_logger)

    def _get_anomalib_temp_path(self, product: str, area: str) -> str | None:
        if not self.result_sink:
            return None
        return self.result_sink.get_annotated_path(
            status="TEMP", detector="anomalib", product=product, area=area
        )

    def _run_backends(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        yolo_engine: Any,
        ano_engine: Any,
        ano_output_path: str | None,
        run_logger: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        inference_timeout: int | float = getattr(self.config, "timeout", 10)

        with ThreadPoolExecutor(max_workers=2) as pool:
            yolo_future = pool.submit(
                self._infer_yolo, frame, product, area, yolo_engine, run_logger
            )
            ano_future = pool.submit(
                self._infer_anomalib,
                frame,
                product,
                area,
                ano_engine,
                ano_output_path,
                run_logger,
            )

            yolo_res = self._resolve_future(
                yolo_future, "YOLO", inference_timeout, run_logger
            )
            ano_res = self._resolve_future(
                ano_future, "Anomalib", inference_timeout, run_logger
            )
        return yolo_res, ano_res

    @staticmethod
    def _infer_yolo(
        frame: np.ndarray,
        product: str,
        area: str,
        yolo_engine: Any,
        run_logger: Any,
    ) -> dict[str, Any]:
        try:
            raw = yolo_engine.infer(
                frame, product, area, InferenceTypeToken("yolo"), force=True
            )
            return normalize_result(raw, "yolo", frame)
        except Exception as exc:
            run_logger.error("YOLO inference failed in fusion: %s", exc, exc_info=True)
            return FusionInferenceRunner.make_error_result(f"YOLO engine error: {exc}")

    @staticmethod
    def _infer_anomalib(
        frame: np.ndarray,
        product: str,
        area: str,
        ano_engine: Any,
        ano_output_path: str | None,
        run_logger: Any,
    ) -> dict[str, Any]:
        try:
            raw = ano_engine.infer(
                frame,
                product,
                area,
                InferenceTypeToken("anomalib"),
                ano_output_path,
                force=True,
            )
            return normalize_result(raw, "anomalib", frame)
        except Exception as exc:
            run_logger.error(
                "Anomalib inference failed in fusion: %s", exc, exc_info=True
            )
            return FusionInferenceRunner.make_error_result(
                f"Anomalib engine error: {exc}"
            )

    @staticmethod
    def _resolve_future(
        future: Any,
        label: str,
        timeout: int | float,
        run_logger: Any,
    ) -> dict[str, Any]:
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            run_logger.error("%s inference timed out after %s s", label, timeout)
            return FusionInferenceRunner.make_error_result(
                f"{label} inference timed out ({timeout}s)"
            )
        except Exception as exc:
            run_logger.error(
                "%s future raised unexpected error: %s", label, exc, exc_info=True
            )
            return FusionInferenceRunner.make_error_result(f"{label} future error: {exc}")

    @staticmethod
    def _merge_results(
        frame: np.ndarray,
        yolo_res: dict[str, Any],
        ano_res: dict[str, Any],
        run_logger: Any,
    ) -> dict[str, Any]:
        status = "PASS"
        if yolo_res.get("status") == "FAIL" or ano_res.get("status") == "FAIL":
            status = "FAIL"
        elif yolo_res.get("status") == "ERROR" or ano_res.get("status") == "ERROR":
            status = "ERROR"

        merged: dict[str, Any] = {
            "status": status,
            "detections": yolo_res.get("detections", [])
            + ano_res.get("detections", []),
            "missing_items": yolo_res.get("missing_items", [])
            + ano_res.get("missing_items", []),
            "unexpected_items": yolo_res.get("unexpected_items", [])
            + ano_res.get("unexpected_items", []),
            "error": " | ".join(
                filter(None, [yolo_res.get("error"), ano_res.get("error")])
            )
            or None,
            "inference_time": yolo_res.get("inference_time", 0.0)
            + ano_res.get("inference_time", 0.0),
            "anomaly_score": ano_res.get("anomaly_score"),
            "original_image": yolo_res.get("original_image", frame),
            "annotated_path": ano_res.get("annotated_path")
            or yolo_res.get("annotated_path", ""),
            "heatmap_path": ano_res.get("heatmap_path", ""),
        }

        result_frame = FusionInferenceRunner._overlay_yolo_on_anomalib_frame(
            yolo_res, ano_res, run_logger
        )
        fallback_frame = yolo_res.get("result_frame")
        if fallback_frame is None:
            fallback_frame = ano_res.get("result_frame")
        merged["result_frame"] = result_frame if result_frame is not None else fallback_frame
        merged["processed_image"] = (
            merged["result_frame"] if merged["result_frame"] is not None else frame
        )
        return merged

    @staticmethod
    def _overlay_yolo_on_anomalib_frame(
        yolo_res: dict[str, Any],
        ano_res: dict[str, Any],
        run_logger: Any,
    ) -> np.ndarray | None:
        ano_frame = ano_res.get("result_frame")
        if not isinstance(ano_frame, np.ndarray) or ano_frame.size == 0:
            return None

        base_frame = ano_frame.copy()
        try:
            import cv2

            for det in yolo_res.get("detections", []):
                bbox = det.get("bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                label = det.get("class", "")
                conf = det.get("confidence", 0.0)
                cv2.rectangle(
                    base_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    base_frame,
                    f"{label} {conf:.2f}",
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            return base_frame
        except Exception as exc:
            run_logger.warning("Image fusion failed: %s", exc)
            return None

    @staticmethod
    def make_error_result(error_msg: str) -> dict[str, Any]:
        """Build a standard error dict with merge-compatible list fields."""
        return {
            "status": "ERROR",
            "error": error_msg,
            "detections": [],
            "missing_items": [],
            "unexpected_items": [],
            "inference_time": 0.0,
            "anomaly_score": None,
            "result_frame": None,
            "original_image": None,
            "annotated_path": "",
            "heatmap_path": "",
        }
