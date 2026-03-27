"""Detection system orchestrator.

Responsibilities:
- Load and merge per-product/area/type model configs via ModelManager
- Initialize camera and inference engine backends
- Build a per-run pipeline (position/color/save) and execute
- Normalize outputs and persist results via sinks
- (Async mode) Manage Producer-Consumer pipeline workers

Public entrypoints:
  - DetectionSystem.detect(...)          — synchronous, single-shot
  - DetectionSystem.start_pipeline(...)  — async continuous mode
  - DetectionSystem.stop_pipeline()      — graceful shutdown
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from camera.camera_controller import CameraController
from core.async_pipeline import AsyncPipelineManager
from core.config import DetectionConfig
from core.inference_engine import InferenceEngine
from core.logging_config import DetectionLogger
from core.logging_utils import context_adapter
from core.path_utils import project_root, resolve_path
from core.pipeline.context import DetectionContext
from core.pipeline.registry import PipelineEnv, build_pipeline, default_pipeline
from core.result_adapter import normalize_result
from core.services.color_checker import ColorCheckerService
from core.services.model_manager import ModelManager
from core.services.result_sink import ExcelImageResultSink
from core.types import DetectionItem, DetectionResult, DetectionTask


class _InferenceTypeToken(str):
    @property
    def value(self) -> str:
        return str(self)

PROJECT_ROOT = project_root()


class DetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        """Initializes the DetectionSystem with project settings.

        Args:
            config_path: Relative or absolute path to the global config.yaml.
        """

        self.logger = DetectionLogger()
        root_dir = project_root()
        if config_path:
            resolved_config = Path(config_path).resolve()
        else:
            # resolve_path checks sys.executable parent, sys._MEIPASS, then fallback
            resolved = resolve_path("config.yaml")
            resolved_config = resolved if resolved and resolved.exists() else root_dir / "config.yaml"
        self.config_path = resolved_config
        self.config = self.load_config(resolved_config)

        self.camera: CameraController | None = None
        self.result_sink: ExcelImageResultSink | None = None
        self._sink_base_dir: Path | None = None
        self._refresh_result_sink()

        self.inference_engine: InferenceEngine | None = None
        self.current_inference_type: str | None = None
        self.model_manager = ModelManager(
            self.logger, max_cache_size=self.config.max_cache_size
        )
        self.color_service = ColorCheckerService()
        self._color_override_cache: dict[str, dict[str, Any]] = {}

        # --- Producer-Consumer pipeline (delegated to AsyncPipelineManager) ---
        self._pipeline = AsyncPipelineManager()

        self.initialize_camera()

    def _resolve_output_dir(self) -> Path:
        output_dir = Path(self.config.output_dir)
        if not output_dir.is_absolute():
            output_dir = (self.config_path.parent / output_dir).resolve()
        self.config.output_dir = str(output_dir)
        return output_dir

    def _refresh_result_sink(self) -> None:
        output_dir = self._resolve_output_dir()
        if getattr(self, "_sink_base_dir", None) == output_dir:
            return
        if self.result_sink:
            try:
                self.result_sink.close()
            except Exception:
                pass
        self.result_sink = ExcelImageResultSink(
            self.config, base_dir=str(output_dir), logger=self.logger
        )
        self._sink_base_dir = output_dir

    def load_config(self, config_path: str | Path) -> DetectionConfig:
        """Load global config from YAML into DetectionConfig dataclass."""
        return DetectionConfig.from_yaml(str(Path(config_path)))

    def shutdown(self) -> None:
        """Release resources: pipeline workers, models, camera, sinks."""
        # Stop async pipeline first (if running)
        if self._pipeline.running:
            self.stop_pipeline()

        if self.inference_engine:
            self.inference_engine.shutdown()
            self.inference_engine = None
        if self.camera:
            self.camera.shutdown()
            self.camera = None
        if self.result_sink:
            try:
                self.result_sink.close()
            except Exception:
                pass
            self.result_sink = None
        self._sink_base_dir = None

    # ------------------------------------------------------------------
    # Producer-Consumer Pipeline API
    # ------------------------------------------------------------------

    def start_pipeline(
        self,
        product: str,
        area: str,
        inference_type: str = "yolo",
        *,
        capture_interval: float = 0.0,
        on_task_captured=None,
        on_task_processed=None,
        on_camera_lost=None,
    ) -> None:
        """Start the async Producer-Consumer detection pipeline.

        Delegates to :class:`AsyncPipelineManager` which manages the
        three-stage worker lifecycle (Acquisition → Inference → Storage).

        Args:
            product: Name of product.
            area: Name of area.
            inference_type: 'yolo', 'anomalib', or 'fusion'.
            capture_interval: Seconds between captures.
            on_task_captured: Optional callback for each captured task.
            on_task_processed: Optional callback for each stored task.
            on_camera_lost: Optional callback when camera disconnects.

        Raises:
            RuntimeError: If the pipeline is already running or camera
                is unavailable.
        """
        if not self.camera:
            raise RuntimeError(
                "Cannot start pipeline: no camera available. "
                "Call initialize_camera() first."
            )

        _logger = logging.getLogger(__name__)

        # Pre-load model configs so InferenceWorker is ready
        self.load_model_configs(product, area, inference_type)
        self._prepare_resources(product, area, inference_type, _logger)

        self._pipeline.start(
            camera=self.camera,
            detection_system=self,
            product=product,
            area=area,
            inference_type=inference_type,
            buffer_limit=getattr(self.config, "buffer_limit", 10),
            capture_interval=capture_interval,
            on_task_captured=on_task_captured,
            on_task_processed=on_task_processed,
            on_camera_lost=on_camera_lost,
        )

    def stop_pipeline(self, timeout: float = 10.0) -> None:
        """Gracefully shut down the async pipeline.

        Delegates to :class:`AsyncPipelineManager` which handles
        poison-pill propagation and worker join.

        Args:
            timeout: Max seconds to wait for each worker to finish.
        """
        self._pipeline.stop(timeout=timeout)

    @property
    def pipeline_running(self) -> bool:
        """Whether the async pipeline is currently active."""
        return self._pipeline.running

    def pipeline_stats(self) -> dict[str, Any]:
        """Return a snapshot of pipeline counters for monitoring."""
        return self._pipeline.stats()

    def capture_image(self) -> np.ndarray | None:
        """Capture a single frame from the active camera."""
        if self.camera:
            return self.camera.capture_frame()
        return None

    def initialize_camera(self) -> None:
        """Initialize camera if available; log and fall back to dummy on failure."""
        self.logger.logger.info("Initializing camera...")
        try:
            self.camera = CameraController(self.config)
            self.camera.initialize()
            self.logger.logger.info("Camera is ready")
        except Exception as e:
            self.logger.logger.error(f"Camera init failed: {str(e)}")
            self.logger.logger.warning(
                "Camera disabled; using dummy image for detection"
            )
            self.camera = None

    def _load_color_override_bundle(
        self, product: str, area: str, inference_type: str
    ) -> tuple[dict[str, float] | None, dict[str, dict[str, Any]] | None]:
        overrides = getattr(self.config, "color_threshold_overrides", None)
        rules_over = getattr(self.config, "color_rules_overrides", None)
        cfg_path = (
            PROJECT_ROOT / "models" / product / area / inference_type / "config.yaml"
        )
        cache_key = str(cfg_path)
        try:
            stat = cfg_path.stat()
        except FileNotFoundError:
            self._color_override_cache.pop(cache_key, None)
            return overrides, rules_over

        cached = self._color_override_cache.get(cache_key)
        if cached and cached.get("mtime") == stat.st_mtime:
            cached_overrides = cached.get("overrides")
            cached_rules = cached.get("rules")
            return (
                cached_overrides if cached_overrides is not None else overrides,
                cached_rules if cached_rules is not None else rules_over,
            )

        try:
            import yaml as _yaml  # type: ignore
        except ImportError:
            return overrides, rules_over

        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                model_cfg = _yaml.safe_load(handle)
            if not isinstance(model_cfg, dict):
                self.logger.logger.warning(
                    "Color override config %s is not a valid mapping, skipping", cfg_path
                )
                return overrides, rules_over
        except (OSError, _yaml.YAMLError) as exc:
            self.logger.logger.warning(
                "Failed to reload color overrides from %s: %s",
                cfg_path,
                exc,
            )
            return overrides, rules_over

        disk_overrides = model_cfg.get("color_threshold_overrides")
        if not isinstance(disk_overrides, dict) or not disk_overrides:
            disk_overrides = None
        disk_rules = model_cfg.get("color_rules_overrides")
        if not isinstance(disk_rules, dict) or not disk_rules:
            disk_rules = None

        self._color_override_cache[cache_key] = {
            "mtime": stat.st_mtime,
            "overrides": disk_overrides,
            "rules": disk_rules,
        }
        if len(self._color_override_cache) > 32:
            try:
                first_key = next(iter(self._color_override_cache.keys()))
                if first_key != cache_key:
                    self._color_override_cache.pop(first_key, None)
            except StopIteration:
                pass

        return (
            disk_overrides if disk_overrides is not None else overrides,
            disk_rules if disk_rules is not None else rules_over,
        )

    def disconnect_camera(self) -> None:
        """Release the active camera instance and mark it unavailable."""
        self.logger.logger.info("Disconnecting camera...")
        if self.camera:
            try:
                self.camera.shutdown()
            except Exception as e:
                self.logger.logger.warning(f"Camera shutdown raised: {e}")
        self.camera = None

    def reconnect_camera(self) -> bool:
        """Attempt to reinitialize the camera after a manual disconnect."""
        self.disconnect_camera()
        self.logger.logger.info("Reconnecting camera...")
        self.initialize_camera()
        connected = self.is_camera_connected()
        if connected:
            self.logger.logger.info("Camera reconnected successfully")
        else:
            self.logger.logger.error("Camera reconnect failed")
        return connected

    def is_camera_connected(self) -> bool:
        """Return True if a camera controller is initialized and ready."""
        return bool(self.camera and getattr(self.camera, "is_initialized", False))

    def load_model_configs(self, product: str, area: str, inference_type: str) -> None:
        """Switch to a specific (product, area, type) model configuration.

        Uses ModelManager to keep an LRU cache of engines and snapshots of configs.
        """
        if inference_type.lower() == "fusion":
            # Load Anomalib first
            self.model_manager.switch(self.config, product, area, "anomalib")
            # Save anomalib_config before it gets overwritten by YOLO's configuration
            import copy
            ano_cfg = copy.deepcopy(getattr(self.config, "anomalib_config", None))
            
            # Load YOLO second, so self.config holds YOLO's configuration for pipeline steps
            self.model_manager.switch(self.config, product, area, "yolo")
            
            # Restore anomalib_config to the merged config
            if ano_cfg is not None:
                self.config.anomalib_config = ano_cfg
                
            self.inference_engine = None
            self.current_inference_type = "fusion"
        else:
            engine, _ = self.model_manager.switch(
                self.config, product, area, inference_type
            )
            self.inference_engine = engine
            self.current_inference_type = inference_type.lower()
        self._refresh_result_sink()

    def _is_canceled(self, cancel_cb) -> bool:
        """Helper to safely check the cancellation callback."""
        try:
            return bool(cancel_cb and cancel_cb())
        except Exception:
            return False

    def _build_result(
        self, product: str, area: str, inference_type: str, status: str,
        *, error: str | None = None,
    ) -> DetectionResult:
        """Builds a standardized DetectionResult for empty, error, or canceled cases."""
        return DetectionResult(
            status=status,  # type: ignore[arg-type]
            product=product,
            area=area,
            inference_type=inference_type,
            error=error,
        )

    def _prepare_resources(
        self, product: str, area: str, inference_type: str, run_logger
    ):
        """Load model configs and initialize the color checker."""
        self.load_model_configs(product, area, inference_type)
        if self.config.enable_color_check and self.config.color_model_path:
            try:
                overrides, rules_over = self._load_color_override_bundle(
                    product, area, inference_type
                )
                checker_type = (
                    getattr(self.config, "color_checker_type", "color_qc") or "color_qc"
                )
                default_threshold = getattr(self.config, "color_score_threshold", None)
                self.color_service.ensure_loaded(
                    self.config.color_model_path,
                    overrides=overrides,
                    rules_overrides=rules_over,
                    checker_type=checker_type,
                    default_threshold=default_threshold,
                )
                run_logger.info(f"Color checker loaded ({checker_type})")
            except Exception as e:
                run_logger.error(f"Color checker init failed: {e}")

    def _acquire_frame(self, frame: np.ndarray | None, run_logger) -> np.ndarray:
        """Capture a frame from the camera if not provided.

        Raises:
            RuntimeError: If the camera returns None or no camera/frame is
                available.  The caller (``detect``) catches this and maps
                it to ``DetectionResult(status='ERROR')``.
        """
        if frame is not None:
            return frame
        if self.camera:
            frame = self.camera.capture_frame()
            if frame is None:
                run_logger.critical(
                    "Hardware IO Error: camera.capture_frame() returned None. "
                    "Possible cable disconnect or driver failure."
                )
                raise RuntimeError(
                    "Failed to read frame from camera. Hardware failure."
                )
            return frame
        # No camera instance and no frame supplied
        run_logger.critical(
            "Hardware IO Error: no camera available and no frame provided."
        )
        raise RuntimeError("No camera available and no frame provided.")

    def _run_inference(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        inference_type: str,
        run_logger,
    ) -> dict[str, Any]:
        """Perform model inference and post-processing on the frame."""
        if inference_type.lower() == "fusion":
            return self._run_fusion_inference(frame, product, area, run_logger)

        if not self.inference_engine:
            return {"status": "ERROR", "error": "Model not loaded"}
        
        inference_type_name = inference_type.lower()
        output_path = None
        if inference_type_name == "anomalib":
            output_path = self.result_sink.get_annotated_path(
                status="TEMP", detector=inference_type, product=product, area=area
            )

        raw_result = self.inference_engine.infer(
            frame, product, area, _InferenceTypeToken(inference_type_name), output_path
        )
        result = normalize_result(raw_result, inference_type_name, frame)

        if result.get("status") == "ERROR":
            run_logger.error(f"Inference failed: {result.get('error')}")
            return result

        if inference_type_name == "anomalib" and output_path:
            self._adjust_anomalib_output_path(result, output_path, run_logger)

        return result

    @staticmethod
    def _make_error_result(error_msg: str) -> dict[str, Any]:
        """Build a standard error dict with ALL keys required by merge logic.

        This prevents ``TypeError`` when the merge step concatenates list
        fields (``missing_items``, ``unexpected_items``, ``detections``).
        """
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

    def _run_fusion_inference(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        run_logger,
    ) -> dict[str, Any]:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        # --- Validate engines before parallel dispatch ---
        yolo_engine = self.model_manager._cache.get((product, area), {}).get("yolo", (None, None))[0]
        if not yolo_engine:
            return {"status": "ERROR", "error": "YOLO model not loaded for fusion"}

        ano_engine = self.model_manager._cache.get((product, area), {}).get("anomalib", (None, None))[0]
        if not ano_engine:
            run_logger.warning("Anomalib model not loaded for fusion, falling back to YOLO-only")
            raw_yolo = yolo_engine.infer(frame, product, area, _InferenceTypeToken("yolo"), force=True)
            return normalize_result(raw_yolo, "yolo", frame)

        ano_output_path = self.result_sink.get_annotated_path(
            status="TEMP", detector="anomalib", product=product, area=area
        )

        # Read timeout from config (default 10 s); production lines can
        # override via config.yaml to match their own Cycle Time.
        inference_timeout: int | float = getattr(self.config, "timeout", 10)

        # --- Run YOLO and Anomalib concurrently ---
        # Both engines use C extensions (PyTorch/OpenCV) that release the GIL,
        # so threading gives real overlap (~120 ms saved).
        def _yolo_task() -> dict[str, Any]:
            try:
                raw = yolo_engine.infer(frame, product, area, _InferenceTypeToken("yolo"), force=True)
                return normalize_result(raw, "yolo", frame)
            except Exception as exc:
                run_logger.error("YOLO inference failed in fusion: %s", exc, exc_info=True)
                return DetectionSystem._make_error_result(f"YOLO engine error: {exc}")

        def _ano_task() -> dict[str, Any]:
            try:
                raw = ano_engine.infer(
                    frame, product, area, _InferenceTypeToken("anomalib"), ano_output_path, force=True
                )
                return normalize_result(raw, "anomalib", frame)
            except Exception as exc:
                run_logger.error("Anomalib inference failed in fusion: %s", exc, exc_info=True)
                return DetectionSystem._make_error_result(f"Anomalib engine error: {exc}")

        with ThreadPoolExecutor(max_workers=2) as pool:
            yolo_future = pool.submit(_yolo_task)
            ano_future = pool.submit(_ano_task)

            try:
                yolo_res = yolo_future.result(timeout=inference_timeout)
            except FutureTimeoutError:
                run_logger.error("YOLO inference timed out after %s s", inference_timeout)
                yolo_res = self._make_error_result(
                    f"YOLO inference timed out ({inference_timeout}s)"
                )
            except Exception as exc:
                run_logger.error("YOLO future raised unexpected error: %s", exc, exc_info=True)
                yolo_res = self._make_error_result(f"YOLO future error: {exc}")

            try:
                ano_res = ano_future.result(timeout=inference_timeout)
            except FutureTimeoutError:
                run_logger.error("Anomalib inference timed out after %s s", inference_timeout)
                ano_res = self._make_error_result(
                    f"Anomalib inference timed out ({inference_timeout}s)"
                )
            except Exception as exc:
                run_logger.error("Anomalib future raised unexpected error: %s", exc, exc_info=True)
                ano_res = self._make_error_result(f"Anomalib future error: {exc}")

        if ano_output_path:
            self._adjust_anomalib_output_path(ano_res, ano_output_path, run_logger)

        # 3. Merge Status
        status = "PASS"
        if yolo_res.get("status") == "FAIL" or ano_res.get("status") == "FAIL":
            status = "FAIL"
        elif yolo_res.get("status") == "ERROR" or ano_res.get("status") == "ERROR":
            status = "ERROR"
        
        # 4. Merge Data
        merged: dict[str, Any] = {
            "status": status,
            "detections": yolo_res.get("detections", []) + ano_res.get("detections", []),
            "missing_items": yolo_res.get("missing_items", []) + ano_res.get("missing_items", []),
            "unexpected_items": yolo_res.get("unexpected_items", []) + ano_res.get("unexpected_items", []),
            "error": " | ".join(filter(None, [yolo_res.get("error"), ano_res.get("error")])) or None,
            "inference_time": yolo_res.get("inference_time", 0.0) + ano_res.get("inference_time", 0.0),
            "anomaly_score": ano_res.get("anomaly_score"),
            "original_image": yolo_res.get("original_image", frame), # base frame
            "annotated_path": ano_res.get("annotated_path") or yolo_res.get("annotated_path", ""),
            "heatmap_path": ano_res.get("heatmap_path", ""),
        }

        # 5. Blend Images
        # Use in-memory result_frame directly instead of re-reading from disk.
        # The anomalib_inference_model already returns the processed image as
        # BGR ndarray in result_frame — no need for expensive disk round-trip.
        import cv2
        ano_frame = ano_res.get("result_frame")
        yolo_frame = yolo_res.get("result_frame")

        # ano_frame from anomalib_inference_model is already BGR (it does
        # cvtColor(RGB→BGR) before returning).  Use it directly as base.
        base_frame_to_draw = ano_frame.copy() if isinstance(ano_frame, np.ndarray) and ano_frame.size > 0 else None

        if base_frame_to_draw is not None and "detections" in yolo_res:
            try:
                for det in yolo_res.get("detections", []):
                    bbox = det.get("bbox")
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        label = det.get("class", "")
                        conf = det.get("confidence", 0.0)
                        cv2.rectangle(base_frame_to_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(base_frame_to_draw, f"{label} {conf:.2f}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                merged["result_frame"] = base_frame_to_draw
            except Exception as e:
                run_logger.warning(f"Image fusion failed: {e}")
                merged["result_frame"] = yolo_frame if yolo_frame is not None else ano_frame
        else:
            merged["result_frame"] = yolo_frame if yolo_frame is not None else ano_frame

        return merged

    def _execute_pipeline(self, ctx: DetectionContext, run_logger, cancel_cb=None):
        """Build and run the post-processing pipeline."""
        env = PipelineEnv(
            color_service=self.color_service,
            result_sink=self.result_sink,
            logger=run_logger,
            product=ctx.product,
            area=ctx.area,
            config=self.config,
        )

        pipe_cfg = getattr(self.config, "pipeline", None)
        step_opts_raw = getattr(self.config, "steps", {}) or {}
        step_opts = {str(k).lower(): v for k, v in step_opts_raw.items()}

        step_names = (
            [str(name) for name in pipe_cfg]
            if isinstance(pipe_cfg, list) and pipe_cfg
            else default_pipeline(env)
        )

        steps = build_pipeline(step_names, env, step_opts)
        if not steps:
            run_logger.warning(
                "Pipeline produced no executable steps; enforcing save_results fallback"
            )
            steps = build_pipeline(["save_results"], env, step_opts)

        for step in steps:
            if self._is_canceled(cancel_cb):
                ctx.status = "CANCELED"
                return
            step.run(ctx)

    def _adjust_anomalib_output_path(self, result: dict, temp_path: str, run_logger):
        """Move anomalib output from TEMP to a final PASS/FAIL directory."""
        correct_path = self.result_sink.get_annotated_path(
            status=result["status"],
            detector="anomalib",
            product=result.get("product"),
            area=result.get("area"),
            anomaly_score=result.get("anomaly_score"),
        )
        result["output_path"] = correct_path
        if temp_path != correct_path and os.path.exists(temp_path):
            os.makedirs(os.path.dirname(correct_path), exist_ok=True)
            try:
                shutil.move(temp_path, correct_path)
                run_logger.info(f"Moved output: {temp_path} -> {correct_path}")
            except Exception as e:
                run_logger.error(f"Move output failed: {e}")
            try:
                old_dir = os.path.dirname(temp_path)
                if os.path.isdir(old_dir) and not os.listdir(old_dir):
                    os.rmdir(old_dir)
            except Exception as cleanup_err:
                run_logger.warning(f"Cleanup temp dir failed: {cleanup_err}")

    def _log_summary(self, ctx: DetectionContext, run_logger):
        """Log a summary of the detection results and failure reasons."""
        if ctx.inference_type.lower() == "anomalib":
            self.logger.log_anomaly(ctx.status, ctx.result.get("anomaly_score", 0.0))
        else:
            self.logger.log_detection(ctx.status, ctx.result.get("detections", []))

        if str(ctx.status).upper() != "FAIL":
            return
        try:
            reasons = []
            if ctx.result.get("missing_items"):
                reasons.append(f"missing items: {ctx.result['missing_items']}")
            if ctx.result.get("over_items"):
                reasons.append(f"extra items: {ctx.result['over_items']}")
            if color_res := ctx.color_result:
                if not color_res.get("is_ok", True):
                    items = color_res.get("items", []) or []
                    fails = sum(1 for i in items if not i.get("is_ok"))
                    reasons.append(f"color mismatch: {fails}/{len(items)} failed")
            seq_res = ctx.result.get("sequence_check")
            if seq_res and not seq_res.get("is_ok", True):
                expected_seq = seq_res.get("expected")
                observed_seq = seq_res.get("observed")
                reasons.append(
                    f"sequence mismatch: expected={expected_seq}, observed={observed_seq}"
                )
            if ctx.result.get("unexpected_items"):
                reasons.append(f"unexpected items: {ctx.result['unexpected_items']}")
            pos_wrong = [
                d.get("class")
                for d in (ctx.result.get("detections", []) or [])
                if d.get("position_status") == "WRONG"
            ]
            if pos_wrong:
                reasons.append(f"position wrong: {pos_wrong}")
            if reasons:
                run_logger.info(f"Fail reasons: {'; '.join(reasons)}")
        except Exception:
            pass  # Avoid logging failures to interfere with main flow

    def detect(
        self,
        product: str,
        area: str,
        inference_type: str,
        frame: np.ndarray | None = None,
        cancel_cb=None,
    ) -> DetectionResult:
        """Runs the complete detection pipeline for a specific product and area.

        This includes image acquisition, model inference, post-processing steps
        (position check, color check), and results saving.

        Args:
            product: Name of the product (e.g., 'LED').
            area: Name of the station or area (e.g., 'A').
            inference_type: Type of model to run ('yolo' or 'anomalib').
            frame: Optional pre-acquired image frame. If None, captures from camera.
            cancel_cb: Optional callback that returns True to abort execution.

        Returns:
            DetectionResult: Strongly-typed result containing status, detections,
            and artifact paths.
        """
        import time as _time
        t0 = _time.time()
        run_logger = context_adapter(self.logger.logger, product, area, inference_type)
        run_logger.info("Start detection")

        try:
            if self._is_canceled(cancel_cb):
                return self._build_result(
                    product, area, inference_type, "CANCELED"
                )

            self._prepare_resources(product, area, inference_type, run_logger)
            frame = self._acquire_frame(frame, run_logger)

            if self._is_canceled(cancel_cb):
                return self._build_result(
                    product, area, inference_type, "CANCELED"
                )

            result = self._run_inference(frame, product, area, inference_type, run_logger)
            if result.get("status") == "ERROR":
                error_msg = result.get("error", "Inference failed")
                return self._build_result(
                    product, area, inference_type, "ERROR", error=error_msg
                )

            ctx = DetectionContext(
                product=product,
                area=area,
                inference_type=inference_type,
                frame=frame,
                processed_image=result.get("processed_image", frame),
                result=result,
                status=result["status"],
                config=self.config,
            )

            self._execute_pipeline(ctx, run_logger, cancel_cb)
            if ctx.status == "CANCELED":
                return self._build_result(
                    product, area, inference_type, "CANCELED"
                )

            self._log_summary(ctx, run_logger)

            # --- Build typed DetectionResult ---
            save_res = ctx.save_result or {}
            raw_dets = result.get("detections", [])
            items: list[DetectionItem] = []
            for d in raw_dets:
                bbox = d.get("bbox", (0.0, 0.0, 0.0, 0.0))
                if isinstance(bbox, list):
                    bbox = tuple(bbox)
                items.append(DetectionItem(
                    label=d.get("class", "unknown"),
                    confidence=float(d.get("confidence", 0.0)),
                    bbox_xyxy=bbox,
                    metadata={k: v for k, v in d.items()
                              if k not in ("class", "confidence", "bbox")},
                ))

            latency = _time.time() - t0
            return DetectionResult(
                status=ctx.status,  # type: ignore[arg-type]
                items=items,
                latency=latency,
                product=product,
                area=area,
                inference_type=inference_type,
                error=result.get("error", "") or None,
                ckpt_path=result.get("ckpt_path", ""),
                anomaly_score=result.get("anomaly_score"),
                missing_items=result.get("missing_items", []),
                unexpected_items=result.get("unexpected_items", []),
                original_image_path=save_res.get("original_path", ""),
                preprocessed_image_path=save_res.get("preprocessed_path", ""),
                annotated_path=save_res.get("annotated_path", ""),
                heatmap_path=save_res.get("heatmap_path", ""),
                cropped_paths=save_res.get("cropped_paths", []),
                color_check=ctx.color_result,
                sequence_check=ctx.result.get("sequence_check"),
                result_frame=result.get("result_frame"),
            )

        except Exception as e:
            run_logger.error(f"Detection failed: {e}", exc_info=True)
            return self._build_result(
                product, area, inference_type, "ERROR", error=str(e)
            )

