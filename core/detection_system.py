"""Detection system orchestrator.

Responsibilities:
- Load and merge per-product/area/type model configs via ModelManager
- Initialize camera and inference engine backends
- Build a per-run pipeline (position/color/save) and execute
- Normalize outputs and persist results via sinks

Public entrypoint: DetectionSystem.detect(product, area, inference_type, frame=None)
"""

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from camera.camera_controller import CameraController
from core.config import DetectionConfig
from core.inference_engine import InferenceEngine
from core.logging_config import DetectionLogger
from core.logging_utils import context_adapter
from core.path_utils import project_root
from core.pipeline.context import DetectionContext
from core.pipeline.registry import PipelineEnv, build_pipeline, default_pipeline
from core.result_adapter import normalize_result
from core.services.color_checker import ColorCheckerService
from core.services.model_manager import ModelManager
from core.services.result_sink import ExcelImageResultSink


class _InferenceTypeToken(str):
    @property
    def value(self) -> str:
        return str(self)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        """Initializes the DetectionSystem with project settings.

        Args:
            config_path: Relative or absolute path to the global config.yaml.
        """

        self.logger = DetectionLogger()
        root_dir = project_root()
        if not (root_dir / "config.yaml").exists():
            root_dir = Path(__file__).resolve().parent.parent
        resolved_config = Path(config_path) if config_path else root_dir / "config.yaml"
        resolved_config = resolved_config.resolve()
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
        """Release resources: models, camera, sinks."""
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
        except Exception:
            return overrides, rules_over

        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                model_cfg = _yaml.safe_load(handle) or {}
        except Exception as exc:
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

    def _build_empty_result(
        self, product: str, area: str, inference_type: str, status: str
    ) -> dict:
        """Builds a standardized dictionary for empty or error results."""
        return {
            "status": status,
            "product": product,
            "area": area,
            "inference_type": inference_type,
            "ckpt_path": "",
            "anomaly_score": "",
            "detections": [],
            "missing_items": [],
            "original_image_path": "",
            "preprocessed_image_path": "",
            "annotated_path": "",
            "heatmap_path": "",
            "cropped_paths": [],
            "color_check": None,
        }

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
        """Capture a frame from the camera if not provided."""
        if frame is not None:
            return frame
        if self.camera:
            frame = self.camera.capture_frame()
            if frame is None:
                run_logger.error("Failed to read frame from camera")
                run_logger.warning("Using dummy image for detection")
                return np.zeros((640, 640, 3), dtype=np.uint8)
            return frame
        else:
            run_logger.warning("Camera unavailable; using dummy image for detection")
            return np.zeros((640, 640, 3), dtype=np.uint8)

    def _run_inference(
        self,
        frame: np.ndarray,
        product: str,
        area: str,
        inference_type: str,
        run_logger,
    ) -> dict:
        """Execute the inference engine and handle results."""
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
    ) -> dict:
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
            dict: Normalized results dictionary containing status, detections, and artifact paths.
        """
        run_logger = context_adapter(self.logger.logger, product, area, inference_type)
        run_logger.info("Start detection")

        try:
            if self._is_canceled(cancel_cb):
                return self._build_empty_result(
                    product, area, inference_type, "CANCELED"
                )

            self._prepare_resources(product, area, inference_type, run_logger)
            frame = self._acquire_frame(frame, run_logger)

            if self._is_canceled(cancel_cb):
                return self._build_empty_result(
                    product, area, inference_type, "CANCELED"
                )

            result = self._run_inference(frame, product, area, inference_type, run_logger)
            if result.get("status") == "ERROR":
                error_msg = result.get("error", "Inference failed")
                final_res = self._build_empty_result(
                    product, area, inference_type, "ERROR"
                )
                final_res["error"] = error_msg
                return final_res

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
                return self._build_empty_result(
                    product, area, inference_type, "CANCELED"
                )

            self._log_summary(ctx, run_logger)

            save_res = ctx.save_result or {}
            return {
                "status": ctx.status,
                "error": result.get("error", ""),
                "product": product,
                "area": area,
                "inference_type": inference_type,
                "ckpt_path": result.get("ckpt_path", ""),
                "anomaly_score": result.get("anomaly_score", ""),
                "detections": result.get("detections", []),
                "missing_items": result.get("missing_items", []),
                "result_frame": result.get("result_frame"),
                "original_image_path": save_res.get("original_path", ""),
                "preprocessed_image_path": save_res.get("preprocessed_path", ""),
                "annotated_path": save_res.get("annotated_path", ""),
                "heatmap_path": save_res.get("heatmap_path", ""),
                "cropped_paths": save_res.get("cropped_paths", []),
                "color_check": ctx.color_result,
            }

        except Exception as e:
            run_logger.error(f"Detection failed: {e}", exc_info=True)
            final_res = self._build_empty_result(product, area, inference_type, "ERROR")
            final_res["error"] = str(e)
            return final_res
