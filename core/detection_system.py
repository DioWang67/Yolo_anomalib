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
from core.services.model_manager import ModelManager
import numpy as np

from core.config import DetectionConfig
from core.logger import DetectionLogger
from core.inference_engine import InferenceEngine
from camera.camera_controller import CameraController
from core.services.color_checker import ColorCheckerService
from core.services.result_sink import ExcelImageResultSink
from core.pipeline.context import DetectionContext
from core.pipeline.registry import PipelineEnv, build_pipeline, default_pipeline
from core.logging_utils import context_adapter
from core.result_adapter import normalize_result


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        """Create a detection system instance.

        Args:
            config_path: Path to the global config YAML (project-level defaults).
        """
        self.logger = DetectionLogger()
        self.config = self.load_config(config_path)
        self.camera: CameraController | None = None
        self.result_sink = ExcelImageResultSink(self.config, base_dir=self.config.output_dir, logger=self.logger)
        self.inference_engine: InferenceEngine | None = None
        self.current_inference_type: str | None = None
        self.model_manager = ModelManager(self.logger, max_cache_size=self.config.max_cache_size)
        self.color_service = ColorCheckerService()
        self.initialize_camera()

    def load_config(self, config_path: str) -> DetectionConfig:
        """Load global config from YAML into DetectionConfig dataclass."""
        return DetectionConfig.from_yaml(config_path)

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

    def initialize_camera(self) -> None:
        """Initialize camera if available; log and fall back to dummy on failure."""
        self.logger.logger.info("Initializing camera...")
        try:
            self.camera = CameraController(self.config)
            self.camera.initialize()
            self.logger.logger.info("Camera is ready")
        except Exception as e:
            self.logger.logger.error(f"Camera init failed: {str(e)}")
            self.logger.logger.warning("Camera disabled; using dummy image for detection")
            self.camera = None

    def load_model_configs(self, product: str, area: str, inference_type: str) -> None:
        """Switch to a specific (product, area, type) model configuration.

        Uses ModelManager to keep an LRU cache of engines and snapshots of configs.
        """
        engine, _ = self.model_manager.switch(self.config, product, area, inference_type)
        self.inference_engine = engine
        self.current_inference_type = inference_type.lower()

    def detect(self, product: str, area: str, inference_type: str, frame: np.ndarray | None = None, cancel_cb=None) -> dict:
        """Run one-shot detection and return a normalized result dict.

        Args:
            product: Product name (top-level folder under models/)
            area: Area/station name (second-level folder)
            inference_type: Backend key (e.g., 'yolo', 'anomalib', or custom)
            frame: Optional BGR image np.ndarray; if None, capture from camera or use dummy

        Returns:
            Dict with normalized keys consumed by sinks and GUI (status, paths, etc.).
        """
        try:
            def _canceled() -> bool:
                try:
                    return bool(cancel_cb and cancel_cb())
                except Exception:
                    return False

            def _cancel_result(status: str = "CANCELED") -> dict:
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
            self.load_model_configs(product, area, inference_type)
            if _canceled():
                return _cancel_result()
            run_logger = context_adapter(self.logger.logger, product, area, inference_type)
            run_logger.info("Start detection")
            inference_type_name = inference_type.lower()

            # Ensure color checker ready if enabled
            if self.config.enable_color_check and self.config.color_model_path:
                try:
                    # Load latest overrides from model-level YAML (even if engine is cached)
                    overrides = getattr(self.config, "color_threshold_overrides", None)
                    rules_over = getattr(self.config, "color_rules_overrides", None)
                    try:
                        import yaml as _yaml
                        _cfg_path = str(PROJECT_ROOT / "models" / product / area / inference_type / "config.yaml")
                        if os.path.exists(_cfg_path):
                            with open(_cfg_path, "r", encoding="utf-8") as _f:
                                _model_cfg = _yaml.safe_load(_f) or {}
                                ov = _model_cfg.get("color_threshold_overrides")
                                if isinstance(ov, dict) and ov:
                                    overrides = ov
                                crov = _model_cfg.get("color_rules_overrides")
                                if isinstance(crov, dict) and crov:
                                    rules_over = crov
                    except Exception:
                        pass
                    self.color_service.ensure_loaded(
                        self.config.color_model_path,
                        overrides=overrides,
                        rules_overrides=rules_over,
                    )
                    run_logger.info("Color checker loaded (LEDQCEnhanced)")
                except Exception as e:
                    run_logger.error(f"Color checker init failed: {str(e)}")

            # Capture frame if not provided
            if frame is None:
                if self.camera:
                    frame = self.camera.capture_frame()
                    if frame is None:
                        run_logger.error("Failed to read frame from camera")
                        frame = np.zeros((640, 640, 3), dtype=np.uint8)
                        run_logger.warning("Using dummy image for detection")
                else:
                    frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    run_logger.warning("Camera unavailable; using dummy image for detection")

            # Temporary path for anomalib output
            output_path = None
            if inference_type_name == "anomalib":
                output_path = self.result_sink.get_annotated_path(
                    status="TEMP", detector=inference_type, product=product, area=area
                )

            # Inference
            if _canceled():
                return _cancel_result()
            raw_result = self.inference_engine.infer(frame, product, area, inference_type_name, output_path)
            result = normalize_result(raw_result, inference_type_name, frame)
            if "status" in result and result["status"] == "ERROR":
                run_logger.error(f"Inference failed: {result.get('error')}")
                return {
                    "status": "ERROR",
                    "error": result.get("error"),
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

            status = result["status"]

            # Build pipeline context
            ctx = DetectionContext(
                product=product,
                area=area,
                inference_type=inference_type,
                frame=frame,
                processed_image=result.get("processed_image", frame),
                result=result,
                status=status,
                config=self.config,
            )

            env = PipelineEnv(
                color_service=self.color_service,
                result_sink=self.result_sink,
                logger=run_logger,
                product=product,
                area=area,
                config=self.config,
            )

            pipe_cfg = getattr(self.config, "pipeline", None)
            step_opts_raw = getattr(self.config, "steps", {}) or {}
            step_opts = {str(k).lower(): v for k, v in step_opts_raw.items()}

            if isinstance(pipe_cfg, list) and pipe_cfg:
                step_names = [str(name) for name in pipe_cfg]
            else:
                step_names = default_pipeline(env)

            steps = build_pipeline(step_names, env, step_opts)
            if not steps:
                run_logger.warning("Pipeline produced no executable steps; enforcing save_results fallback")
                steps = build_pipeline(["save_results"], env, step_opts)

            # Adjust anomalib output path from TEMP to PASS/FAIL
            if inference_type_name == "anomalib" and output_path:
                correct_output_path = self.result_sink.get_annotated_path(
                    status=status,
                    detector=inference_type,
                    product=product,
                    area=area,
                    anomaly_score=result.get("anomaly_score"),
                )
                if output_path != correct_output_path and os.path.exists(output_path):
                    os.makedirs(os.path.dirname(correct_output_path), exist_ok=True)
                    try:
                        shutil.move(output_path, correct_output_path)
                        result["output_path"] = correct_output_path
                        run_logger.info(f"Moved output: {output_path} -> {correct_output_path}")
                    except Exception as e:
                        run_logger.error(f"Move output failed: {str(e)}")
                    # Cleanup temp directory if empty
                    try:
                        old_dir = os.path.dirname(output_path)
                        if os.path.isdir(old_dir) and not os.listdir(old_dir):
                            os.rmdir(old_dir)
                    except Exception as cleanup_err:
                        run_logger.warning(f"Cleanup temp dir failed: {cleanup_err}")
                else:
                    run_logger.info(f"Output path OK: {correct_output_path}")
                    result["output_path"] = correct_output_path

            # Persist results via pipeline
            for step in steps:
                if _canceled():
                    return _cancel_result()
                step.run(ctx)

            # Use possibly-updated status from pipeline (e.g., color/position checks)
            status = ctx.status

            # Logging summary with final status
            if inference_type_name == "anomalib":
                self.logger.log_anomaly(status, result.get("anomaly_score", 0.0))
            else:
                self.logger.log_detection(status, result.get("detections", []))

            # Extra debugging: print reasons when FAIL
            try:
                if str(status).upper() == "FAIL":
                    miss = result.get("missing_items", [])
                    if miss:
                        run_logger.info(f"Fail reason - missing items: {miss}")
                    c = ctx.color_result or {}
                    if isinstance(c, dict) and (not c.get("is_ok", True)):
                        items = c.get("items", []) or []
                        fail_cnt = sum(1 for it in items if not it.get("is_ok", False))
                        run_logger.info(f"Fail reason - color mismatch: {fail_cnt}/{len(items)} items failed")
                    unexp = result.get("unexpected_items", [])
                    if unexp:
                        run_logger.info(f"Fail reason - unexpected items: {unexp}")
                    pos_wrong = [d.get("class") for d in (result.get("detections", []) or []) if d.get("position_status") == "WRONG"]
                    if pos_wrong:
                        run_logger.info(f"Fail reason - position wrong: {pos_wrong}")
            except Exception:
                pass

            return {
                "status": status,
                "product": product,
                "area": area,
                "inference_type": inference_type,
                "ckpt_path": result.get("ckpt_path", ""),
                "anomaly_score": result.get("anomaly_score", ""),
                "detections": result.get("detections", []),
                "missing_items": result.get("missing_items", []),
                "result_frame": result.get("result_frame"),
                "original_image_path": (ctx.save_result or {}).get("original_path", ""),
                "preprocessed_image_path": (ctx.save_result or {}).get("preprocessed_path", ""),
                "annotated_path": (ctx.save_result or {}).get("annotated_path", ""),
                "heatmap_path": (ctx.save_result or {}).get("heatmap_path", ""),
                "cropped_paths": (ctx.save_result or {}).get("cropped_paths", []),
                "color_check": ctx.color_result,
            }

        except Exception as e:
            self.logger.logger.error(f"Detection failed: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e),
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

