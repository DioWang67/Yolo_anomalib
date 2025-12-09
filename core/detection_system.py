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
        """Create a detection system instance.

        Args:
            config_path: Path to the global config YAML (project-level defaults).
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

    def detect(
        self,
        product: str,
        area: str,
        inference_type: str,
        frame: np.ndarray | None = None,
        cancel_cb=None,
    ) -> dict:
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
            run_logger = context_adapter(
                self.logger.logger, product, area, inference_type
            )
            run_logger.info("Start detection")
            inference_type_name = inference_type.lower()

            # Ensure color checker ready if enabled
            if self.config.enable_color_check and self.config.color_model_path:
                try:
                    overrides, rules_over = self._load_color_override_bundle(
                        product, area, inference_type
                    )
                    checker_type = (
                        getattr(self.config, "color_checker_type", "color_qc")
                        or "color_qc"
                    )
                    default_threshold = getattr(
                        self.config, "color_score_threshold", None
                    )
                    self.color_service.ensure_loaded(
                        self.config.color_model_path,
                        overrides=overrides,
                        rules_overrides=rules_over,
                        checker_type=checker_type,
                        default_threshold=default_threshold,
                    )
                    run_logger.info(f"Color checker loaded ({checker_type})")
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
                    run_logger.warning(
                        "Camera unavailable; using dummy image for detection"
                    )

            # Temporary path for anomalib output
            output_path = None
            if inference_type_name == "anomalib":
                output_path = self.result_sink.get_annotated_path(
                    status="TEMP", detector=inference_type, product=product, area=area
                )

            # Inference
            if _canceled():
                return _cancel_result()
            raw_result = self.inference_engine.infer(
                frame,
                product,
                area,
                _InferenceTypeToken(inference_type_name),
                output_path,
            )
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
                run_logger.warning(
                    "Pipeline produced no executable steps; enforcing save_results fallback"
                )
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
                        run_logger.info(
                            f"Moved output: {output_path} -> {correct_output_path}"
                        )
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
                        run_logger.info(
                            f"Fail reason - color mismatch: {fail_cnt}/{len(items)} items failed"
                        )
                    unexp = result.get("unexpected_items", [])
                    if unexp:
                        run_logger.info(f"Fail reason - unexpected items: {unexp}")
                    pos_wrong = [
                        d.get("class")
                        for d in (result.get("detections", []) or [])
                        if d.get("position_status") == "WRONG"
                    ]
                    if pos_wrong:
                        run_logger.info(f"Fail reason - position wrong: {pos_wrong}")
            except Exception:
                pass

            return {
                "status": status,
                "error": result.get("error", ""),
                "product": product,
                "area": area,
                "inference_type": inference_type,
                "ckpt_path": result.get("ckpt_path", ""),
                "anomaly_score": result.get("anomaly_score", ""),
                "detections": result.get("detections", []),
                "missing_items": result.get("missing_items", []),
                "result_frame": result.get("result_frame"),
                "original_image_path": (ctx.save_result or {}).get("original_path", ""),
                "preprocessed_image_path": (ctx.save_result or {}).get(
                    "preprocessed_path", ""
                ),
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
