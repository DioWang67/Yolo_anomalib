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
from core.pipeline.steps import ColorCheckStep, SaveResultsStep, PositionCheckStep
from core.logging_utils import context_adapter
from core.result_adapter import normalize_result


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
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
        return DetectionConfig.from_yaml(config_path)

    def shutdown(self) -> None:
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
        self.logger.logger.info("Initializing camera...")
        try:
            self.camera = CameraController(self.config)
            self.camera.initialize()
            self.logger.logger.info("Camera is ready")
        except Exception as e:
            self.logger.logger.error(f"Camera init failed: {str(e)}")
            self.logger.logger.warning("Camera disabled; using dummy image for detection")
            self.camera = None

    def initialize_inference_engine(self) -> None:
        self.logger.logger.info("Initializing inference engine...")
        if not self.inference_engine or not self.inference_engine.initialize():
            self.logger.logger.error("Inference engine init failed")
            raise RuntimeError("Inference engine init failed")
        self.logger.logger.info("Inference engine is ready")

    def initialize_product_models(self, product: str) -> None:
        if self.config.enable_yolo:
            self.logger.logger.info(f"YOLO is enabled for {product}")
        if self.config.enable_anomalib:
            try:
                from core.anomalib_lightning_inference import initialize_product_models as _anoma_init
                _anoma_init(self.config.anomalib_config, product)
                self.logger.logger.info(f"Anomalib models initialized for {product}")
            except Exception as e:
                self.logger.logger.error(f"Anomalib init failed for {product}: {str(e)}")
                raise

    def load_model_configs(self, product: str, area: str, inference_type: str) -> None:
        engine, _ = self.model_manager.switch(self.config, product, area, inference_type)
        self.inference_engine = engine
        self.current_inference_type = inference_type.lower()

    def detect(self, product: str, area: str, inference_type: str, frame: np.ndarray | None = None) -> dict:
        try:
            self.load_model_configs(product, area, inference_type)
            run_logger = context_adapter(self.logger.logger, product, area, inference_type)
            run_logger.info("Start detection")
            inference_type_name = inference_type.lower()

            # Ensure color checker ready if enabled
            if self.config.enable_color_check and self.config.color_model_path:
                try:
                    self.color_service.ensure_loaded(self.config.color_model_path)
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

            # Build pipeline steps from config if provided; fallback to default
            steps = []
            pipe_cfg = getattr(self.config, "pipeline", None)
            step_opts = getattr(self.config, "steps", {}) or {}
            def _add_color():
                if self.color_service.is_ready():
                    steps.append(ColorCheckStep(self.color_service, run_logger, options=step_opts.get("color_check", {})))
                else:
                    run_logger.warning(f"Color checker not ready; skip color check (path={self.config.color_model_path})")
            def _add_save():
                steps.append(SaveResultsStep(self.result_sink, run_logger, options=step_opts.get("save_results", {})))
            def _add_position():
                steps.append(PositionCheckStep(run_logger, product=product, area=area, options=step_opts.get("position_check", {})))
            if isinstance(pipe_cfg, list) and pipe_cfg:
                for name in pipe_cfg:
                    key = str(name).lower().strip()
                    if key == "color_check" and self.config.enable_color_check and self.config.color_model_path:
                        _add_color()
                    elif key == "position_check":
                        _add_position()
                    elif key == "save_results":
                        _add_save()
                    else:
                        run_logger.warning(f"Unknown or disabled pipeline step: {name}")
            else:
                # default pipeline: optional color check then save
                if self.config.enable_color_check and self.config.color_model_path:
                    _add_color()
                # 預設不強制位置檢查，保持與既有行為一致；可在 pipeline 配置中開啟
                _add_save()

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
                step.run(ctx)
            # Logging summary
            if inference_type_name == "anomalib":
                self.logger.log_anomaly(status, result.get("anomaly_score", 0.0))
            else:
                self.logger.log_detection(status, result.get("detections", []))

            return {
                "status": status,
                "product": product,
                "area": area,
                "inference_type": inference_type,
                "ckpt_path": result.get("ckpt_path", ""),
                "anomaly_score": result.get("anomaly_score", ""),
                "detections": result.get("detections", []),
                "missing_items": result.get("missing_items", []),
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
