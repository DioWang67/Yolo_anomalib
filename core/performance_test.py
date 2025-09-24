
import os
import tempfile
import time
from typing import Any, Dict

import cv2

from core.anomalib_lightning_inference import lightning_inference
from core.config import DetectionConfig
from core.logger import DetectionLogger
from core.services.results.handler import ResultHandler
from core.utils import ImageUtils


class PerformanceTester:
    """Utility for running ad-hoc performance checks on the anomalib pipeline."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.logger = DetectionLogger()
        self.config_path = config_path
        self.config = DetectionConfig.from_yaml(config_path)
        self.result_handler = ResultHandler(self.config)
        self.image_utils = ImageUtils()

    def single_inference_test(self, image_path: str | None = None, product: str = "PCBA1", area: str = "E") -> Dict[str, Any]:
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("Starting anomalib single inference benchmark")
        self.logger.logger.info("=" * 80)

        if image_path:
            frame = cv2.imread(image_path)
            if frame is None:
                self.logger.logger.error("Failed to read image from path")
                return {"status": "ERROR", "error_message": "Unable to read input image"}
        else:
            from camera.camera_controller import CameraController  # local import to avoid optional dependency at import time

            camera = CameraController(self.config)
            camera.initialize()
            frame = camera.capture_frame()
            camera.shutdown()
            if frame is None:
                self.logger.logger.error("Failed to capture frame from camera")
                return {"status": "ERROR", "error_message": "Unable to capture frame from camera"}

        processed_image = self.image_utils.letterbox(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            size=self.config.imgsz,
            fill_color=(0, 0, 0),
        )

        input_desc = image_path if image_path else "camera input"
        self.logger.logger.info(
            f"Running single inference on {input_desc} (product={product}, area={area})"
        )
        self.logger.logger.info("-" * 80)

        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            result = lightning_inference(tmp_path, thread_safe=True, enable_timing=True)
            os.unlink(tmp_path)
        inference_time = time.time() - start_time

        if "error" in result:
            self.logger.logger.error(f"Inference failed: {result['error']}")
            return {"status": "ERROR", "error_message": result["error"], "inference_time": inference_time}

        save_result = self.result_handler.save_results(
            frame=frame,
            detections=[],
            status="PASS" if result.get("anomaly_score", 0.0) <= self.config.anomalib_config.get("threshold", 0.5) else "FAIL",
            detector="anomalib",
            missing_items=[],
            processed_image=processed_image,
            anomaly_score=result.get("anomaly_score"),
            heatmap_path=result.get("output_path"),
            product=product,
            area=area,
        )

        if save_result.get("status") == "ERROR":
            self.logger.logger.error(f"Persisting result failed: {save_result.get('error_message')}")
            return {
                "status": "ERROR",
                "error_message": save_result.get("error_message"),
                "inference_time": inference_time,
            }

        self.logger.logger.info(f"Inference time: {inference_time:.2f}s")
        timings = result.get("timings")
        if timings:
            self.logger.logger.info(
                "       Detailed timings: dataset_creation=%.2fs, model_inference=%.2fs, result_processing=%.2fs"
                % (
                    timings.get("dataset_creation", 0.0),
                    timings.get("model_inference", 0.0),
                    timings.get("result_processing", 0.0),
                )
            )
        self.logger.logger.info(
            f"Final anomaly score: {result.get('anomaly_score', 0.0):.4f}"
        )
        self.logger.logger.info(f"Heatmap saved to: {result.get('output_path', '')}")
        self.logger.logger.info(f"Original image saved to: {save_result.get('original_path')}")

        return {
            "status": "SUCCESS",
            "image_path": image_path if image_path else "camera_input",
            "anomaly_score": result.get("anomaly_score"),
            "output_path": result.get("output_path"),
            "inference_time": inference_time,
            "timings": timings or {},
            "original_image_path": save_result.get("original_path", ""),
            "preprocessed_image_path": save_result.get("preprocessed_path", ""),
            "heatmap_image_path": save_result.get("heatmap_path", ""),
            "product": product,
            "area": area,
        }
