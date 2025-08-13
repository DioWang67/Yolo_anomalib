import cv2
import time
import numpy as np
from core.base_model import BaseInferenceModel

from core.anomalib_lightning_inference import initialize, lightning_inference
from core.utils import ImageUtils


class AnomalibInferenceModel(BaseInferenceModel):
    def __init__(self, config):
        super().__init__(config)
        self.image_utils = ImageUtils()

    def initialize(self, product: str = None, area: str = None) -> bool:
        try:
            self.logger.logger.info(f"正在初始化 Anomalib 模型 (產品: {product}, 區域: {area})...")
            anomalib_config = self.config.anomalib_config
            if not anomalib_config:
                raise ValueError("Anomalib 配置缺失")
            initialize(config=anomalib_config, product=product, area=area)
            self.is_initialized = True
            self.logger.logger.info(f"Anomalib 模型初始化成功 (產品: {product}, 區域: {area})")
            return True
        except Exception as e:
            self.logger.logger.error(f"Anomalib 模型初始化失敗: {str(e)}")
            return False

    def infer(self, image: np.ndarray, product: str, area: str, output_path: str = None):
        if not self.is_initialized:
            self.initialize(product, area)
            if not self.is_initialized:
                raise RuntimeError("Anomalib 模型初始化失敗")

        try:
            start_time = time.time()

            processed_image = self.image_utils.letterbox(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                size=self.config.imgsz,
                fill_color=(0, 0, 0)
            )

            result = lightning_inference(
                image=processed_image,
                thread_safe=True,
                enable_timing=True,
                product=product,
                area=area,
                output_path=output_path,
                num_workers=self.config.anomalib_config.get('num_workers', 1) if self.config.anomalib_config else 1
            )

            if 'error' in result:
                raise RuntimeError(result['error'])

            anomaly_score = result.get('anomaly_score', 0.0)
            threshold = self.config.anomalib_config.get('threshold', 0.5)
            is_anomaly = anomaly_score > threshold
            status = "FAIL" if is_anomaly else "PASS"

            inference_time = time.time() - start_time

            self.logger.logger.debug(
                f"Anomalib 推理時間: {inference_time:.3f}s, 異常分數: {anomaly_score:.4f}, 圖像尺寸: {processed_image.shape}"
            )

            return {
                "inference_type": "anomalib",
                "status": status,
                "anomaly_score": anomaly_score,
                "is_anomaly": is_anomaly,
                "inference_time": inference_time,
                "processed_image": processed_image,
                "result_frame": processed_image,
                "output_path": result.get('heatmap_path', ''),
                "expected_items": [],
                "product": product,
                "area": area
            }

        except Exception as e:
            self.logger.logger.error(f"Anomalib 推理失敗: {str(e)}")
            raise
