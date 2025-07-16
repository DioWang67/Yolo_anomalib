import time
import cv2
import os
import tempfile
from core.anomalib_lightning_inference import lightning_inference
from core.logger import DetectionLogger
from core.config import DetectionConfig
from core.result_handler import ResultHandler
from core.utils import ImageUtils

class PerformanceTester:
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = DetectionLogger()
        self.config_path = config_path
        self.config = DetectionConfig.from_yaml(config_path)
        self.result_handler = ResultHandler(self.config)
        self.image_utils = ImageUtils()

    def single_inference_test(self, image_path: str = None, product: str = "PCBA1", area: str = "E"):
        """執行單次推理效能測試"""
        self.logger.logger.info("=" * 80)
        self.logger.logger.info("📊 異常檢測單次推理效能測試")
        self.logger.logger.info("=" * 80)

        # 讀取圖片或從相機捕獲
        if image_path:
            frame = cv2.imread(image_path)
            if frame is None:
                self.logger.logger.error("無法讀取圖片")
                return {"status": "ERROR", "error_message": "無法讀取圖片"}
        else:
            from camera.camera_controller import CameraController
            camera = CameraController(self.config)
            camera.initialize()
            frame = camera.capture_frame()
            camera.shutdown()
            if frame is None:
                self.logger.logger.error("無法從相機獲取圖像")
                return {"status": "ERROR", "error_message": "無法從相機獲取圖像"}

        # 預處理圖像到 640x640，黑色填充 (0, 0, 0) for Anomalib
        processed_image = self.image_utils.letterbox(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            size=self.config.imgsz,
            fill_color=(0, 0, 0)  # Anomalib 使用黑色填充
        )

        # 執行單次推理
        self.logger.logger.info(f"\n🎯 對{'圖片 ' + image_path if image_path else '相機輸入'}進行單次推理測試 (產品: {product}, 區域: {area})")
        self.logger.logger.info("-" * 80)

        start_time = time.time()
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
            result = lightning_inference(tmp_path, thread_safe=True, enable_timing=True)
            os.unlink(tmp_path)

        inference_time = time.time() - start_time

        if "error" in result:
            self.logger.logger.error(f"❌ 推理失敗: {result['error']}")
            return {"status": "ERROR", "error_message": result["error"], "inference_time": inference_time}

        # 保存結果
        save_result = self.result_handler.save_results(
            frame=frame,
            detections=[],
            status="PASS" if result.get("anomaly_score", 0.0) <= self.config.anomalib_config.get("threshold", 0.5) else "FAIL",
            detector=None,
            missing_items=[],
            processed_image=processed_image,
            anomaly_score=result.get("anomaly_score"),
            heatmap_path=result.get("output_path"),
            product=product,
            area=area
        )

        if save_result.get("status") == "ERROR":
            self.logger.logger.error(f"保存結果失敗: {save_result.get('error_message')}")
            return {
                "status": "ERROR",
                "error_message": save_result.get("error_message"),
                "inference_time": inference_time
            }

        self.logger.logger.info(f"📊 推理時間: {inference_time:.2f}s")
        if "timings" in result:
            timings = result["timings"]
            self.logger.logger.info(
                f"       📊 詳細時間: 資料集建立={timings.get('dataset_creation', 0):.2f}s, "
                f"模型推理={timings.get('model_inference', 0):.2f}s, "
                f"結果處理={timings.get('result_processing', 0):.2f}s"
            )
        self.logger.logger.info(f"✅ 推理完成 - 異常分數: {result.get('anomaly_score', 0.0):.4f}")
        self.logger.logger.info(f"💾 熱圖儲存至: {result.get('output_path', '')}")
        self.logger.logger.info(f"💾 結果已保存至: {save_result.get('original_path')}")

        return {
            "status": "SUCCESS",
            "image_path": image_path if image_path else "camera_input",
            "anomaly_score": result.get("anomaly_score"),
            "output_path": result.get("output_path"),
            "inference_time": inference_time,
            "timings": result.get("timings", {}),
            "original_image_path": save_result.get("original_path", ""),
            "preprocessed_image_path": save_result.get("preprocessed_path", ""),
            "heatmap_image_path": save_result.get("heatmap_path", ""),
            "product": product,
            "area": area
        }