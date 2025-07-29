
import sys
import logging
from datetime import datetime
import numpy as np
from core.anomalib_lightning_inference import initialize_product_models, lightning_inference, get_status
from core.result_handler import ResultHandler
from core.config import DetectionConfig
from core.logger import DetectionLogger
from core.inference_engine import InferenceEngine, InferenceType
from camera.camera_controller import CameraController

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/detection_{datetime.now().strftime('%Y%m%d')}.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(encoding='utf-8')

class DetectionSystem:
    def __init__(self, config_path="config.yaml"):
        self.logger = DetectionLogger()
        self.config = DetectionConfig.from_yaml(config_path)
        self.camera = None
        self.result_handler = ResultHandler(self.config, base_dir=self.config.output_dir, logger=self.logger)
        self.inference_engine = InferenceEngine(self.config)
        self.initialize_camera()
        self.initialize_inference_engine()

    def load_config(self, config_path):
        return DetectionConfig.from_yaml(config_path)

    def initialize_camera(self):
        self.logger.logger.info("正在初始化相機...")
        try:
            self.camera = CameraController(self.config)
            self.camera.initialize()
            self.logger.logger.info("相機初始化成功")
        except Exception as e:
            self.logger.logger.error(f"相機初始化失敗: {str(e)}")
            self.logger.logger.warning("相機不可用，將使用模擬圖像進行測試")
            self.camera = None

    def initialize_inference_engine(self):
        self.logger.logger.info("正在初始化推理引擎...")
        if not self.inference_engine.initialize():
            self.logger.logger.error("推理引擎初始化失敗")
            raise RuntimeError("推理引擎初始化失敗")
        self.logger.logger.info("推理引擎初始化成功")

    def initialize_product_models(self, product):
        """初始化指定機種的所有模型"""
        if self.config.enable_yolo:
            self.logger.logger.info(f"YOLO 模型已為 {product} 準備")
        
        if self.config.enable_anomalib:
            try:
                initialize_product_models(self.config.anomalib_config, product)
                self.logger.logger.info(f"機種 {product} 的 Anomalib 模型初始化完成")
            except Exception as e:
                self.logger.logger.error(f"機種 {product} 的 Anomalib 模型初始化失敗: {str(e)}")
                raise

    def detect(self, product, area, inference_type):
        try:
            self.logger.logger.info(f"開始檢測 - 產品: {product}, 區域: {area}, 推理類型: {inference_type}")
            inference_type_enum = InferenceType.from_string(inference_type)
            
            # 嘗試使用真實相機捕獲圖像
            if self.camera:
                frame = self.camera.capture_frame()
                if frame is None:
                    self.logger.logger.error("無法從相機獲取圖像")
                    frame = np.zeros((640, 640, 3), dtype=np.uint8)  # 回退到模擬圖像
                    self.logger.logger.warning("使用模擬圖像進行檢測")
            else:
                frame = np.zeros((640, 640, 3), dtype=np.uint8)  # 模擬圖像
                self.logger.logger.warning("相機不可用，使用模擬圖像進行檢測")
            
            image_path = f"tmp_{product}_{area}_{datetime.now().strftime('%H%M%S')}.jpg"
            # cv2.imwrite(image_path, frame)
            
            result = self.inference_engine.infer(frame, product, area, inference_type_enum)
            if "status" in result and result["status"] == "ERROR":
                self.logger.logger.error(f"推理失敗: {result.get('error')}")
                return result
            
            status = result["status"]
            if inference_type_enum == InferenceType.ANOMALIB:
                self.result_handler.save_results(
                    frame=frame,
                    detections=[],
                    status=status,
                    detector=inference_type,
                    missing_items=[],
                    processed_image=result["processed_image"],
                    anomaly_score=result.get("anomaly_score"),
                    heatmap_path=result.get("heatmap_path"),
                    product=product,
                    area=area,
                    ckpt_path=result.get("ckpt_path")
                )
                self.logger.log_anomaly(status, result.get("anomaly_score", 0.0))
            else:  # YOLO
                self.result_handler.save_results(
                    frame=frame,
                    detections=result.get("detections", []),
                    status=status,
                    detector=inference_type,
                    missing_items=result.get("missing_items", []),
                    processed_image=result["processed_image"],
                    anomaly_score=None,
                    heatmap_path=None,
                    product=product,
                    area=area,
                    ckpt_path=self.config.weights
                )
                self.logger.log_detection(status, result.get("detections", []))
            
            self.logger.logger.info(f"檢測完成 - 狀態: {status}")
            return {
                "status": status,
                "product": product,
                "area": area,
                "inference_type": inference_type,
                "ckpt_path": result.get("ckpt_path", self.config.weights),
                "anomaly_score": result.get("anomaly_score", ""),
                "detections": result.get("detections", []),
                "missing_items": result.get("missing_items", []),
                "original_image_path": f"Result/{datetime.now().strftime('%Y%m%d')}/{status}/original/{product}_{area}_{datetime.now().strftime('%H%M%S')}.jpg",
                "preprocessed_image_path": f"Result/{datetime.now().strftime('%Y%m%d')}/{status}/preprocessed/{product}_{area}_{datetime.now().strftime('%H%M%S')}.jpg",
                "heatmap_path": result.get("heatmap_path", ""),
                "cropped_paths": result.get("cropped_paths", [])
            }
            
        except Exception as e:
            self.logger.logger.error(f"檢測失敗: {str(e)}")
            return {"status": "ERROR", "error": str(e)}

    def run(self):
        self.logger.logger.info("檢測系統已啟動，等待輸入訊號...")
        
        available_products = list(self.config.expected_items.keys())
        if not available_products:
            self.logger.logger.error("配置中未定義任何機種")
            return
        
        print(f"可用機種: {', '.join(available_products)}")
        while True:
            product = input("請輸入要檢測的機種 (或輸入 'quit' 退出): ").strip()
            if product.lower() == "quit":
                self.logger.logger.info("退出檢測系統")
                self.inference_engine.shutdown()
                if self.camera:
                    self.camera.shutdown()
                return
            if product not in available_products:
                print(f"無效的機種: {product}，請選擇: {', '.join(available_products)}")
                continue
            break
        
        try:
            self.initialize_product_models(product)
        except Exception as e:
            self.logger.logger.error(f"初始化 {product} 模型失敗: {str(e)}")
            if self.camera:
                self.camera.shutdown()
            return
        
        available_areas = list(self.config.expected_items.get(product, {}).keys())
        
        while True:
            print(f"可用區域: {', '.join(available_areas)}")
            cmd = input("請輸入檢測指令 (格式: area,inference_type 或 quit): ").strip()
            if cmd.lower() == "quit":
                self.logger.logger.info("退出檢測系統")
                self.inference_engine.shutdown()
                if self.camera:
                    self.camera.shutdown()
                break
            
            try:
                parts = cmd.split(",")
                if len(parts) != 2:
                    print("指令格式錯誤，應為: area,inference_type")
                    continue
                
                area, inference_type = parts
                if area not in available_areas:
                    print(f"無效的區域: {area}，請選擇: {', '.join(available_areas)}")
                    continue
                if inference_type.lower() not in ["yolo", "anomalib"]:
                    print("無效的推理類型，應為: yolo 或 anomalib")
                    continue
                
                result = self.detect(product, area, inference_type.lower())
                print("\n=== 檢測結果 ===")
                print(f"狀態: {result['status']}")
                print(f"產品: {result.get('product', '')}")
                print(f"區域: {result.get('area', '')}")
                print(f"推理類型: {result.get('inference_type', '')}")
                print(f"檢查點路徑: {result.get('ckpt_path', '')}")
                print(f"異常分數: {result.get('anomaly_score', '')}")
                print(f"檢測項目: {result.get('detections', [])}")
                print(f"缺少項目: {result.get('missing_items', [])}")
                print(f"原始圖像路徑: {result.get('original_image_path', '')}")
                print(f"預處理圖像路徑: {result.get('preprocessed_image_path', '')}")
                print(f"異常熱圖路徑: {result.get('heatmap_path', '')}")
                print(f"裁剪圖像路徑: {result.get('cropped_paths', [])}")
                print("====================")
                
            except Exception as e:
                self.logger.logger.error(f"處理指令失敗: {str(e)}")
                print(f"錯誤: {str(e)}")

if __name__ == "__main__":
    system = DetectionSystem()
    system.run()