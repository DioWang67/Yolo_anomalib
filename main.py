import os
import shutil
import sys
import logging
from datetime import datetime
import numpy as np
import yaml
import copy
from core.anomalib_lightning_inference import initialize_product_models, lightning_inference, get_status
from core.result_handler import ResultHandler
from core.config import DetectionConfig
from core.logger import DetectionLogger
from core.inference_engine import InferenceEngine, InferenceType
from camera.camera_controller import CameraController
from collections import OrderedDict

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
        self.config = self.load_config(config_path)
        self.camera = None
        self.result_handler = ResultHandler(self.config, base_dir=self.config.output_dir, logger=self.logger)
        self.inference_engine = None
        self.current_inference_type = None
        self.model_cache = OrderedDict()
        self.max_cache_size = self.config.max_cache_size
        self.initialize_camera()

    def load_config(self, config_path):
        """載入 YAML 組態並回傳 DetectionConfig 物件"""
        return DetectionConfig.from_yaml(config_path)

    def shutdown(self):
        """釋放推理引擎與相機資源"""
        if self.inference_engine:
            self.inference_engine.shutdown()
            self.inference_engine = None
        if self.camera:
            self.camera.shutdown()
            self.camera = None

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

    def load_model_configs(self, product: str, area: str, inference_type: str) -> None:
        """根據指定產品、區域與模型載入設定並初始化推理引擎"""
        cache_key = (product, area)

        if cache_key in self.model_cache and inference_type in self.model_cache[cache_key]:
            self.logger.logger.info(
                f"使用快取模型: 產品 {product}, 區域 {area}, 模型 {inference_type}"
            )
            self.inference_engine, cached_config = self.model_cache[cache_key][inference_type]
            self.model_cache.move_to_end(cache_key)
            self.config.__dict__.update(copy.deepcopy(cached_config.__dict__))
            self.current_inference_type = inference_type
            return

        self.logger.logger.info(
            f"切換至產品: {product}, 區域: {area}，模型: {inference_type}",
        )

        model_config_path = os.path.join(
            "models", product, area, inference_type, "config.yaml"
        )
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"找不到模型設定檔: {model_config_path}")

        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.config.current_product = product
        self.config.current_area = area

        if inference_type == "yolo":
            self.config.enable_yolo = True
            self.config.enable_anomalib = False
            self.config.weights = cfg.get("weights")
            self.config.device = cfg.get("device", self.config.device)
            self.config.conf_thres = cfg.get("conf_thres", self.config.conf_thres)
            self.config.iou_thres = cfg.get("iou_thres", self.config.iou_thres)
            self.config.imgsz = tuple(cfg.get("imgsz", self.config.imgsz))
            self.config.timeout = cfg.get("timeout", self.config.timeout)
            self.config.expected_items = cfg.get("expected_items", {})
            self.config.output_dir = cfg.get("output_dir", self.config.output_dir)
            self.config.position_config = cfg.get("position_config", {})
            self.config.anomalib_config = None
        else:  # anomalib
            self.config.enable_yolo = False
            self.config.enable_anomalib = True
            self.config.device = cfg.get("device", self.config.device)
            self.config.imgsz = tuple(cfg.get("imgsz", self.config.imgsz))
            self.config.width = cfg.get("width", self.config.width)
            self.config.height = cfg.get("height", self.config.height)
            self.config.output_dir = cfg.get("output_dir", self.config.output_dir)
            self.config.anomalib_config = cfg.get("anomalib_config")
            self.config.expected_items = {}
            self.config.position_config = {}
            self.config.weights = ""

        self.inference_engine = InferenceEngine(self.config)
        self.initialize_inference_engine()

        if inference_type == "anomalib":
            try:
                self.initialize_product_models(product)
            except Exception as e:
                self.logger.logger.error(f"重新初始化 {product} 模型失敗: {str(e)}")
                raise

        self.current_inference_type = inference_type
        if cache_key not in self.model_cache:
            self.model_cache[cache_key] = {}
        self.model_cache[cache_key][inference_type] = (self.inference_engine, copy.deepcopy(self.config))
        self.model_cache.move_to_end(cache_key)
        if len(self.model_cache) > self.max_cache_size:
            old_key, engines = self.model_cache.popitem(last=False)
            for eng, _ in engines.values():
                eng.shutdown()
            self.logger.logger.info(f"釋放快取模型: 產品 {old_key[0]}, 區域 {old_key[1]}")

    def detect(self, product, area, inference_type):
        try:
            self.load_model_configs(product, area, inference_type)
            self.logger.logger.info(f"開始檢測 - 產品: {product}, 區域: {area}, 推理類型: {inference_type}")
            inference_type_enum = InferenceType.from_string(inference_type)
            
            # 嘗試使用真實相機捕獲圖像
            if self.camera:
                frame = self.camera.capture_frame()
                if frame is None:
                    self.logger.logger.error("無法從相機獲取圖像")
                    frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    self.logger.logger.warning("使用模擬圖像進行檢測")
            else:
                frame = np.zeros((640, 640, 3), dtype=np.uint8)
                self.logger.logger.warning("相機不可用，使用模擬圖像進行檢測")
            
            # 為 Anomalib 預先生成標註影像路徑
            output_path = None
            if inference_type_enum == InferenceType.ANOMALIB:
                output_path = self.result_handler.get_annotated_path(
                    status="TEMP", detector=inference_type, product=product, area=area
                )
            
            # 執行推理
            result = self.inference_engine.infer(frame, product, area, inference_type_enum, output_path)
            if "status" in result and result["status"] == "ERROR":
                self.logger.logger.error(f"推理失敗: {result.get('error')}")
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
                    "heatmap_path": "",
                    "cropped_paths": []
                }
            
            status = result["status"]
            
            # 處理 Anomalib 熱圖路徑
            if inference_type_enum == InferenceType.ANOMALIB and output_path:
                correct_output_path = self.result_handler.get_annotated_path(
                    status=status,
                    detector=inference_type,
                    product=product,
                    area=area,
                    anomaly_score=result.get("anomaly_score")
                )
                if output_path != correct_output_path and os.path.exists(output_path):
                    os.makedirs(os.path.dirname(correct_output_path), exist_ok=True)
                    try:
                        shutil.move(output_path, correct_output_path)
                        result["output_path"] = correct_output_path
                        self.logger.logger.info(f"熱圖檔案已從 {output_path} 移動到 {correct_output_path}")
                    except Exception as e:
                        self.logger.logger.error(f"移動熱圖檔案失敗: {str(e)}")
                    try:
                        old_dir = os.path.dirname(output_path)
                        if not os.listdir(old_dir):
                            os.rmdir(old_dir)
                    except Exception as cleanup_err:
                        self.logger.logger.warning(f"清理舊目錄失敗: {cleanup_err}")
                else:
                    self.logger.logger.info(f"無需移動熱圖檔案，路徑已正確: {correct_output_path}")
                    result["output_path"] = correct_output_path
            
            # 保存結果
            if inference_type_enum == InferenceType.ANOMALIB:
                save_result = self.result_handler.save_results(
                    frame=frame,
                    detections=[],
                    status=status,
                    detector=inference_type,
                    missing_items=[],
                    processed_image=result["processed_image"],
                    anomaly_score=result.get("anomaly_score"),
                    heatmap_path=result.get("output_path"),
                    product=product,
                    area=area,
                    ckpt_path=result.get("ckpt_path")
                )
                self.logger.log_anomaly(status, result.get("anomaly_score", 0.0))
            else:  # YOLO
                save_result = self.result_handler.save_results(
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
            
            if save_result.get("status") == "ERROR":
                self.logger.logger.error(f"保存結果失敗: {save_result.get('error_message')}")
                return {
                    "status": "ERROR",
                    "error": save_result.get("error_message"),
                    "product": product,
                    "area": area,
                    "inference_type": inference_type,
                    "ckpt_path": result.get("ckpt_path", self.config.weights),
                    "anomaly_score": result.get("anomaly_score", ""),
                    "detections": result.get("detections", []),
                    "missing_items": result.get("missing_items", []),
                    "original_image_path": "",
                    "preprocessed_image_path": "",
                    "heatmap_path": "",
                    "cropped_paths": []
                }
            
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
                "original_image_path": save_result.get("original_path", ""),
                "preprocessed_image_path": save_result.get("preprocessed_path", ""),
                "heatmap_path": save_result.get("heatmap_path", ""),
                "cropped_paths": save_result.get("cropped_paths", [])
            }
        
        except Exception as e:
            self.logger.logger.error(f"檢測失敗: {str(e)}")
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
                "heatmap_path": "",
                "cropped_paths": []
            }
    def run(self):
        self.logger.logger.info("檢測系統已啟動，等待輸入訊號...")

        models_base = os.path.join(os.path.dirname(__file__), "models")
        available_products = [
            d for d in os.listdir(models_base)
            if os.path.isdir(os.path.join(models_base, d))
        ]
        if not available_products:
            self.logger.logger.error("models 資料夾中未找到任何機種")
            return

        print(f"可用機種: {', '.join(available_products)}")
        while True:
            product = input("請輸入要檢測的機種 (或輸入 'quit' 退出): ").strip()
            if product.lower() == "quit":
                self.logger.logger.info("退出檢測系統")
                self.shutdown()
                return
            if product not in available_products:
                print(f"無效的機種: {product}，請選擇: {', '.join(available_products)}")
                continue
            break
        
        product_dir = os.path.join(models_base, product)
        available_areas = [
            d for d in os.listdir(product_dir)
            if os.path.isdir(os.path.join(product_dir, d))
        ]
        
        while True:
            print(f"可用區域: {', '.join(available_areas)}")
            cmd = input("請輸入檢測指令 (格式: area,inference_type 或 quit): ").strip()
            if cmd.lower() == "quit":
                self.logger.logger.info("退出檢測系統")
                self.shutdown()
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

                config_path = os.path.join(
                    models_base, product, area, inference_type.lower(), "config.yaml"
                )
                if not os.path.exists(config_path):
                    print(
                        f"選擇的模型不存在: {config_path}"
                    )
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
