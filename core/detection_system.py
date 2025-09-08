import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import copy
import yaml
import numpy as np

from core.result_handler import ResultHandler
from core.config import DetectionConfig
from core.logger import DetectionLogger
from core.inference_engine import InferenceEngine, InferenceType
from camera.camera_controller import CameraController
from core.led_qc_enhanced import LEDQCEnhanced


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DetectionSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.logger = DetectionLogger()
        self.config = self.load_config(config_path)
        self.camera = None
        self.result_handler = ResultHandler(self.config, base_dir=self.config.output_dir, logger=self.logger)
        self.inference_engine: InferenceEngine | None = None
        self.current_inference_type: InferenceType | None = None
        self.model_cache: OrderedDict[tuple[str, str], dict[str, tuple[InferenceEngine, DetectionConfig]]] = OrderedDict()
        self.max_cache_size = self.config.max_cache_size
        self.color_checker: LEDQCEnhanced | None = None
        self._color_model_path: str | None = None
        self._color_checker_mode: str | None = None
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
        if self.result_handler:
            self.result_handler.close()
            self.result_handler = None

    def initialize_camera(self) -> None:
        self.logger.logger.info("初始化相機中...")
        try:
            self.camera = CameraController(self.config)
            self.camera.initialize()
            self.logger.logger.info("相機已就緒")
        except Exception as e:
            self.logger.logger.error(f"相機初始化失敗: {str(e)}")
            self.logger.logger.warning("相機不可用，將使用模擬影像進行測試")
            self.camera = None

    def initialize_inference_engine(self) -> None:
        self.logger.logger.info("初始化推理引擎...")
        if not self.inference_engine or not self.inference_engine.initialize():
            self.logger.logger.error("推理引擎初始化失敗")
            raise RuntimeError("推理引擎初始化失敗")
        self.logger.logger.info("推理引擎已就緒")

    def initialize_product_models(self, product: str) -> None:
        if self.config.enable_yolo:
            self.logger.logger.info(f"YOLO 模型已為 {product} 準備")
        if self.config.enable_anomalib:
            try:
                from core.anomalib_lightning_inference import initialize_product_models as _anoma_init
                _anoma_init(self.config.anomalib_config, product)
                self.logger.logger.info(f"機種 {product} 的 Anomalib 模型已初始化完成")
            except Exception as e:
                self.logger.logger.error(f"機種 {product} 的 Anomalib 模型初始化失敗: {str(e)}")
                raise

    def load_model_configs(self, product: str, area: str, inference_type: str) -> None:
        """根據 product/area/type 載入模型設定並初始化推理引擎。"""
        cache_key = (product, area)

        if cache_key in self.model_cache and inference_type in self.model_cache[cache_key]:
            self.logger.logger.info(
                f"使用快取模型: 機種 {product}, 區域 {area}, 模型 {inference_type}"
            )
            self.inference_engine, cached_config = self.model_cache[cache_key][inference_type]
            self.model_cache.move_to_end(cache_key)
            self.config.__dict__.update(copy.deepcopy(cached_config.__dict__))
            self.current_inference_type = InferenceType.from_string(inference_type)
            return

        self.logger.logger.info(
            f"切換產線: {product}, 區域 {area}，模型 {inference_type}"
        )

        model_config_path = os.path.join("models", product, area, inference_type, "config.yaml")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"找不到模型設定檔: {model_config_path}")

        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # 覆寫可變更的設定（保持簡單及一致）
        self.config.device = cfg.get("device", self.config.device)
        self.config.conf_thres = cfg.get("conf_thres", self.config.conf_thres)
        self.config.iou_thres = cfg.get("iou_thres", self.config.iou_thres)
        self.config.imgsz = tuple(cfg.get("imgsz", self.config.imgsz))
        self.config.enable_yolo = cfg.get("enable_yolo", self.config.enable_yolo)
        self.config.enable_anomalib = cfg.get("enable_anomalib", self.config.enable_anomalib)
        self.config.expected_items = cfg.get("expected_items", self.config.expected_items)
        self.config.position_config = cfg.get("position_config", self.config.position_config)
        self.config.output_dir = cfg.get("output_dir", self.config.output_dir)
        self.config.anomalib_config = cfg.get("anomalib_config")
        self.config.weights = cfg.get("weights", "")
        self.config.enable_color_check = cfg.get("enable_color_check", False)
        color_model_path = cfg.get("color_model_path", None)
        if color_model_path and not os.path.isabs(color_model_path):
            color_model_path = os.path.join(PROJECT_ROOT, color_model_path)
        self.config.color_model_path = color_model_path

        self.inference_engine = InferenceEngine(self.config)
        self.initialize_inference_engine()

        if inference_type == "anomalib":
            try:
                self.initialize_product_models(product)
            except Exception as e:
                self.logger.logger.error(f"重新初始化 {product} 模型失敗: {str(e)}")
                raise

        self.current_inference_type = InferenceType.from_string(inference_type)

        # 快取當前設定與引擎
        if cache_key not in self.model_cache:
            self.model_cache[cache_key] = {}
        self.model_cache[cache_key][inference_type] = (self.inference_engine, copy.deepcopy(self.config))
        self.model_cache.move_to_end(cache_key)
        if len(self.model_cache) > self.max_cache_size:
            old_key, engines = self.model_cache.popitem(last=False)
            for eng, _ in engines.values():
                try:
                    eng.shutdown()
                except Exception:
                    pass
            self.logger.logger.info(f"已釋放快取模型: 機種 {old_key[0]}, 區域 {old_key[1]}")

    def detect(self, product: str, area: str, inference_type: str, frame: np.ndarray | None = None) -> dict:
        try:
            self.load_model_configs(product, area, inference_type)
            self.logger.logger.info(f"開始檢測 - 機種: {product}, 區域: {area}, 類型: {inference_type}")
            inference_type_enum = InferenceType.from_string(inference_type)

            # 顏色檢測器（使用 LEDQCEnhanced）
            if self.config.enable_color_check and self.config.color_model_path:
                if self.color_checker is None or self._color_model_path != self.config.color_model_path:
                    try:
                        self.color_checker = LEDQCEnhanced.from_json(self.config.color_model_path)
                        self._color_checker_mode = "advanced"
                        self._color_model_path = self.config.color_model_path
                        self.logger.logger.info("顏色檢測器已載入 (LEDQCEnhanced)")
                    except Exception as e:
                        self.logger.logger.error(f"顏色檢測初始化失敗: {str(e)}")
                        self.color_checker = None
            else:
                self.color_checker = None

            # 嘗試使用外部給定影像，否則使用相機
            if frame is None:
                if self.camera:
                    frame = self.camera.capture_frame()
                    if frame is None:
                        self.logger.logger.error("無法從相機獲取影像")
                        frame = np.zeros((640, 640, 3), dtype=np.uint8)
                        self.logger.logger.warning("使用模擬影像進行檢測")
                else:
                    frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    self.logger.logger.warning("相機不可用，使用模擬影像進行檢測")

            # Anomalib 臨時標註輸出路徑
            output_path = None
            if inference_type_enum == InferenceType.ANOMALIB:
                output_path = self.result_handler.get_annotated_path(
                    status="TEMP", detector=inference_type, product=product, area=area
                )

            # 推理
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
                    "annotated_path": "",
                    "heatmap_path": "",
                    "cropped_paths": [],
                    "color_check": None,
                }

            status = result["status"]

            # 顏色檢測（基於 YOLO 偵測框）
            color_result = None
            if self.color_checker is not None:
                try:
                    detections = result.get("detections", [])
                    per_items = []
                    all_ok = True
                    if detections:
                        proc = result.get("processed_image", frame)
                        for idx, det in enumerate(detections):
                            x1, y1, x2, y2 = det["bbox"]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(proc.shape[1], x2), min(proc.shape[0], y2)
                            roi = proc[y1:y2, x1:x2]
                            allowed = [det.get("class")] if det.get("class") else None
                            c_res = self.color_checker.check(roi, allowed_colors=allowed)
                            item = {
                                "index": idx,
                                "class": det.get("class"),
                                "bbox": det.get("bbox"),
                                "best_color": c_res.best_color,
                                "diff": c_res.diff,
                                "threshold": c_res.threshold,
                                "is_ok": c_res.is_ok,
                            }
                            per_items.append(item)
                            if not c_res.is_ok:
                                all_ok = False
                            state = "通過" if c_res.is_ok else "不通過"
                            self.logger.logger.info(
                                f"顏色檢測{state} (idx={idx}, 類別:{det.get('class')}, 預測:{c_res.best_color}, diff={c_res.diff:.2f}, thr={c_res.threshold:.2f})"
                            )
                    else:
                        # 無偵測，以整張圖估色
                        c_res = self.color_checker.check(frame)
                        per_items = [{
                            "index": -1,
                            "class": None,
                            "bbox": None,
                            "best_color": c_res.best_color,
                            "diff": c_res.diff,
                            "threshold": c_res.threshold,
                            "is_ok": c_res.is_ok,
                        }]
                        all_ok = c_res.is_ok

                    diff_str = ";".join([f"{it.get('diff', 0):.2f}" for it in per_items])
                    color_result = {"is_ok": all_ok, "items": per_items, "diff": diff_str}
                except Exception as e:
                    self.logger.logger.error(f"顏色檢測失敗: {str(e)}")
                    color_result = {"is_ok": False, "items": [], "error": str(e)}

            # 調整 Anomalib 輸出路徑（將 TEMP 移到 PASS/FAIL）
            if inference_type_enum == InferenceType.ANOMALIB and output_path:
                correct_output_path = self.result_handler.get_annotated_path(
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
                        self.logger.logger.info(f"輸出檔已由 {output_path} 移至 {correct_output_path}")
                    except Exception as e:
                        self.logger.logger.error(f"移動輸出檔案失敗: {str(e)}")
                    try:
                        old_dir = os.path.dirname(output_path)
                        if os.path.isdir(old_dir) and not os.listdir(old_dir):
                            os.rmdir(old_dir)
                    except Exception as cleanup_err:
                        self.logger.logger.warning(f"清理暫存目錄失敗: {cleanup_err}")
                else:
                    self.logger.logger.info(f"輸出路徑已正確: {correct_output_path}")
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
                    ckpt_path=result.get("ckpt_path"),
                    color_result=color_result,
                )
                # 立即寫入 Excel
                try:
                    self.result_handler.flush()
                except Exception as _e:
                    self.logger.logger.warning(f"Excel flush 失敗: {_e}")
                self.logger.log_anomaly(status, result.get("anomaly_score", 0.0))
            else:  # YOLO
                save_result = self.result_handler.save_results(
                    frame=frame,
                    detections=result.get("detections", []),
                    status=status,
                    detector=inference_type,
                    missing_items=result.get("missing_items", []),
                    processed_image=result.get("processed_image", frame),
                    anomaly_score=None,
                    heatmap_path=None,
                    product=product,
                    area=area,
                    ckpt_path=None,
                    color_result=color_result,
                )
                # 立即寫入 Excel
                try:
                    self.result_handler.flush()
                except Exception as _e:
                    self.logger.logger.warning(f"Excel flush 失敗: {_e}")
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
                "original_image_path": save_result.get("original_path", ""),
                "preprocessed_image_path": save_result.get("preprocessed_path", ""),
                "annotated_path": save_result.get("annotated_path", ""),
                "heatmap_path": save_result.get("heatmap_path", ""),
                "cropped_paths": save_result.get("cropped_paths", []),
                "color_check": color_result,
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
                "annotated_path": "",
                "heatmap_path": "",
                "cropped_paths": [],
                "color_check": None,
            }
