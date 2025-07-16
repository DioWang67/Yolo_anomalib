import os
import cv2
import numpy as np
import warnings
from jsonargparse import ActionConfigFile, Namespace
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import AnomalyModule
from importlib import import_module
import pathlib
import sys
import threading
import time
import logging
from datetime import datetime
from matplotlib import colormaps

logging.getLogger("anomalib").setLevel(logging.INFO)
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")
warnings.filterwarnings("ignore", message=".*F1Score class exists for backwards compatibility.*")
warnings.filterwarnings("ignore", message=".*ckpt_path is not provided.*")

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath
sys.stdout.reconfigure(encoding='utf-8')

_engine = None
_models = {}
_output_dir = None
_transform = None
_inference_lock = threading.Lock()
_initialization_lock = threading.Lock()
_is_initialized = {}
_current_product = None
_current_area = None
_current_ckpt_path = None

def initialize(config: dict = None, product: str = None, area: str = None) -> None:
    global _engine, _models, _output_dir, _transform, _is_initialized, _current_product, _current_area, _current_ckpt_path

    with _initialization_lock:
        model_key = (product, area)
        if model_key in _is_initialized and _is_initialized[model_key]:
            logging.getLogger("anomalib").info(
                f"模型已針對 {product},{area} 初始化，檢查點: {_models[model_key]['ckpt_path']}，跳過重複初始化"
            )
            _current_product = product
            _current_area = area
            _current_ckpt_path = _models[model_key]['ckpt_path']
            return

        try:
            logging.getLogger("anomalib").info(f"開始初始化 Anomalib 模型 (產品: {product}, 區域: {area})")
            parser = LightningArgumentParser(description="Inference on Anomaly models in Lightning format.")
            parser.add_lightning_class_args(AnomalyModule, "model", subclass_mode=True)
            parser.add_lightning_class_args(Callback, "--callbacks", subclass_mode=True, required=False)
            parser.add_argument("--ckpt_path", type=str, required=False, help="Path to model weights.")
            parser.add_argument("--output", type=str, required=False, default="./patchcore_outputs")
            parser.add_argument("--show", action="store_true", required=False)
            parser.add_class_arguments(PredictDataset, "--data", instantiate=False)

            args = Namespace()
            args.output = config.get('output', './patchcore_outputs')
            args.data = config.get('data', {})
            args.show = config.get('show', False)
            args.model = Namespace(**config.get('model', {'class_path': 'anomalib.models.Patchcore'}))

            if product and area and config.get('models'):
                models = config.get('models', {})
                if product not in models or area not in models[product]:
                    logging.getLogger("anomalib").warning(f"未找到 {product},{area} 的檢查點，使用預設檢查點")
                    args.ckpt_path = config.get('ckpt_path', '')
                else:
                    args.ckpt_path = models[product][area].get('ckpt_path')
                    logging.getLogger("anomalib").info(f"選擇檢查點: {args.ckpt_path} (產品: {product}, 區域: {area})")
            else:
                args.ckpt_path = config.get('ckpt_path', '')
                logging.getLogger("anomalib").info(f"使用預設檢查點: {args.ckpt_path}")

            if not args.ckpt_path:
                raise ValueError("檢查點路徑未提供")
            if not os.path.exists(args.ckpt_path):
                raise ValueError(f"檢查點檔案不存在: {args.ckpt_path}")

            logging.getLogger("anomalib").info(f"正在載入檢查點: {args.ckpt_path}, 模型類型: {args.model.class_path}")

            current_date = datetime.now().strftime("%Y%m%d")
            args.output = args.output.replace("YYYYMMDD", current_date)
            
            if _engine is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _engine = Engine(
                        default_root_dir=args.output,
                        callbacks=None,
                        devices="auto",
                        enable_progress_bar=False,
                        logger=False
                    )
                    logging.getLogger("anomalib").info("Engine 初始化成功")

            model_class_path = args.model.class_path
            module_name, class_name = model_class_path.rsplit(".", 1)
            model_class = getattr(import_module(module_name), class_name)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model = model_class.load_from_checkpoint(args.ckpt_path)
                logging.getLogger("anomalib").info(f"成功載入檢查點: {args.ckpt_path}")

            if _model is None:
                raise ValueError(f"模型初始化失敗: {args.ckpt_path}, 模型物件為 None")

            _models[model_key] = {'model': _model, 'ckpt_path': args.ckpt_path}
            _output_dir = args.output
            _transform = args.data.get("transform")
            os.makedirs(_output_dir, exist_ok=True)

            _is_initialized[model_key] = True
            _current_product = product
            _current_area = area
            _current_ckpt_path = args.ckpt_path
            logging.getLogger("anomalib").info(f"模型初始化完成 (產品: {product}, 區域: {area}, 檢查點: {args.ckpt_path})")
            logging.getLogger("anomalib").info(f"輸出目錄: {_output_dir}")
            
        except Exception as e:
            logging.getLogger("anomalib").error(f"模型初始化失敗 (產品: {product}, 區域: {area}): {str(e)}")
            raise

def initialize_product_models(config: dict = None, product: str = None) -> None:
    if not product or not config.get('models') or product not in config['models']:
        raise ValueError(f"無效的機種: {product} 或未定義模型")
    
    logging.getLogger("anomalib").info(f"正在初始化機種 {product} 的所有區域模型")
    for area in config['models'][product]:
        initialize(config, product, area)

def lightning_inference(image_path: str, thread_safe: bool = True, enable_timing: bool = True, product: str = None, area: str = None) -> dict:
    global _engine, _models, _output_dir, _transform, _inference_lock

    model_key = (product, area)
    if model_key not in _is_initialized or not _is_initialized[model_key]:
        raise RuntimeError(f"模型未針對 {product},{area} 初始化，請先呼叫 initialize()")

    if not os.path.exists(image_path):
        return {"image_path": image_path, "error": "圖片檔案不存在"}

    img = cv2.imread(image_path)
    if img.shape[:2] != (640, 640):
        logging.getLogger("anomalib").warning(f"輸入圖像尺寸 {img.shape[:2]} 不符合預期 640x640")

    timings = {}
    start_time = time.time()
    
    try:
        dataset_start = time.time()
        dataset = PredictDataset(path=image_path, transform=_transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=16, 
            num_workers=0,
            pin_memory=False,
            persistent_workers=False
        )
        if enable_timing:
            timings["dataset_creation"] = round(time.time() - dataset_start, 4)

        predict_start = time.time()
        if thread_safe:
            with _inference_lock:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    predictions = _engine.predict(model=_models[model_key]['model'], dataloaders=[dataloader])
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = _engine.predict(model=_models[model_key]['model'], dataloaders=[dataloader])
        
        if enable_timing:
            timings["model_inference"] = round(time.time() - predict_start, 4)

        process_start = time.time()
        for pred in predictions:
            result = _process_prediction(pred, image_path, product, area, enable_timing=enable_timing)
            if enable_timing:
                timings["result_processing"] = round(time.time() - process_start, 4)
                result["timings"] = timings
            
            total_time = time.time() - start_time
            result["inference_time"] = round(total_time, 4)
            result["ckpt_path"] = _models[model_key]['ckpt_path']
            return result

    except Exception as e:
        return {
            "image_path": image_path, 
            "error": f"圖片處理失敗: {str(e)}",
            "inference_time": round(time.time() - start_time, 4)
        }

def _process_prediction(pred, image_path: str, product: str, area: str, enable_timing: bool = False):
    try:
        anomaly_score = pred.get("pred_scores", 0.0)
        if anomaly_score is not None:
            anomaly_score = anomaly_score.item()
        anomaly_map = pred.get("anomaly_maps", None)
        if anomaly_map is None:
            logging.getLogger("anomalib").error(f"未找到異常熱圖，預測內容: {pred.keys()}")
            return {"image_path": image_path, "error": "未找到異常熱圖", "product": product, "area": area}

        original_image = cv2.imread(image_path)
        if original_image is None:
            logging.getLogger("anomalib").error(f"無法讀取圖片: {image_path}")
            return {"image_path": image_path, "error": "無法讀取圖片", "product": product, "area": area}

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        heatmap_3d = anomaly_map[0].cpu().numpy()
        heatmap_3d = np.squeeze(heatmap_3d)
        if heatmap_3d.ndim == 3 and heatmap_3d.shape[-1] > 1:
            heatmap_3d = heatmap_3d.mean(axis=-1)

        heatmap_3d = cv2.resize(heatmap_3d, (original_image.shape[1], original_image.shape[0]))
        min_val, max_val = heatmap_3d.min(), heatmap_3d.max()
        if max_val - min_val == 0:
            normalized_heatmap = np.zeros_like(heatmap_3d, dtype=np.float32)
            logging.getLogger("anomalib").warning(f"熱圖正規化失敗，min={min_val}, max={max_val}")
        else:
            normalized_heatmap = (heatmap_3d - min_val) / (max_val - min_val + 1e-8)

        threshold = 0.5  # 從 config 中讀取閾值
        mask = (normalized_heatmap >= threshold).astype(np.float32)
        filtered_heatmap = normalized_heatmap * mask

        colormap = colormaps.get_cmap('jet')
        colored_heatmap = (colormap(filtered_heatmap)[:, :, :3] * 255).astype(np.uint8)
        overlay_image = cv2.addWeighted(original_image, 0.8, colored_heatmap, 0.4, 0)

        current_date = datetime.now().strftime("%Y%m%d")
        heatmap_path = os.path.join("Result", current_date, "annotated", "anomalib", f"anomalib_{product}_{area}_{datetime.now().strftime('%H%M%S')}_{anomaly_score:.4f}.jpg")
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, overlay_image_bgr)
        logging.getLogger("anomalib").info(f"熱圖儲存至: {heatmap_path}")

        return {
            "image_path": image_path,
            "anomaly_score": anomaly_score,
            "heatmap_path": heatmap_path,
            "product": product,
            "area": area,
            "ckpt_path": _models.get((product, area), {}).get("ckpt_path", "")
        }

    except Exception as e:
        logging.getLogger("anomalib").error(f"處理預測結果失敗: {str(e)}")
        return {"image_path": image_path, "error": f"處理預測結果失敗: {str(e)}", "product": product, "area": area}

def get_status():
    return {
        "initialized": _is_initialized,
        "model_loaded": {k: v['model'] is not None for k, v in _models.items()},
        "engine_ready": _engine is not None,
        "output_dir": _output_dir,
        "current_product": _current_product,
        "current_area": _current_area,
        "current_ckpt_path": _current_ckpt_path
    }