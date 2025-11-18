import os
from typing import Any, Dict, Optional, Tuple, Type, cast
import cv2
import numpy as np
import warnings
# imgaug (pulled in by anomalib) still expects np.sctypes, which was removed in NumPy 2.x.
# Shim the attr early so anomalib/imgaug can import when the exe bundles NumPy 2.
if not hasattr(np, "sctypes"):  # pragma: no cover - exercised only on NumPy>=2
    _float_types = [t for t in (getattr(np, name, None) for name in (
        "half",
        "single",
        "double",
        "longdouble",
    )) if t is not None]
    _complex_types = [t for t in (getattr(np, name, None) for name in (
        "csingle",
        "cdouble",
        "clongdouble",
    )) if t is not None]
    _int_types = [t for t in (getattr(np, name, None) for name in (
        "byte",
        "short",
        "intc",
        "int_",
        "longlong",
    )) if t is not None]
    _uint_types = [t for t in (getattr(np, name, None) for name in (
        "ubyte",
        "ushort",
        "uintc",
        "uint",
        "ulonglong",
    )) if t is not None]
    _other_types = [
        t
        for t in (
            getattr(np, name, None)
            for name in (
                "bool_",
                "bytes_",
                "str_",
                "void",
                "datetime64",
                "timedelta64",
            )
        )
        if t is not None
    ]
    np.sctypes = {  # type: ignore[attr-defined]
        "int": _int_types,
        "uint": _uint_types,
        "float": _float_types,
        "complex": _complex_types,
        "others": _other_types,
    }
from jsonargparse import Namespace
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
# ---- Anomalib / jsonargparse 依賴檢查與強制載入 ----
import jsonargparse

logger = logging.getLogger("anomalib")

try:
    # 強制載入 signatures extra，讓 PyInstaller 確定收這個子模組
    import jsonargparse.signatures  # type: ignore  # noqa: F401

    logger.info(
        "jsonargparse loaded OK (version=%s, file=%s)",
        getattr(jsonargparse, "__version__", "N/A"),
        getattr(jsonargparse, "__file__", "N/A"),
    )
except Exception as e:  # pragma: no cover - 只用來在 exe 裡除錯
    logger.error(
        "Failed to import jsonargparse.signatures: %s (version=%s, file=%s)",
        e,
        getattr(jsonargparse, "__version__", "N/A"),
        getattr(jsonargparse, "__file__", "N/A"),
    )
# ---- end jsonargparse check ----


logging.getLogger("anomalib").setLevel(logging.INFO)
logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="anomalib")
warnings.filterwarnings(
    "ignore", message=".*F1Score class exists for backwards compatibility.*"
)
warnings.filterwarnings("ignore", message=".*ckpt_path is not provided.*")

if sys.platform == "win32":
    pathlib.PosixPath = cast(Type[pathlib.Path], pathlib.WindowsPath)  # type: ignore[misc,assignment]
if hasattr(sys.stdout, "reconfigure"):
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")

ModelKey = Tuple[str, str]

_engine: Optional[Engine] = None
_models: Dict[ModelKey, Dict[str, Any]] = {}
_output_dir: Optional[str] = None
_transform: Any | None = None
_inference_lock = threading.Lock()
_initialization_lock = threading.Lock()
_is_initialized: Dict[ModelKey, bool] = {}
_current_product: Optional[str] = None
_current_area: Optional[str] = None
_current_ckpt_path: Optional[str] = None
_thresholds: Dict[ModelKey, float] = {}  # Stores thresholds per product/area


# anomalib_lightning_inference.py, initialize 方法中
def initialize(
    config: Optional[Dict[str, Any]] = None,
    product: Optional[str] = None,
    area: Optional[str] = None,
) -> None:
    global _engine, _models, _output_dir, _transform, _is_initialized
    global _current_product, _current_area, _current_ckpt_path, _thresholds

    if config is None:
        raise ValueError("config is required for initialize()")
    if product is None or area is None:
        raise ValueError("product and area are required for initialize()")

    model_key: ModelKey = (product, area)

    with _initialization_lock:
        if _is_initialized.get(model_key):
            prior_ckpt = _models[model_key]["ckpt_path"]
            message = (
                f"Models for {product},{area} already initialized; checkpoint: {prior_ckpt}. Skipping reinitialization."
            )
            logging.getLogger("anomalib").info(message)
            _current_product = product
            _current_area = area
            _current_ckpt_path = prior_ckpt
            return

        try:
            logging.getLogger("anomalib").info(
                f"Initializing Anomalib model (product: {product}, area: {area})"
            )

            # Build the minimal namespace manually to avoid Lightning CLI's optional dependency on jsonargparse[signatures].
            args = Namespace()
            args.output = str(config.get("output", "./patchcore_outputs"))
            args.data = config.get("data", {}) or {}
            args.show = bool(config.get("show", False))
            args.model = Namespace(
                **config.get("model", {"class_path": "anomalib.models.Patchcore"})
            )

            models_cfg = config.get("models") or {}
            if product not in models_cfg or area not in models_cfg.get(product, {}):
                raise ValueError(
                    f"Missing checkpoint for {product},{area}; please configure it in config.yaml"
                )
            model_config = models_cfg[product][area]
            args.ckpt_path = model_config.get("ckpt_path")
            threshold = float(model_config.get("threshold", 0.5))
            _thresholds[model_key] = threshold
            logging.getLogger("anomalib").info(
                f"Using checkpoint: {args.ckpt_path} (product: {product}, area: {area}, threshold: {threshold})"
            )

            if not args.ckpt_path:
                raise ValueError(
                    "Checkpoint path not provided; configure a valid ckpt_path in config.yaml"
                )
            if not os.path.exists(str(args.ckpt_path)):
                raise ValueError(f"Checkpoint file not found: {args.ckpt_path}")

            message = (
                f"Loading checkpoint: {args.ckpt_path}, model type: {args.model.class_path}"
            )
            logging.getLogger("anomalib").info(message)
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
                        logger=False,
                    )
                    logging.getLogger("anomalib").info("Engine initialized")

            model_class_path = args.model.class_path
            module_name, class_name = model_class_path.rsplit(".", 1)
            model_class = getattr(import_module(module_name), class_name)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _model = model_class.load_from_checkpoint(args.ckpt_path)
                if _model is None:
                    raise ValueError(f"Failed to read image: {args.ckpt_path}")
                logging.getLogger("anomalib").info(
                    f"Loaded checkpoint: {args.ckpt_path}"
                )

            _models[model_key] = {"model": _model, "ckpt_path": args.ckpt_path}
            _output_dir = args.output
            _transform = args.data.get("transform")
            os.makedirs(_output_dir, exist_ok=True)

            _is_initialized[model_key] = True
            _current_product = product
            _current_area = area
            _current_ckpt_path = str(args.ckpt_path)
            message = (
                f"Failed to read image? (??: {product}, ??: {area}, ???: {args.ckpt_path}, "
                f"??: {_thresholds[model_key]})"
            )
            logging.getLogger("anomalib").info(message)
            logging.getLogger("anomalib").info(f"Output directory: {_output_dir}")

        except Exception as exc:
            logging.getLogger("anomalib").error(
                f"Model initialization failed (product: {product}, area: {area}): {str(exc)}"
            )
            raise


def initialize_product_models(
    config: Optional[Dict[str, Any]] = None,
    product: Optional[str] = None,
) -> None:
    if config is None:
        raise ValueError("config is required for initialize_product_models()")
    if product is None:
        raise ValueError("product is required for initialize_product_models()")

    models_cfg = config.get("models") or {}
    if product not in models_cfg:
        raise ValueError(f"No model configuration found for product {product}")

    logging.getLogger("anomalib").info(
        f"Initializing Anomalib models for all areas of {product}"
    )
    for area_name in models_cfg[product]:
        initialize(config, product, area_name)


def lightning_inference(
    image_path: str,
    thread_safe: bool = True,
    enable_timing: bool = True,
    product: Optional[str] = None,
    area: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    global _engine, _models, _output_dir, _transform, _inference_lock, _is_initialized

    if product is None or area is None:
        raise ValueError("product and area are required for inference")

    model_key: ModelKey = (product, area)
    if not _is_initialized.get(model_key):
        raise RuntimeError(
            f"Failed to read image? {product},{area}????? initialize()"
        )

    if not os.path.exists(image_path):
        return {
            "image_path": image_path,
            "product": product,
            "area": area,
            "error": "Failed to read image?",
        }

    img = cv2.imread(image_path)
    if img is None:
        return {
            "image_path": image_path,
            "product": product,
            "area": area,
            "error": "Failed to read image",
        }

    if img.shape[:2] != (640, 640):
        warn_msg = f"Failed to read image {img.shape[:2]} ????? 640x640"
        logging.getLogger("anomalib").warning(warn_msg)

    timings: Dict[str, float] = {}
    start_time = time.time()

    try:
        dataset_start = time.time()
        dataset = PredictDataset(path=image_path, transform=_transform)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        if enable_timing:
            timings["dataset_creation"] = round(time.time() - dataset_start, 4)

        predict_start = time.time()

        if _engine is None:
            raise RuntimeError("Engine is not initialized")

        def _do_predict() -> tuple[list[Any], str]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_info = _models[model_key]
                outputs = _engine.predict(
                    model=model_info["model"],
                    dataloaders=[dataloader],
                    ckpt_path=model_info["ckpt_path"],
                )
                return list(outputs or []), str(model_info["ckpt_path"])

        if thread_safe:
            with _inference_lock:
                predictions, ckpt_path = _do_predict()
        else:
            predictions, ckpt_path = _do_predict()

        if not predictions:
            return {
                "image_path": image_path,
                "product": product,
                "area": area,
                "error": "Failed to read image????",
            }

        if enable_timing:
            timings["model_inference"] = round(time.time() - predict_start, 4)

        process_start = time.time()
        pred = predictions[0]
        result = _process_prediction(
            pred,
            image_path,
            product,
            area,
            output_path=output_path,
            enable_timing=enable_timing,
        )

        if enable_timing:
            timings["result_processing"] = round(time.time() - process_start, 4)
            result["timings"] = timings

        total_time = time.time() - start_time
        result["inference_time"] = round(total_time, 4)
        result["ckpt_path"] = ckpt_path
        result.setdefault("image_path", image_path)
        result.setdefault("product", product)
        result.setdefault("area", area)
        return result

    except Exception as exc:
        return {
            "image_path": image_path,
            "product": product,
            "area": area,
            "error": f"Failed to read image: {str(exc)}",
            "inference_time": round(time.time() - start_time, 4),
        }


# anomalib_lightning_inference.py, _process_prediction 方法中
def _process_prediction(
    pred: Dict[str, Any],
    image_path: str,
    product: str,
    area: str,
    output_path: Optional[str] = None,
    enable_timing: bool = False,
) -> Dict[str, Any]:
    try:
        model_key: ModelKey = (product, area)
        threshold = _thresholds.get(model_key, 0.5)
        anomaly_score_raw = pred.get("pred_scores", 0.0)
        if anomaly_score_raw is not None and hasattr(anomaly_score_raw, "item"):
            anomaly_score = float(anomaly_score_raw.item())
        else:
            anomaly_score = float(anomaly_score_raw or 0.0)

        anomaly_map = pred.get("anomaly_maps")
        if anomaly_map is None:
            logging.getLogger("anomalib").error(
                f"Anomaly heatmap missing; keys: {list(pred.keys())}"
            )
            return {
                "image_path": image_path,
                "error": "Anomaly heatmap missing",
                "product": product,
                "area": area,
            }

        original_image = cv2.imread(image_path)
        if original_image is None:
            logging.getLogger("anomalib").error(f"Failed to read image: {image_path}")
            return {
                "image_path": image_path,
                "error": "Failed to read image",
                "product": product,
                "area": area,
            }

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        heatmap_3d = anomaly_map[0].cpu().numpy()
        heatmap_3d = np.squeeze(heatmap_3d)
        if heatmap_3d.ndim == 3 and heatmap_3d.shape[-1] > 1:
            heatmap_3d = heatmap_3d.mean(axis=-1)

        heatmap_3d = cv2.resize(
            heatmap_3d, (original_image.shape[1], original_image.shape[0])
        )

        dynamic_threshold = threshold
        if heatmap_3d.max() - heatmap_3d.min() == 0:
            normalized_heatmap = np.zeros_like(heatmap_3d, dtype=np.float32)
            threshold_mask = np.zeros_like(heatmap_3d, dtype=bool)
            message = (
                f"Heatmap has no variation (min={heatmap_3d.min()}, max={heatmap_3d.max()})"
            )
            logging.getLogger("anomalib").warning(message)
        else:
            normalized_heatmap = (heatmap_3d - heatmap_3d.min()) / np.ptp(heatmap_3d)
            dynamic_threshold = 0.95 if anomaly_score < 0.1 else threshold
            threshold_mask = normalized_heatmap > dynamic_threshold
            message = (
                "Score-based threshold adjustment, dynamic threshold: "
                f"{dynamic_threshold:.4f} (base: {threshold}, "
                f"anomaly score: {anomaly_score:.4f})"
            )
            logging.getLogger("anomalib").info(message)

        colored_heatmap = cv2.applyColorMap(
            (normalized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
        overlay_image = original_image.copy()

        if np.any(threshold_mask):
            overlay_image[threshold_mask] = cv2.addWeighted(
                colored_heatmap[threshold_mask].reshape(-1, 3),
                0.4,
                original_image[threshold_mask].reshape(-1, 3),
                0.6,
                0,
            ).reshape(-1, 3)
            anomaly_pixel_count = int(np.sum(threshold_mask))
            total_pixels = int(threshold_mask.size)
            anomaly_ratio = anomaly_pixel_count / total_pixels
            message = (
                "Pixel-level threshold filtering (method: score_based, threshold: "
                f"{dynamic_threshold}), anomaly pixels: {anomaly_pixel_count}/{total_pixels} "
                f"({anomaly_ratio:.2%})"
            )
            logging.getLogger("anomalib").info(message)
        else:
            logging.getLogger("anomalib").info(
                "No pixel exceeds the threshold (method: score_based); displaying original image"
            )
            anomaly_pixel_count = 0
            total_pixels = int(threshold_mask.size)
            anomaly_ratio = 0.0

        is_anomaly = anomaly_score > threshold
        status_str = "anomaly" if is_anomaly else "normal"

        if (
            output_path
            and os.path.dirname(output_path)
            != "Result/YYYYMMDD/TEMP/annotated/anomalib"
        ):
            heatmap_path = output_path
        else:
            current_date = datetime.now().strftime("%Y%m%d")
            time_stamp = datetime.now().strftime("%H%M%S")
            heatmap_path = os.path.join(
                "Result",
                current_date,
                status_str,
                "annotated",
                "anomalib",
                f"anomalib_{product}_{area}_{time_stamp}_{anomaly_score:.4f}.jpg",
            )

        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, overlay_image_bgr)
        logging.getLogger("anomalib").info(f"Saved annotated heatmap to: {heatmap_path}")

        return {
            "image_path": image_path,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "anomaly_pixel_count": anomaly_pixel_count,
            "total_pixels": total_pixels,
            "anomaly_pixel_ratio": anomaly_ratio,
            "heatmap_path": heatmap_path,
            "product": product,
            "area": area,
            "ckpt_path": _models.get(model_key, {}).get("ckpt_path", ""),
        }

    except Exception as exc:
        logging.getLogger("anomalib").error(
            f"Processing prediction failed: {str(exc)}"
        )
        return {
            "image_path": image_path,
            "error": f"Processing prediction failed: {str(exc)}",
            "product": product,
            "area": area,
        }


def set_threshold(product: str, area: str, threshold: float) -> None:
    """動態設定特定產品區域的閾值"""
    global _thresholds
    model_key = (product, area)
    _thresholds[model_key] = threshold
    logging.getLogger("anomalib").info(
        f"已設定 {product},{area} 的閾值為: {threshold}")


def get_threshold(product: str, area: str) -> float:
    """取得特定產品區域的閾值"""
    model_key = (product, area)
    return _thresholds.get(model_key, 0.5)


def get_status():
    return {
        "initialized": _is_initialized,
        "model_loaded": {k: v["model"] is not None for k, v in _models.items()},
        "engine_ready": _engine is not None,
        "output_dir": _output_dir,
        "current_product": _current_product,
        "current_area": _current_area,
        "current_ckpt_path": _current_ckpt_path,
        "thresholds": _thresholds,  # 新增：顯示所有閾值設定
    }
