# Import order in this module is a validated native-loader contract across the
# supported Windows Conda environment; do not let an auto-sort move project
# modules across the Anomalib/Torch boundary.
# ruff: noqa: I001
"""Anomalib 2.x inference for the station.

Models arrive as packages written by the training project. Their
``anomalib_config.compatibility`` block is a stamp written by
``Yolo11_auto_train/src/picture_tool/train/anomalib_compat.py``: the runtime that
produced the checkpoint, the checkpoint's sha256, and the raw score of a
closed-form probe image. A model is loaded only when the stamp matches this
installation, the file is the one that was stamped, and the probe still scores
the same. A checkpoint from another Anomalib, a changed resize or a silently
different default pre-processor would otherwise load and judge parts with a
different scale. The repositories never import each other, so the probe formula
here is a deliberate copy; every parameter comes from the stamp.
"""

import hashlib
import logging
import os
import sys
import threading
import time
from importlib import import_module, metadata
from typing import Any, cast

import cv2
import numpy as np

from core.services.results.path_manager import ResultPathManager
from core.station_data import load_station_data_paths

import torch  # noqa: E402
from anomalib.data.utils import read_image  # noqa: E402

logger = logging.getLogger("anomalib")
logging.getLogger("anomalib").setLevel(logging.INFO)
logging.getLogger("lightning").setLevel(logging.ERROR)

if hasattr(sys.stdout, "reconfigure"):
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")

# Limit PyTorch CPU threads to prevent UI starvation (e.g. combo box stutters) when running locally
torch.set_num_threads(int(max(1, (os.cpu_count() or 2) // 2)))

SUPPORTED_ARTIFACT_SCHEMA = 2

ModelKey = tuple[str, str]


class AnomalibModelIncompatible(ValueError):
    """The packaged Anomalib model cannot be trusted with this installation."""


_models: dict[ModelKey, dict[str, Any]] = {}
_output_dir: str | None = None
_inference_lock = threading.Lock()
_initialization_lock = threading.Lock()
_is_initialized: dict[ModelKey, bool] = {}
_current_product: str | None = None
_current_area: str | None = None
_current_ckpt_path: str | None = None
_thresholds: dict[ModelKey, float] = {}  # Stores thresholds per product/area


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_image(probe: dict[str, Any]) -> np.ndarray:
    """Rebuild the training project's probe image from the stamp's parameters."""
    size = int(probe["size"])
    rows = np.arange(size, dtype=np.float64)[None, :, None]
    columns = np.arange(size, dtype=np.float64)[None, None, :]
    channels = np.arange(3, dtype=np.float64)[:, None, None]
    values = (
        float(probe["base"])
        + float(probe["amplitude"])
        * np.sin(float(probe["row_frequency"]) * rows + float(probe["channel_phase"]) * channels)
        * np.cos(float(probe["column_frequency"]) * columns)
        + float(probe["detail_amplitude"])
        * np.sin(float(probe["detail_frequency"]) * rows)
        * np.sin(float(probe["detail_frequency"]) * columns)
    )
    return values.astype(np.float32)


def _raw_output(model: Any, image_chw: torch.Tensor) -> Any:
    """Run one ``(3, H, W)`` image through the training-time transform and the model.

    ``model(batch)`` would apply ``pre_processor.export_transform`` (resize
    without antialiasing) and score up to ~0.05 away from training-time
    prediction, enough to flip a verdict near the threshold.
    """
    transformed = model.pre_processor.transform(image_chw[None])
    return model.model(transformed)


def _check_stamp(stamp: Any, class_path: str, ckpt_path: str) -> None:
    """Refuse a package whose stamp does not match this installation or file."""
    if not isinstance(stamp, dict):
        raise AnomalibModelIncompatible(
            "No compatibility stamp in anomalib_config. The model was not packaged "
            "by the Anomalib 2.x training project; Anomalib 1.x checkpoints cannot "
            "be loaded. Retrain it and deploy the new package."
        )
    if stamp.get("artifact_schema") != SUPPORTED_ARTIFACT_SCHEMA:
        raise AnomalibModelIncompatible(
            f"Unsupported anomalib artifact_schema {stamp.get('artifact_schema')!r}; "
            f"this station reads schema {SUPPORTED_ARTIFACT_SCHEMA}."
        )
    if stamp.get("class_path") != class_path:
        raise AnomalibModelIncompatible(
            f"Stamp is for {stamp.get('class_path')!r} but the config loads {class_path!r}."
        )
    stamped = str((stamp.get("runtime") or {}).get("anomalib") or "")
    installed = metadata.version("anomalib")
    if stamped.split(".")[:2] != installed.split(".")[:2]:
        raise AnomalibModelIncompatible(
            f"Model was trained with anomalib {stamped or 'unknown'}, this station runs "
            f"{installed}. Retrain with the station's version."
        )
    actual_sha = _sha256_file(ckpt_path)
    if stamp.get("checkpoint_sha256") != actual_sha:
        raise AnomalibModelIncompatible(
            f"Checkpoint {ckpt_path} is not the stamped file "
            f"(sha256 {actual_sha} != {stamp.get('checkpoint_sha256')})."
        )


def _check_probe(model: Any, stamp: dict[str, Any]) -> float:
    """Re-score the stamped probe; a drift means the model would judge differently."""
    probe = stamp["probe"]
    with torch.inference_mode():
        raw = _raw_output(model, torch.from_numpy(_probe_image(probe)))
    score = float(raw.pred_score.reshape(-1)[0])
    expected = float(probe["raw_pred_score"])
    tolerance = float(probe["abs_tolerance"]) + float(probe["rel_tolerance"]) * abs(expected)
    if abs(score - expected) > tolerance:
        raise AnomalibModelIncompatible(
            f"Probe score {score:.6f} differs from the stamped {expected:.6f} by more "
            f"than {tolerance:.2e}: the model or its pre-processing changed since packaging."
        )
    return score


def _load_model(class_path: str, ckpt_path: str) -> Any:
    module_name, class_name = class_path.rsplit(".", 1)
    model_class = getattr(import_module(module_name), class_name)
    # These files are copied between machines; refuse anything that needs
    # arbitrary unpickling.
    model = model_class.load_from_checkpoint(ckpt_path, weights_only=True)
    if model is None:
        raise ValueError(f"Failed to load checkpoint: {ckpt_path}")
    return model.eval()


def initialize(
    config: dict[str, Any] | None = None,
    product: str | None = None,
    area: str | None = None,
) -> None:
    global _output_dir, _current_product, _current_area, _current_ckpt_path

    if config is None:
        raise ValueError("config is required for initialize()")
    if product is None or area is None:
        raise ValueError("product and area are required for initialize()")

    model_key: ModelKey = (product, area)

    with _initialization_lock:
        if _is_initialized.get(model_key):
            prior_ckpt = _models[model_key]["ckpt_path"]
            logger.info(
                f"Models for {product},{area} already initialized; checkpoint: {prior_ckpt}. "
                "Skipping reinitialization."
            )
            _current_product = product
            _current_area = area
            _current_ckpt_path = prior_ckpt
            return

        try:
            logger.info(f"Initializing Anomalib model (product: {product}, area: {area})")
            output_dir = str(config.get("output", "./patchcore_outputs"))
            class_path = str(
                (config.get("model") or {}).get("class_path", "anomalib.models.Patchcore")
            )

            models_cfg = config.get("models") or {}
            if product not in models_cfg or area not in models_cfg.get(product, {}):
                raise ValueError(
                    f"Missing checkpoint for {product},{area}; please configure it in config.yaml"
                )
            model_config = models_cfg[product][area]
            ckpt_path = model_config.get("ckpt_path")
            default_threshold = float(config.get("threshold", 0.5))
            threshold = float(model_config.get("threshold", default_threshold))
            if not ckpt_path:
                raise ValueError(
                    "Checkpoint path not provided; configure a valid ckpt_path in config.yaml"
                )
            if not os.path.exists(str(ckpt_path)):
                raise ValueError(f"Checkpoint file not found: {ckpt_path}")

            stamp = config.get("compatibility")
            _check_stamp(stamp, class_path, str(ckpt_path))
            logger.info(f"Loading checkpoint: {ckpt_path}, model type: {class_path}")
            model = _load_model(class_path, str(ckpt_path))
            probe_score = _check_probe(model, stamp)
            logger.info(
                f"Compatibility stamp verified (anomalib {stamp['runtime']['anomalib']}, "
                f"probe {probe_score:.6f})"
            )

            _thresholds[model_key] = threshold
            _models[model_key] = {"model": model, "ckpt_path": ckpt_path}
            _output_dir = output_dir.replace("YYYYMMDD", time.strftime("%Y%m%d"))
            os.makedirs(_output_dir, exist_ok=True)

            _is_initialized[model_key] = True
            _current_product = product
            _current_area = area
            _current_ckpt_path = str(ckpt_path)
            logger.info(
                f"Model initialized (product: {product}, area: {area}, "
                f"ckpt: {ckpt_path}, threshold: {threshold})"
            )
            logger.info(f"Output directory: {_output_dir}")

        except Exception as exc:
            logger.error(
                f"Model initialization failed (product: {product}, area: {area}): {str(exc)}",
                exc_info=True,
            )
            raise


def initialize_product_models(
    config: dict[str, Any] | None = None,
    product: str | None = None,
) -> None:
    if config is None:
        raise ValueError("config is required for initialize_product_models()")
    if product is None:
        raise ValueError("product is required for initialize_product_models()")

    models_cfg = config.get("models") or {}
    if product not in models_cfg:
        raise ValueError(f"No model configuration found for product {product}")

    logger.info(f"Initializing Anomalib models for all areas of {product}")
    for area_name in models_cfg[product]:
        initialize(config, product, area_name)


def lightning_inference(
    image_path: str,
    thread_safe: bool = True,
    enable_timing: bool = True,
    product: str | None = None,
    area: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    if product is None or area is None:
        raise ValueError("product and area are required for inference")

    model_key: ModelKey = (product, area)
    if not _is_initialized.get(model_key):
        raise RuntimeError(f"Anomalib model for {product},{area} is not initialized; call initialize()")

    if not os.path.exists(image_path):
        return {
            "image_path": image_path,
            "product": product,
            "area": area,
            "error": f"Image not found: {image_path}",
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
        logger.warning(f"Image size {img.shape[:2]} differs from the expected 640x640")

    timings: dict[str, float] = {}
    start_time = time.time()

    try:
        read_start = time.time()
        # The same reader training-time prediction uses, so decoding matches.
        image_chw = read_image(image_path, as_tensor=True)
        if enable_timing:
            timings["image_read"] = round(time.time() - read_start, 4)

        predict_start = time.time()

        def _do_predict() -> tuple[Any, str]:
            model_info = _models[model_key]
            model = model_info["model"]
            with torch.inference_mode():
                output = model.post_processor(_raw_output(model, image_chw))
            return output, str(model_info["ckpt_path"])

        if thread_safe:
            with _inference_lock:
                output, ckpt_path = _do_predict()
        else:
            output, ckpt_path = _do_predict()

        if enable_timing:
            timings["model_inference"] = round(time.time() - predict_start, 4)

        process_start = time.time()
        result = _process_prediction(
            {"pred_scores": output.pred_score, "anomaly_maps": output.anomaly_map},
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
        logger.error(f"Anomalib inference failed for {image_path}: {exc}", exc_info=True)
        return {
            "image_path": image_path,
            "product": product,
            "area": area,
            "error": f"Anomalib inference failed: {str(exc)}",
            "inference_time": round(time.time() - start_time, 4),
        }


# anomalib_lightning_inference.py, _process_prediction 方法中
def _process_prediction(
    pred: dict[str, Any],
    image_path: str,
    product: str,
    area: str,
    output_path: str | None = None,
    enable_timing: bool = False,
) -> dict[str, Any]:
    try:
        model_key: ModelKey = (product, area)
        threshold = _thresholds.get(model_key, 0.5)
        anomaly_score_raw = pred.get("pred_scores", 0.0)
        if anomaly_score_raw is not None and hasattr(anomaly_score_raw, "item"):
            anomaly_score = float(anomaly_score_raw.reshape(-1)[0].item())
        else:
            anomaly_score = float(anomaly_score_raw or 0.0)

        anomaly_map = pred.get("anomaly_maps")
        if anomaly_map is None:
            logger.error(f"Anomaly heatmap missing; keys: {list(pred.keys())}")
            return {
                "image_path": image_path,
                "error": "Anomaly heatmap missing",
                "product": product,
                "area": area,
            }

        original_image = cv2.imread(image_path)
        if original_image is None:
            logger.error(f"Failed to read image: {image_path}")
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

        is_anomaly = anomaly_score > threshold
        total_pixels = int(heatmap_3d.size)
        overlay_image = original_image.copy()

        heatmap_range = heatmap_3d.max() - heatmap_3d.min()
        logger.info(
            f"anomaly_score={anomaly_score:.4f}, threshold={threshold}, "
            f"heatmap raw min={heatmap_3d.min():.4f} max={heatmap_3d.max():.4f}"
        )

        if not is_anomaly or heatmap_range == 0:
            # PASS or flat heatmap: return original image without any heat overlay.
            threshold_mask = np.zeros_like(heatmap_3d, dtype=bool)
            anomaly_pixel_count = 0
            anomaly_ratio = 0.0
            if heatmap_range == 0 and is_anomaly:
                logger.warning("Heatmap has no variation; skipping overlay")
            else:
                logger.info("PASS: no heat overlay applied")
        else:
            # FAIL: normalize per-image for colormap, mask by threshold on normalized scale.
            normalized_heatmap = (heatmap_3d - heatmap_3d.min()) / heatmap_range
            threshold_mask = normalized_heatmap > threshold
            colored_heatmap = cv2.applyColorMap(
                (normalized_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)
            if np.any(threshold_mask):
                overlay_image[threshold_mask] = cv2.addWeighted(
                    colored_heatmap[threshold_mask].reshape(-1, 3),
                    0.4,
                    original_image[threshold_mask].reshape(-1, 3),
                    0.6,
                    0,
                ).reshape(-1, 3)
            anomaly_pixel_count = int(np.sum(threshold_mask))
            anomaly_ratio = anomaly_pixel_count / total_pixels
            logger.info(
                f"Anomaly pixels: {anomaly_pixel_count}/{total_pixels} ({anomaly_ratio:.2%})"
            )

        if output_path:
            heatmap_path = output_path
        else:
            status_label = "FAIL" if is_anomaly else "PASS"
            bundle = ResultPathManager(
                str(load_station_data_paths().results)
            ).build_paths(
                status=status_label,
                detector="anomalib",
                product=product,
                area=area,
                anomaly_score=anomaly_score,
            )
            heatmap_path = bundle.annotated_path

        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_path, overlay_image_bgr)
        logger.info(f"Saved annotated heatmap to: {heatmap_path}")

        return {
            "image_path": image_path,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            "is_anomaly": is_anomaly,
            "anomaly_pixel_count": anomaly_pixel_count,
            "total_pixels": total_pixels,
            "anomaly_pixel_ratio": anomaly_ratio,
            "heatmap_path": heatmap_path,
            "overlay_image": overlay_image_bgr,  # BGR ndarray, avoids disk re-read
            "product": product,
            "area": area,
            "ckpt_path": _models.get(model_key, {}).get("ckpt_path", ""),
        }

    except Exception as exc:
        logger.error(
            f"Processing prediction failed: {str(exc)}",
            exc_info=True,
        )
        return {
            "image_path": image_path,
            "error": f"Processing prediction failed: {str(exc)}",
            "product": product,
            "area": area,
        }


def set_threshold(product: str, area: str, threshold: float) -> None:
    """動態設定特定產品區域的閾值"""
    model_key = (product, area)
    _thresholds[model_key] = threshold
    logger.info(f"已設定 {product},{area} 的閾值為: {threshold}")


def get_threshold(product: str, area: str) -> float:
    """取得特定產品區域的閾值"""
    model_key = (product, area)
    return _thresholds.get(model_key, 0.5)


def get_status():
    return {
        "initialized": _is_initialized,
        "model_loaded": {k: v["model"] is not None for k, v in _models.items()},
        "output_dir": _output_dir,
        "current_product": _current_product,
        "current_area": _current_area,
        "current_ckpt_path": _current_ckpt_path,
        "thresholds": _thresholds,  # 新增：顯示所有閾值設定
    }
