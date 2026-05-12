from __future__ import annotations

"""高階 YOLO 推論封裝，提供快取、前處理與驗證工具。"""

import os
import threading
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.cuda.amp import autocast
from ultralytics import YOLO

from core.base_model import BaseInferenceModel
from core.detector import YOLODetector
from core.exceptions import (
    BackendInitializationError,
    ConfigurationError,
    ModelInferenceError,
    ModelInitializationError,
    ResourceExhaustionError,
)
from core.runtime_preflight import validate_runtime_for_model
from core.position_validator import PositionValidator
from core.utils import ImageUtils
from core.yolo_runtime import YoloRuntimeInfo, detect_yolo_runtime


class YOLOInferenceModel(BaseInferenceModel):
    """Inference wrapper for YOLO models with built-in caching and position validation.

    This class handles model loading, device placement (CPU/GPU), and provides
    a structured interface for image inference and pre-processing.

    Attributes:
        model_cache (OrderedDict): LRU cache storing (Product, Area) -> (YOLO, Detector) tuples.
        max_cache_size (int): Maximum number of models to keep in memory.
        image_utils (ImageUtils): Utility class for image transformations.
    """
    def __init__(self, config):
        super().__init__(config)
        self._cache_lock = threading.Lock()
        self._warmup_registry: set[tuple[str, ...]] = set()
        self.runtime_info: YoloRuntimeInfo | None = None
        self.model_cache: OrderedDict[tuple[str, str], tuple[YOLO, YOLODetector]] = (
            OrderedDict()
        )
        self.max_cache_size = getattr(config, "max_cache_size", 3)
        if getattr(config, "disable_internal_cache", False):
            self.max_cache_size = 0
        self.image_utils = ImageUtils()

    def initialize(self, product: str | None = None, area: str | None = None) -> bool:
        """Initializes or retrieves the YOLO model for a specific product and area.

        Args:
            product: The product identifier. Defaults to 'default'.
            area: The area/station identifier. Defaults to 'default'.

        Returns:
            bool: True if initialization was successful.

        Raises:
            ModelInitializationError: If weights are missing or a runtime hardware error occurs.
        """
        key = (product or "default", area or "default")
        with self._cache_lock:
            if self.max_cache_size > 0 and key in self.model_cache:
                self.model, self.detector = self.model_cache[key]
                self.model_cache.move_to_end(key)
                self.is_initialized = True
                self.logger.logger.info(
                    f"Loaded YOLO model from cache (product={product}, area={area})"
                )
                return True

        try:
            self.logger.logger.info("Initializing YOLO model...")
            self.runtime_info = detect_yolo_runtime(
                self.config.weights,
                getattr(self.config, "device", None),
                cuda_available=torch.cuda.is_available(),
            )
            validate_runtime_for_model(self.config.weights)
            weights_key = (
                os.path.abspath(str(self.config.weights)),
                str(self.runtime_info.setup_device or ""),
                str(self.runtime_info.predict_device or ""),
                self.runtime_info.runtime,
            )
            should_warmup = weights_key not in self._warmup_registry

            self.model = YOLO(self.config.weights)
            if self.runtime_info.setup_device:
                self.model.to(self.runtime_info.setup_device)
                self.config.device = self.runtime_info.setup_device
            elif self.runtime_info.predict_device:
                self.config.device = self.runtime_info.predict_device

            if self.runtime_info.call_fuse and hasattr(self.model, "fuse"):
                self.model.fuse()
            if self.runtime_info.use_torch_half:
                inner_model = getattr(self.model, "model", None)
                if hasattr(inner_model, "half"):
                    inner_model.half()
                torch.backends.cudnn.benchmark = True
            self.detector = YOLODetector(self.model, self.config)

            if should_warmup:
                try:
                    h, w = self.config.imgsz
                    dummy = np.zeros((h, w, 3), dtype=np.uint8)
                    with torch.inference_mode():
                        _ = self._predict_yolo(dummy)
                    self._warmup_registry.add(weights_key)
                except Exception as warmup_err:
                    if self.runtime_info.runtime == "onnx":
                        raise ModelInitializationError(
                            "YOLO ONNX warmup failed after runtime preflight: "
                            f"{warmup_err}"
                        ) from warmup_err
                    self.logger.logger.warning(
                        "YOLO warmup failed: %s", warmup_err, exc_info=warmup_err
                    )

            with self._cache_lock:
                if self.max_cache_size > 0:
                    self.model_cache[key] = (self.model, self.detector)
                    self.model_cache.move_to_end(key)
                    if len(self.model_cache) > self.max_cache_size:
                        old_key, (old_model, _) = self.model_cache.popitem(
                            last=False)
                        try:
                            del old_model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as release_err:
                            self.logger.logger.warning(
                                "Failed to release YOLO model resources: %s",
                                release_err,
                                exc_info=release_err,
                            )
                        self.logger.logger.info(
                            f"Evicted cached YOLO model: product={old_key[0]}, area={old_key[1]}"
                        )

            self.is_initialized = True
            self.logger.logger.info(
                "YOLO model initialized "
                f"(runtime={self.runtime_info.runtime}, device={self.config.device}, "
                f"product={product}, area={area})"
            )
            return True
        except FileNotFoundError as exc:
            self.logger.logger.error(
                f"YOLO model weights file not found: {self.config.weights}"
            )
            raise ConfigurationError(
                f"模型權重檔案不存在: {self.config.weights}。請檢查路徑或配置。"
            ) from exc
        except (BackendInitializationError, ModelInitializationError):
            raise
        except RuntimeError as exc:
            self.logger.logger.exception(
                "Runtime error (e.g., CUDA issue) during YOLO model initialization"
            )
            if "out of memory" in str(exc).lower():
                raise ResourceExhaustionError(
                    "顯存不足 (CUDA OOM)，無法加載 YOLO 模型。"
                ) from exc
            raise ModelInitializationError(
                "模型加載過程中發生運行時錯誤，詳情請見日誌。"
            ) from exc

    def preprocess_image(
        self, frame: np.ndarray, product: str, area: str
    ) -> np.ndarray:
        """Prepares a raw image for YOLO inference using letterboxing.

        Args:
            frame: Raw input image (BGR).
            product: Product identifier.
            area: Area identifier.

        Returns:
            np.ndarray: Resized and padded image ready for the model.
        """
        target_size = self.config.imgsz
        resized_img = self.image_utils.letterbox(
            frame, size=target_size, fill_color=(128, 128, 128)
        )
        self.logger.logger.debug(
            f"Preprocessing image: original={frame.shape[:2]}, target={target_size}"
        )
        return resized_img

    def infer(
        self, image: np.ndarray, product: str, area: str, output_path: str | None = None
    ) -> dict[str, Any]:
        """Performs inference on the provided image for a given product and area.

        Includes preprocessing, model forward pass, detection processing, and
        optional position validation.

        Args:
            image: The raw input image (BGR).
            product: Name of the product being inspected.
            area: Station or area ID.
            output_path: Optional path for saving temporary inference results.

        Returns:
            dict: Structured result containing detections, missing items, and status.

        Raises:
            ModelInferenceError: If inference fails due to configuration or runtime issues.
        """
        if not self.is_initialized:
            raise RuntimeError("YOLO model is not initialized")

        try:
            start_time = time.time()
            processed_image = self.preprocess_image(image, product, area)
            expected_items = self.config.get_items_by_area(product, area)
            if not expected_items:
                raise ModelInferenceError(
                    f"Invalid product/area combination: {product},{area}"
                )

            runtime_info = self.runtime_info or detect_yolo_runtime(
                self.config.weights,
                getattr(self.config, "device", None),
                cuda_available=torch.cuda.is_available(),
            )
            amp_ctx = autocast if runtime_info.use_torch_amp else nullcontext
            with torch.inference_mode():
                with amp_ctx():
                    prediction = self._predict_yolo(processed_image)

            result_frame, detections, missing_items = self.detector.process_detections(
                prediction, processed_image, image, expected_items
            )

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["image_height"] = image.shape[0]
                det["image_width"] = image.shape[1]
                det["cx"] = (x1 + x2) / 2
                det["cy"] = (y1 + y2) / 2

            # Position check can be toggled per product/area; skip entirely when disabled
            position_enabled = False
            try:
                position_enabled = self.config.is_position_check_enabled(product, area)
            except Exception:
                position_enabled = False

            if position_enabled:
                validator = PositionValidator(self.config, product, area)
                detections = validator.validate(detections)
                status = validator.evaluate_status(detections, missing_items)
            else:
                # When disabled, keep status purely based on missing items here
                status = "FAIL" if missing_items else "PASS"

            unexpected_items: list[str] = []
            try:
                expected_set = {str(x).strip() for x in (expected_items or [])}
                detected_names = [
                    str(d.get("class", "")).strip() for d in (detections or [])
                ]
                unexpected_items = sorted(
                    {n for n in detected_names if n and n not in expected_set}
                )
                if unexpected_items and getattr(
                    self.config, "fail_on_unexpected", True
                ):
                    status = "FAIL"
            except Exception as item_err:
                self.logger.logger.warning(
                    "Failed to compute unexpected items: %s",
                    item_err,
                    exc_info=item_err,
                )
                unexpected_items = []

            inference_time = time.time() - start_time
            self.logger.logger.debug(
                f"YOLO inference time: {inference_time:.3f}s, detections={len(detections)}"
            )

            return {
                "inference_type": "yolo",
                "status": status,
                "detections": detections,
                "missing_items": list(missing_items),
                "unexpected_items": unexpected_items,
                "inference_time": inference_time,
                "processed_image": processed_image,
                "result_frame": result_frame,
                "expected_items": expected_items,
                "ckpt_path": str(self.config.weights),
            }
        except Exception as exc:
            self.logger.logger.exception("YOLO inference failed")
            raise ModelInferenceError(str(exc)) from exc

    def _predict_yolo(self, image: np.ndarray):
        """Run the loaded YOLO model with runtime-specific device arguments."""
        if self.runtime_info and self.runtime_info.predict_device:
            return self.model(
                image,
                conf=self.config.conf_thres,
                iou=self.config.iou_thres,
                device=self.runtime_info.predict_device,
            )
        return self.model(
            image,
            conf=self.config.conf_thres,
            iou=self.config.iou_thres,
        )
