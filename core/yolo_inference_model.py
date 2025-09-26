from __future__ import annotations


import os
import time
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict

import numpy as np
import torch
from torch.cuda.amp import autocast
from ultralytics import YOLO

from core.base_model import BaseInferenceModel
from core.exceptions import ModelInitializationError, ModelInferenceError
from core.detector import YOLODetector
from core.position_validator import PositionValidator
from core.utils import ImageUtils


class YOLOInferenceModel(BaseInferenceModel):
    _warmup_registry: set[tuple[str, str]] = set()

    """Inference wrapper that keeps a lightweight cache of YOLO models."""

    def __init__(self, config):
        super().__init__(config)
        self.model_cache: OrderedDict[tuple[str, str], tuple[YOLO, YOLODetector]] = (
            OrderedDict()
        )
        self.max_cache_size = getattr(config, "max_cache_size", 3)
        if getattr(config, "disable_internal_cache", False):
            self.max_cache_size = 0
        self.image_utils = ImageUtils()

    def initialize(self, product: str | None = None, area: str | None = None) -> bool:
        key = (product or "default", area or "default")
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
            weights_key = (
                os.path.abspath(str(self.config.weights)),
                str(self.config.device),
            )
            should_warmup = weights_key not in self._warmup_registry

            self.model = YOLO(self.config.weights)
            self.model.to(self.config.device)
            self.model.fuse()
            if self.config.device != "cpu":
                self.model.model.half()
                torch.backends.cudnn.benchmark = True
            self.detector = YOLODetector(self.model, self.config)

            if should_warmup:
                try:
                    h, w = self.config.imgsz
                    dummy = np.zeros((h, w, 3), dtype=np.uint8)
                    with torch.inference_mode():
                        _ = self.model(
                            dummy,
                            conf=self.config.conf_thres,
                            iou=self.config.iou_thres,
                        )
                    self._warmup_registry.add(weights_key)
                except Exception as warmup_err:
                    self.logger.logger.warning(
                        "YOLO warmup failed: %s", warmup_err, exc_info=warmup_err
                    )

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
                f"YOLO model initialized (device={self.config.device}, product={product}, area={area})"
            )
            return True
        except (FileNotFoundError, RuntimeError) as exc:
            self.logger.logger.exception("Failed to initialize YOLO model")
            raise ModelInitializationError(str(exc)) from exc
        except Exception as exc:
            self.logger.logger.exception(
                "Unexpected error during YOLO model initialization"
            )
            raise ModelInitializationError(
                "YOLO initialization failed") from exc

    def preprocess_image(
        self, frame: np.ndarray, product: str, area: str
    ) -> np.ndarray:
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
    ) -> Dict[str, Any]:
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

            amp_ctx = autocast if self.config.device != "cpu" else nullcontext
            with torch.inference_mode():
                with amp_ctx():
                    prediction = self.model(
                        processed_image,
                        conf=self.config.conf_thres,
                        iou=self.config.iou_thres,
                    )

            result_frame, detections, missing_items = self.detector.process_detections(
                prediction, processed_image, image, expected_items
            )

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                det["image_height"] = image.shape[0]
                det["image_width"] = image.shape[1]
                det["cx"] = (x1 + x2) / 2
                det["cy"] = (y1 + y2) / 2

            validator = PositionValidator(self.config, product, area)
            detections = validator.validate(detections)
            status = validator.evaluate_status(detections, missing_items)

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
            }
        except Exception as exc:
            self.logger.logger.exception("YOLO inference failed")
            raise ModelInferenceError(str(exc)) from exc
