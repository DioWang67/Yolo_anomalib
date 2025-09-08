from __future__ import annotations

import os
from pathlib import Path
from collections import OrderedDict
import copy
import yaml

from core.config import DetectionConfig
from core.logger import DetectionLogger
from core.inference_engine import InferenceEngine
from core.path_utils import project_root, resolve_path
from core.config_validation import validate_model_cfg


# Repository root (two levels up from core/services)
# Determine repository root (can be overridden by YOLO11_ROOT env var)
PROJECT_ROOT = project_root()


class ModelManager:
    def __init__(self, logger: DetectionLogger, max_cache_size: int = 3) -> None:
        self.logger = logger
        self.max_cache_size = max_cache_size
        # cache key: (product, area) -> { type: (engine, config_snapshot) }
        self._cache: OrderedDict[tuple[str, str], dict[str, tuple[InferenceEngine, DetectionConfig]]] = OrderedDict()

    def _initialize_product_models(self, config: DetectionConfig, product: str) -> None:
        if not getattr(config, "enable_anomalib", False):
            return
        try:
            from core.anomalib_lightning_inference import initialize_product_models as _anoma_init
            _anoma_init(config.anomalib_config, product)
            self.logger.logger.info(f"Anomalib models initialized for {product}")
        except Exception as e:
            self.logger.logger.error(f"Anomalib init failed for {product}: {str(e)}")
            raise

    def _apply_model_config(self, base_config: DetectionConfig, cfg: dict) -> None:
        base_config.device = cfg.get("device", base_config.device)
        base_config.conf_thres = cfg.get("conf_thres", base_config.conf_thres)
        base_config.iou_thres = cfg.get("iou_thres", base_config.iou_thres)
        base_config.imgsz = tuple(cfg.get("imgsz", base_config.imgsz))
        base_config.enable_yolo = cfg.get("enable_yolo", base_config.enable_yolo)
        base_config.enable_anomalib = cfg.get("enable_anomalib", base_config.enable_anomalib)
        base_config.expected_items = cfg.get("expected_items", base_config.expected_items)
        base_config.position_config = cfg.get("position_config", base_config.position_config)
        base_config.output_dir = cfg.get("output_dir", base_config.output_dir)
        base_config.anomalib_config = cfg.get("anomalib_config")
        base_config.weights = cfg.get("weights", getattr(base_config, "weights", ""))
        base_config.enable_color_check = cfg.get("enable_color_check", getattr(base_config, "enable_color_check", False))
        color_model_path = cfg.get("color_model_path", None)
        if color_model_path:
            base_config.color_model_path = str(resolve_path(color_model_path))
        else:
            base_config.color_model_path = None
        # optional custom backends config (name -> {class_path, enabled, ...})
        base_config.backends = cfg.get("backends", getattr(base_config, "backends", None))
        # optional pipeline and steps overrides
        if "pipeline" in cfg:
            base_config.pipeline = cfg.get("pipeline")
        # merge steps (model-level overrides take precedence)
        steps_cfg = cfg.get("steps", {}) or {}
        merged_steps = dict(getattr(base_config, "steps", {}) or {})
        merged_steps.update(steps_cfg)
        base_config.steps = merged_steps

    def switch(self, base_config: DetectionConfig, product: str, area: str, inference_type: str) -> tuple[InferenceEngine, DetectionConfig]:
        key = (product, area)
        if key in self._cache and inference_type in self._cache[key]:
            self.logger.logger.info(
                f"Using cached model: product={product}, area={area}, type={inference_type}"
            )
            engine, cfg_snapshot = self._cache[key][inference_type]
            base_config.__dict__.update(copy.deepcopy(cfg_snapshot.__dict__))
            self._cache.move_to_end(key)
            return engine, base_config

        model_config_path = os.path.join("models", product, area, inference_type, "config.yaml")
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")

        with open(model_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Optional pydantic normalization for model-level config
        try:
            from core.config_schema import ModelConfigSchema, _to_dict  # type: ignore
        except Exception:
            ModelConfigSchema = None  # type: ignore
            _to_dict = None  # type: ignore
        if ModelConfigSchema is not None:
            try:
                cfg = _to_dict(ModelConfigSchema(**cfg))  # type: ignore
            except Exception as e:
                self.logger.logger.warning(f"Model config schema validation failed: {e}")

        # Validate critical fields/paths early with helpful messages
        try:
            validate_model_cfg(cfg or {}, product, area)
        except Exception as e:
            self.logger.logger.error(f"Model config validation failed: {e}")
            raise

        self._apply_model_config(base_config, cfg)

        engine = InferenceEngine(base_config)
        if not engine.initialize():
            raise RuntimeError("Inference engine init failed")

        if inference_type == "anomalib":
            self._initialize_product_models(base_config, product)

        if key not in self._cache:
            self._cache[key] = {}
        self._cache[key][inference_type] = (engine, copy.deepcopy(base_config))
        self._cache.move_to_end(key)
        if len(self._cache) > self.max_cache_size:
            old_key, engines = self._cache.popitem(last=False)
            for eng, _ in engines.values():
                try:
                    eng.shutdown()
                except Exception:
                    pass
            self.logger.logger.info(f"Evicted cached model: product={old_key[0]}, area={old_key[1]}")

        return engine, base_config
