from __future__ import annotations

import copy
import os
import threading
from collections import OrderedDict
from pathlib import Path

import yaml  # type: ignore[import]

from core.config import DetectionConfig
from core.config_validation import validate_model_cfg
from core.inference_engine import InferenceEngine
from core.logging_config import DetectionLogger
from core.path_utils import project_root, resolve_path
from core.version_utils import (
    ModelVersionError,
    check_compatibility,
    find_latest_version,
    parse_model_version,
    parse_version_string,
    version_to_string,
)

# Repository root (two levels up from core/services)
# Determine repository root (can be overridden by YOLO11_ROOT env var)
PROJECT_ROOT = project_root()


class ModelManager:
    def __init__(self, logger: DetectionLogger, max_cache_size: int = 3) -> None:
        """Create a model manager.

        Args:
            logger: DetectionLogger wrapper
            max_cache_size: Max number of (product, area) entries to keep
        """
        self.logger = logger
        self.max_cache_size = max_cache_size
        self._cache_lock = threading.Lock()
        # cache key: (product, area) -> { type: (engine, config_snapshot) }
        self._cache: OrderedDict[
            tuple[str, str], dict[str, tuple[InferenceEngine, DetectionConfig]]
        ] = OrderedDict()

    def _initialize_product_models(self, config: DetectionConfig, product: str) -> None:
        """Preload anomalib models for all areas of a product (optional)."""
        if not getattr(config, "enable_anomalib", False):
            return
        try:
            from core.anomalib_lightning_inference import (
                initialize_product_models as _anoma_init,
            )

            anomalib_cfg = config.anomalib_config or {}
            _anoma_init(anomalib_cfg, product)
            self.logger.logger.info(f"Anomalib models initialized for {product}")
        except Exception as e:
            self.logger.logger.error(f"Anomalib init failed for {product}: {str(e)}")
            raise

    @staticmethod
    def _resolve_model_path(
        raw: str, model_cfg_dir: Path | None
    ) -> str:
        """Resolve a relative model path against project root, then config dir."""
        p = Path(raw)
        if p.is_absolute():
            return raw
        resolved = resolve_path(raw)
        if model_cfg_dir and (resolved is None or not resolved.exists()):
            config_relative = (model_cfg_dir / p).resolve()
            if config_relative.exists():
                return str(config_relative)
        return str(resolved) if resolved else raw

    def _apply_model_config(
        self,
        base_config: DetectionConfig,
        cfg: dict,
        context: str | None = None,
        model_cfg_dir: Path | None = None,
    ) -> None:
        """Apply per-model overrides into the shared DetectionConfig instance."""
        # --- Simple scalar overrides ---
        _SCALAR_FIELDS = [
            "device", "conf_thres", "iou_thres", "enable_yolo", "enable_anomalib",
            "enable_color_check",
        ]
        for field in _SCALAR_FIELDS:
            if field in cfg:
                setattr(base_config, field, cfg.get(field, getattr(base_config, field)))

        # --- Fields that only apply when present and non-None ---
        _OPTIONAL_FIELDS = [
            "expected_items", "position_config", "anomalib_config",
            "color_threshold_overrides", "color_rules_overrides",
            "backends", "pipeline",
        ]
        for field in _OPTIONAL_FIELDS:
            if field in cfg and cfg.get(field) is not None:
                setattr(base_config, field, cfg[field])

        # --- imgsz needs tuple conversion ---
        if "imgsz" in cfg and cfg.get("imgsz") is not None:
            base_config.imgsz = tuple(cfg["imgsz"])  # type: ignore[arg-type]

        # --- output_dir: resolve relative paths against model config dir ---
        if "output_dir" in cfg:
            raw_output_dir = cfg.get("output_dir")
            if raw_output_dir:
                path_str = str(raw_output_dir).strip()
                if path_str:
                    resolved = Path(path_str)
                    if not resolved.is_absolute():
                        base = model_cfg_dir if model_cfg_dir else PROJECT_ROOT
                        resolved = (base / resolved).resolve()
                    base_config.output_dir = str(resolved)
                else:
                    self.logger.logger.warning(
                        "Model config %s provided whitespace output_dir; keeping %s",
                        context or "unknown", base_config.output_dir,
                    )
            else:
                self.logger.logger.warning(
                    "Model config %s provided empty output_dir; keeping %s",
                    context or "unknown", base_config.output_dir,
                )

        # --- weights / color_model_path: resolve via shared helper ---
        raw_weights = cfg.get("weights", getattr(base_config, "weights", ""))
        if raw_weights:
            base_config.weights = self._resolve_model_path(raw_weights, model_cfg_dir)

        color_model_path = cfg.get("color_model_path")
        if color_model_path:
            base_config.color_model_path = self._resolve_model_path(
                color_model_path, model_cfg_dir
            )

        # --- color checker type with fallback ---
        base_config.color_checker_type = str(
            cfg.get(
                "color_checker_type",
                getattr(base_config, "color_checker_type", "color_qc"),
            ) or "color_qc"
        )
        base_config.color_score_threshold = cfg.get(
            "color_score_threshold",
            getattr(base_config, "color_score_threshold", None),
        )

        # --- merge steps (model-level overrides take precedence) ---
        steps_cfg = cfg.get("steps", {}) or {}
        merged_steps = dict(getattr(base_config, "steps", {}) or {})
        merged_steps.update(steps_cfg)
        base_config.steps = merged_steps

    def switch(
        self, base_config: DetectionConfig, product: str, area: str, inference_type: str
    ) -> tuple[InferenceEngine, DetectionConfig]:
        """Switch engine to (product, area, type), with LRU cache.

        Returns the engine and the (mutated) base_config snapshot after
        overrides.
        """
        key = (product, area)
        with self._cache_lock:
            if key in self._cache and inference_type in self._cache[key]:
                self.logger.logger.info(
                    f"Using cached model: product={product}, "
                    f"area={area}, type={inference_type}"
                )
                engine, cfg_snapshot = self._cache[key][inference_type]
                base_config.__dict__.update(copy.deepcopy(cfg_snapshot.__dict__))
                self._cache.move_to_end(key)
                return engine, base_config

        model_config_path = os.path.join(
            "models", product, area, inference_type, "config.yaml"
        )
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config not found: {model_config_path}")

        with open(model_config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Optional pydantic normalization for model-level config
        try:
            from core.config_schema import (
                ModelConfigSchema,  # type: ignore
                _to_dict,  # type: ignore
            )
        except Exception:
            ModelConfigSchema = None  # type: ignore
            _to_dict = None  # type: ignore
        if ModelConfigSchema is not None:
            try:
                cfg = _to_dict(ModelConfigSchema(**cfg))  # type: ignore
            except Exception as e:
                self.logger.logger.warning(
                    f"Model config schema validation failed: {e}"
                )

        model_cfg_dir = Path(model_config_path).resolve().parent

        # Validate critical fields/paths early with helpful messages
        try:
            validate_model_cfg(
                cfg or {}, product, area, selected_backend=inference_type, model_cfg_dir=model_cfg_dir
            )
        except Exception as e:
            self.logger.logger.error(f"Model config validation failed: {e}")
            raise

        context = f"{product}/{area}/{inference_type}"
        self._apply_model_config(base_config, cfg, context, model_cfg_dir=model_cfg_dir)

        # Version validation (if model uses versioned naming)
        self._validate_model_version(base_config, cfg, product, area, inference_type)

        engine = InferenceEngine(base_config)
        if not engine.initialize():
            raise RuntimeError("Inference engine init failed")

        if inference_type == "anomalib":
            self._initialize_product_models(base_config, product)

        with self._cache_lock:
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
                self.logger.logger.info(
                    f"Evicted cached model: product={old_key[0]}, area={old_key[1]}"
                )

        return engine, base_config

    def _validate_model_version(
        self,
        config: DetectionConfig,
        model_cfg: dict,
        product: str,
        area: str,
        inference_type: str,
    ) -> None:
        """Validate model version compatibility.

        Args:
            config: DetectionConfig with weights path
            model_cfg: Model-specific config dict
            product: Product name
            area: Area name
            inference_type: Inference type (yolo/anomalib)

        Raises:
            ModelVersionError: If version is incompatible
        """
        weights_path = getattr(config, "weights", None)
        if not weights_path:
            return

        # Parse version from filename
        current_version = parse_model_version(weights_path)
        if not current_version:
            # Legacy non-versioned model, skip validation
            return

        # Check minimum supported version (if specified in config)
        min_version_str = model_cfg.get("min_supported_version")
        if min_version_str:
            try:
                min_version = parse_version_string(min_version_str)
                if not check_compatibility(current_version, min_version):
                    raise ModelVersionError(
                        f"Model version {version_to_string(current_version)} "
                        f"is below minimum supported version {min_version_str} "
                        f"for {product}/{area}/{inference_type}"
                    )
                self.logger.logger.info(
                    f"Model version validated: {version_to_string(current_version)} "
                    f">= {min_version_str} for {product}/{area}/{inference_type}"
                )
            except ValueError as e:
                self.logger.logger.warning(
                    f"Invalid min_supported_version format: {min_version_str}: {e}"
                )
        else:
            # Just log the version
            self.logger.logger.info(
                f"Loaded model version: {version_to_string(current_version)} "
                f"for {product}/{area}/{inference_type}"
            )
