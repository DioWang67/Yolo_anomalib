from __future__ import annotations

import copy
import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml  # type: ignore[import]

from core.config import DetectionConfig
from core.config_validation import validate_model_cfg
from core.exceptions import ModelConfigError
from core.logging_config import DetectionLogger
from core.path_utils import project_root, resolve_path
from core.security import resolve_output_dir, safe_segment
from core.version_utils import (
    ModelVersionError,
    check_compatibility,
    parse_model_version,
    parse_version_string,
    version_to_string,
)

if TYPE_CHECKING:  # pragma: no cover
    from core.inference_engine import InferenceEngine

EngineFactory = Callable[[DetectionConfig], "InferenceEngine"]

# Repository root (two levels up from core/services)
# Determine repository root (can be overridden by YOLO11_ROOT env var)
PROJECT_ROOT = project_root()

_NON_ENGINE_CONFIG_FIELDS = {
    "exposure_time",
    "gain",
    "light_brightness",
    "calibration",
    "output_dir",
    "save_original",
    "save_processed",
    "save_annotated",
    "save_crops",
    "save_pass_crops",
    "save_fail_only",
    "jpeg_quality",
    "png_compression",
    "max_crops_per_frame",
    "buffer_limit",
    "storage_queue_maxsize",
    "image_queue_maxsize",
    "image_queue_max_mb",
    "image_write_timeout_seconds",
    "min_free_disk_mb",
    "flush_interval",
}


class ModelManager:
    def __init__(
        self,
        logger: DetectionLogger,
        max_cache_size: int = 3,
        engine_factory: EngineFactory | None = None,
        models_root: str | Path | None = None,
        output_root: str | Path | None = None,
        model_config_overrides: Mapping[
            tuple[str, str, str], str | Path
        ]
        | None = None,
    ) -> None:
        """Create a model manager.

        Args:
            logger: DetectionLogger wrapper
            max_cache_size: Max number of (product, area) entries to keep
            engine_factory: Optional inference-engine constructor. Production
                uses the real engine lazily; tests can inject a lightweight
                implementation without importing native ML runtimes.
            models_root: Optional explicit models directory. Offline candidate
                validation uses a job-scoped root so it never reads or mutates
                the deployed station bundle.
            output_root: Writable root used to resolve and validate model-level
                result paths. Production injects the station-data root; the
                project root remains the compatibility default for callers
                that do not persist station results.
            model_config_overrides: Optional exact config paths keyed by
                ``(product, area, inference_type)``. Acceptance matrices use
                immutable historical config snapshots without changing the
                deployed ``config.yaml`` pointer.
        """
        self.logger = logger
        self.max_cache_size = max_cache_size
        self._engine_factory = engine_factory
        self._models_root = (
            Path(models_root).expanduser().resolve()
            if models_root is not None
            else None
        )
        self._output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else PROJECT_ROOT
        )
        self._model_config_overrides = self._validate_config_overrides(
            model_config_overrides or {}
        )
        self._cache_lock = threading.Lock()
        # Engine construction may load native runtimes and must not run while
        # holding the cache lock. Serializing the rare activation path avoids
        # duplicate candidates while ordinary cache reads remain concurrent.
        self._activation_lock = threading.Lock()
        # cache key: (product, area) -> { type: (engine, config_snapshot) }
        self._cache: OrderedDict[
            tuple[str, str], dict[str, tuple[InferenceEngine, DetectionConfig]]
        ] = OrderedDict()
        self._cache_signatures: dict[
            tuple[str, str, str], str
        ] = {}

    @staticmethod
    def _validate_config_overrides(
        overrides: Mapping[tuple[str, str, str], str | Path],
    ) -> dict[tuple[str, str, str], Path]:
        validated: dict[tuple[str, str, str], Path] = {}
        for raw_key, raw_path in overrides.items():
            if not isinstance(raw_key, tuple) or len(raw_key) != 3:
                raise ModelConfigError(
                    "Model config override keys must be "
                    "(product, area, inference_type)."
                )
            product, area, inference_type = (
                safe_segment(str(value), field_name=label)
                for value, label in zip(
                    raw_key,
                    ("product", "area", "inference_type"),
                    strict=True,
                )
            )
            key = (product, area, inference_type.lower())
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or path.is_symlink():
                raise ModelConfigError(
                    f"Model config override is unavailable: {path}"
                )
            validated[key] = path
        return validated

    @staticmethod
    def _engine_config_signature(
        cfg: dict,
        merged: DetectionConfig,
        inference_type: str,
    ) -> str:
        """Hash only fields that require rebuilding an inference backend.

        Camera calibration and result-storage settings are consumed outside
        the engine.  Excluding them prevents an exposure/light adjustment from
        unloading and reloading a multi-megabyte model.
        """
        engine_cfg = {
            key: value
            for key, value in cfg.items()
            if key not in _NON_ENGINE_CONFIG_FIELDS
        }
        weights_path = Path(str(getattr(merged, "weights", "") or ""))
        weight_identity: tuple[int, int, int] | None = None
        try:
            weight_stat = weights_path.stat()
            weight_identity = (
                weight_stat.st_mtime_ns,
                weight_stat.st_size,
                weight_stat.st_ino,
            )
        except OSError:
            pass
        serialized = json.dumps(
            {
                "backend": inference_type,
                "config": engine_cfg,
                "weight_identity": weight_identity,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _create_engine(self, config: DetectionConfig) -> InferenceEngine:
        """Construct an engine without importing native runtimes at module load."""
        if self._engine_factory is not None:
            return self._engine_factory(config)

        from core.inference_engine import InferenceEngine

        return InferenceEngine(config)

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
        """Apply per-model overrides onto a private DetectionConfig copy.

        ``base_config`` here is always the caller-owned merged copy created
        in :meth:`switch` — never the shared global config.
        """
        # --- Simple scalar overrides ---
        _SCALAR_FIELDS = [
            "device", "conf_thres", "iou_thres", "enable_yolo", "enable_anomalib",
            "enable_color_check", "color_fail_closed", "enable_custom_backends",
        ]
        for field in _SCALAR_FIELDS:
            if field in cfg and cfg.get(field) is not None:
                setattr(base_config, field, cfg[field])

        # --- Fields that only apply when present and non-None ---
        _OPTIONAL_FIELDS = [
            "exposure_time", "gain", "light_brightness", "calibration",
            "expected_items", "position_config", "anomalib_config",
            "color_threshold_overrides", "color_rules_overrides",
            "backends", "pipeline", "defect_coverage",
        ]
        for field in _OPTIONAL_FIELDS:
            if field in cfg and cfg.get(field) is not None:
                setattr(base_config, field, cfg[field])

        # Bounded operational settings may be overridden per model.  Schema
        # validation owns their numeric ranges; absent values keep the global
        # station defaults.
        _OPERATIONAL_FIELDS = [
            "buffer_limit",
            "storage_queue_maxsize",
            "image_queue_maxsize",
            "image_queue_max_mb",
            "image_write_timeout_seconds",
            "min_free_disk_mb",
            "flush_interval",
        ]
        for field in _OPERATIONAL_FIELDS:
            if cfg.get(field) is not None:
                setattr(base_config, field, cfg[field])

        # --- imgsz needs tuple conversion ---
        if "imgsz" in cfg and cfg.get("imgsz") is not None:
            base_config.imgsz = tuple(cfg["imgsz"])  # type: ignore[arg-type]

        # --- output_dir: all inspection outputs stay under the injected root ---
        if "output_dir" in cfg:
            raw_output_dir = cfg.get("output_dir")
            if raw_output_dir:
                path_str = str(raw_output_dir).strip()
                if path_str:
                    resolved = resolve_output_dir(
                        path_str,
                        base_dir=self._output_root,
                        allowed_root=self._output_root,
                    )
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

    def _locate_model_config(
        self,
        product: str,
        area: str,
        inference_type: str,
        config_path_override: str | Path | None = None,
    ) -> str:
        """Locate models/<product>/<area>/<type>/config.yaml.

        Search order: current working directory first (backward compatible
        with existing callers and tests), then the project root, so the
        bundle is still found when the app is launched from another cwd or
        as a frozen executable.

        Raises:
            FileNotFoundError: If the config exists in neither location.
        """
        relative = Path(product) / area / inference_type / "config.yaml"
        if config_path_override is not None:
            exact = Path(config_path_override).expanduser().resolve()
            if exact.is_symlink() or not exact.is_file():
                raise ModelConfigError(
                    f"Model config override is unavailable: {exact}"
                )
            return str(exact)
        override = self._model_config_overrides.get(
            (product, area, inference_type)
        )
        if override is not None:
            return str(override)
        candidates = (
            [self._models_root / relative]
            if self._models_root is not None
            else [
                Path.cwd() / "models" / relative,
                PROJECT_ROOT / "models" / relative,
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Model config not found: {relative} "
            f"(searched: {[str(c) for c in candidates]})"
        )

    def switch(
        self,
        base_config: DetectionConfig,
        product: str,
        area: str,
        inference_type: str,
        *,
        config_path_override: str | Path | None = None,
    ) -> tuple[InferenceEngine, DetectionConfig]:
        """Switch engine to (product, area, type), with LRU cache.

        ``base_config`` is treated as read-only: overrides are applied onto a
        deep copy, so the shared global config is never mutated mid-switch
        (workers and the GUI may be reading it concurrently). Callers adopt
        the returned merged config by assignment, which is an atomic
        reference swap.

        Returns:
            The inference engine and a caller-owned merged DetectionConfig.
        """
        safe_product = safe_segment(product, field_name="product")
        safe_area = safe_segment(area, field_name="area")
        safe_inference_type = safe_segment(
            inference_type.lower(), field_name="inference_type"
        )
        key = (safe_product, safe_area)
        model_config_path = self._locate_model_config(
            safe_product,
            safe_area,
            safe_inference_type,
            config_path_override=config_path_override,
        )
        signature_key = (safe_product, safe_area, safe_inference_type)

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
                cfg or {},
                safe_product,
                safe_area,
                selected_backend=safe_inference_type,
                model_cfg_dir=model_cfg_dir,
            )
        except Exception as e:
            self.logger.logger.error(f"Model config validation failed: {e}")
            raise

        context = f"{safe_product}/{safe_area}/{safe_inference_type}"
        merged = copy.deepcopy(base_config)
        self._apply_model_config(merged, cfg, context, model_cfg_dir=model_cfg_dir)
        if safe_inference_type == "yolo":
            self._validate_inspection_scope(merged, safe_product, safe_area, context)

        # Version validation (if model uses versioned naming)
        self._validate_model_version(
            merged, cfg, safe_product, safe_area, safe_inference_type
        )

        config_signature = self._engine_config_signature(
            cfg,
            merged,
            safe_inference_type,
        )
        with self._cache_lock:
            if key in self._cache and safe_inference_type in self._cache[key]:
                engine, _ = self._cache[key][safe_inference_type]
                if self._cache_signatures.get(signature_key) == config_signature:
                    self.logger.logger.info(
                        "Using cached model: product=%s, area=%s, type=%s",
                        safe_product,
                        safe_area,
                        safe_inference_type,
                    )
                    self._cache.move_to_end(key)
                    return engine, merged

        with self._activation_lock:
            # Another caller may have completed the same activation while this
            # caller waited. Recheck before loading a second native backend.
            with self._cache_lock:
                cached = self._cache.get(key, {}).get(safe_inference_type)
                if (
                    cached is not None
                    and self._cache_signatures.get(signature_key)
                    == config_signature
                ):
                    self._cache.move_to_end(key)
                    return cached[0], merged
                stale_engine = cached[0] if cached is not None else None

            if stale_engine is not None:
                self.logger.logger.info(
                    "Inference engine settings changed; preparing replacement "
                    "for %s/%s/%s",
                    safe_product,
                    safe_area,
                    safe_inference_type,
                )

            # The old engine remains reachable until every candidate
            # initialization step succeeds. This is the activation rollback.
            candidate = self._create_engine(merged)
            try:
                if not candidate.initialize():
                    raise RuntimeError("Inference engine init failed")
                if safe_inference_type == "anomalib":
                    self._initialize_product_models(merged, safe_product)
            except (ImportError, OSError, RuntimeError, TypeError, ValueError):
                try:
                    candidate.shutdown()
                except (OSError, RuntimeError, TypeError, ValueError):
                    self.logger.logger.warning(
                        "Failed candidate engine cleanup raised an error",
                        exc_info=True,
                    )
                raise

            engines_to_shutdown: list[InferenceEngine] = []
            evicted_key: tuple[str, str] | None = None
            with self._cache_lock:
                previous = self._cache.get(key, {}).get(safe_inference_type)
                if key not in self._cache:
                    self._cache[key] = {}
                self._cache[key][safe_inference_type] = (
                    candidate,
                    copy.deepcopy(merged),
                )
                self._cache_signatures[signature_key] = config_signature
                self._cache.move_to_end(key)
                if previous is not None and previous[0] is not candidate:
                    engines_to_shutdown.append(previous[0])

                if len(self._cache) > self.max_cache_size:
                    evicted_key, evicted_engines = self._cache.popitem(last=False)
                    for backend, (evicted_engine, _) in evicted_engines.items():
                        self._cache_signatures.pop(
                            (evicted_key[0], evicted_key[1], backend), None
                        )
                        if evicted_engine is not candidate:
                            engines_to_shutdown.append(evicted_engine)

            # Native shutdown can block, so it runs outside all synchronization
            # locks after the new engine is already the active cache entry.
            retired_ids: set[int] = set()
            for old_engine in engines_to_shutdown:
                if id(old_engine) in retired_ids:
                    continue
                retired_ids.add(id(old_engine))
                try:
                    old_engine.shutdown()
                except (OSError, RuntimeError, TypeError, ValueError):
                    self.logger.logger.warning(
                        "Retired inference engine shutdown failed",
                        exc_info=True,
                    )
            if evicted_key is not None:
                self.logger.logger.info(
                    "Evicted cached model: product=%s, area=%s",
                    evicted_key[0],
                    evicted_key[1],
                )

            return candidate, merged

    def _validate_inspection_scope(
        self,
        config: DetectionConfig,
        product: str,
        area: str,
        context: str,
    ) -> None:
        """Fail fast when a model bundle cannot define the inspection scope."""
        expected_items = config.get_items_by_area(product, area)
        if expected_items:
            return
        raise ModelConfigError(
            "Model config missing expected_items for "
            f"{product}/{area} ({context}). "
            "Add expected_items.<product>.<area> to the model config."
        )

    def get_cached_engine(
        self, product: str, area: str, inference_type: str
    ) -> InferenceEngine | None:
        """Return a cached engine without exposing the internal cache layout.

        Args:
            product: Product name.
            area: Area/station name.
            inference_type: Backend type, for example ``yolo`` or ``anomalib``.

        Returns:
            The cached inference engine, or ``None`` when it has not been loaded.
        """
        safe_product = safe_segment(product, field_name="product")
        safe_area = safe_segment(area, field_name="area")
        safe_inference_type = safe_segment(
            inference_type.lower(), field_name="inference_type"
        )
        with self._cache_lock:
            engines = self._cache.get((safe_product, safe_area), {})
            cached = engines.get(safe_inference_type)
            return cached[0] if cached else None

    def clear_cache(
        self,
        product: str | None = None,
        area: str | None = None,
        inference_type: str | None = None,
    ) -> None:
        """Shutdown and remove cached engines.

        Args:
            product: Optional product filter.
            area: Optional area filter.
            inference_type: Optional backend filter.
        """
        target_product = (
            safe_segment(product, field_name="product") if product else None
        )
        target_area = safe_segment(area, field_name="area") if area else None
        target_type = (
            safe_segment(inference_type.lower(), field_name="inference_type")
            if inference_type
            else None
        )
        with self._cache_lock:
            for key in list(self._cache.keys()):
                key_product, key_area = key
                if target_product is not None and key_product != target_product:
                    continue
                if target_area is not None and key_area != target_area:
                    continue

                engines = self._cache[key]
                for backend in list(engines.keys()):
                    if target_type is not None and backend.lower() != target_type:
                        continue
                    engine, _ = engines.pop(backend)
                    self._cache_signatures.pop(
                        (key_product, key_area, backend), None
                    )
                    try:
                        engine.shutdown()
                    except Exception:
                        pass
                if not engines:
                    self._cache.pop(key, None)

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
