from __future__ import annotations

"""負責載入並驗證偵測流程與後端設定的配置管理器。"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:  # pragma: no cover - runtime optional depending on pydantic version
    from pydantic import ValidationError as _ValidationError  # type: ignore
except Exception:  # pragma: no cover
    try:
        from pydantic.v1 import (
            ValidationError as _ValidationError,  # type: ignore
        )
    except Exception:  # pragma: no cover
        _ValidationError = None  # type: ignore

try:  # schema helpers (require pydantic)
    from .config_schema import (
        GlobalConfigSchema,
        ModelConfigSchema,
        _to_dict,  # type: ignore
    )
except Exception:  # pragma: no cover
    GlobalConfigSchema = None  # type: ignore
    ModelConfigSchema = None  # type: ignore
    _to_dict = None  # type: ignore


class ConfigError(RuntimeError):
    """Base class for configuration related failures."""


class ConfigLoadError(ConfigError):
    """Raised when a configuration file cannot be read or parsed."""


class ConfigValidationError(ConfigError):
    """Raised when a configuration payload fails validation."""


def _format_validation_error(exc: Exception) -> str:
    if _ValidationError is not None and isinstance(exc, _ValidationError):
        try:
            lines = []
            for err in exc.errors():  # type: ignore[attr-defined]
                loc = ".".join(str(x) for x in err.get("loc", ()))
                msg = err.get("msg", str(exc))
                lines.append(f"{loc}: {msg}" if loc else msg)
            if lines:
                return "; ".join(lines)
        except Exception:
            return str(exc)
    return str(exc)


def _ensure_schema(schema: Any, label: str) -> Any:
    if schema is None or _to_dict is None:
        logger.warning("Pydantic schemas unavailable; skipping %s validation", label)
        return None
    return schema


def _normalize_with_schema(
    schema: Any, payload: dict[str, Any] | None, source: str, scope: str
) -> dict[str, Any]:
    data = payload or {}
    if not isinstance(data, dict):
        raise ConfigValidationError(
            f"{scope} configuration must be a mapping ({source})"
        )
    model = _ensure_schema(schema, scope)
    if model is None:
        return dict(data)
    try:
        normalized = model(**data)
    except Exception as exc:  # pragma: no cover - pydantic error formatting
        raise ConfigValidationError(

                f"{scope} configuration invalid ({source}): "
                f"{_format_validation_error(exc)}"

        ) from exc
    return _to_dict(normalized)  # type: ignore[func-returns-value]


def _coerce_imgsz(
    value: Any, *, default: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ConfigValidationError("imgsz must contain exactly two numeric values")


@dataclass
class DetectionConfig:
    """Unified configuration registry for the detection system.

    This dataclass holds global defaults (from config.yaml) and per-model
    overrides. It governs model parameters (weights, conf_thres), hardware 
    settings (device, camera exposure), and pipeline behavior (position check).

    Attributes:
        weights: Path to the default YOLO weights file.
        device: Computation device ('cpu', 'cuda:0', etc.).
        conf_thres: Confidence threshold for object detection.
        iou_thres: Intersection-over-union threshold for NMS.
        imgsz: Target image size (width, height) for inference.
        expected_items: Nested mapping of Product -> Area -> List of expected classes.
        enable_yolo: Whether to run YOLO detection.
        enable_anomalib: Whether to run anomaly detection.
        position_config: Configuration for position validation rules.
        fail_on_unexpected: If True, detection status is FAIL if unseen classes appear.
    """

    weights: str
    device: str = "cpu"
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: tuple[int, int] = (640, 640)
    timeout: int = 2
    exposure_time: str = "1000"
    gain: str = "1.0"
    width: int = 3072
    height: int = 2048
    MV_CC_GetImageBuffer_nMsec: int = 10000
    current_product: str | None = None
    current_area: str | None = None
    expected_items: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    enable_yolo: bool = True
    enable_anomalib: bool = False
    enable_color_check: bool = False
    color_model_path: str | None = None
    color_threshold_overrides: dict[str, float] | None = None
    # Optional per-color rules overrides:
    # { ColorName: { s_p90_max, s_p10_min, v_p50_min, v_p95_max } }
    color_rules_overrides: dict[str, dict[str, float | None]] | None = None
    color_checker_type: str = "color_qc"
    color_score_threshold: float | None = None
    output_dir: str = "Result"
    anomalib_config: dict[str, Any] | None = None
    position_config: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    max_cache_size: int = 3
    buffer_limit: int = 1
    flush_interval: float | None = None
    pipeline: list[str] | None = None
    steps: dict[str, Any] = field(default_factory=dict)
    backends: dict[str, dict[str, Any]] | None = None  # extra/custom backends
    # Avoid duplicating cache with YOLO internal cache (default: disable)
    disable_internal_cache: bool = True
    # Saving controls
    save_original: bool = True
    save_processed: bool = True
    save_annotated: bool = True
    save_crops: bool = True
    save_fail_only: bool = False
    jpeg_quality: int = 95
    png_compression: int = 3
    max_crops_per_frame: int | None = None
    fail_on_unexpected: bool = True

    @classmethod
    def normalize_global_dict(
        cls, payload: dict[str, Any], source: str
    ) -> dict[str, Any]:
        normalized = _normalize_with_schema(
            GlobalConfigSchema, payload, source, "Global"
        )
        normalized["imgsz"] = _coerce_imgsz(normalized.get("imgsz"), default=(640, 640))
        return normalized

    @staticmethod
    def normalize_model_dict(payload: dict[str, Any], source: str) -> dict[str, Any]:
        normalized = _normalize_with_schema(ModelConfigSchema, payload, source, "Model")
        normalized["imgsz"] = _coerce_imgsz(normalized.get("imgsz"), default=None)
        return normalized

    @classmethod
    def from_yaml(cls, path: str) -> DetectionConfig:
        """Loads and validates a global configuration YAML file.

        Args:
            path: Path to the config.yaml file.

        Returns:
            DetectionConfig: A validated configuration instance.

        Raises:
            ConfigLoadError: If the file is missing or unreadable.
            ConfigValidationError: If the YAML content fails schema validation.
        """
        # Import path validator for security checks
        try:
            from core.security import SecurityError, path_validator
        except ImportError:
            # Fallback if security module is not available
            logger.warning("Security module not available, skipping path validation")
            path_validator = None
            SecurityError = Exception

        # Validate path to prevent directory traversal attacks
        if path_validator is not None:
            try:
                config_path = path_validator.validate_path(path, must_exist=True)
            except SecurityError as exc:
                raise ConfigLoadError(f"Security error loading config: {exc}") from exc
            except FileNotFoundError as exc:
                raise ConfigLoadError(f"Config file not found: {path}") from exc
        else:
            config_path = Path(path)
            if not config_path.exists():
                raise ConfigLoadError(f"Config file not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except yaml.YAMLError as exc:  # pragma: no cover -
            # unlikely parse error
            raise ConfigLoadError(f"Failed to parse YAML {config_path}: {exc}") from exc
        except OSError as exc:
            raise ConfigLoadError(
                f"Unable to read config {config_path}: {exc}"
            ) from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigValidationError(
                f"Global configuration must be a mapping ({config_path})"
            )

        normalized = cls.normalize_global_dict(raw, str(config_path))
        weights = normalized.get("weights")
        if not weights:
            raise ConfigValidationError(
                f"'weights' is required in global config ({config_path})"
            )

        pipeline_raw = normalized.get("pipeline")
        pipeline_value = (
            list(pipeline_raw) if isinstance(pipeline_raw, (list, tuple)) else None
        )
        backends_raw = normalized.get("backends")
        backends_value = dict(backends_raw) if isinstance(backends_raw, dict) else None

        kwargs: dict[str, Any] = {
            "weights": str(weights),
            "device": normalized.get("device", "cpu"),
            "conf_thres": float(normalized.get("conf_thres", 0.25)),
            "iou_thres": float(normalized.get("iou_thres", 0.45)),
            "imgsz": _coerce_imgsz(normalized.get("imgsz"), default=(640, 640))
            or (640, 640),
            "timeout": int(normalized.get("timeout", 2)),
            "exposure_time": str(normalized.get("exposure_time", "1000")),
            "gain": str(normalized.get("gain", "1.0")),
            "width": int(normalized.get("width", 640)),
            "height": int(normalized.get("height", 640)),
            "MV_CC_GetImageBuffer_nMsec": int(
                normalized.get("MV_CC_GetImageBuffer_nMsec", 10000)
            ),
            "current_product": normalized.get("current_product"),
            "current_area": normalized.get("current_area"),
            "expected_items": dict(normalized.get("expected_items", {})),
            "enable_yolo": bool(normalized.get("enable_yolo", True)),
            "enable_anomalib": bool(normalized.get("enable_anomalib", False)),
            "enable_color_check": bool(normalized.get("enable_color_check", False)),
            "color_model_path": normalized.get("color_model_path"),
            "color_threshold_overrides": normalized.get("color_threshold_overrides"),
            "color_rules_overrides": normalized.get("color_rules_overrides"),
            "color_checker_type": str(
                normalized.get("color_checker_type") or "color_qc"
            ),
            "color_score_threshold": normalized.get("color_score_threshold"),
            "output_dir": str(normalized.get("output_dir", "Result")),
            "anomalib_config": normalized.get("anomalib_config"),
            "position_config": dict(normalized.get("position_config", {})),
            "max_cache_size": int(normalized.get("max_cache_size", 3)),
            "buffer_limit": int(normalized.get("buffer_limit", 10)),
            "flush_interval": normalized.get("flush_interval"),
            "pipeline": pipeline_value,
            "steps": dict(normalized.get("steps", {})),
            "backends": backends_value,
            "disable_internal_cache": bool(
                normalized.get("disable_internal_cache", True)
            ),
            "save_original": bool(normalized.get("save_original", True)),
            "save_processed": bool(normalized.get("save_processed", True)),
            "save_annotated": bool(normalized.get("save_annotated", True)),
            "save_crops": bool(normalized.get("save_crops", True)),
            "save_fail_only": bool(normalized.get("save_fail_only", False)),
            "jpeg_quality": int(normalized.get("jpeg_quality", 95)),
            "png_compression": int(normalized.get("png_compression", 3)),
            "max_crops_per_frame": normalized.get("max_crops_per_frame"),
            "fail_on_unexpected": bool(normalized.get("fail_on_unexpected", True)),
        }

        max_crops = kwargs.get("max_crops_per_frame")
        if max_crops is not None:
            kwargs["max_crops_per_frame"] = int(max_crops)

        return cls(**kwargs)

    def get_items_by_area(self, product: str, area: str) -> list[str] | None:
        return self.expected_items.get(product, {}).get(area)

    def get_position_config(self, product: str, area: str) -> dict[str, Any]:
        """Retrieves position validation settings for a product area.

        Args:
            product: Product name.
            area: Area name.

        Returns:
            dict: The 'position_check' dictionary from the model config.
        """
        return self.position_config.get(product, {}).get(area, {})

    def is_position_check_enabled(self, product: str, area: str) -> bool:
        """Checks if position validation is active for a product area.

        Args:
            product: Product name.
            area: Area name.

        Returns:
            bool: True if 'enabled' is True in the position config.
        """
        cfg = self.get_position_config(product, area)
        return bool(cfg.get("enabled", False))

    def get_tolerance_ratio(self, product: str, area: str) -> float:
        """Retrieves the tolerance ratio for position validation.

        Args:
            product: Product name.
            area: Area name.

        Returns:
            float: Tolerance ratio (e.g., 0.05 for 5%). Defaults to 0.05.
        """
        config = self.get_position_config(product, area)
        if not config:  # Use 'not config' to handle empty dict as well as None
            return 0.05 # Default to 0.05 if no config or empty config

        val = config.get("tolerance", 5) # Default to 5% if 'tolerance' key is missing
        if val is None or val <= 0:
            return 0.0
        if val <= 100:
            return float(val) / 100.0  # convert percentage to decimal
        return 0.0
