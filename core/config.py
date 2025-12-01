from __future__ import annotations

"""負責載入並驗證偵測流程與後端設定的配置管理器。"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    schema: Any, payload: Dict[str, Any] | None, source: str, scope: str
) -> Dict[str, Any]:
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
            (
                f"{scope} configuration invalid ({source}): "
                f"{_format_validation_error(exc)}"
            )
        ) from exc
    return _to_dict(normalized)  # type: ignore[func-returns-value]


def _coerce_imgsz(
    value: Any, *, default: Optional[Tuple[int, int]] = None
) -> Optional[Tuple[int, int]]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ConfigValidationError("imgsz must contain exactly two numeric values")


@dataclass
class DetectionConfig:
    """Shared runtime configuration.

    Loaded from global config.yaml and then overridden by per-model configs
    (models/<product>/<area>/<type>/config.yaml). This instance is mutated
    in-place by ModelManager.switch() to reflect the active model settings.
    """

    weights: str
    device: str = "cpu"
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    imgsz: Tuple[int, int] = (640, 640)
    timeout: int = 2
    exposure_time: str = "1000"
    gain: str = "1.0"
    width: int = 3072
    height: int = 2048
    MV_CC_GetImageBuffer_nMsec: int = 10000
    current_product: Optional[str] = None
    current_area: Optional[str] = None
    expected_items: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    enable_yolo: bool = True
    enable_anomalib: bool = False
    enable_color_check: bool = False
    color_model_path: Optional[str] = None
    color_threshold_overrides: Optional[Dict[str, float]] = None
    # Optional per-color rules overrides:
    # { ColorName: { s_p90_max, s_p10_min, v_p50_min, v_p95_max } }
    color_rules_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    color_checker_type: str = "color_qc"
    color_score_threshold: Optional[float] = None
    output_dir: str = "Result"
    anomalib_config: Optional[Dict[str, Any]] = None
    position_config: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    max_cache_size: int = 3
    buffer_limit: int = 1
    flush_interval: Optional[float] = None
    pipeline: Optional[List[str]] = None
    steps: Dict[str, Any] = field(default_factory=dict)
    backends: Optional[Dict[str, Dict[str, Any]]] = None  # extra/custom backends
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
    max_crops_per_frame: Optional[int] = None
    fail_on_unexpected: bool = True

    @classmethod
    def normalize_global_dict(
        cls, payload: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        normalized = _normalize_with_schema(
            GlobalConfigSchema, payload, source, "Global"
        )
        normalized["imgsz"] = _coerce_imgsz(normalized.get("imgsz"), default=(640, 640))
        return normalized

    @staticmethod
    def normalize_model_dict(payload: Dict[str, Any], source: str) -> Dict[str, Any]:
        normalized = _normalize_with_schema(ModelConfigSchema, payload, source, "Model")
        normalized["imgsz"] = _coerce_imgsz(normalized.get("imgsz"), default=None)
        return normalized

    @classmethod
    def from_yaml(cls, path: str) -> "DetectionConfig":
        """Load and validate the global configuration YAML file."""
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

        kwargs: Dict[str, Any] = {
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

    def get_items_by_area(self, product: str, area: str) -> Optional[List[str]]:
        return self.expected_items.get(product, {}).get(area)

    def get_position_config(self, product: str, area: str) -> Optional[Dict[str, Any]]:
        return self.position_config.get(product, {}).get(area)

    def is_position_check_enabled(self, product: str, area: str) -> bool:
        config = self.get_position_config(product, area)
        return config is not None and bool(config.get("enabled", False))

    def get_tolerance_ratio(self, product: str, area: str) -> float:
        """Return tolerance ratio for a given product/area (0.05 == 5%)."""
        config = self.get_position_config(product, area)
        if config is None:
            return 0.0

        val = config.get("tolerance", 0)
        if val is None or val <= 0:
            return 0.0
        if val <= 100:
            return float(val) / 100.0  # convert percentage to decimal
        return 0.0
