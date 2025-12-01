from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # pragma: no cover - depends on pydantic availability/version
    from pydantic import BaseModel, Field

    try:  # pydantic v2
        from pydantic import ConfigDict, field_validator as _field_validator

        _VALIDATOR_MODE = "v2"
    except ImportError:  # pragma: no cover - pydantic v1 fallback
        ConfigDict = None  # type: ignore
        from pydantic import validator as _validator  # type: ignore

        _VALIDATOR_MODE = "v1"
except Exception:  # pragma: no cover - pydantic not installed
    try:
        from pydantic.v1 import BaseModel, Field, validator as _validator  # type: ignore

        ConfigDict = None  # type: ignore
        _VALIDATOR_MODE = "v1"
    except Exception:  # pragma: no cover - no pydantic available
        BaseModel = None  # type: ignore
        ConfigDict = None  # type: ignore
        _VALIDATOR_MODE = None


def _to_dict(model: Any) -> Dict[str, Any]:  # compatible dump helper
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _normalize_sequence(value: Any, *, expect_len: int | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if expect_len is not None and len(value) != expect_len:
            raise ValueError(f"expected sequence length {expect_len}")
        return [v for v in value]
    raise ValueError("expected list/tuple")


def _normalize_output_dir(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    raise ValueError("output_dir must be a non-empty string")


if BaseModel is not None:  # pragma: no cover - runtime optional

    class GlobalConfigSchema(BaseModel):
        weights: str
        device: Optional[str] = "cpu"
        conf_thres: Optional[float] = 0.25
        iou_thres: Optional[float] = 0.45
        imgsz: Optional[List[int]] = Field(default_factory=lambda: [640, 640])
        timeout: Optional[int] = 2
        exposure_time: Optional[str] = "1000"
        gain: Optional[str] = "1.0"
        width: Optional[int] = 640
        height: Optional[int] = 640
        MV_CC_GetImageBuffer_nMsec: Optional[int] = 10000
        current_product: Optional[str] = None
        current_area: Optional[str] = None
        expected_items: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
        enable_yolo: Optional[bool] = True
        enable_anomalib: Optional[bool] = False
        enable_color_check: Optional[bool] = False
        color_model_path: Optional[str] = None
        color_threshold_overrides: Optional[Dict[str, float]] = None
        color_rules_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None
        color_checker_type: Optional[str] = "color_qc"
        color_score_threshold: Optional[float] = None
        output_dir: Optional[str] = "Result"
        anomalib_config: Optional[Dict[str, Any]] = None
        position_config: Dict[str, Dict[str, Dict[str, Any]]] = Field(
            default_factory=dict
        )
        max_cache_size: Optional[int] = 3
        buffer_limit: Optional[int] = 10
        flush_interval: Optional[float] = None
        pipeline: Optional[List[str]] = None
        steps: Dict[str, Any] = Field(default_factory=dict)
        backends: Optional[Dict[str, Dict[str, Any]]] = None
        disable_internal_cache: Optional[bool] = True
        save_original: Optional[bool] = True
        save_processed: Optional[bool] = True
        save_annotated: Optional[bool] = True
        save_crops: Optional[bool] = True
        save_fail_only: Optional[bool] = False
        jpeg_quality: Optional[int] = 95
        png_compression: Optional[int] = 3
        max_crops_per_frame: Optional[int] = None
        fail_on_unexpected: Optional[bool] = True

        if _VALIDATOR_MODE == "v2":
            # type: ignore[assignment]
            model_config = ConfigDict(extra="allow")

            @_field_validator("imgsz", mode="before")  # type: ignore[misc]
            def _coerce_imgsz(cls, value: Any) -> Any:
                return _normalize_sequence(value, expect_len=2)

            @_field_validator("pipeline", mode="before")  # type: ignore[misc]
            def _coerce_pipeline(cls, value: Any) -> Any:
                if value is None:
                    return None
                return _normalize_sequence(value)

            @_field_validator("output_dir", mode="before")  # type: ignore[misc]
            def _validate_output_dir(cls, value: Any) -> Any:
                return _normalize_output_dir(value)

        elif _VALIDATOR_MODE == "v1":

            class Config:
                extra = "allow"

            @_validator("imgsz", pre=True)  # type: ignore[misc]
            def _coerce_imgsz(cls, value: Any) -> Any:
                return _normalize_sequence(value, expect_len=2)

            @_validator("pipeline", pre=True)  # type: ignore[misc]
            def _coerce_pipeline(cls, value: Any) -> Any:
                if value is None:
                    return None
                return _normalize_sequence(value)

            @_validator("output_dir", pre=True)  # type: ignore[misc]
            def _validate_output_dir(cls, value: Any) -> Any:
                return _normalize_output_dir(value)

    class ModelConfigSchema(BaseModel):
        device: Optional[str] = None
        weights: Optional[str] = None
        conf_thres: Optional[float] = None
        iou_thres: Optional[float] = None
        imgsz: Optional[List[int]] = None
        timeout: Optional[int] = None
        exposure_time: Optional[str] = None
        gain: Optional[str] = None
        width: Optional[int] = None
        height: Optional[int] = None
        MV_CC_GetImageBuffer_nMsec: Optional[int] = None
        output_dir: Optional[str] = None
        enable_yolo: Optional[bool] = None
        enable_anomalib: Optional[bool] = None
        enable_color_check: Optional[bool] = None
        color_model_path: Optional[str] = None
        color_checker_type: Optional[str] = None
        color_score_threshold: Optional[float] = None
        expected_items: Optional[Dict[str, Dict[str, List[str]]]] = None
        position_config: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
        anomalib_config: Optional[Dict[str, Any]] = None
        backends: Optional[Dict[str, Dict[str, Any]]] = None
        pipeline: Optional[List[str]] = None
        steps: Dict[str, Any] = Field(default_factory=dict)
        disable_internal_cache: Optional[bool] = None
        save_original: Optional[bool] = None
        save_processed: Optional[bool] = None
        save_annotated: Optional[bool] = None
        save_crops: Optional[bool] = None
        save_fail_only: Optional[bool] = None
        jpeg_quality: Optional[int] = None
        png_compression: Optional[int] = None
        max_crops_per_frame: Optional[int] = None
        fail_on_unexpected: Optional[bool] = None
        buffer_limit: Optional[int] = None
        flush_interval: Optional[float] = None
        max_cache_size: Optional[int] = None

        if _VALIDATOR_MODE == "v2":
            # type: ignore[assignment]
            model_config = ConfigDict(extra="allow")

            @_field_validator("imgsz", mode="before")  # type: ignore[misc]
            def _coerce_imgsz(cls, value: Any) -> Any:
                return _normalize_sequence(value, expect_len=2)

            @_field_validator("pipeline", mode="before")  # type: ignore[misc]
            def _coerce_pipeline(cls, value: Any) -> Any:
                if value is None:
                    return None
                return _normalize_sequence(value)

            @_field_validator("output_dir", mode="before")  # type: ignore[misc]
            def _validate_model_output_dir(cls, value: Any) -> Any:
                if value is None:
                    return None
                return _normalize_output_dir(value)

        elif _VALIDATOR_MODE == "v1":

            class Config:
                extra = "allow"

            @_validator("imgsz", pre=True)  # type: ignore[misc]
            def _coerce_imgsz(cls, value: Any) -> Any:
                return _normalize_sequence(value, expect_len=2)

            @_validator("pipeline", pre=True)  # type: ignore[misc]
            def _coerce_pipeline(cls, value: Any) -> Any:
                if value is None:
                    return None
                return _normalize_sequence(value)

            @_validator("output_dir", pre=True)  # type: ignore[misc]
            def _validate_model_output_dir(cls, value: Any) -> Any:
                if value is None:
                    return None
                return _normalize_output_dir(value)

else:
    GlobalConfigSchema = None  # type: ignore
    ModelConfigSchema = None  # type: ignore
