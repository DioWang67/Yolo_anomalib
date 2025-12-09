from __future__ import annotations

from typing import Any

try:  # pragma: no cover - depends on pydantic availability/version
    from pydantic import BaseModel, Field

    try:  # pydantic v2
        from pydantic import ConfigDict
        from pydantic import field_validator as _field_validator

        _VALIDATOR_MODE = "v2"
    except ImportError:  # pragma: no cover - pydantic v1 fallback
        ConfigDict = None  # type: ignore
        from pydantic import validator as _validator  # type: ignore

        _VALIDATOR_MODE = "v1"
except Exception:  # pragma: no cover - pydantic not installed
    try:
        from pydantic.v1 import BaseModel, Field  # type: ignore
        from pydantic.v1 import validator as _validator

        ConfigDict = None  # type: ignore
        _VALIDATOR_MODE = "v1"
    except Exception:  # pragma: no cover - no pydantic available
        BaseModel = None  # type: ignore
        ConfigDict = None  # type: ignore
        _VALIDATOR_MODE = None


def _to_dict(model: Any) -> dict[str, Any]:  # compatible dump helper
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
        device: str | None = "cpu"
        conf_thres: float | None = 0.25
        iou_thres: float | None = 0.45
        imgsz: list[int] | None = Field(default_factory=lambda: [640, 640])
        timeout: int | None = 2
        exposure_time: str | None = "1000"
        gain: str | None = "1.0"
        width: int | None = 640
        height: int | None = 640
        MV_CC_GetImageBuffer_nMsec: int | None = 10000
        current_product: str | None = None
        current_area: str | None = None
        expected_items: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
        enable_yolo: bool | None = True
        enable_anomalib: bool | None = False
        enable_color_check: bool | None = False
        color_model_path: str | None = None
        color_threshold_overrides: dict[str, float] | None = None
        color_rules_overrides: dict[str, dict[str, float | None]] | None = None
        color_checker_type: str | None = "color_qc"
        color_score_threshold: float | None = None
        output_dir: str | None = "Result"
        anomalib_config: dict[str, Any] | None = None
        position_config: dict[str, dict[str, dict[str, Any]]] = Field(
            default_factory=dict
        )
        max_cache_size: int | None = 3
        buffer_limit: int | None = 10
        flush_interval: float | None = None
        pipeline: list[str] | None = None
        steps: dict[str, Any] = Field(default_factory=dict)
        backends: dict[str, dict[str, Any]] | None = None
        disable_internal_cache: bool | None = True
        save_original: bool | None = True
        save_processed: bool | None = True
        save_annotated: bool | None = True
        save_crops: bool | None = True
        save_fail_only: bool | None = False
        jpeg_quality: int | None = 95
        png_compression: int | None = 3
        max_crops_per_frame: int | None = None
        fail_on_unexpected: bool | None = True

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
        device: str | None = None
        weights: str | None = None
        conf_thres: float | None = None
        iou_thres: float | None = None
        imgsz: list[int] | None = None
        timeout: int | None = None
        exposure_time: str | None = None
        gain: str | None = None
        width: int | None = None
        height: int | None = None
        MV_CC_GetImageBuffer_nMsec: int | None = None
        output_dir: str | None = None
        enable_yolo: bool | None = None
        enable_anomalib: bool | None = None
        enable_color_check: bool | None = None
        color_model_path: str | None = None
        color_checker_type: str | None = None
        color_score_threshold: float | None = None
        expected_items: dict[str, dict[str, list[str]]] | None = None
        position_config: dict[str, dict[str, dict[str, Any]]] | None = None
        anomalib_config: dict[str, Any] | None = None
        backends: dict[str, dict[str, Any]] | None = None
        pipeline: list[str] | None = None
        steps: dict[str, Any] = Field(default_factory=dict)
        disable_internal_cache: bool | None = None
        save_original: bool | None = None
        save_processed: bool | None = None
        save_annotated: bool | None = None
        save_crops: bool | None = None
        save_fail_only: bool | None = None
        jpeg_quality: int | None = None
        png_compression: int | None = None
        max_crops_per_frame: int | None = None
        fail_on_unexpected: bool | None = None
        buffer_limit: int | None = None
        flush_interval: float | None = None
        max_cache_size: int | None = None

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
