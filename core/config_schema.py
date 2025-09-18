
from __future__ import annotations

from typing import Dict, List, Optional, Any

try:
    from pydantic import BaseModel, Field
    try:
        from pydantic import field_validator as _field_validator
        _VALIDATOR_MODE = 'v2'
    except ImportError:  # pragma: no cover - pydantic v1 path
        from pydantic import validator as _validator  # type: ignore
        _VALIDATOR_MODE = 'v1'
except Exception:  # pragma: no cover
    try:
        from pydantic.v1 import BaseModel, Field, validator as _validator  # type: ignore
        _VALIDATOR_MODE = 'v1'
    except Exception:  # pragma: no cover
        BaseModel = None  # type: ignore
        _VALIDATOR_MODE = None


def _to_dict(model: Any) -> Dict[str, Any]:  # compatible dump
    if hasattr(model, 'dict'):
        return model.dict()
    if hasattr(model, 'model_dump'):
        return model.model_dump()
    return dict(model)


if BaseModel is not None:  # pragma: no cover - runtime optional

    class GlobalConfigSchema(BaseModel):
        exposure_time: Optional[str] = None
        gain: Optional[str] = None
        MV_CC_GetImageBuffer_nMsec: Optional[int] = None
        timeout: Optional[int] = 1
        width: Optional[int] = 640
        height: Optional[int] = 640
        enable_yolo: Optional[bool] = False
        enable_anomalib: Optional[bool] = False
        max_cache_size: Optional[int] = 3
        buffer_limit: Optional[int] = 1
        flush_interval: Optional[float] = None
        pipeline: Optional[List[str]] = None
        steps: Dict[str, Any] = Field(default_factory=dict)
        backends: Optional[Dict[str, Dict[str, Any]]] = None

    class ModelConfigSchema(BaseModel):
        device: Optional[str] = None
        conf_thres: Optional[float] = None
        iou_thres: Optional[float] = None
        imgsz: Optional[List[int]] = None
        output_dir: Optional[str] = None
        enable_yolo: Optional[bool] = None
        enable_anomalib: Optional[bool] = None
        enable_color_check: Optional[bool] = None
        color_model_path: Optional[str] = None
        weights: Optional[str] = None
        expected_items: Optional[Dict[str, Dict[str, List[str]]]] = None
        position_config: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
        anomalib_config: Optional[Dict[str, Any]] = None
        backends: Optional[Dict[str, Dict[str, Any]]] = None
        pipeline: Optional[List[str]] = None
        steps: Dict[str, Any] = Field(default_factory=dict)

        if _VALIDATOR_MODE == 'v2':

            @_field_validator('imgsz', mode='before')
            def _coerce_imgsz(cls, v):
                if v is None:
                    return v
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return [int(v[0]), int(v[1])]
                raise ValueError('imgsz must be [h, w]')

        elif _VALIDATOR_MODE == 'v1':

            @_validator('imgsz', pre=True)  # type: ignore[misc]
            def _coerce_imgsz(cls, v):
                if v is None:
                    return v
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return [int(v[0]), int(v[1])]
                raise ValueError('imgsz must be [h, w]')
else:
    # No pydantic available; expose sentinels
    GlobalConfigSchema = None  # type: ignore
    ModelConfigSchema = None  # type: ignore
