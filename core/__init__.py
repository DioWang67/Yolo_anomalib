"""Core package exports."""

from __future__ import annotations

# 可選：用標準 logging 發出偵錯訊息
import logging
from typing import Optional

from .color_qc_enhanced import ColorQCEnhanced  # noqa: F401

logger = logging.getLogger(__name__)

# -- Optional import: anomalib 相關可能缺 --
AnomalibInferenceModel: type | None  # 明確型別註解
try:  # pragma: no cover
    from .anomalib_inference_model import (
        AnomalibInferenceModel as _AIM,  # noqa: F401
    )

    AnomalibInferenceModel = _AIM
except ImportError:  # 精準攔截
    AnomalibInferenceModel = None  # type: ignore[assignment]
    logger.debug(
        "anomalib_inference_model not available (optional dependency missing)."
    )
except Exception:
    # 若你擔心其他初始化錯誤，建議至少記錄下來，方便排查
    logger.exception(
        "Failed to import anomalib_inference_model due to non-import error."
    )
    AnomalibInferenceModel = None  # type: ignore[assignment]


# -- Always available utilities --

__all__ = ["ColorQCEnhanced"]
if AnomalibInferenceModel is not None:
    __all__.append("AnomalibInferenceModel")

__version__ = "0.1.0"
