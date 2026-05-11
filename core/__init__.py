"""Core package exports."""

from __future__ import annotations

import logging

from .color_qc_enhanced import ColorQCEnhanced

logger = logging.getLogger(__name__)


def __getattr__(name: str):
    """Lazy-load optional heavy exports only when explicitly requested."""
    if name != "AnomalibInferenceModel":
        raise AttributeError(f"module 'core' has no attribute {name!r}")
    try:  # pragma: no cover
        from .anomalib_inference_model import AnomalibInferenceModel
    except ImportError:
        logger.debug(
            "anomalib_inference_model not available (optional dependency missing)."
        )
        return None
    except Exception:
        logger.exception(
            "Failed to import anomalib_inference_model due to non-import error."
        )
        return None
    return AnomalibInferenceModel


__all__: list[str] = ["ColorQCEnhanced", "AnomalibInferenceModel"]
__version__ = "0.1.0"
