"""Core package exports."""

# Optional import: anomalib dependencies might be missing in some envs
try:  # pragma: no cover
    from .anomalib_inference_model import AnomalibInferenceModel  # noqa: F401
except Exception:  # pragma: no cover
    AnomalibInferenceModel = None  # type: ignore

# Always available local utilities
from .led_qc_enhanced import LEDQCEnhanced  # noqa: F401

__all__ = [
    "LEDQCEnhanced",
]
if AnomalibInferenceModel is not None:
    __all__.append("AnomalibInferenceModel")

__version__ = "0.1.0"

