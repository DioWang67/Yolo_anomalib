# core/__init__.py
try:  # 匯入可能因缺少相依套件而失敗（如 cv2）
    from .anomalib_inference_model import AnomalibInferenceModel
except Exception:  # pragma: no cover
    AnomalibInferenceModel = None  # type: ignore

from .led_qc_enhanced import LEDQCEnhanced

__all__ = ["LEDQCEnhanced"]
if AnomalibInferenceModel is not None:
    __all__.append("AnomalibInferenceModel")
__version__ = "0.1.0"  # 可選
