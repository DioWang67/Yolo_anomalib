from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


def _ensure_list(v, default=None):
    if v is None:
        return default or []
    if isinstance(v, list):
        return v
    return list(v)


def normalize_result(result: Dict[str, Any], inference_type: str, fallback_frame: np.ndarray) -> Dict[str, Any]:
    """Normalize backend-specific outputs into a common shape.

    Ensures presence of keys consumed by downstream pipeline/sinks:
    - status: str
    - detections: list[dict]
    - missing_items: list[str]
    - processed_image: np.ndarray
    - anomaly_score: float | None
    - output_path: str | '' (for anomalib-like)
    - ckpt_path: str | ''
    - inference_type: original name (lower)
    """
    out: Dict[str, Any] = dict(result or {})
    out["inference_type"] = str(inference_type).lower()

    # Status
    status = out.get("status") or out.get("Status") or "PASS"
    out["status"] = status

    # Detections / missing
    out["detections"] = _ensure_list(out.get("detections"), default=[])
    out["missing_items"] = _ensure_list(out.get("missing_items"), default=[])

    # Processed image
    proc = out.get("processed_image")
    if proc is None or not isinstance(proc, np.ndarray):
        proc = fallback_frame
    out["processed_image"] = proc

    # Anomaly fields (for anomalib-like backends)
    anomaly_score = out.get("anomaly_score", None)
    if anomaly_score is not None:
        try:
            anomaly_score = float(anomaly_score)
        except Exception:
            anomaly_score = None
    out["anomaly_score"] = anomaly_score
    out["output_path"] = out.get("output_path", "") or ""
    out["ckpt_path"] = out.get("ckpt_path", "") or ""

    return out
"""
Result adapter to normalize backend-specific outputs.

All downstream steps/sinks expect the following keys:
- status: 'PASS' | 'FAIL' | 'ERROR'
- detections: list of dicts
- missing_items: list of str
- processed_image: np.ndarray (BGR)
- anomaly_score: float | None
- output_path: path to heatmap/overlay (anomalib-like) or ''
- ckpt_path: model checkpoint path or ''
- inference_type: backend key
"""
