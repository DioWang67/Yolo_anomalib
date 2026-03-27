from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from core.path_utils import resolve_path

logger = logging.getLogger(__name__)


def _resolve_with_fallback(rel_path: str, model_cfg_dir: Path | None) -> Path | None:
    """Try resolve_path first, then fall back to model_cfg_dir-relative lookup."""
    resolved = resolve_path(rel_path)
    if resolved and resolved.exists():
        return resolved
    if model_cfg_dir:
        candidate = (model_cfg_dir / rel_path).resolve()
        if candidate.exists():
            return candidate
    return None


def validate_model_cfg(
    cfg: dict[str, Any], product: str, area: str, selected_backend: str | None = None, model_cfg_dir: Path | None = None
) -> None:
    """Lightweight validation for model-level config.

    Raises ValueError for critical issues (missing YOLO weights).
    For optional features (color_check, anomalib), auto-disables and warns
    when required files are missing, so the detection pipeline can continue.
    """
    # YOLO weights — critical, cannot proceed without
    enable_yolo = cfg.get("enable_yolo", False)
    if enable_yolo:
        weights = cfg.get("weights")
        if not weights:
            raise ValueError(
                "YOLO 'weights' is required when enable_yolo=True")
        if not _resolve_with_fallback(weights, model_cfg_dir):
            raise ValueError(f"YOLO weights not found: {weights} (checked root and {model_cfg_dir})")

    # Color checker model — optional, degrade gracefully
    if cfg.get("enable_color_check", False):
        color_path = cfg.get("color_model_path")
        if not color_path:
            logger.warning("enable_color_check=True but 'color_model_path' not set — disabling color check")
            cfg["enable_color_check"] = False
        elif not _resolve_with_fallback(color_path, model_cfg_dir):
            logger.warning(
                f"Color model not found: {color_path} (checked root and {model_cfg_dir}) "
                f"— disabling color check for this run"
            )
            cfg["enable_color_check"] = False

    # Anomalib ckpt — optional, degrade gracefully
    if cfg.get("enable_anomalib", False):
        acfg = cfg.get("anomalib_config") or {}
        models = (acfg.get("models") or {}).get(product, {})
        m = models.get(area)
        if not m or not m.get("ckpt_path"):
            logger.warning(
                f"Anomalib ckpt_path missing for {product},{area} — disabling anomalib"
            )
            cfg["enable_anomalib"] = False
        elif not _resolve_with_fallback(str(m["ckpt_path"]), model_cfg_dir):
            logger.warning(
                f"Anomalib ckpt not found: {m['ckpt_path']} (checked root and {model_cfg_dir}) "
                f"— disabling anomalib for this run"
            )
            cfg["enable_anomalib"] = False

    # Custom backend (when a non-builtin type is selected)
    if selected_backend:
        name = str(selected_backend).lower().strip()
        if name not in ("yolo", "anomalib"):
            backs = cfg.get("backends") or {}
            spec = backs.get(name) if isinstance(backs, dict) else None
            if not spec:
                raise ValueError(
                    f"Custom backend '{name}' not configured under backends"
                )
            if not spec.get("class_path"):
                raise ValueError(
                    f"Custom backend '{name}' missing class_path in backends"
                )
