from __future__ import annotations

import os
from typing import Any, Dict

from core.path_utils import resolve_path


def validate_model_cfg(
    cfg: Dict[str, Any], product: str, area: str, selected_backend: str | None = None
) -> None:
    """Lightweight validation for model-level config.

    Raises ValueError with helpful message on critical issues.
    """
    # YOLO weights
    enable_yolo = cfg.get("enable_yolo", False)
    if enable_yolo:
        weights = cfg.get("weights")
        if not weights:
            raise ValueError(
                "YOLO 'weights' is required when enable_yolo=True")
        wpath = resolve_path(weights)
        if not wpath or not os.path.exists(str(wpath)):
            raise ValueError(f"YOLO weights not found: {wpath}")

    # Color checker model (optional)
    if cfg.get("enable_color_check", False):
        color_path = cfg.get("color_model_path")
        if not color_path:
            raise ValueError(
                "enable_color_check=True but 'color_model_path' not set")
        cpath = resolve_path(color_path)
        if not cpath or not os.path.exists(str(cpath)):
            raise ValueError(f"Color model not found: {cpath}")

    # Anomalib ckpt (optional)
    if cfg.get("enable_anomalib", False):
        acfg = cfg.get("anomalib_config") or {}
        models = (acfg.get("models") or {}).get(product, {})
        m = models.get(area)
        if not m or not m.get("ckpt_path"):
            raise ValueError(
                f"Anomalib ckpt_path missing for {product},{area}")
        ckp = resolve_path(m.get("ckpt_path"))
        if not ckp or not os.path.exists(str(ckp)):
            raise ValueError(f"Anomalib ckpt not found: {ckp}")

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


"""Lightweight validation helpers for model-level configs.

These checks provide clear error messages early for missing/invalid paths.
Use together with optional pydantic schema for type/range validation.
"""
