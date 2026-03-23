from __future__ import annotations

import os
from typing import Any
from pathlib import Path

from core.path_utils import resolve_path


def validate_model_cfg(
    cfg: dict[str, Any], product: str, area: str, selected_backend: str | None = None, model_cfg_dir: Path | None = None
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
        if (not wpath or not wpath.exists()) and model_cfg_dir:
            rel_wpath = (model_cfg_dir / weights).resolve()
            if rel_wpath.exists():
                wpath = rel_wpath
        if not wpath or not wpath.exists():
            raise ValueError(f"YOLO weights not found: {weights} (checked root and {model_cfg_dir})")

    # Color checker model (optional)
    if cfg.get("enable_color_check", False):
        color_path = cfg.get("color_model_path")
        if not color_path:
            raise ValueError(
                "enable_color_check=True but 'color_model_path' not set")
        cpath = resolve_path(color_path)
        if (not cpath or not cpath.exists()) and model_cfg_dir:
            rel_cpath = (model_cfg_dir / color_path).resolve()
            if rel_cpath.exists():
                cpath = rel_cpath
        if not cpath or not cpath.exists():
            raise ValueError(f"Color model not found: {color_path} (checked root and {model_cfg_dir})")

    # Anomalib ckpt (optional)
    if cfg.get("enable_anomalib", False):
        acfg = cfg.get("anomalib_config") or {}
        models = (acfg.get("models") or {}).get(product, {})
        m = models.get(area)
        if not m or not m.get("ckpt_path"):
            raise ValueError(
                f"Anomalib ckpt_path missing for {product},{area}")
        ckp = resolve_path(m.get("ckpt_path"))
        if (not ckp or not ckp.exists()) and model_cfg_dir:
            rel_ckp = (model_cfg_dir / str(m.get("ckpt_path"))).resolve()
            if rel_ckp.exists():
                ckp = rel_ckp
        if not ckp or not ckp.exists():
            raise ValueError(f"Anomalib ckpt not found: {m.get('ckpt_path')} (checked root and {model_cfg_dir})")

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
