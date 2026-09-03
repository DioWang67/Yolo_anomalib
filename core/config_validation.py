from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from core.color_baseline_contract import color_model_vocabulary
from core.path_utils import resolve_path

CUSTOM_BACKEND_PREFIX = "core.backends."
#: Checker type whose baseline carries one statistics block per color name.
_STATS_CHECKER_TYPE = "stats"

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


def _requires_color_baseline(cfg: dict[str, Any]) -> bool:
    """Return whether a missing color artifact is a configuration failure."""
    configured = cfg.get("color_baseline_algorithm_enforcement")
    if configured is None:
        return False
    return str(configured).strip().casefold() != "warn"


def _validate_configured_colors(
    cfg: dict[str, Any], product: str, area: str, color_model_path: Path
) -> None:
    """Reject a configured color the deployed baseline cannot score.

    The runtime folds an unscoreable color into *every* item's verdict, so one
    unknown or misspelled name in ``expected_items`` makes every board fail the
    color check for as long as it stays in the config -- and under ``warn`` as
    much as under ``strict``. Failing here costs a startup that names the color;
    failing at the first board costs a shift and looks like a camera problem.
    """
    checker_type = str(cfg.get("color_checker_type") or "").strip().casefold()
    if checker_type != _STATS_CHECKER_TYPE:
        return
    # Imported here rather than at module scope: this module is also loaded by
    # the runtime environment checks, which have no reason to pull in the image
    # stack that the station color settings depend on.
    from core.services.station_color_settings import expected_color_names

    step_options = (cfg.get("steps") or {}).get("color_check") or {}
    configured = expected_color_names(
        cfg,
        product,
        area,
        generic_classes=step_options.get("generic_classes"),
    )
    if not configured:
        return
    vocabulary = color_model_vocabulary(color_model_path)
    if vocabulary is None:
        # Unreadable, or no summary at all. The loader reports that precisely;
        # a complaint about every color would only bury it.
        return
    unknown = sorted(
        {name for name in configured if name.casefold() not in vocabulary}
    )
    if unknown:
        raise ValueError(
            f"expected_items for {product}/{area} names colors the deployed "
            f"color baseline cannot score: {', '.join(unknown)}. "
            f"Every configured color must exist in {color_model_path}, which "
            f"scores: {', '.join(sorted(vocabulary))}. Correct the name, or "
            f"rebuild the baseline with evidence for the new color before "
            f"adding it here."
        )


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

    # Color checker model — optional in legacy/warn mode, mandatory when the
    # station has explicitly enabled strict baseline provenance enforcement.
    if cfg.get("enable_color_check", False):
        color_path = cfg.get("color_model_path")
        resolved_color_path = (
            _resolve_with_fallback(color_path, model_cfg_dir) if color_path else None
        )
        if not color_path:
            if _requires_color_baseline(cfg):
                raise ValueError(
                    "color_model_path is required when strict color baseline "
                    "enforcement is enabled"
                )
            logger.warning("enable_color_check=True but 'color_model_path' not set — disabling color check")
            cfg["enable_color_check"] = False
        elif resolved_color_path is None:
            if _requires_color_baseline(cfg):
                raise ValueError(
                    f"Color model not found under strict baseline enforcement: "
                    f"{color_path} (checked root and {model_cfg_dir})"
                )
            logger.warning(
                f"Color model not found: {color_path} (checked root and {model_cfg_dir}) "
                f"— disabling color check for this run"
            )
            cfg["enable_color_check"] = False
        else:
            _validate_configured_colors(cfg, product, area, resolved_color_path)

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
            if not bool(cfg.get("enable_custom_backends", False)):
                raise ValueError(
                    f"Custom backend '{name}' requires enable_custom_backends=True"
                )
            class_path = str(spec.get("class_path"))
            if not class_path.startswith(CUSTOM_BACKEND_PREFIX):
                raise ValueError(
                    "Custom backend class_path must start with "
                    f"'{CUSTOM_BACKEND_PREFIX}'"
                )
