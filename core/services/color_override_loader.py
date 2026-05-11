"""Load per-model color threshold/rule overrides with a small mtime cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ColorOverrideLoader:
    """Loads color checker overrides from model-level config files.

    Args:
        models_root: Root directory containing ``<product>/<area>/<type>`` model
            folders.
        max_cache_size: Maximum number of config files to keep in memory.
    """

    def __init__(self, models_root: Path, max_cache_size: int = 32) -> None:
        self.models_root = models_root
        self.max_cache_size = max_cache_size
        self._cache: dict[str, dict[str, Any]] = {}

    def load(
        self,
        config: Any,
        product: str,
        area: str,
        inference_type: str,
        logger: Any,
    ) -> tuple[dict[str, float] | None, dict[str, dict[str, Any]] | None]:
        """Return color threshold/rule overrides for the active model config.

        Args:
            config: Current detection config; global overrides are used as
                fallback values.
            product: Product name.
            area: Area/station name.
            inference_type: Model type folder, for example ``yolo``.
            logger: Logger-compatible object used for warnings.

        Returns:
            A tuple of ``(threshold_overrides, rule_overrides)``.
        """
        fallback_overrides = getattr(config, "color_threshold_overrides", None)
        fallback_rules = getattr(config, "color_rules_overrides", None)
        cfg_path = self.models_root / product / area / inference_type / "config.yaml"
        cache_key = str(cfg_path)

        try:
            stat = cfg_path.stat()
        except FileNotFoundError:
            self._cache.pop(cache_key, None)
            return fallback_overrides, fallback_rules

        cached = self._cache.get(cache_key)
        if cached and cached.get("mtime") == stat.st_mtime:
            return self._with_fallback(cached, fallback_overrides, fallback_rules)

        yaml_module = _import_yaml()
        if yaml_module is None:
            logger.warning(
                "PyYAML is not available; skipping color overrides from %s", cfg_path
            )
            return fallback_overrides, fallback_rules

        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                model_cfg = yaml_module.safe_load(handle)
        except (OSError, yaml_module.YAMLError) as exc:
            logger.warning("Failed to reload color overrides from %s: %s", cfg_path, exc)
            return fallback_overrides, fallback_rules

        if not isinstance(model_cfg, dict):
            logger.warning("Color override config %s is not a valid mapping, skipping", cfg_path)
            return fallback_overrides, fallback_rules

        disk_overrides = self._non_empty_mapping_or_none(
            model_cfg.get("color_threshold_overrides")
        )
        disk_rules = self._non_empty_mapping_or_none(
            model_cfg.get("color_rules_overrides")
        )

        self._cache[cache_key] = {
            "mtime": stat.st_mtime,
            "overrides": disk_overrides,
            "rules": disk_rules,
        }
        self._trim_cache(latest_key=cache_key)
        return (
            disk_overrides if disk_overrides is not None else fallback_overrides,
            disk_rules if disk_rules is not None else fallback_rules,
        )

    @staticmethod
    def _non_empty_mapping_or_none(value: Any) -> dict | None:
        if isinstance(value, dict) and value:
            return value
        return None

    @staticmethod
    def _with_fallback(
        cached: dict[str, Any],
        fallback_overrides: dict[str, float] | None,
        fallback_rules: dict[str, dict[str, Any]] | None,
    ) -> tuple[dict[str, float] | None, dict[str, dict[str, Any]] | None]:
        cached_overrides = cached.get("overrides")
        cached_rules = cached.get("rules")
        return (
            cached_overrides if cached_overrides is not None else fallback_overrides,
            cached_rules if cached_rules is not None else fallback_rules,
        )

    def _trim_cache(self, latest_key: str) -> None:
        if len(self._cache) <= self.max_cache_size:
            return
        for key in list(self._cache.keys()):
            if key != latest_key:
                self._cache.pop(key, None)
                return


def _import_yaml() -> Any | None:
    """Return the optional PyYAML module when available."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return None
    return yaml
