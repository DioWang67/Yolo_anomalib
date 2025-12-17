from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _list_subdirectories(path: Path) -> list[str]:
    if not path.exists():
        return []
    entries = [
        entry.name
        for entry in path.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    ]
    entries.sort()
    return entries


@dataclass
class ModelCatalog:
    """Discover available products/areas/inference types from the models directory."""

    root: Path
    _cache: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def refresh(self) -> None:
        self._cache.clear()

    def products(self) -> list[str]:
        products = _list_subdirectories(self.root)
        # simple cache invalidation if counts changed
        if set(products) != set(self._cache.keys()):
            self._cache.clear()
        return products

    def areas(self, product: str) -> list[str]:
        if product not in self._cache:
            product_dir = self.root / product
            self._cache[product] = {
                "__areas__": _list_subdirectories(product_dir)
            }
        return self._cache[product]["__areas__"]

    def inference_types(self, product: str, area: str) -> list[str]:
        self.areas(product)  # ensure product cached
        key = f"{product}:{area}"
        if key not in self._cache:
            self._cache[key] = {}
        if "__types__" not in self._cache[key]:
            area_dir = self.root / product / area
            self._cache[key]["__types__"] = _list_subdirectories(area_dir)
        return self._cache[key]["__types__"]

    def config_path(self, product: str, area: str, inference_type: str) -> Path:
        return self.root / product / area / inference_type / "config.yaml"

    def config_exists(self, product: str, area: str, inference_type: str) -> bool:
        return self.config_path(product, area, inference_type).exists()
