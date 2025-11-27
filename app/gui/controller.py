from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from app.gui.workers import DetectionWorker

if TYPE_CHECKING:  # pragma: no cover
    from core.detection_system import DetectionSystem


def _list_subdirectories(path: Path) -> List[str]:
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
    _cache: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def refresh(self) -> None:
        self._cache.clear()

    def products(self) -> List[str]:
        products = _list_subdirectories(self.root)
        # simple cache invalidation if counts changed
        if set(products) != set(self._cache.keys()):
            self._cache.clear()
        return products

    def areas(self, product: str) -> List[str]:
        if product not in self._cache:
            product_dir = self.root / product
            self._cache[product] = {
                "__areas__": _list_subdirectories(product_dir)
            }
        return self._cache[product]["__areas__"]

    def inference_types(self, product: str, area: str) -> List[str]:
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


class DetectionController:
    """Coordinate detection-system lifecycle and file operations for the GUI."""

    def __init__(
        self,
        config_path: Path,
        catalog: ModelCatalog,
        logger: Optional[logging.Logger] = None,
        detection_cls: Optional[Any] = None,
    ) -> None:
        self._config_path = config_path
        self.catalog = catalog
        self._logger = logger or logging.getLogger(__name__)
        self._system: Optional[Any] = None
        self._detection_cls = detection_cls

    @property
    def detection_system(self) -> Any:
        if self._system is None:
            cls = self._detection_cls
            if cls is None:
                from core.detection_system import DetectionSystem as _DS
                cls = _DS
            self._system = cls(config_path=str(self._config_path))
            self._logger.debug("Detection system initialized with %s", self._config_path)
        return self._system

    def has_system(self) -> bool:
        return self._system is not None

    def is_camera_connected(self) -> bool:
        try:
            return self.detection_system.is_camera_connected()
        except Exception as exc:
            self._logger.exception("Camera status check failed: %s", exc)
            return False

    def reconnect_camera(self) -> bool:
        return self.detection_system.reconnect_camera()

    def disconnect_camera(self) -> None:
        if self._system:
            self._system.disconnect_camera()

    def shutdown(self) -> None:
        if self._system:
            try:
                self._system.shutdown()
            finally:
                self._system = None

    def build_worker(
        self,
        product: str,
        area: str,
        inference_type: str,
        *,
        frame,
    ) -> DetectionWorker:
        system = self.detection_system
        return DetectionWorker(system, product, area, inference_type, frame=frame)

    def save_result_json(self, file_path: Path, result: Dict[str, object]) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        self._logger.debug("Result written to %s", file_path)

    def load_image(self, path: Path) -> Optional[object]:
        if not path.exists():
            return None
        try:
            import cv2

            image = cv2.imread(str(path))
            if image is not None and getattr(image, "size", 0) > 0:
                return image
        except Exception as exc:  # pragma: no cover - best effort
            self._logger.exception("Image load failed for %s: %s", path, exc)
        return None
