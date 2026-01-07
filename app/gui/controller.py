from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.gui.workers import CameraInitWorker, DetectionWorker, ModelLoaderWorker

if TYPE_CHECKING:  # pragma: no cover
    from core.services.model_catalog import ModelCatalog
    from core.types import DetectionResult


class DetectionController:
    """
    Application Coordinator.
    Responsible for:
    1. System Lifecycle (Init / Shutdown)
    2. Worker Factory (DI Injection)
    3. Bridge between GUI and Core
    
    NO Business Logic allowed here.
    """

    def __init__(
        self,
        config_path: Path,
        catalog: ModelCatalog,
        logger: logging.Logger | None = None,
        detection_cls: Any | None = None,
    ) -> None:
        self._config_path = config_path
        self.catalog = catalog
        self._logger = logger or logging.getLogger(__name__)
        self._system: Any | None = None
        self._detection_cls = detection_cls

    @property
    def detection_system(self) -> Any:
        """
        Lazy initialization of the detection system.
        In a stricter DI setup, this should be passed in __init__, 
        but for now we maintain lazy-load behavior but managed here.
        """
        if self._system is None:
            self._logger.info("Initializing Detection System...")
            try:
                cls = self._detection_cls
                if cls is None:
                    from core.detection_system import DetectionSystem as _DS
                    cls = _DS
                self._system = cls(config_path=str(self._config_path))
                self._logger.debug("Detection system initialized with %s", self._config_path)
            except Exception as e:
                self._logger.error(f"Failed to initialize DetectionSystem: {e}")
                raise e
        return self._system

    def has_system(self) -> bool:
        return self._system is not None

    def is_camera_connected(self) -> bool:
        if not self.has_system():
             return False
        try:
            return self.detection_system.is_camera_connected()
        except Exception as exc:
            self._logger.exception("Camera status check failed: %s", exc)
            return False

    def reconnect_camera(self) -> bool:
        if not self.has_system():
             return False
        return self.detection_system.reconnect_camera()

    def disconnect_camera(self) -> None:
        if self._system:
            self._system.disconnect_camera()

    def shutdown(self) -> None:
        """Full system shutdown."""
        if self._system:
            try:
                self._system.shutdown()
            finally:
                self._system = None

    # --- Worker Factory Methods (Dependency Injection) ---

    def build_worker(
        self,
        product: str,
        area: str,
        inference_type: str,
        *,
        frame: Any = None,
        continuous: bool = False,
    ) -> DetectionWorker:
        """Creates a detection worker with injected system."""
        system = self.detection_system
        return DetectionWorker(
            detection_system=system,
            product=product,
            area=area,
            inference_type=inference_type,
            frame=frame,
            continuous=continuous
        )

    def build_model_loader(self) -> ModelLoaderWorker:
        return ModelLoaderWorker(self.catalog)

    def build_camera_initializer(self) -> CameraInitWorker:
        return CameraInitWorker(self)

    # --- Utilities (Should eventually move to services) ---

    def save_result_json(self, file_path: Path, result: DetectionResult) -> None:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
             with file_path.open("w", encoding="utf-8") as f:
                 json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
             self._logger.debug("Result written to %s", file_path)
        except Exception as e:
             self._logger.error(f"Failed to save result: {e}")

    def load_image(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            import cv2
            # TODO: Move image IO to a dedicated ImageService
            image = cv2.imread(str(path))
            if image is not None and getattr(image, "size", 0) > 0:
                return image
        except Exception as exc:
            self._logger.exception("Image load failed for %s: %s", path, exc)
        return None
