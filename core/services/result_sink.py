from __future__ import annotations

from typing import Dict, Any

from core.services.results.handler import ResultHandler
from core.logging_config import DetectionLogger


class ResultSink:
    """Abstract sink for persisting results (images, excel, db, etc.)."""

    def get_annotated_path(self, *args, **kwargs) -> str:  # pragma: no cover
        """Return an output path for annotated images (per status/product/area)."""
        raise NotImplementedError

    def save(self, *args, **kwargs) -> Dict[str, Any]:  # pragma: no cover
        """Persist results and return paths (original, processed, annotated, ...)."""
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover
        """Flush buffered writes (if any)."""
        raise NotImplementedError


class ExcelImageResultSink(ResultSink):
    def __init__(
        self, config, base_dir: str, logger: DetectionLogger | None = None
    ) -> None:
        """Result sink backed by ResultHandler (images + Excel workbook)."""
        self._handler = ResultHandler(config, base_dir=base_dir, logger=logger)

    def get_annotated_path(self, *args, **kwargs) -> str:
        return self._handler.get_annotated_path(*args, **kwargs)

    def save(self, *args, **kwargs) -> Dict[str, Any]:
        return self._handler.save_results(*args, **kwargs)

    def flush(self) -> None:
        self._handler.flush()

    def close(self) -> None:
        self._handler.close()
