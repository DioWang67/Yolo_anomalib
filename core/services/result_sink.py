from __future__ import annotations

from typing import Dict, Any

from core.result_handler import ResultHandler
from core.logger import DetectionLogger


class ResultSink:
    def get_annotated_path(self, *args, **kwargs) -> str:  # pragma: no cover
        raise NotImplementedError

    def save(self, *args, **kwargs) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def flush(self) -> None:  # pragma: no cover
        raise NotImplementedError


class ExcelImageResultSink(ResultSink):
    def __init__(self, config, base_dir: str, logger: DetectionLogger | None = None) -> None:
        self._handler = ResultHandler(config, base_dir=base_dir, logger=logger)

    def get_annotated_path(self, *args, **kwargs) -> str:
        return self._handler.get_annotated_path(*args, **kwargs)

    def save(self, *args, **kwargs) -> Dict[str, Any]:
        return self._handler.save_results(*args, **kwargs)

    def flush(self) -> None:
        self._handler.flush()

    def close(self) -> None:
        self._handler.close()

