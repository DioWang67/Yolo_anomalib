"""Public exports for results services."""

from .excel_buffer import ExcelWorkbookBuffer
from .handler import ResultHandler
from .image_queue import ImageWriteQueue
from .path_manager import ResultPathManager, SavePathBundle

# 僅公開穩定 API，避免外部接觸內部實作細節
__all__ = [
    "ResultHandler",
    "ExcelWorkbookBuffer",
    "ImageWriteQueue",
    "ResultPathManager",
    "SavePathBundle",
]
__version__ = "0.1.0"
