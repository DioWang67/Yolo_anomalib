from __future__ import annotations

import atexit
import os
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.logging_config import DetectionLogger
from core.utils import ImageUtils, DetectionResults
from core.exceptions import (
    ResultExcelWriteError,
    ResultImageWriteError,
    ResultPersistenceError,
)

from .annotations import annotate_yolo_frame
from .crops import save_detection_crops
from .excel_buffer import ExcelWorkbookBuffer
from .excel_formatter import build_excel_row
from .image_queue import ImageWriteError, ImageWriteQueue
from .path_manager import ResultPathManager, SavePathBundle

# Minimal color helper (compatible with ultralytics.colors signature)
def colors(class_id, bgr=True):
    return (0, 255, 0)

# Excel column titles (stored as unicode escapes to avoid encoding issues)
COLUMN_NAMES: List[str] = [
    "\u6642\u9593\u6233\u8a18",  # 時間戳記
    "\u6e2c\u8a66\u7de8\u865f",  # 測試編號
    "\u7522\u54c1",  # 產品
    "\u5340\u57df",  # 區域
    "\u6a21\u578b\u985e\u578b",  # 模型類型
    "\u7d50\u679c",  # 結果
    "\u4fe1\u5fc3\u5206\u6578",  # 信心分數
    "\u7570\u5e38\u5206\u6578",  # 異常分數
    "\u984f\u8272\u6aa2\u6e2c\u72c0\u614b",  # 顏色檢測狀態
    "\u984f\u8272\u5dee\u7570\u503c",  # 顏色差異值
    "\u932f\u8aa4\u8a0a\u606f",  # 錯誤訊息
    "\u6a19\u8a3b\u5f71\u50cf\u8def\u5f91",  # 標註影像路徑
    "\u539f\u59cb\u5f71\u50cf\u8def\u5f91",  # 原始影像路徑
    "\u9810\u8655\u7406\u5716\u50cf\u8def\u5f91",  # 預處理圖像路徑
    "\u7570\u5e38\u71b1\u5716\u8def\u5f91",  # 異常熱圖路徑
    "\u88c1\u526a\u5716\u50cf\u8def\u5f91",  # 裁剪圖像路徑
    "\u6aa2\u67e5\u9ede\u8def\u5f91",  # 檢查點路徑
]


class ResultHandler:
    """Handle result persistence (images + Excel workbook with buffering)."""

    def __init__(self, config, base_dir: str = "Result",
                 logger: DetectionLogger | None = None) -> None:
        if is_dataclass(config) and not isinstance(config, type):
            cfg_dict = asdict(config)
        elif isinstance(config, dict):
            cfg_dict = dict(config)
        else:
            cfg_dict = config
        self._config_source = config
        self.config = cfg_dict
        self.base_dir = base_dir
        self.logger = logger or DetectionLogger()
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(config)
        self.path_manager = ResultPathManager(base_dir)
        self.path_manager.ensure_base()

        self.columns = list(COLUMN_NAMES)
        self.excel_path = os.path.join(self.base_dir, "results.xlsx")
        buffer_limit = int(self._cfg_get("buffer_limit", 10) or 10)
        flush_interval = self._cfg_get("flush_interval", None)
        self._excel = ExcelWorkbookBuffer(
            path=self.excel_path,
            columns=self.columns,
            buffer_limit=buffer_limit,
            flush_interval=flush_interval,
            logger=self.logger.logger,
        )

        queue_size = int(self._cfg_get("image_queue_maxsize", 1000) or 1000)
        warn_threshold = float(
            self._cfg_get(
                "image_queue_warn_threshold",
                0.8) or 0.8)
        self._img_queue = ImageWriteQueue(
            self.logger.logger,
            maxsize=queue_size,
            warn_threshold=warn_threshold)

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_annotated_path(
        self,
        status: str,
        detector: str,
        product: Optional[str],
        area: Optional[str],
        anomaly_score: Optional[float] = None,
    ) -> str:
        return self.path_manager.get_annotated_path(
            status, detector, product, area, anomaly_score
        )

    def save_results(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        status: str,
        detector: str,
        missing_items: List[str],
        processed_image: np.ndarray,
        anomaly_score: float | None = None,
        heatmap_path: str | None = None,
        product: str | None = None,
        area: str | None = None,
        ckpt_path: str | None = None,
        color_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        try:
            timestamp = datetime.now()
            bundle = self.path_manager.build_paths(
                status=status,
                detector=detector,
                product=product,
                area=area,
                anomaly_score=anomaly_score,
                timestamp=timestamp,
            )

            save_flags = self._resolve_save_flags(status)
            jpeg_quality = int(self._cfg_get("jpeg_quality", 95) or 95)
            png_compression = int(self._cfg_get("png_compression", 3) or 3)
            imwrite_params_jpg = [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(max(1, min(100, jpeg_quality))),
            ]
            imwrite_params_png = [
                int(cv2.IMWRITE_PNG_COMPRESSION),
                int(max(0, min(9, png_compression))),
            ]

            original_path = (
                bundle.original_path if save_flags["original"] else ""
            )
            preprocessed_path = (
                (
                    bundle.preprocessed_path
                    if save_flags["processed"]
                    else ""
                )
            )
            annotated_path = (
                bundle.annotated_path if save_flags["annotated"] else ""
            )
            cropped_paths: List[str] = []
            heatmap_dest_path = ""

            if save_flags["original"]:
                self._img_queue.enqueue(
                    original_path, frame, imwrite_params_jpg)
            if save_flags["processed"]:
                target_params = (
                    imwrite_params_png
                    if preprocessed_path.lower().endswith(".png")
                    else imwrite_params_jpg
                )
                self._img_queue.enqueue(
                    preprocessed_path,
                    processed_image,
                    target_params,
                )

            detector_lower = (detector or "").lower()
            if detector_lower == "yolo" and save_flags["annotated"]:
                annotated_frame = processed_image.copy()
                crop_source = processed_image
                annotate_yolo_frame(
                    self.image_utils, annotated_frame, detections, color_result, status
                )
                self._img_queue.write_sync(
                    annotated_path,
                    annotated_frame,
                    (
                        imwrite_params_jpg
                        if annotated_path.lower().endswith(".jpg")
                        else imwrite_params_png
                    ),
                )
                if save_flags["crops"] and detections:
                    max_crops = self._cfg_get("max_crops_per_frame", None)
                    limit = int(max_crops) if max_crops is not None else None
                    cropped_paths = save_detection_crops(
                        self._img_queue,
                        crop_source=crop_source,
                        detections=detections,
                        bundle=bundle,
                        product=product,
                        area=area,
                        timestamp_text=bundle.timestamp,
                        params=imwrite_params_png,
                        limit=limit,
                    )
            elif (
                save_flags["annotated"]
                and detector_lower == "anomalib"
                and heatmap_path
                and os.path.exists(heatmap_path)
            ):
                heatmap_dest_path = annotated_path or bundle.annotated_path
                if heatmap_dest_path:
                    src_norm = os.path.normcase(os.path.abspath(heatmap_path))
                    dest_norm = os.path.normcase(os.path.abspath(heatmap_dest_path))
                    if src_norm != dest_norm:
                        try:
                            shutil.copy2(heatmap_path, heatmap_dest_path)
                        except Exception as copy_exc:
                            self.logger.logger.warning(
                                f"Heatmap copy failed: {copy_exc}")
                            heatmap_dest_path = heatmap_path
                    else:
                        heatmap_dest_path = heatmap_path
                else:
                    heatmap_dest_path = heatmap_path
            elif save_flags["annotated"]:
                heatmap_dest_path = annotated_path

            test_id = self._excel.next_test_id(self._excel.pending_rows())
            excel_row = build_excel_row(
                self.columns,
                timestamp=timestamp,
                status=status,
                detector=detector,
                product=product,
                area=area,
                detections=detections,
                missing_items=missing_items,
                anomaly_score=anomaly_score,
                annotated_path=annotated_path,
                original_path=original_path,
                preprocessed_path=preprocessed_path,
                heatmap_path=heatmap_dest_path,
                cropped_paths=cropped_paths,
                ckpt_path=ckpt_path,
                color_result=color_result,
                test_id=test_id,
            )
            try:
                self._excel.append(excel_row)
            except Exception as exc:
                self.logger.logger.exception("Excel append failed")
                raise ResultExcelWriteError(str(exc)) from exc

            return {
                "status": "SUCCESS",
                "original_path": original_path,
                "preprocessed_path": preprocessed_path,
                "annotated_path": annotated_path,
                "heatmap_path": heatmap_dest_path,
                "cropped_paths": cropped_paths,
                "product": product,
                "area": area,
            }
        except ImageWriteError as exc:
            self.logger.logger.exception("Image write failed")
            raise ResultImageWriteError(str(exc)) from exc
        except ResultPersistenceError:
            raise
        except Exception as exc:  # pragma: no cover
            self.logger.logger.exception("Result persistence failed")
            raise ResultPersistenceError(str(exc)) from exc

    def flush(self) -> None:
        self._excel.flush()

    def close(self) -> None:
        def _warn(action: str, exc: Exception) -> None:
            self.logger.logger.warning(
                f"{action} during ResultHandler.close failed: {exc}",
                exc_info=exc)

        operations = [
            ("Excel flush", self._excel.flush),
            ("Image queue flush", self._img_queue.flush),
            ("Image queue shutdown", self._img_queue.shutdown),
            ("Excel close", self._excel.close),
        ]
        for label, fn in operations:
            try:
                fn()
            except Exception as exc:
                _warn(label, exc)

        stats = self._img_queue.stats
        if stats.overflows:
            self.logger.logger.warning(
                f"Image queue overflow occurred {stats.overflows} times"
            )
        if stats.errors:
            self.logger.logger.warning(
                f"Image writer encountered {stats.errors} errors"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cfg_get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self._config_source, key, default)

    def _resolve_save_flags(self, status: str) -> Dict[str, bool]:
        only_fail = bool(self._cfg_get("save_fail_only", False))
        should_save_images = (status != "PASS") if only_fail else True
        return {
            "original": bool(self._cfg_get("save_original", True))
            and should_save_images,
            "processed": bool(self._cfg_get("save_processed", True))
            and should_save_images,
            "annotated": bool(self._cfg_get("save_annotated", True))
            and should_save_images,
            "crops": (
                bool(self._cfg_get("save_crops", True)) and should_save_images
            ),
        }

