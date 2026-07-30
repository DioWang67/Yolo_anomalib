from __future__ import annotations

import atexit
import hashlib
import json
import os
import shutil
import socket
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.exceptions import (
    ResultExcelWriteError,
    ResultImageWriteError,
    ResultPersistenceError,
)
from core.logging_config import DetectionLogger
from core.position_validator import build_missing_item_locations
from core.security import ensure_subpath
from core.services.decision_engine import collect_fail_reasons
from core.services.inspection_maintenance import (
    InspectionMaintenanceScheduler,
    InspectionRetentionPolicy,
)
from core.services.inspection_repository import InspectionRepository
from core.services.inspection_sync import (
    InspectionSyncConfigurationError,
    InspectionSyncWorker,
    build_inspection_sync_worker,
)
from core.utils import DetectionResults, ImageUtils

from .annotations import annotate_yolo_frame
from .crops import save_detection_crops, save_failure_crops
from .excel_buffer import ExcelWorkbookBuffer
from .excel_formatter import build_excel_row
from .image_queue import ImageWriteError, ImageWriteQueue, ImageWriteReceipt
from .path_manager import ResultPathManager


# Minimal color helper (compatible with ultralytics.colors signature)
def colors(class_id, bgr=True):
    return (0, 255, 0)

# Excel column titles (stored as unicode escapes to avoid encoding issues)
COLUMN_NAMES: list[str] = [
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
        self._set_config(config)
        resolved_base_dir = Path(base_dir).resolve()
        self.base_dir = str(
            ensure_subpath(resolved_base_dir, resolved_base_dir, must_exist=False)
        )
        self.allowed_root = self.base_dir
        self.logger = logger or DetectionLogger()
        self.image_utils = ImageUtils()
        self.detection_results = DetectionResults(config)
        self.path_manager = ResultPathManager(
            self.base_dir, allowed_root=self.allowed_root
        )
        self.path_manager.ensure_base()
        self._inspection_repository = InspectionRepository(
            Path(self.base_dir) / "inspection_records.sqlite3"
        )
        retention_policy = InspectionRetentionPolicy(
            pass_image_days=int(
                self._cfg_get("inspection_pass_image_days", 30) or 30
            ),
            fail_preprocessed_days=int(
                self._cfg_get("inspection_fail_preprocessed_days", 90) or 90
            ),
            fail_all_image_days=int(
                self._cfg_get("inspection_fail_all_image_days", 180) or 180
            ),
        )
        backup_interval_hours = int(
            self._cfg_get("inspection_backup_interval_hours", 24) or 24
        )
        self._inspection_maintenance = InspectionMaintenanceScheduler(
            self.base_dir,
            database_path=self._inspection_repository.path,
            cleanup_enabled=bool(
                self._cfg_get("inspection_retention_cleanup_enabled", False)
            ),
            policy=retention_policy,
            interval=timedelta(hours=max(1, backup_interval_hours)),
            logger=self.logger.logger,
        )
        self._inspection_sync: InspectionSyncWorker | None = None
        if bool(self._cfg_get("inspection_sync_enabled", False)):
            try:
                self._inspection_sync = build_inspection_sync_worker(
                    self._inspection_repository.path,
                    endpoint=str(
                        self._cfg_get("inspection_sync_endpoint", "") or ""
                    ),
                    api_token_env=str(
                        self._cfg_get(
                            "inspection_sync_api_token_env",
                            "YOLO11_INSPECTION_SYNC_TOKEN",
                        )
                        or ""
                    ),
                    timeout_seconds=float(
                        self._cfg_get("inspection_sync_timeout_seconds", 10.0)
                        or 10.0
                    ),
                    interval_seconds=float(
                        self._cfg_get("inspection_sync_interval_seconds", 30.0)
                        or 30.0
                    ),
                    batch_size=int(
                        self._cfg_get("inspection_sync_batch_size", 20) or 20
                    ),
                    max_attempts=int(
                        self._cfg_get("inspection_sync_max_attempts", 12) or 12
                    ),
                    allow_insecure_http=bool(
                        self._cfg_get(
                            "inspection_sync_allow_insecure_http",
                            False,
                        )
                    ),
                    logger=self.logger.logger,
                )
            except (InspectionSyncConfigurationError, ValueError):
                self._inspection_maintenance.close()
                raise

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
            allowed_root=self.allowed_root,
        )

        queue_size_value = self._cfg_get("image_queue_maxsize", 8)
        queue_size = max(0, int(8 if queue_size_value is None else queue_size_value))
        if queue_size > 32:
            self.logger.logger.warning(
                "image_queue_maxsize=%d exceeds the safe limit; capping at 32",
                queue_size,
            )
            queue_size = 32
        queue_max_mb_value = self._cfg_get("image_queue_max_mb", 256)
        queue_max_mb = max(
            0, int(256 if queue_max_mb_value is None else queue_max_mb_value)
        )
        warn_threshold = float(
            self._cfg_get(
                "image_queue_warn_threshold",
                0.8) or 0.8)
        self._img_queue = ImageWriteQueue(
            self.logger.logger,
            maxsize=queue_size,
            max_bytes=queue_max_mb * 1024 * 1024,
            warn_threshold=warn_threshold,
            allowed_root=self.allowed_root,
        )

        atexit.register(self.close)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_config(self, config) -> None:
        """Update runtime config used for result metadata and missing boxes.

        Args:
            config: DetectionConfig-like object or dictionary. Existing output
                queues and workbook remain open.
        """
        self._set_config(config)
        self.detection_results = DetectionResults(config)

    def get_annotated_path(
        self,
        status: str,
        detector: str,
        product: str | None,
        area: str | None,
        anomaly_score: float | None = None,
    ) -> str:
        return self.path_manager.get_annotated_path(
            status, detector, product, area, anomaly_score
        )

    def save_results(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        status: str,
        detector: str,
        missing_items: list[str],
        processed_image: np.ndarray,
        anomaly_score: float | None = None,
        heatmap_path: str | None = None,
        product: str | None = None,
        area: str | None = None,
        ckpt_path: str | None = None,
        color_result: dict[str, Any] | None = None,
        sequence_check: dict[str, Any] | None = None,
        error_message: str | None = None,
        decision: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
        inference_time: float | None = None,
        slot_mismatches: list[dict[str, Any]] | None = None,
        duplicate_filter: dict[str, Any] | None = None,
        raw_detections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            self._ensure_disk_capacity(frame, processed_image)
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

                    bundle.preprocessed_path
                    if save_flags["processed"]
                    else ""

            )
            annotated_path = (
                bundle.annotated_path if save_flags["annotated"] else ""
            )
            cropped_paths: list[str] = []
            heatmap_dest_path = ""
            missing_locations = build_missing_item_locations(
                self._config_source,
                product,
                area,
                missing_items,
                detections,
            )

            image_receipts: list[ImageWriteReceipt] = []
            if save_flags["original"]:
                image_receipts.append(
                    self._img_queue.enqueue(
                        original_path, frame, imwrite_params_jpg
                    )
                )
            if save_flags["processed"]:
                target_params = (
                    imwrite_params_png
                    if preprocessed_path.lower().endswith(".png")
                    else imwrite_params_jpg
                )
                image_receipts.append(
                    self._img_queue.enqueue(
                        preprocessed_path,
                        processed_image,
                        target_params,
                    )
                )

            detector_lower = (detector or "").lower()
            if detector_lower == "yolo" and save_flags["annotated"]:
                annotated_frame = processed_image.copy()
                crop_source = processed_image
                expected_boxes = self._get_expected_boxes(product, area)
                annotate_yolo_frame(
                    self.image_utils,
                    annotated_frame,
                    detections,
                    color_result,
                    status,
                    missing_items=missing_items,
                    expected_boxes=expected_boxes,
                    missing_locations=missing_locations,
                    duplicate_filter=duplicate_filter,
                    raw_detections=raw_detections,
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
                        receipts=image_receipts,
                    )
                if save_flags["crops"] and str(status).upper() != "PASS":
                    failure_crop_paths = save_failure_crops(
                        self._img_queue,
                        crop_source=crop_source,
                        bundle=bundle,
                        product=product,
                        area=area,
                        timestamp_text=bundle.timestamp,
                        params=imwrite_params_png,
                        missing_locations=missing_locations,
                        detections=detections,
                        slot_mismatches=slot_mismatches,
                        receipts=image_receipts,
                    )
                    cropped_paths.extend(failure_crop_paths)
            elif detector_lower == "fusion" and save_flags["annotated"]:
                if processed_image is not None and processed_image.size > 0:
                    annotated_frame = processed_image.copy()
                    annotate_yolo_frame(
                        self.image_utils,
                        annotated_frame,
                        detections,
                        color_result,
                        status,
                        missing_locations=missing_locations,
                        duplicate_filter=duplicate_filter,
                        raw_detections=raw_detections,
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
                    heatmap_dest_path = annotated_path
            elif (
                save_flags["annotated"]
                and detector_lower == "anomalib"
                and heatmap_path
                and os.path.exists(heatmap_path)
            ):
                heatmap_dest_path = annotated_path or bundle.annotated_path
                if heatmap_dest_path:
                    ensure_subpath(
                        heatmap_dest_path, self.allowed_root, must_exist=False
                    )
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

            # Original/processed/crop writes overlap with annotation work, but
            # every required artifact must be confirmed before publishing the
            # traceability record or returning SUCCESS.
            write_timeout_value = self._cfg_get("image_write_timeout_seconds", 30.0)
            write_timeout = max(
                0.1,
                float(30.0 if write_timeout_value is None else write_timeout_value),
            )
            self._img_queue.wait_for(image_receipts, timeout=write_timeout)

            test_id = self._excel.next_test_id()
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
                sequence_check=sequence_check,
                error_message=error_message,
                test_id=test_id,
            )
            try:
                excel_result = self._excel.append(excel_row)
                if excel_result is not None and not excel_result.success:
                    raise ResultExcelWriteError(
                        excel_result.error or "Excel flush failed"
                    )
            except ResultExcelWriteError:
                raise
            except Exception as exc:
                self.logger.logger.exception("Excel append failed")
                raise ResultExcelWriteError(str(exc)) from exc

            config_snapshot_path = self._write_config_snapshot(
                bundle,
                timestamp=timestamp,
                status=status,
                detector=detector,
                product=product,
                area=area,
                decision=decision,
                model_info=model_info,
                inference_time=inference_time,
                detections=detections,
                missing_items=missing_items,
                anomaly_score=anomaly_score,
                color_result=color_result,
                sequence_check=sequence_check,
                error_message=error_message,
                duplicate_filter=duplicate_filter,
                raw_detections=raw_detections,
                artifacts={
                    "original_path": original_path,
                    "preprocessed_path": preprocessed_path,
                    "annotated_path": annotated_path,
                    "heatmap_path": heatmap_dest_path,
                    "cropped_paths": list(cropped_paths),
                    "mask_paths": [
                        str(detection.get("mask_path"))
                        for detection in detections
                        if str(detection.get("mask_path") or "").strip()
                    ],
                },
            )
            if config_snapshot_path:
                try:
                    self._inspection_repository.upsert_snapshot_file(
                        config_snapshot_path
                    )
                    if self._inspection_sync is not None:
                        self._inspection_sync.notify()
                    self._inspection_maintenance.maybe_schedule()
                except (
                    OSError,
                    ValueError,
                    json.JSONDecodeError,
                    sqlite3.Error,
                ) as exc:
                    self.logger.logger.warning(
                        "Inspection database indexing failed: %s", exc
                    )

            return {
                "status": "SUCCESS",
                "inspection_id": bundle.inspection_id,
                "original_path": original_path,
                "preprocessed_path": preprocessed_path,
                "annotated_path": annotated_path,
                "heatmap_path": heatmap_dest_path,
                "cropped_paths": cropped_paths,
                "failure_crop_paths": [
                    path for path in cropped_paths if "_NG_" in os.path.basename(path)
                ],
                "product": product,
                "area": area,
                "missing_locations": missing_locations,
                "decision": dict(decision or {}),
                "model_info": dict(model_info or {}),
                "inference_time": inference_time,
                "duplicate_filter": dict(duplicate_filter or {}),
                "config_snapshot_path": config_snapshot_path,
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
        excel_result = self._excel.flush()
        if not excel_result.success:
            raise ResultExcelWriteError(excel_result.error or "Excel flush failed")
        self._img_queue.flush()

    def flush_async(self) -> None:
        """Schedule the Excel export without blocking the detection path."""
        self._excel.flush_async()

    def close(self) -> None:
        def _warn(action: str, exc: Exception) -> None:
            import sys
            print(f"WARNING: {action} during ResultHandler.close failed: {exc}", file=sys.stderr)

        operations = [
            ("Excel flush", self._excel.flush),
            ("Image queue flush", self._img_queue.flush),
            ("Image queue shutdown", self._img_queue.shutdown),
            ("Excel close", self._excel.close),
            ("Inspection maintenance", self._inspection_maintenance.close),
        ]
        if self._inspection_sync is not None:
            operations.insert(
                0,
                ("Inspection company sync", self._inspection_sync.close),
            )
        for label, fn in operations:
            try:
                fn()
            except Exception as exc:
                _warn(label, exc)

        try:
            stats = self._img_queue.stats
            import sys
            if stats.overflows:
                print(f"WARNING: Image queue overflow occurred {stats.overflows} times", file=sys.stderr)
            if stats.errors:
                print(f"WARNING: Image writer encountered {stats.errors} errors", file=sys.stderr)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_config(self, config) -> None:
        if is_dataclass(config) and not isinstance(config, type):
            cfg_dict = asdict(config)
        elif isinstance(config, dict):
            cfg_dict = dict(config)
        else:
            cfg_dict = config
        self._config_source = config
        self.config = cfg_dict

    def _cfg_get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.config, dict):
            value = self.config.get(key, default)
        else:
            value = getattr(self._config_source, key, default)
        if value.__class__.__module__ == "unittest.mock":
            return default
        return value

    def _write_config_snapshot(
        self,
        bundle,
        *,
        timestamp: datetime,
        status: str,
        detector: str,
        product: str | None,
        area: str | None,
        decision: dict[str, Any] | None,
        model_info: dict[str, Any] | None,
        inference_time: float | None,
        detections: list[dict[str, Any]] | None = None,
        missing_items: list[str] | None = None,
        anomaly_score: float | None = None,
        color_result: dict[str, Any] | None = None,
        sequence_check: dict[str, Any] | None = None,
        error_message: str | None = None,
        duplicate_filter: dict[str, Any] | None = None,
        raw_detections: list[dict[str, Any]] | None = None,
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        """Persist the full per-inspection result record (result.json).

        The file keeps the historical ``*_config_snapshot.json`` suffix because
        tools/collect_review_cases.py globs on it, but since schema_version 2 it
        is the complete traceability record for one inspection: verdict with
        merged fail reasons, detections, check outputs, artifact paths, and the
        runtime config (plus its hash) that produced them.
        """
        metadata_dir = os.path.join(bundle.base_path, "metadata", bundle.detector_prefix)
        ensure_subpath(metadata_dir, self.allowed_root, must_exist=False)
        os.makedirs(metadata_dir, exist_ok=True)
        stem, _ = os.path.splitext(bundle.image_name)
        snapshot_path = os.path.join(metadata_dir, f"{stem}_config_snapshot.json")
        ensure_subpath(snapshot_path, self.allowed_root, must_exist=False)
        safe_config = self._json_safe(self.config)
        payload = {
            "schema_version": 2,
            "inspection_id": bundle.inspection_id,
            "timestamp": timestamp.isoformat(),
            "status": status,
            "detector": detector,
            "product": product,
            "area": area,
            "equipment": self._equipment_metadata(area),
            "decision": dict(decision or {}),
            "fail_reasons": collect_fail_reasons(
                status=status,
                decision=decision,
                color_result=color_result,
                sequence_check=sequence_check,
                detector=detector,
                anomaly_score=anomaly_score,
                error_message=error_message,
            ),
            "model_info": dict(model_info or {}),
            "inference_time": inference_time,
            "detections": self._json_safe(list(detections or [])),
            "raw_detections": self._json_safe(
                list(raw_detections if raw_detections is not None else detections or [])
            ),
            "missing_items": list(missing_items or []),
            "anomaly_score": (
                float(anomaly_score) if anomaly_score is not None else None
            ),
            "color_result": self._json_safe(color_result or {}),
            "sequence_check": self._json_safe(sequence_check or {}),
            "duplicate_filter": self._json_safe(duplicate_filter or {}),
            "error_message": error_message or "",
            "artifacts": self._json_safe(artifacts or {}),
            "config_hash": self._hash_config(safe_config),
            "config": safe_config,
        }
        temporary_path = os.path.join(
            metadata_dir,
            f".{stem}_config_snapshot.{bundle.inspection_id}.tmp",
        )
        ensure_subpath(temporary_path, self.allowed_root, must_exist=False)
        try:
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, snapshot_path)
        except OSError as exc:
            self.logger.logger.exception("Config snapshot write failed")
            raise ResultPersistenceError(str(exc)) from exc
        finally:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass
        return snapshot_path

    def _ensure_disk_capacity(self, frame: Any, processed_image: Any) -> None:
        """Fail before persistence when the configured disk reserve is unsafe."""
        reserve_value = self._cfg_get("min_free_disk_mb", 1024)
        reserve_mb = max(0, int(1024 if reserve_value is None else reserve_value))
        if reserve_mb == 0:
            return
        estimated_bytes = sum(
            max(0, int(getattr(image, "nbytes", 0) or 0))
            for image in (frame, processed_image)
            if image is not None
        )
        try:
            free_bytes = shutil.disk_usage(self.base_dir).free
        except OSError as exc:
            raise ResultPersistenceError(
                f"Unable to verify result disk capacity: {exc}"
            ) from exc
        required_bytes = reserve_mb * 1024 * 1024 + estimated_bytes
        if free_bytes < required_bytes:
            raise ResultPersistenceError(
                "Insufficient result disk space: "
                f"free={free_bytes // (1024 * 1024)} MiB, "
                f"required={required_bytes // (1024 * 1024)} MiB"
            )

    def _equipment_metadata(self, area: str | None) -> dict[str, str]:
        """Return stable equipment identifiers stored with every inspection."""
        return {
            "machine_id": str(
                self._cfg_get("machine_id", None) or socket.gethostname()
            ),
            "station": str(self._cfg_get("station_id", None) or area or ""),
            "work_order": str(self._cfg_get("work_order", None) or ""),
            "camera_id": str(self._cfg_get("camera_id", None) or ""),
        }

    @staticmethod
    def _hash_config(safe_config: Any) -> str:
        """Return a short stable hash so runs can be compared without diffing."""
        serialized = json.dumps(
            safe_config, ensure_ascii=False, sort_keys=True, default=str
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]

    def _json_safe(self, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return self._json_safe(asdict(value))
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): self._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if hasattr(value, "__fspath__"):
            return os.fspath(value)
        if hasattr(value, "__dict__"):
            public_attrs = {
                key: item
                for key, item in vars(value).items()
                if not str(key).startswith("_")
            }
            return self._json_safe(public_attrs)
        return str(value)

    def _resolve_save_flags(self, status: str) -> dict[str, bool]:
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

    def _get_expected_boxes(
        self,
        product: str | None,
        area: str | None,
    ) -> dict[str, dict[str, Any]]:
        if not product or not area:
            return {}

        source = self._config_source
        try:
            if hasattr(source, "get_position_config"):
                cfg = source.get_position_config(product, area)
                boxes = cfg.get("expected_boxes", {}) if isinstance(cfg, dict) else {}
                return boxes if isinstance(boxes, dict) else {}
        except Exception:
            return {}

        try:
            position_config = self._cfg_get("position_config", {})
            area_cfg = position_config.get(product, {}).get(area, {})
            boxes = area_cfg.get("expected_boxes", {}) if isinstance(area_cfg, dict) else {}
            return boxes if isinstance(boxes, dict) else {}
        except Exception:
            return {}

