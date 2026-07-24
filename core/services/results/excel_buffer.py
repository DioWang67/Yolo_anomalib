"""具備備份與批次寫入的 Excel 緩衝寫手，降低寫入失敗風險。"""

from __future__ import annotations

import os
import shutil
import threading
import time
import uuid
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd  # type: ignore[import]
from openpyxl import load_workbook  # type: ignore[import]

from core.security import ensure_subpath


@dataclass
class ExcelFlushResult:
    success: bool
    rows_written: int = 0
    error: str | None = None


@dataclass
class ExcelWorkbookBuffer:
    """Buffered Excel writer with backups and optional periodic flush."""

    path: str
    columns: list[str]
    buffer_limit: int
    logger: Any
    flush_interval: float | None = None
    allowed_root: str | None = None
    workbook_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.buffer: list[list[Any]] = []
        self._buffer_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._async_threads_lock = threading.Lock()
        self._async_threads: set[threading.Thread] = set()
        if self.allowed_root:
            ensure_subpath(self.path, self.allowed_root, must_exist=False)
        self.backup_path = self.path + ".bak"
        if self.allowed_root:
            ensure_subpath(self.backup_path, self.allowed_root, must_exist=False)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            self._initialize_excel()
        self.wb = self._load_or_rebuild_workbook()
        self.ws = self.wb.active
        self._committed_max_row = int(self.ws.max_row)
        self._inflight_rows = 0
        self._closed = False
        self._timer: threading.Timer | None = None
        if self.flush_interval:
            self._timer = threading.Timer(
                self.flush_interval, self._periodic_flush
            )
            self._timer.daemon = True
            self._timer.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, row: Iterable[Any]) -> ExcelFlushResult | None:
        row_list = list(row)
        with self._buffer_lock:
            if self._closed:
                return ExcelFlushResult(
                    success=False,
                    error="Excel buffer is closed",
                )
            self.buffer.append(row_list)
            should_flush = len(self.buffer) >= max(1, self.buffer_limit)
        if should_flush:
            self.flush_async()
        return None

    def flush(self) -> ExcelFlushResult:
        # The whole flush (buffer drain + workbook mutation + save) must run
        # under the lock: the periodic timer thread, StorageWorker, and
        # close() can flush concurrently, and openpyxl workbooks are not
        # thread-safe — interleaved append/save corrupts the file.
        with self._write_lock:
            with self._buffer_lock:
                if not self.buffer:
                    return ExcelFlushResult(success=True, rows_written=0)
                rows = list(self.buffer)
                del self.buffer[: len(rows)]
                self._inflight_rows += len(rows)

            result = self._write_rows(rows)
            with self._buffer_lock:
                self._inflight_rows -= len(rows)
                if result.success:
                    self._committed_max_row += len(rows)
                else:
                    self.buffer[0:0] = rows
            return result

    def flush_async(self) -> None:
        """Flush buffered rows on a daemon thread without blocking detection."""
        with self._buffer_lock:
            if self._closed or not self.buffer:
                return

        thread = threading.Thread(
            target=self._async_flush_worker,
            daemon=True,
            name="excel-flush",
        )
        with self._async_threads_lock:
            self._async_threads.add(thread)
        thread.start()

    def _async_flush_worker(self) -> None:
        current = threading.current_thread()
        try:
            result = self.flush()
            if not result.success:
                self.logger.error(
                    "Background Excel flush failed; %d rows remain buffered: %s",
                    self.pending_rows(),
                    result.error,
                )
        finally:
            with self._async_threads_lock:
                self._async_threads.discard(current)

    def _write_rows(self, rows: list[list[Any]]) -> ExcelFlushResult:
        """Append one serialized batch to the workbook."""
        last_error = "flush_failed"
        for attempt in range(3):
            temporary_path = (
                f"{self.path}.{uuid.uuid4().hex}.tmp.xlsx"
            )
            if self.allowed_root:
                ensure_subpath(temporary_path, self.allowed_root, must_exist=False)
            candidate = None
            try:
                # Reload the committed workbook for every retry. Reusing the
                # previously mutated object would append the same rows again
                # after a failed save and create duplicate inspection records.
                candidate = load_workbook(self.path, **self.workbook_kwargs)
                worksheet = candidate.active
                for row in rows:
                    worksheet.append(row)
                candidate.save(temporary_path)
                os.replace(temporary_path, self.path)
                try:
                    self.wb.close()
                except (AttributeError, OSError):
                    pass
                self.wb = candidate
                self.ws = worksheet
                candidate = None
                self.logger.info(f"Excel 已更新: {self.path}")
                return ExcelFlushResult(success=True, rows_written=len(rows))
            except PermissionError as exc:
                last_error = str(exc)
                self.logger.error(

                        f"權限不足，無法寫入 {self.path}，"
                        "請檢查檔案是否開啟或權限設定"

                )
            except Exception as exc:
                last_error = str(exc)
                self.logger.error(f"寫入 Excel 發生錯誤 (第{attempt + 1}次重試): {exc}")
            finally:
                if candidate is not None:
                    candidate.close()
                try:
                    os.remove(temporary_path)
                except FileNotFoundError:
                    pass
            if attempt < 2:
                time.sleep(0.5)
        return ExcelFlushResult(
            success=False,
            rows_written=0,
            error=last_error,
        )

    def next_test_id(self, pending_count: int | None = None) -> int:
        with self._buffer_lock:
            pending = (
                self._inflight_rows + len(self.buffer)
                if pending_count is None
                else pending_count
            )
            return self._committed_max_row + pending

    def pending_rows(self) -> int:
        with self._buffer_lock:
            return self._inflight_rows + len(self.buffer)

    def close(self) -> None:
        with self._buffer_lock:
            if self._closed:
                return
            self._closed = True
            timer = self._timer
        if timer:
            timer.cancel()
        self._wait_for_async_flushes()
        self.flush()
        self.wb.close()

    def _wait_for_async_flushes(self) -> None:
        """Wait for already-scheduled writers before closing the workbook."""
        while True:
            with self._async_threads_lock:
                threads = list(self._async_threads)
            if not threads:
                return
            for thread in threads:
                thread.join()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _initialize_excel(self) -> None:
        df = pd.DataFrame(columns=self.columns)
        df.to_excel(self.path, index=False, engine="openpyxl")

    def _load_or_rebuild_workbook(self):
        try:
            return load_workbook(self.path, **self.workbook_kwargs)
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            corrupt_path = f"{self.path}.corrupt_{timestamp}"
            try:
                shutil.move(self.path, corrupt_path)
                self.logger.warning(
                    "Corrupt Excel workbook moved to %s; creating a new one (%s)",
                    corrupt_path,
                    exc,
                )
            except OSError:
                self.logger.warning(
                    "Corrupt Excel workbook could not be moved; recreating %s (%s)",
                    self.path,
                    exc,
                )
            self._initialize_excel()
            return load_workbook(self.path, **self.workbook_kwargs)

    def _periodic_flush(self) -> None:
        try:
            if not self._closed:
                self.flush()
        finally:
            next_timer = None
            with self._buffer_lock:
                if self.flush_interval and not self._closed:
                    next_timer = threading.Timer(
                        self.flush_interval, self._periodic_flush
                    )
                    next_timer.daemon = True
                    self._timer = next_timer
            if next_timer is not None:
                next_timer.start()


def format_excel_row(columns: list[str], data: dict[str, Any]) -> list[Any]:
    row: list[Any] = []
    for col in columns:
        value = data.get(col, "")
        if isinstance(value, datetime):
            value = value.strftime("%Y-%m-%d %H:%M:%S")
        row.append(value)
    return row
