"""以執行緒佇列非同步寫入影像，避免阻塞主流程。"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2

from core.security import ensure_subpath


class ImageWriteError(RuntimeError):
    """Raised when cv2.imwrite fails (even if fallback succeeds)."""

    def __init__(
        self, path: str, original_exc: Exception, recovered: bool = False
    ) -> None:
        self.path = path
        self.original_exc = original_exc
        self.recovered = recovered
        msg = f"Image write failed for {path}: {original_exc}"
        if recovered:
            msg += " (recovered)"
        super().__init__(msg)


@dataclass
class ImageWriteStats:
    overflows: int = 0
    errors: int = 0
    peak_queued_bytes: int = 0


@dataclass
class ImageWriteReceipt:
    """Completion handle for one queued image write."""

    path: str
    _done: threading.Event = field(default_factory=threading.Event, repr=False)
    error: Exception | None = None

    def wait(self, timeout: float | None = None) -> None:
        if not self._done.wait(timeout):
            raise ImageWriteError(
                self.path,
                TimeoutError(f"Timed out waiting for image write: {self.path}"),
            )
        if self.error is not None:
            raise self.error


class ImageWriteQueue:
    """Asynchronous image writer with overflow fallback."""

    def __init__(
        self,
        logger,
        maxsize: int = 8,
        max_bytes: int = 256 * 1024 * 1024,
        warn_threshold: float = 0.8,
        allowed_root: str | None = None,
    ) -> None:
        self.logger = logger
        self.allowed_root = allowed_root
        self.maxsize = max(0, maxsize)
        self.max_bytes = max(0, int(max_bytes))
        self.warn_threshold = None
        if self.maxsize > 0 and 0 < warn_threshold < 1:
            self.warn_threshold = int(self.maxsize * warn_threshold)
        self._queue: queue.Queue = queue.Queue(maxsize=self.maxsize)
        self._stop_event = threading.Event()
        self._shutdown_lock = threading.Lock()
        self._bytes_lock = threading.Lock()
        self._queued_bytes = 0
        self._is_shutdown = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._stats = ImageWriteStats()
        self._worker.start()

    @property
    def stats(self) -> ImageWriteStats:
        return self._stats

    def write_sync(
        self, path: str, image, params: Sequence[int] | None = None
    ) -> None:
        self._write_sync(path, image, params)

    def enqueue(
        self,
        path: str,
        image: Any,
        params: Sequence[int] | None = None,
    ) -> ImageWriteReceipt:
        receipt = ImageWriteReceipt(path=path)
        if self.maxsize == 0:
            self._complete_sync(receipt, path, image, params)
            return receipt

        image_bytes = max(0, int(getattr(image, "nbytes", 0) or 0))
        if not self._reserve_bytes(image_bytes):
            self._stats.overflows += 1
            self.logger.warning(
                "Image queue memory budget exceeded; writing synchronously for %s",
                path,
            )
            self._complete_sync(receipt, path, image, params)
            return receipt
        try:
            self._queue.put_nowait((path, image, params, receipt, image_bytes))
            if self.warn_threshold:
                try:
                    qsize = self._queue.qsize()
                except NotImplementedError:
                    qsize = None
                if qsize is not None and qsize >= self.warn_threshold:
                    maxsize = self.maxsize or "unbounded"
                    self.logger.warning(
                        f"Image queue backlog at {qsize}/{maxsize} items"
                    )
        except queue.Full:
            self._release_bytes(image_bytes)
            self._stats.overflows += 1
            self.logger.warning(
                f"Image queue full ({self._stats.overflows} overflows); writing synchronously for {path}"
            )
            self._complete_sync(receipt, path, image, params)
        return receipt

    @staticmethod
    def wait_for(
        receipts: Sequence[ImageWriteReceipt], timeout: float | None = None
    ) -> None:
        """Raise when any write in ``receipts`` failed or did not finish."""
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        for receipt in receipts:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            receipt.wait(remaining)

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for queued writes with a bounded total shutdown budget."""
        deadline = time.monotonic() + max(0.0, timeout)
        while getattr(self._queue, "unfinished_tasks", 0):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for the image write queue to drain"
                )
            time.sleep(0.05)

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._is_shutdown:
                return
            self._is_shutdown = True
            self._stop_event.set()
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
        try:
            self._worker.join(timeout=5)
        except Exception:
            pass
        remaining = getattr(self._queue, "unfinished_tasks", 0)
        if remaining:
            import sys
            print(f"WARNING: Image queue shutdown with {remaining} pending tasks", file=sys.stderr)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_sync(
        self, path: str, image, params: Sequence[int] | None = None
    ) -> None:
        try:
            if self.allowed_root:
                path = str(ensure_subpath(path, self.allowed_root, must_exist=False))
            if params is not None:
                ok = cv2.imwrite(path, image, params)
            else:
                ok = cv2.imwrite(path, image)
            if ok is False:
                raise OSError("cv2.imwrite returned False")
        except Exception as exc:
            self._stats.errors += 1
            self.logger.error(f"Image write failed ({path}): {exc}")
            recovered = False
            try:
                ok = cv2.imwrite(path, image)
                if ok is False:
                    raise OSError("cv2.imwrite returned False")
                recovered = True
            except Exception as fallback_exc:
                self._stats.errors += 1
                self.logger.error(
                    f"Image write fallback failed ({path}): {fallback_exc}"
                )
                raise ImageWriteError(
                    path, fallback_exc, recovered=False
                ) from fallback_exc
            raise ImageWriteError(path, exc, recovered=recovered) from exc

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break
            path, image, params, receipt, image_bytes = item
            self._release_bytes(image_bytes)
            try:
                self._write_sync(path, image, params)
            except Exception as exc:
                receipt.error = exc
            finally:
                receipt._done.set()
                self._queue.task_done()

        # Drain remaining items synchronously when stopping
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._queue.task_done()
                continue
            path, image, params, receipt, image_bytes = item
            self._release_bytes(image_bytes)
            try:
                self._write_sync(path, image, params)
            except Exception as exc:
                receipt.error = exc
            finally:
                receipt._done.set()
                self._queue.task_done()

    def _complete_sync(
        self,
        receipt: ImageWriteReceipt,
        path: str,
        image: Any,
        params: Sequence[int] | None,
    ) -> None:
        try:
            self._write_sync(path, image, params)
        except Exception as exc:
            receipt.error = exc
            raise
        finally:
            receipt._done.set()

    def _reserve_bytes(self, image_bytes: int) -> bool:
        with self._bytes_lock:
            if self.max_bytes and self._queued_bytes + image_bytes > self.max_bytes:
                return False
            self._queued_bytes += image_bytes
            self._stats.peak_queued_bytes = max(
                self._stats.peak_queued_bytes, self._queued_bytes
            )
            return True

    def _release_bytes(self, image_bytes: int) -> None:
        with self._bytes_lock:
            self._queued_bytes = max(0, self._queued_bytes - image_bytes)
