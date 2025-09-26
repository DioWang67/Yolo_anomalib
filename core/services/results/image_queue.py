from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2


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


class ImageWriteQueue:
    """Asynchronous image writer with overflow fallback."""

    def __init__(
        self,
        logger,
        maxsize: int = 1000,
        warn_threshold: float = 0.8,
    ) -> None:
        self.logger = logger
        self.maxsize = max(0, maxsize)
        self.warn_threshold = None
        if self.maxsize > 0 and 0 < warn_threshold < 1:
            self.warn_threshold = int(self.maxsize * warn_threshold)
        self._queue: queue.Queue = queue.Queue(maxsize=self.maxsize)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._stats = ImageWriteStats()
        self._worker.start()

    @property
    def stats(self) -> ImageWriteStats:
        return self._stats

    def write_sync(
        self, path: str, image, params: Optional[Sequence[int]] = None
    ) -> None:
        self._write_sync(path, image, params)

    def enqueue(self, path: str, image, params: Optional[Sequence[int]] = None) -> None:
        if self.maxsize == 0:
            self._write_sync(path, image, params)
            return
        try:
            self._queue.put_nowait((path, image, params))
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
            self._stats.overflows += 1
            self.logger.warning(
                f"Image queue full ({self._stats.overflows} overflows); writing synchronously for {path}"
            )
            self._write_sync(path, image, params)

    def flush(self) -> None:
        try:
            self._queue.join()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        try:
            self._worker.join(timeout=5)
        except Exception:
            pass
        remaining = getattr(self._queue, "unfinished_tasks", 0)
        if remaining:
            self.logger.warning(
                f"Image queue shutdown with {remaining} pending tasks")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_sync(
        self, path: str, image, params: Optional[Sequence[int]] = None
    ) -> None:
        try:
            if params is not None:
                ok = cv2.imwrite(path, image, params)
            else:
                ok = cv2.imwrite(path, image)
            if ok is False:
                raise IOError("cv2.imwrite returned False")
        except Exception as exc:
            self._stats.errors += 1
            self.logger.error(f"Image write failed ({path}): {exc}")
            recovered = False
            try:
                ok = cv2.imwrite(path, image)
                if ok is False:
                    raise IOError("cv2.imwrite returned False")
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
            path, image, params = item
            try:
                self._write_sync(path, image, params)
            except Exception:
                pass
            finally:
                self._queue.task_done()

        # Drain remaining items synchronously when stopping
        while True:
            try:
                item = self._queue.get_nowait()
            except Exception:
                break
            if item is None:
                continue
            path, image, params = item
            try:
                self._write_sync(path, image, params)
            except Exception:
                pass
            finally:
                self._queue.task_done()
