"""Bounded asynchronous image decoding shared by review UI components."""

from __future__ import annotations

import itertools
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QImageReader

logger = logging.getLogger(__name__)

IMAGE_STATUS_LOADING = "loading"
IMAGE_STATUS_LOADED = "loaded"
IMAGE_STATUS_MISSING = "missing"
IMAGE_STATUS_PERMISSION_DENIED = "permission_denied"
IMAGE_STATUS_UNSUPPORTED = "unsupported"
IMAGE_STATUS_CORRUPT = "corrupt"
IMAGE_STATUS_CANCELLED = "cancelled"
IMAGE_STATUS_STALE = "stale"
IMAGE_STATUS_UNEXPECTED = "unexpected"

NEGATIVE_CACHE_STATUSES = frozenset(
    {
        IMAGE_STATUS_MISSING,
        IMAGE_STATUS_PERMISSION_DENIED,
        IMAGE_STATUS_UNSUPPORTED,
        IMAGE_STATUS_CORRUPT,
        IMAGE_STATUS_UNEXPECTED,
    }
)
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    f".{bytes(image_format).decode('ascii').lower()}"
    for image_format in QImageReader.supportedImageFormats()
)


@dataclass(frozen=True)
class ImageLoadResult:
    request_id: int
    token: Any
    purpose: str
    path: str
    status: str
    image: QImage | None
    error: str = ""
    from_cache: bool = False


@dataclass(frozen=True)
class _CacheValue:
    status: str
    image: QImage | None
    error: str
    cost_bytes: int
    expires_at: float | None


class BoundedImageCache:
    """UI-thread LRU constrained by both entries and estimated native bytes."""

    def __init__(self, *, max_entries: int, max_bytes: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self.max_bytes = max(1, int(max_bytes))
        self._entries: OrderedDict[str, _CacheValue] = OrderedDict()
        self._total_bytes = 0

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def get(self, key: str, *, now: float | None = None) -> _CacheValue | None:
        value = self._entries.pop(key, None)
        if value is None:
            return None
        current_time = time.monotonic() if now is None else now
        if value.expires_at is not None and value.expires_at <= current_time:
            self._total_bytes -= value.cost_bytes
            return None
        self._entries[key] = value
        return value

    def put(self, key: str, value: _CacheValue) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._total_bytes -= previous.cost_bytes
        self._entries[key] = value
        self._total_bytes += value.cost_bytes
        while (
            len(self._entries) > self.max_entries
            or self._total_bytes > self.max_bytes
        ):
            _old_key, old_value = self._entries.popitem(last=False)
            self._total_bytes -= old_value.cost_bytes

    def discard(self, key: str) -> None:
        value = self._entries.pop(key, None)
        if value is not None:
            self._total_bytes -= value.cost_bytes

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0

    def keys(self) -> tuple[str, ...]:
        return tuple(self._entries)


@dataclass(frozen=True)
class _DecodeSpec:
    cache_key: str
    path: str
    target_width: int
    target_height: int


@dataclass(frozen=True)
class _DecodedPayload:
    cache_key: str
    path: str
    status: str
    image: QImage | None
    error: str


class _WorkerSignals(QObject):
    finished = pyqtSignal(object)


class _DecodeTask(QRunnable):
    def __init__(self, spec: _DecodeSpec) -> None:
        super().__init__()
        self.spec = spec
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            payload = _decode_image(self.spec)
        except Exception as exc:  # defensive worker boundary
            payload = _DecodedPayload(
                cache_key=self.spec.cache_key,
                path=self.spec.path,
                status=IMAGE_STATUS_UNEXPECTED,
                image=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        self.signals.finished.emit(payload)


class AsyncImageService(QObject):
    """Decode QImage in a bounded pool and publish results on the UI thread."""

    result_ready = pyqtSignal(object)
    idle = pyqtSignal()

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        thumbnail_cache_entries: int = 128,
        thumbnail_cache_bytes: int = 24 * 1024 * 1024,
        full_cache_entries: int = 4,
        full_cache_bytes: int = 64 * 1024 * 1024,
        negative_ttl_seconds: float = 2.0,
        synchronous: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.synchronous = bool(synchronous)
        self.negative_ttl_seconds = max(0.0, float(negative_ttl_seconds))
        self.thumbnail_cache = BoundedImageCache(
            max_entries=thumbnail_cache_entries,
            max_bytes=thumbnail_cache_bytes,
        )
        self.full_cache = BoundedImageCache(
            max_entries=full_cache_entries,
            max_bytes=full_cache_bytes,
        )
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, max_workers or 2))
        self._request_ids = itertools.count(1)
        self._active = True
        self._inflight: dict[str, list[tuple[int, Any, str, str]]] = {}
        self._tasks: dict[str, _DecodeTask] = {}
        self._sha_source_fingerprints: dict[tuple[str, str], tuple[int, int]] = {}
        self.decode_count = 0

    @classmethod
    def from_environment(cls, *, parent: QObject | None = None) -> AsyncImageService:
        return cls(
            max_workers=_environment_int("YOLO_REVIEW_IMAGE_WORKERS", 2, minimum=1, maximum=8),
            thumbnail_cache_entries=_environment_int(
                "YOLO_REVIEW_THUMBNAIL_CACHE_ENTRIES", 128, minimum=16, maximum=4096
            ),
            thumbnail_cache_bytes=_environment_int(
                "YOLO_REVIEW_THUMBNAIL_CACHE_MB", 24, minimum=4, maximum=1024
            )
            * 1024
            * 1024,
            full_cache_entries=_environment_int(
                "YOLO_REVIEW_FULL_IMAGE_CACHE_ENTRIES", 4, minimum=1, maximum=128
            ),
            full_cache_bytes=_environment_int(
                "YOLO_REVIEW_FULL_IMAGE_CACHE_MB", 64, minimum=8, maximum=4096
            )
            * 1024
            * 1024,
            synchronous=_environment_flag("YOLO_REVIEW_SYNC_IMAGE_LOADING"),
            parent=parent,
        )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def pending_count(self) -> int:
        return sum(len(waiters) for waiters in self._inflight.values())

    def request_image(
        self,
        path: str | Path,
        *,
        purpose: str,
        token: Any,
        target_size: QSize | None = None,
        image_sha: str = "",
        priority: int = 0,
    ) -> int | None:
        if not self._active:
            return None
        if purpose not in {"thumbnail", "full"}:
            raise ValueError(f"Unsupported image purpose: {purpose}")
        request_id = next(self._request_ids)
        normalized_path, fingerprint, stat_status, stat_error = _source_fingerprint(path)
        width = max(0, target_size.width()) if target_size is not None else 0
        height = max(0, target_size.height()) if target_size is not None else 0
        cache_key = self._cache_key(
            normalized_path,
            fingerprint,
            purpose=purpose,
            width=width,
            height=height,
            image_sha=str(image_sha or "").strip(),
        )
        cache = self.thumbnail_cache if purpose == "thumbnail" else self.full_cache
        cached = cache.get(cache_key)
        if cached is not None:
            self._queue_result(
                ImageLoadResult(
                    request_id=request_id,
                    token=token,
                    purpose=purpose,
                    path=normalized_path,
                    status=cached.status,
                    image=cached.image,
                    error=cached.error,
                    from_cache=True,
                )
            )
            return request_id

        if stat_status is not None:
            self._store_and_queue_terminal_result(
                cache=cache,
                cache_key=cache_key,
                request_id=request_id,
                token=token,
                purpose=purpose,
                path=normalized_path,
                status=stat_status,
                error=stat_error,
            )
            return request_id

        waiter = (request_id, token, purpose, normalized_path)
        if cache_key in self._inflight:
            self._inflight[cache_key].append(waiter)
            return request_id
        self._inflight[cache_key] = [waiter]
        spec = _DecodeSpec(cache_key, normalized_path, width, height)
        if self.synchronous:
            self.decode_count += 1
            self._handle_decoded(_decode_image(spec))
            return request_id
        task = _DecodeTask(spec)
        task.signals.finished.connect(self._handle_decoded)
        self._tasks[cache_key] = task
        self.decode_count += 1
        self._pool.start(task, int(priority))
        return request_id

    def release_full_cache(self) -> None:
        self.full_cache.clear()

    def shutdown(self, *, wait_ms: int = 0) -> bool:
        if not self._active:
            return self._pool.waitForDone(max(0, int(wait_ms))) if wait_ms else True
        self._active = False
        self._pool.clear()
        self._inflight.clear()
        self._tasks.clear()
        self.thumbnail_cache.clear()
        self.full_cache.clear()
        if wait_ms <= 0:
            return True
        return self._pool.waitForDone(max(0, int(wait_ms)))

    def _cache_key(
        self,
        path: str,
        fingerprint: tuple[int, int] | None,
        *,
        purpose: str,
        width: int,
        height: int,
        image_sha: str,
    ) -> str:
        variant = f"{purpose}:{width}x{height}"
        if image_sha:
            source_key = (image_sha, path)
            previous = self._sha_source_fingerprints.get(source_key)
            if fingerprint is not None and previous is not None and previous != fingerprint:
                cache = self.thumbnail_cache if purpose == "thumbnail" else self.full_cache
                cache.discard(f"sha:{image_sha}|{variant}")
            if fingerprint is not None:
                self._sha_source_fingerprints[source_key] = fingerprint
            return f"sha:{image_sha}|{variant}"
        fingerprint_text = "missing" if fingerprint is None else f"{fingerprint[0]}:{fingerprint[1]}"
        return f"path:{path}|{fingerprint_text}|{variant}"

    def _store_and_queue_terminal_result(
        self,
        *,
        cache: BoundedImageCache,
        cache_key: str,
        request_id: int,
        token: Any,
        purpose: str,
        path: str,
        status: str,
        error: str,
    ) -> None:
        value = self._cache_value(status=status, image=None, error=error)
        cache.put(cache_key, value)
        self._queue_result(
            ImageLoadResult(
                request_id=request_id,
                token=token,
                purpose=purpose,
                path=path,
                status=status,
                image=None,
                error=error,
            )
        )

    def _handle_decoded(self, payload: _DecodedPayload) -> None:
        waiters = self._inflight.pop(payload.cache_key, [])
        self._tasks.pop(payload.cache_key, None)
        if not self._active:
            return
        purpose = waiters[0][2] if waiters else "thumbnail"
        cache = self.thumbnail_cache if purpose == "thumbnail" else self.full_cache
        cache.put(
            payload.cache_key,
            self._cache_value(
                status=payload.status,
                image=payload.image,
                error=payload.error,
            ),
        )
        if payload.status == IMAGE_STATUS_UNEXPECTED:
            logger.error("Image worker failed path=%s error=%s", payload.path, payload.error)
        for request_id, token, waiter_purpose, path in waiters:
            self.result_ready.emit(
                ImageLoadResult(
                    request_id=request_id,
                    token=token,
                    purpose=waiter_purpose,
                    path=path,
                    status=payload.status,
                    image=payload.image,
                    error=payload.error,
                )
            )
        if not self._inflight:
            self.idle.emit()

    def _cache_value(
        self,
        *,
        status: str,
        image: QImage | None,
        error: str,
    ) -> _CacheValue:
        expires_at = (
            time.monotonic() + self.negative_ttl_seconds
            if status in NEGATIVE_CACHE_STATUSES
            else None
        )
        cost_bytes = 1
        if image is not None and not image.isNull():
            cost_bytes = max(1, image.bytesPerLine() * image.height())
        return _CacheValue(status, image, error, cost_bytes, expires_at)

    def _queue_result(self, result: ImageLoadResult) -> None:
        if self.synchronous:
            self._emit_if_active(result)
        else:
            QTimer.singleShot(0, lambda: self._emit_if_active(result))

    def _emit_if_active(self, result: ImageLoadResult) -> None:
        if self._active:
            self.result_ready.emit(result)


def _decode_image(spec: _DecodeSpec) -> _DecodedPayload:
    reader = QImageReader(spec.path)
    reader.setAutoTransform(True)
    if spec.target_width > 0 and spec.target_height > 0:
        source_size = reader.size()
        if source_size.isValid():
            reader.setScaledSize(
                source_size.scaled(
                    QSize(spec.target_width, spec.target_height),
                    Qt.KeepAspectRatio,
                )
            )
    image = reader.read()
    if not image.isNull():
        return _DecodedPayload(
            spec.cache_key,
            spec.path,
            IMAGE_STATUS_LOADED,
            image,
            "",
        )
    status = (
        IMAGE_STATUS_CORRUPT
        if Path(spec.path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        else IMAGE_STATUS_UNSUPPORTED
    )
    return _DecodedPayload(
        spec.cache_key,
        spec.path,
        status,
        None,
        reader.errorString(),
    )


def _source_fingerprint(
    path_value: str | Path,
) -> tuple[str, tuple[int, int] | None, str | None, str]:
    path = Path(str(path_value or ""))
    try:
        normalized = str(path.resolve(strict=False))
        stat = path.stat()
    except FileNotFoundError:
        return str(path.resolve(strict=False)), None, IMAGE_STATUS_MISSING, "File does not exist"
    except PermissionError as exc:
        return str(path.resolve(strict=False)), None, IMAGE_STATUS_PERMISSION_DENIED, str(exc)
    except OSError as exc:
        return str(path.resolve(strict=False)), None, IMAGE_STATUS_UNEXPECTED, str(exc)
    if not path.is_file():
        return normalized, None, IMAGE_STATUS_MISSING, "Path is not a file"
    return normalized, (stat.st_mtime_ns, stat.st_size), None, ""


def _environment_flag(name: str) -> bool:
    return str(os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _environment_int(
    name: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(str(os.environ.get(name) or default))
    except ValueError:
        value = default
    return min(max(value, minimum), maximum)
