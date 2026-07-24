import os
import threading

import pytest
from PyQt5 import sip
from PyQt5.QtCore import QObject, QSize, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication

from app.gui import async_image_service as image_service_module
from app.gui.async_image_service import (
    IMAGE_STATUS_CORRUPT,
    IMAGE_STATUS_LOADED,
    IMAGE_STATUS_MISSING,
    IMAGE_STATUS_UNEXPECTED,
    AsyncImageService,
    BoundedImageCache,
    ImageLoadResult,
    _CacheValue,
)
from app.gui.review_selection_gallery import GALLERY_PAGE_SIZE, ReviewSelectionGallery
from app.gui.review_workspace import ReviewImageViewer


class ControlledImageService(QObject):
    result_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.active = True
        self.synchronous = False
        self.requests = []
        self.shutdown_calls = 0

    def request_image(self, path, **kwargs):
        request_id = len(self.requests) + 1
        self.requests.append((request_id, str(path), kwargs))
        return request_id

    def emit_result(self, request_position, *, image=None, status=IMAGE_STATUS_LOADED):
        request_id, path, kwargs = self.requests[request_position]
        self.result_ready.emit(
            ImageLoadResult(
                request_id=request_id,
                token=kwargs["token"],
                purpose=kwargs["purpose"],
                path=path,
                status=status,
                image=image,
            )
        )

    def shutdown(self, **_kwargs):
        self.shutdown_calls += 1
        self.active = False
        return True

    def release_full_cache(self):
        return None


def _save_image(path, *, width=32, height=24, color=Qt.white):
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(color)
    assert image.save(str(path))
    return image


@pytest.fixture
def async_service():
    service = AsyncImageService(max_workers=2)
    yield service
    service.shutdown(wait_ms=3000)


def test_lru_cache_enforces_entry_and_byte_limits():
    cache = BoundedImageCache(max_entries=2, max_bytes=5)
    value = _CacheValue(IMAGE_STATUS_LOADED, None, "", 2, None)
    cache.put("a", value)
    cache.put("b", value)
    assert cache.get("a") is value

    cache.put("c", value)

    assert cache.keys() == ("a", "c")
    assert cache.entry_count == 2
    assert cache.total_bytes == 4


def test_same_image_requests_share_one_decode(tmp_path, qtbot, async_service):
    path = tmp_path / "shared.png"
    _save_image(path)
    results = []
    async_service.result_ready.connect(results.append)

    async_service.request_image(
        path,
        purpose="thumbnail",
        token="first",
        target_size=QSize(16, 16),
    )
    async_service.request_image(
        path,
        purpose="thumbnail",
        token="second",
        target_size=QSize(16, 16),
    )

    qtbot.waitUntil(lambda: len(results) == 2)
    assert async_service.decode_count == 1
    assert {result.token for result in results} == {"first", "second"}


def test_filter_reuses_cached_thumbnail_without_new_decode(tmp_path, qtbot):
    path = tmp_path / "cached.png"
    _save_image(path)
    service = AsyncImageService(max_workers=1)
    gallery = ReviewSelectionGallery(language="en", image_service=service)
    qtbot.addWidget(gallery)
    gallery.resize(900, 600)
    gallery.show()
    gallery.set_entries(
        [
            (index, {"status": "FAIL", "original_path": str(path)})
            for index in range(5)
        ],
        set(),
    )
    try:
        qtbot.waitUntil(lambda: gallery.loaded_thumbnail_count == 5)
        assert service.decode_count == 1

        gallery.filter_combo.setCurrentIndex(
            gallery.filter_combo.findData("unreviewed")
        )

        qtbot.waitUntil(lambda: gallery.loaded_thumbnail_count == 5)
        assert service.decode_count == 1
    finally:
        service.shutdown(wait_ms=3000)


def test_missing_and_corrupt_results_are_distinct(tmp_path, qtbot, async_service):
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    results = []
    async_service.result_ready.connect(results.append)

    async_service.request_image(
        tmp_path / "missing.png",
        purpose="full",
        token="missing",
    )
    async_service.request_image(corrupt, purpose="full", token="corrupt")

    qtbot.waitUntil(lambda: len(results) == 2)
    statuses = {result.token: result.status for result in results}
    assert statuses == {"missing": IMAGE_STATUS_MISSING, "corrupt": IMAGE_STATUS_CORRUPT}


def test_file_change_invalidates_cached_decode(tmp_path, qtbot):
    service = AsyncImageService(max_workers=1, synchronous=True)
    path = tmp_path / "mutable.png"
    _save_image(path, width=16, height=16)
    results = []
    service.result_ready.connect(results.append)
    try:
        service.request_image(path, purpose="full", token="before")
        assert service.decode_count == 1
        _save_image(path, width=48, height=32, color=Qt.black)
        os.utime(path, None)

        service.request_image(path, purpose="full", token="after")

        assert service.decode_count == 2
        assert [result.token for result in results] == ["before", "after"]
    finally:
        service.shutdown()


def test_expired_negative_cache_entry_is_removed():
    cache = BoundedImageCache(max_entries=2, max_bytes=10)
    cache.put(
        "negative",
        _CacheValue(IMAGE_STATUS_MISSING, None, "missing", 1, 5.0),
    )

    assert cache.get("negative", now=4.0) is not None
    assert cache.get("negative", now=5.0) is None
    assert cache.entry_count == 0


def test_full_image_cache_is_bounded_and_releasable(tmp_path):
    service = AsyncImageService(
        max_workers=1,
        synchronous=True,
        full_cache_entries=2,
        full_cache_bytes=8 * 1024 * 1024,
    )
    try:
        for index in range(3):
            path = tmp_path / f"full-{index}.png"
            _save_image(path, color=(Qt.red, Qt.green, Qt.blue)[index])
            service.request_image(path, purpose="full", token=index)

        assert service.full_cache.entry_count == 2
        assert service.full_cache.total_bytes <= service.full_cache.max_bytes
        service.release_full_cache()
        assert service.full_cache.entry_count == 0
    finally:
        service.shutdown()


def test_worker_exception_does_not_stop_following_request(
    tmp_path, qtbot, monkeypatch, async_service
):
    broken = tmp_path / "broken.png"
    valid = tmp_path / "valid.png"
    _save_image(broken)
    _save_image(valid)
    real_decode = image_service_module._decode_image

    def controlled_decode(spec):
        if spec.path == str(broken.resolve()):
            raise RuntimeError("controlled decode failure")
        return real_decode(spec)

    monkeypatch.setattr(image_service_module, "_decode_image", controlled_decode)
    results = []
    async_service.result_ready.connect(results.append)

    async_service.request_image(broken, purpose="full", token="broken")
    async_service.request_image(valid, purpose="full", token="valid")

    qtbot.waitUntil(lambda: len(results) == 2)
    statuses = {result.token: result.status for result in results}
    assert statuses["broken"] == IMAGE_STATUS_UNEXPECTED
    assert statuses["valid"] == IMAGE_STATUS_LOADED


def test_results_are_delivered_on_qt_ui_thread(tmp_path, qtbot, async_service):
    path = tmp_path / "thread.png"
    _save_image(path)
    delivered_on_ui_thread = []
    async_service.result_ready.connect(
        lambda _result: delivered_on_ui_thread.append(
            QThread.currentThread() is QApplication.instance().thread()
        )
    )

    async_service.request_image(path, purpose="full", token="thread")

    qtbot.waitUntil(lambda: bool(delivered_on_ui_thread))
    assert delivered_on_ui_thread == [True]


def test_shutdown_discards_late_running_result(tmp_path, qtbot, monkeypatch):
    service = AsyncImageService(max_workers=1)
    path = tmp_path / "late.png"
    _save_image(path)
    entered = threading.Event()
    release = threading.Event()
    real_decode = image_service_module._decode_image

    def delayed_decode(spec):
        entered.set()
        release.wait(timeout=2.0)
        return real_decode(spec)

    monkeypatch.setattr(image_service_module, "_decode_image", delayed_decode)
    results = []
    service.result_ready.connect(results.append)
    service.request_image(path, purpose="full", token="late")
    qtbot.waitUntil(entered.is_set)

    service.shutdown()
    release.set()
    service._pool.waitForDone(3000)
    QApplication.processEvents()

    assert results == []


def test_queued_result_ignores_deleted_service(qtbot):
    service = AsyncImageService(max_workers=1)
    result = ImageLoadResult(
        request_id=1,
        token="deleted",
        purpose="thumbnail",
        path="missing.png",
        status=IMAGE_STATUS_MISSING,
        image=None,
    )

    service._queue_result(result)
    sip.delete(service)
    QApplication.processEvents()


def test_gallery_ignores_queued_work_after_list_is_deleted(qtbot):
    service = ControlledImageService()
    gallery = ReviewSelectionGallery(language="en", image_service=service)
    qtbot.addWidget(gallery)
    result = ImageLoadResult(
        request_id=1,
        token=(id(gallery), gallery._generation, 0, "missing.png"),
        purpose="thumbnail",
        path="missing.png",
        status=IMAGE_STATUS_MISSING,
        image=None,
    )

    sip.delete(gallery.thumbnail_list)
    gallery._request_visible_thumbnails()
    gallery._request_thumbnail_at(0, priority=0)
    gallery._on_image_result(result)


def test_gallery_pages_large_result_before_requesting_visible_images(qtbot):
    service = ControlledImageService()
    gallery = ReviewSelectionGallery(language="en", image_service=service)
    qtbot.addWidget(gallery)
    gallery.resize(1200, 700)
    gallery.show()
    entries = [
        (
            index,
            {
                "status": "FAIL",
                "timestamp": str(index),
                "original_path": f"missing-{index}.png",
            },
        )
        for index in range(1000)
    ]

    gallery.set_entries(entries, set())

    assert gallery.thumbnail_list.count() == GALLERY_PAGE_SIZE
    assert len(gallery.source_entry_indices()) == 1000
    assert gallery.load_more_button.isHidden() is False
    assert service.requests == []
    qtbot.waitUntil(lambda: bool(service.requests), timeout=1000)
    assert 0 < len(service.requests) < GALLERY_PAGE_SIZE
    assert service.requests[0][2]["priority"] == 10


def test_scrolling_requests_a_new_thumbnail_range(qtbot):
    service = ControlledImageService()
    gallery = ReviewSelectionGallery(language="en", image_service=service)
    qtbot.addWidget(gallery)
    gallery.resize(900, 600)
    gallery.show()
    gallery.set_entries(
        [
            (index, {"status": "FAIL", "original_path": f"image-{index}.png"})
            for index in range(100)
        ],
        set(),
    )
    qtbot.waitUntil(lambda: bool(service.requests), timeout=1000)
    qtbot.waitUntil(
        lambda: gallery.thumbnail_list.verticalScrollBar().maximum() > 0,
        timeout=1000,
    )
    initially_requested = {
        request[2]["token"][2] for request in service.requests
    }

    gallery.thumbnail_list.verticalScrollBar().setValue(
        gallery.thumbnail_list.verticalScrollBar().maximum()
    )
    qtbot.waitUntil(
        lambda: any(
            request[2]["token"][2] not in initially_requested
            for request in service.requests
        ),
        timeout=1000,
    )
    requested_after_scroll = {
        request[2]["token"][2] for request in service.requests
    }

    assert requested_after_scroll - initially_requested
    assert max(requested_after_scroll) > max(initially_requested)


def test_fast_full_image_switch_only_renders_latest_result(tmp_path, qtbot):
    service = ControlledImageService()
    viewer = ReviewImageViewer(language="en", image_service=service)
    qtbot.addWidget(viewer)
    paths = [tmp_path / f"{name}.png" for name in ("a", "b", "c")]
    images = [
        _save_image(path, color=color)
        for path, color in zip(paths, (Qt.red, Qt.green, Qt.blue), strict=True)
    ]

    for path in paths:
        viewer.set_images(original_path=path, overlay_path=path, sample_id=path.stem)
    service.emit_result(2, image=images[2])
    assert viewer.image_state == IMAGE_STATUS_LOADED
    latest_cache_key = viewer._pixmap.cacheKey()

    service.emit_result(1, image=images[1])
    service.emit_result(0, image=images[0])

    assert viewer.image_state == IMAGE_STATUS_LOADED
    assert viewer._pixmap.cacheKey() == latest_cache_key


def test_shared_service_ignores_other_full_image_token_shapes(qtbot):
    service = ControlledImageService()
    viewer = ReviewImageViewer(language="en", image_service=service)
    qtbot.addWidget(viewer)

    service.result_ready.emit(
        ImageLoadResult(
            request_id=99,
            token=("large-preview", "path"),
            purpose="full",
            path="path",
            status=IMAGE_STATUS_MISSING,
            image=None,
        )
    )

    assert viewer.image_state == IMAGE_STATUS_MISSING


def test_viewer_close_rejects_late_controlled_result(tmp_path, qtbot):
    service = ControlledImageService()
    viewer = ReviewImageViewer(language="en", image_service=service)
    qtbot.addWidget(viewer)
    path = tmp_path / "late.png"
    image = _save_image(path)
    viewer.set_images(original_path=path, overlay_path=path)

    viewer.close()
    service.emit_result(0, image=image)

    assert viewer._pixmap.isNull()


def test_sync_loading_feature_flag_is_opt_in(monkeypatch):
    monkeypatch.setenv("YOLO_REVIEW_SYNC_IMAGE_LOADING", "1")
    service = AsyncImageService.from_environment()
    try:
        assert service.synchronous is True
    finally:
        service.shutdown()


def test_sync_fallback_decodes_gallery_before_set_entries_returns(tmp_path, qtbot):
    path = tmp_path / "sync.png"
    _save_image(path)
    service = AsyncImageService(max_workers=1, synchronous=True)
    gallery = ReviewSelectionGallery(language="en", image_service=service)
    qtbot.addWidget(gallery)
    try:
        gallery.set_entries(
            [
                (index, {"status": "FAIL", "original_path": str(path)})
                for index in range(5)
            ],
            set(),
        )

        assert gallery.loaded_thumbnail_count == 5
    finally:
        service.shutdown()
