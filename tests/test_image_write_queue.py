import pytest

from core.services.results import image_queue as image_queue_module
from core.services.results.image_queue import ImageWriteError, ImageWriteQueue


class DummyLogger:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


class FakeImage:
    nbytes = 16


def test_shutdown_is_idempotent_and_completes_sentinel_task():
    image_queue = ImageWriteQueue(DummyLogger())

    image_queue.shutdown()
    image_queue.shutdown()

    assert image_queue._queue.unfinished_tasks == 0
    assert not image_queue._worker.is_alive()


def test_queued_write_receipt_reports_background_failure(monkeypatch):
    monkeypatch.setattr(image_queue_module.cv2, "imwrite", lambda *_a, **_k: False)
    image_queue = ImageWriteQueue(DummyLogger(), maxsize=2, max_bytes=1024)

    receipt = image_queue.enqueue("broken.jpg", FakeImage())

    with pytest.raises(ImageWriteError):
        receipt.wait(timeout=1.0)
    image_queue.shutdown()


def test_memory_budget_falls_back_to_completed_sync_write(monkeypatch):
    monkeypatch.setattr(image_queue_module.cv2, "imwrite", lambda *_a, **_k: True)
    image_queue = ImageWriteQueue(DummyLogger(), maxsize=2, max_bytes=1)

    receipt = image_queue.enqueue("sync.jpg", FakeImage())
    receipt.wait(timeout=0.1)

    assert image_queue.stats.overflows == 1
    assert image_queue._queue.unfinished_tasks == 0
    image_queue.shutdown()
