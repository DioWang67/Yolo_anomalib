from core.services.results.image_queue import ImageWriteQueue


class DummyLogger:
    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def test_shutdown_is_idempotent_and_completes_sentinel_task():
    image_queue = ImageWriteQueue(DummyLogger())

    image_queue.shutdown()
    image_queue.shutdown()

    assert image_queue._queue.unfinished_tasks == 0
    assert not image_queue._worker.is_alive()
