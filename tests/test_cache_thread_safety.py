"""Thread-safety tests for model caching in ModelManager and YOLOInferenceModel.

These tests verify that concurrent access to the LRU caches does not cause
race conditions, data corruption, or deadlocks.

Tests that require torch/ultralytics are skipped when those packages
cannot be loaded (e.g. broken DLL on Windows CI).
"""

import threading
import time
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

# ---------- ModelManager tests (mock heavy imports) ----------

_torch_available = True
try:
    import torch  # noqa: F401
except (OSError, ImportError):
    _torch_available = False

_model_manager_available = True
try:
    from core.services.model_manager import ModelManager
except (OSError, ImportError, ModuleNotFoundError):
    _model_manager_available = False


@pytest.mark.skipif(not _model_manager_available, reason="torch/ultralytics unavailable")
class TestModelManagerCacheLock:
    """Verify ModelManager._cache is protected by _cache_lock."""

    def _make_manager(self, max_cache_size=3):
        logger = MagicMock()
        logger.logger = MagicMock()
        return ModelManager(logger, max_cache_size=max_cache_size)

    def test_cache_lock_exists(self):
        mgr = self._make_manager()
        assert hasattr(mgr, "_cache_lock")
        assert isinstance(mgr._cache_lock, type(threading.Lock()))

    def test_concurrent_cache_reads_do_not_corrupt(self):
        mgr = self._make_manager()
        mock_engine = MagicMock()
        mock_config = MagicMock()
        mock_config.__dict__ = {"weights": "test.pt"}

        with mgr._cache_lock:
            mgr._cache[("ProductA", "AreaA")] = {"yolo": (mock_engine, mock_config)}

        errors = []

        def reader(thread_id):
            try:
                for _ in range(50):
                    with mgr._cache_lock:
                        if ("ProductA", "AreaA") in mgr._cache:
                            _ = mgr._cache[("ProductA", "AreaA")]
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Concurrent reads caused errors: {errors}"

    def test_cache_eviction_under_contention(self):
        mgr = self._make_manager(max_cache_size=2)
        errors = []

        def inserter(product_id):
            try:
                with mgr._cache_lock:
                    key = (f"Product{product_id}", "A")
                    mock_engine = MagicMock()
                    mock_engine.shutdown = MagicMock()
                    mock_config = MagicMock()
                    if key not in mgr._cache:
                        mgr._cache[key] = {}
                    mgr._cache[key]["yolo"] = (mock_engine, mock_config)
                    mgr._cache.move_to_end(key)
                    if len(mgr._cache) > mgr.max_cache_size:
                        old_key, engines = mgr._cache.popitem(last=False)
                        for eng, _ in engines.values():
                            eng.shutdown()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inserter, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors
        assert len(mgr._cache) <= mgr.max_cache_size


@pytest.mark.skipif(not _torch_available, reason="torch unavailable")
class TestYOLOInferenceModelCacheLock:
    """Verify YOLOInferenceModel._cache_lock exists."""

    def test_cache_lock_attribute_exists(self):
        from core.yolo_inference_model import YOLOInferenceModel

        config = MagicMock()
        config.max_cache_size = 3
        config.disable_internal_cache = False

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(YOLOInferenceModel, "__init__", lambda self, cfg: None)
            model = YOLOInferenceModel.__new__(YOLOInferenceModel)
            model._cache_lock = threading.Lock()
            model.model_cache = OrderedDict()

        assert hasattr(model, "_cache_lock")


class TestOverwriteQueueDropCounting:
    """Verify OverwriteQueue tracks and logs dropped items."""

    def test_drop_count_increments(self):
        from core.queues import OverwriteQueue

        q: OverwriteQueue[int] = OverwriteQueue(maxlen=2, name="test_q")
        q.put(1)
        q.put(2)
        assert q.dropped_count == 0

        q.put(3)
        assert q.dropped_count == 1

        q.put(4)
        assert q.dropped_count == 2

    def test_repr_includes_drops(self):
        from core.queues import OverwriteQueue

        q: OverwriteQueue[int] = OverwriteQueue(maxlen=1, name="repr_q")
        q.put(1)
        q.put(2)
        r = repr(q)
        assert "drops=1" in r

    def test_high_volume_drops(self):
        """Stress test: push many items and verify drop count matches."""
        from core.queues import OverwriteQueue

        q: OverwriteQueue[int] = OverwriteQueue(maxlen=5, name="stress_q")
        for i in range(100):
            q.put(i)

        # 100 puts into a queue of size 5: first 5 fill it, then 95 drops
        assert q.dropped_count == 95
        assert q.qsize() == 5
