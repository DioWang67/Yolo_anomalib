import queue
import threading
import time

import numpy as np
import pytest

from core.types import DetectionTask
from core.queues import OverwriteQueue


# =====================================================================
# DetectionTask DTO Tests
# =====================================================================

def test_detection_task_creation():
    """測試 DTO 是否能正確封裝上下文且預設為非毒藥丸"""
    dummy_frame = np.zeros((10, 10, 3), dtype=np.uint8)
    task = DetectionTask(
        task_id="test-trigger-001",
        timestamp=time.time(),
        product="LED",
        area="A",
        frame=dummy_frame
    )
    assert task.task_id == "test-trigger-001"
    assert task.product == "LED"
    assert task.is_poison_pill is False


def test_detection_task_poison_pill_factory():
    """毒藥丸工廠方法：應建立最小化的終止信號。"""
    pill = DetectionTask.poison_pill()
    assert pill.is_poison_pill is True
    assert pill.task_id == "__POISON_PILL__"
    assert pill.timestamp == 0.0
    assert pill.frame is None  # 不浪費記憶體


def test_detection_task_default_fields():
    """驗證 DTO 的預設欄位：inference_type, error, result。"""
    task = DetectionTask(
        task_id="t", timestamp=1.0, product="P", area="A",
    )
    assert task.inference_type == "yolo"
    assert task.error is None
    assert task.result is None


# =====================================================================
# OverwriteQueue: Core Behaviour
# =====================================================================

def test_overwrite_queue_drops_oldest():
    """核心防禦驗證：測試滿載時是否正確丟棄最舊的資料 (防止殭屍幀)"""
    q = OverwriteQueue(maxlen=3)

    # 刻意塞入 5 筆資料，預期 0, 1 會被擠掉，留下 2, 3, 4
    for i in range(5):
        q.put(i)

    assert q.qsize() == 3
    assert q.get() == 2
    assert q.get() == 3
    assert q.get() == 4
    assert q.empty()


def test_overwrite_queue_blocking_get():
    """驗證 Consumer 取件時的 Blocking 阻擋機制與 Timeout"""
    q = OverwriteQueue(maxlen=5)

    # Queue 是空的，預期 get 會觸發 Empty 例外
    with pytest.raises(queue.Empty):
        q.get(timeout=0.1)


def test_overwrite_queue_thread_safety():
    """併發壓力測試：多個 Producer 狂塞資料，驗證不會引發死鎖或長度越界"""
    q = OverwriteQueue(maxlen=100)

    def producer(start_idx: int):
        for i in range(50):
            q.put(start_idx + i)
            time.sleep(0.001)

    # 啟動 3 個執行緒，總共會塞入 150 筆資料
    threads = [threading.Thread(target=producer, args=(j * 100,)) for j in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Queue 的上限是 100，所以即使塞了 150 筆，最後大小絕對不能超過 100
    assert q.qsize() == 100

    # 若有實作 dropped_count，確保有記錄到丟棄的 50 筆
    if hasattr(q, 'dropped_count'):
        assert q.dropped_count == 50


# =====================================================================
# OverwriteQueue: Extended Edge Cases
# =====================================================================

def test_overwrite_queue_size_one():
    """Size-1 Queue: 永遠只保留最新一筆資料（最小延遲場景）。"""
    q = OverwriteQueue(maxlen=1)
    q.put("old")
    q.put("new")
    assert q.get(timeout=0.1) == "new"
    assert q.empty()
    assert q.dropped_count == 1


def test_overwrite_queue_fifo_order_under_stress():
    """壓力下取出的資料維持 FIFO 順序（不會亂序）。"""
    q = OverwriteQueue(maxlen=10)
    results = []

    def producer():
        for i in range(200):
            q.put(i)

    def consumer():
        while True:
            try:
                val = q.get(timeout=0.3)
                results.append(val)
            except queue.Empty:
                break

    tp = threading.Thread(target=producer)
    tc = threading.Thread(target=consumer)
    tc.start()
    tp.start()
    tp.join()
    tc.join()

    # 取到的值必須單調遞增
    for i in range(len(results) - 1):
        assert results[i] < results[i + 1], (
            f"FIFO 順序違反: results[{i}]={results[i]} >= results[{i+1}]={results[i+1]}"
        )


def test_overwrite_queue_maxlen_validation():
    """maxlen < 1 應拋出 ValueError。"""
    with pytest.raises(ValueError):
        OverwriteQueue(maxlen=0)
    with pytest.raises(ValueError):
        OverwriteQueue(maxlen=-1)


def test_overwrite_queue_counter_accuracy():
    """put_count 與 get_count 計數器應精確。"""
    q = OverwriteQueue(maxlen=3)
    for i in range(10):
        q.put(i)

    assert q.put_count == 10
    assert q.dropped_count == 7  # 10 - 3

    for _ in range(3):
        q.get(timeout=0.1)
    assert q.get_count == 3


def test_overwrite_queue_blocking_get_wakes_on_put():
    """Consumer 阻塞中 → Producer put() → Consumer 立即被喚醒。"""
    q = OverwriteQueue(maxlen=5)
    result = []

    def consumer():
        val = q.get(timeout=2.0)  # 最多等 2 秒
        result.append(val)

    tc = threading.Thread(target=consumer)
    tc.start()

    time.sleep(0.1)  # 確保 consumer 正在阻塞
    q.put("wakeup")
    tc.join(timeout=1.0)

    assert len(result) == 1
    assert result[0] == "wakeup"


def test_overwrite_queue_repr():
    """__repr__ 應包含有用的狀態資訊。"""
    q = OverwriteQueue(maxlen=5, name="test_repr")
    q.put("a")
    r = repr(q)
    assert "test_repr" in r
    assert "1/5" in r
