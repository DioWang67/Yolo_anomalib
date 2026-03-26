"""Unit tests for core/workers.py.

All tests use mock objects to avoid importing torch/ultralytics.
Focuses on:
- BaseWorker: poison pill propagation (normal + crash scenarios)
- AcquisitionWorker: stop_event, frame counting, pill emission
- InferenceWorker: result attachment, error handling
- StorageWorker: persistence counting, flush on shutdown
"""

import queue
import threading
import time

import numpy as np
import pytest

from core.queues import OverwriteQueue
from core.types import DetectionTask
from core.workers import AcquisitionWorker, BaseWorker


# =====================================================================
# Helpers & Fixtures
# =====================================================================

class FakeCamera:
    """Fake camera returning tiny frames."""

    def __init__(self, *, fail_after: int | None = None):
        self._count = 0
        self._fail_after = fail_after

    def capture_frame(self) -> np.ndarray | None:
        self._count += 1
        if self._fail_after and self._count > self._fail_after:
            return None
        return np.zeros((4, 4, 3), dtype=np.uint8)


class EchoWorker(BaseWorker):
    """Copies task to out_queue unchanged."""

    def __init__(self, in_q, out_q):
        super().__init__(in_q, out_q, name="EchoWorker")
        self.processed: list[str] = []

    def process(self, task: DetectionTask) -> None:
        self.processed.append(task.task_id)
        if self.out_queue:
            self.out_queue.put(task)


class CrashOnNthWorker(BaseWorker):
    """Crashes on the Nth task (per-task exception, not thread death)."""

    def __init__(self, in_q, out_q, *, crash_on: int = 2):
        super().__init__(in_q, out_q, name="CrashOnNthWorker")
        self._count = 0
        self._crash_on = crash_on

    def process(self, task: DetectionTask) -> None:
        self._count += 1
        if self._count == self._crash_on:
            raise RuntimeError("Simulated GPU OOM")
        if self.out_queue:
            task.result = {"status": "PASS"}
            self.out_queue.put(task)


def _make_task(task_id: str = "t0") -> DetectionTask:
    return DetectionTask(
        task_id=task_id,
        timestamp=time.time(),
        product="P",
        area="A",
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
    )


def _drain(q: queue.Queue) -> list[DetectionTask]:
    """Drain all items from a stdlib Queue."""
    items: list[DetectionTask] = []
    while not q.empty():
        items.append(q.get_nowait())
    return items


# =====================================================================
# BaseWorker Tests
# =====================================================================

class TestBaseWorker:
    """驗證 BaseWorker 的 Poison Pill 傳遞與例外處理機制。"""

    def test_normal_shutdown_with_poison_pill(self):
        """正常關閉：3 筆任務 + 毒藥丸 → 全數處理 + 下游收到 pill。"""
        in_q: queue.Queue[DetectionTask] = queue.Queue()
        out_q: queue.Queue[DetectionTask] = queue.Queue()
        w = EchoWorker(in_q, out_q)
        w.start()

        for i in range(3):
            in_q.put(_make_task(f"n{i}"))
        in_q.put(DetectionTask.poison_pill())

        w.join(timeout=5)
        assert not w.is_alive(), "Worker 應在毒藥丸後退出"
        assert len(w.processed) == 3

        results = _drain(out_q)
        regular = [r for r in results if not r.is_poison_pill]
        pills = [r for r in results if r.is_poison_pill]
        assert len(regular) == 3, "3 筆正常任務被轉發"
        assert len(pills) == 1, "毒藥丸被傳遞到下游"

    def test_crash_still_sends_pill_downstream(self):
        """Worker process() 拋錯 → 繼續處理 → 仍轉發毒藥丸。"""
        in_q: queue.Queue[DetectionTask] = queue.Queue()
        out_q: queue.Queue[DetectionTask] = queue.Queue()
        w = CrashOnNthWorker(in_q, out_q, crash_on=2)
        w.start()

        for i in range(4):
            in_q.put(_make_task(f"c{i}"))
        in_q.put(DetectionTask.poison_pill())

        w.join(timeout=5)
        assert not w.is_alive()

        results = _drain(out_q)
        pills = [r for r in results if r.is_poison_pill]
        assert len(pills) >= 1, "Crash 後毒藥丸仍傳遞（防死鎖）"

        # The crashed task (c1) should arrive with error set
        err_tasks = [r for r in results if r.error and not r.is_poison_pill]
        assert len(err_tasks) == 1, "失敗任務帶 error 欄位被轉發"

    def test_no_out_queue_still_exits_cleanly(self):
        """Terminal worker (out_queue=None) 也能乾淨退出。"""
        in_q: queue.Queue[DetectionTask] = queue.Queue()

        class SinkWorker(BaseWorker):
            def __init__(self, q):
                super().__init__(q, out_queue=None, name="Sink")
                self.count = 0

            def process(self, task):
                self.count += 1

        w = SinkWorker(in_q)
        w.start()

        in_q.put(_make_task())
        in_q.put(DetectionTask.poison_pill())

        w.join(timeout=5)
        assert not w.is_alive()
        assert w.count == 1


# =====================================================================
# AcquisitionWorker Tests
# =====================================================================

class TestAcquisitionWorker:
    """驗證 AcquisitionWorker 取像、停止與毒藥丸行為。"""

    def test_captures_and_stops(self):
        """正常取像 → stop() → 毒藥丸送入 Queue。"""
        inf_q = OverwriteQueue(maxlen=50)
        cam = FakeCamera()
        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.01,
        )
        aw.start()
        time.sleep(0.15)
        aw.stop()
        aw.join(timeout=3)

        assert not aw.is_alive()
        assert aw.frame_count > 0, f"捕獲了 {aw.frame_count} 幀"

        # Drain to find pill
        found_pill = False
        while not inf_q.empty():
            t = inf_q.get(timeout=0.1)
            if t.is_poison_pill:
                found_pill = True
        assert found_pill, "Queue 尾部應有毒藥丸"

    def test_camera_returns_none_triggers_camera_lost(self):
        """相機回傳 None 超過閾值 → 觸發 on_camera_lost 並自動停止。"""
        cam = FakeCamera(fail_after=3)
        inf_q = OverwriteQueue(maxlen=10)
        lost_events = []
        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.01,
            on_camera_lost=lambda: lost_events.append(True),
        )
        aw.start()
        # Worker should auto-stop after 5 consecutive None frames
        aw.join(timeout=5)

        assert not aw.is_alive()
        assert aw.frame_count == 3, f"只能捕獲 3 幀 (之後 None): got {aw.frame_count}"
        assert len(lost_events) == 1, "on_camera_lost 應被調用一次"

    def test_task_has_correct_metadata(self):
        """DetectionTask 應攜帶正確的 product/area/timestamp。"""
        inf_q = OverwriteQueue(maxlen=5)
        cam = FakeCamera()
        aw = AcquisitionWorker(
            cam, inf_q, product="LED", area="B",
            inference_type="anomalib", capture_interval=0.05,
        )
        aw.start()
        time.sleep(0.1)
        aw.stop()
        aw.join(timeout=3)

        task = inf_q.get(timeout=0.5)
        if not task.is_poison_pill:
            assert task.product == "LED"
            assert task.area == "B"
            assert task.inference_type == "anomalib"
            assert task.timestamp > 0
            assert task.frame is not None


# =====================================================================
# End-to-End Pipeline Tests
# =====================================================================

class TestEndToEndPipeline:
    """端對端管線測試 (mock Inference + Storage)。"""

    def test_full_pipeline_processes_tasks(self):
        """Acquisition → Inference(mock) → Storage(mock) 全通。"""

        class MockInf(BaseWorker):
            def process(self, task):
                task.result = {"status": "PASS"}
                if self.out_queue:
                    self.out_queue.put(task)

        class MockSto(BaseWorker):
            def __init__(self, q):
                super().__init__(q, out_queue=None, name="MockSto")
                self.saved: list[str] = []

            def process(self, task):
                if task.result:
                    self.saved.append(task.task_id)

        inf_q = OverwriteQueue(maxlen=5)
        io_q: queue.Queue[DetectionTask] = queue.Queue(maxsize=50)
        cam = FakeCamera()

        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.02,
        )
        iw = MockInf(inf_q, io_q, name="MockInf")
        sw = MockSto(io_q)

        sw.start()
        iw.start()
        aw.start()

        time.sleep(0.25)
        aw.stop()
        aw.join(timeout=3)
        iw.join(timeout=3)
        sw.join(timeout=3)

        assert not aw.is_alive()
        assert not iw.is_alive()
        assert not sw.is_alive()
        assert len(sw.saved) > 0, f"儲存了 {len(sw.saved)} 筆結果"

    def test_pipeline_under_stress_no_deadlock(self):
        """高頻取像 + 慢推論 → 不死鎖，Queue 有丟幀。"""

        class SlowInf(BaseWorker):
            def process(self, task):
                time.sleep(0.05)
                task.result = {"status": "PASS"}
                if self.out_queue:
                    self.out_queue.put(task)

        class Counter(BaseWorker):
            def __init__(self, q):
                super().__init__(q, out_queue=None, name="Counter")
                self.count = 0

            def process(self, task):
                self.count += 1

        inf_q = OverwriteQueue(maxlen=3)
        io_q: queue.Queue[DetectionTask] = queue.Queue(maxsize=100)
        cam = FakeCamera()

        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.001,
        )
        iw = SlowInf(inf_q, io_q, name="SlowInf")
        sw = Counter(io_q)

        sw.start()
        iw.start()
        aw.start()

        time.sleep(0.5)
        aw.stop()
        aw.join(timeout=5)
        iw.join(timeout=5)
        sw.join(timeout=5)

        assert not aw.is_alive()
        assert not iw.is_alive()
        assert not sw.is_alive()
        assert inf_q.dropped_count > 0, f"應有丟幀: {inf_q.dropped_count}"
        assert sw.count > 0, f"仍有任務被處理: {sw.count}"

    def test_camera_lost_pipeline_graceful_shutdown(self):
        """相機在管線執行中斷線 → 全部 Worker 正常退出且回呼觸發。"""

        class MockInf(BaseWorker):
            def process(self, task):
                task.result = {"status": "PASS"}
                if self.out_queue:
                    self.out_queue.put(task)

        class MockSto(BaseWorker):
            def __init__(self, q):
                super().__init__(q, out_queue=None, name="MockSto")
                self.saved: list[str] = []

            def process(self, task):
                if task.result:
                    self.saved.append(task.task_id)

        # Camera returns 3 good frames then None forever
        cam = FakeCamera(fail_after=3)
        inf_q = OverwriteQueue(maxlen=10)
        io_q: queue.Queue[DetectionTask] = queue.Queue(maxsize=50)

        lost_events: list[bool] = []
        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.01,
            on_camera_lost=lambda: lost_events.append(True),
        )
        iw = MockInf(inf_q, io_q, name="MockInf")
        sw = MockSto(io_q)

        # Start in reverse order (consumers first)
        sw.start()
        iw.start()
        aw.start()

        # All workers should exit on their own (camera lost → pill cascade)
        aw.join(timeout=5)
        iw.join(timeout=5)
        sw.join(timeout=5)

        assert not aw.is_alive(), "AcquisitionWorker 應已停止"
        assert not iw.is_alive(), "InferenceWorker 應已停止"
        assert not sw.is_alive(), "StorageWorker 應已停止"
        assert len(lost_events) == 1, "on_camera_lost 應被調用一次"
        assert aw.frame_count == 3, f"應捕獲 3 幀: got {aw.frame_count}"
        assert len(sw.saved) == 3, f"應儲存 3 筆: got {len(sw.saved)}"
