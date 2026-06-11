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
from core.exceptions import ModelInferenceError
from core.types import DetectionTask
from core.async_pipeline import AsyncPipelineManager
from core.workers import AcquisitionWorker, BaseWorker, InferenceWorker, StorageWorker


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


class FakeResultSink:
    def __init__(self):
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1


class PipelineSystem:
    def __init__(self, status: str = "PASS", *, delay: float = 0.0):
        self.config = object()
        self.result_sink = FakeResultSink()
        self.status = status
        self.delay = delay
        self.inference_calls = 0
        self.saved_statuses: list[str] = []

    def _run_inference(self, *args, **kwargs):
        self.inference_calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.status == "RAISE_INFERENCE_ERROR":
            raise ModelInferenceError("backend failed")
        return {
            "status": self.status,
            "detections": [],
            "missing_items": [],
            "processed_image": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def _execute_pipeline(self, ctx, *_args, **_kwargs):
        ctx.result["status"] = ctx.status
        ctx.save_result = {
            "original_path": "Result/original.jpg",
            "preprocessed_path": "Result/processed.png",
            "annotated_path": "Result/annotated.jpg",
            "heatmap_path": "",
            "cropped_paths": [],
        }

    def _log_summary(self, ctx, *_args, **_kwargs):
        self.saved_statuses.append(ctx.status)


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

    def test_reconnect_recovers_camera_and_continues(self):
        """重連成功 → 連續失敗計數歸零、管線繼續、不觸發 on_camera_lost。"""

        class ReconnectableCamera(FakeCamera):
            def __init__(self):
                super().__init__()
                self.broken = False
                self.reconnect_calls = 0

            def capture_frame(self):
                if self.broken:
                    return None
                return super().capture_frame()

            def reconnect(self) -> bool:
                self.reconnect_calls += 1
                self.broken = False
                return True

        cam = ReconnectableCamera()
        inf_q = OverwriteQueue(maxlen=50)
        lost_events = []
        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.01,
            on_camera_lost=lambda: lost_events.append(True),
            lost_threshold=2,
            reconnect_attempts=3,
            reconnect_backoff=0.01,
        )
        aw.start()
        time.sleep(0.1)
        frames_before = aw.frame_count
        cam.broken = True  # simulate cable pull
        time.sleep(0.5)    # threshold reached -> reconnect() repairs camera
        aw.stop()
        aw.join(timeout=3)

        assert not aw.is_alive()
        assert cam.reconnect_calls >= 1, "應嘗試自動重連"
        assert aw.frame_count > frames_before, "重連後應繼續取像"
        assert not lost_events, "重連成功不應回報 camera lost"

    def test_reconnect_exhausted_reports_camera_lost(self):
        """重連全部失敗 → 回報 on_camera_lost 並停止。"""

        class DeadCamera(FakeCamera):
            def __init__(self):
                super().__init__()
                self.reconnect_calls = 0

            def capture_frame(self):
                return None

            def reconnect(self) -> bool:
                self.reconnect_calls += 1
                return False

        cam = DeadCamera()
        inf_q = OverwriteQueue(maxlen=10)
        lost_events = []
        aw = AcquisitionWorker(
            cam, inf_q, product="P", area="A", capture_interval=0.01,
            on_camera_lost=lambda: lost_events.append(True),
            lost_threshold=2,
            reconnect_attempts=2,
            reconnect_backoff=0.01,
        )
        aw.start()
        aw.join(timeout=5)

        assert not aw.is_alive()
        assert cam.reconnect_calls == 2, "應嘗試設定的重連次數"
        assert len(lost_events) == 1, "重連耗盡後應回報 camera lost"

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
# AsyncPipelineManager Tests
# =====================================================================

class TestAsyncPipelineManager:
    def test_stop_returns_when_inference_backend_is_blocked(self):
        """stop() must not wait forever for a backend stuck in one inference call."""

        class BlockingSystem:
            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()
                self.config = object()
                self.result_sink = None
                self.calls = 0

            def _run_inference(self, *args, **kwargs):
                self.calls += 1
                self.entered.set()
                self.release.wait(timeout=5.0)
                return {"status": "PASS", "detections": []}

            def _execute_pipeline(self, *args, **kwargs):
                return None

            def _log_summary(self, *args, **kwargs):
                return None

        manager = AsyncPipelineManager()
        system = BlockingSystem()
        manager.start(
            camera=FakeCamera(),
            detection_system=system,
            product="P",
            area="A",
            inference_type="yolo",
            capture_interval=0.001,
        )
        assert system.entered.wait(timeout=2.0)

        started = time.perf_counter()
        manager.stop(timeout=0.2)
        elapsed = time.perf_counter() - started

        system.release.set()
        time.sleep(0.2)
        assert elapsed < 1.0
        assert manager.running is False
        assert system.calls == 1

    def test_single_shot_pipeline_stops_after_first_pass(self):
        manager = AsyncPipelineManager()
        system = PipelineSystem(status="PASS")
        processed: list[DetectionTask] = []

        manager.start(
            camera=FakeCamera(),
            detection_system=system,
            product="P",
            area="A",
            inference_type="yolo",
            capture_interval=0.01,
            mode="single",
            on_task_processed=processed.append,
        )

        deadline = time.time() + 3.0
        while time.time() < deadline and manager.running:
            time.sleep(0.01)

        stats = manager.stats()
        manager.stop(timeout=1.0)

        assert manager.running is False
        assert stats["frames_captured"] == 1
        assert system.inference_calls == 1
        assert stats["tasks_saved"] == 1
        assert len(processed) == 1
        assert processed[0].result["status"] == "PASS"
        assert processed[0].result["original_image_path"] == "Result/original.jpg"
        assert (
            processed[0].result["preprocessed_image_path"]
            == "Result/processed.png"
        )

    def test_single_shot_pipeline_stops_after_first_detection_fail(self):
        manager = AsyncPipelineManager()
        system = PipelineSystem(status="DETECTION_FAIL")
        processed: list[DetectionTask] = []

        manager.start(
            camera=FakeCamera(),
            detection_system=system,
            product="P",
            area="A",
            inference_type="yolo",
            capture_interval=0.01,
            mode="single",
            on_task_processed=processed.append,
        )

        deadline = time.time() + 3.0
        while time.time() < deadline and manager.running:
            time.sleep(0.01)

        stats = manager.stats()
        manager.stop(timeout=1.0)

        assert manager.running is False
        assert stats["frames_captured"] == 1
        assert system.inference_calls == 1
        assert stats["tasks_saved"] == 1
        assert processed[0].result["status"] == "DETECTION_FAIL"

    def test_single_shot_pipeline_stops_after_inference_error(self):
        manager = AsyncPipelineManager()
        system = PipelineSystem(status="RAISE_INFERENCE_ERROR")
        processed: list[DetectionTask] = []

        manager.start(
            camera=FakeCamera(),
            detection_system=system,
            product="P",
            area="A",
            inference_type="yolo",
            capture_interval=0.01,
            mode="single",
            on_task_processed=processed.append,
        )

        deadline = time.time() + 3.0
        while time.time() < deadline and manager.running:
            time.sleep(0.01)

        stats = manager.stats()
        manager.stop(timeout=1.0)

        assert manager.running is False
        assert stats["frames_captured"] == 1
        assert system.inference_calls == 1
        assert stats["tasks_saved"] == 1
        assert processed[0].result["status"] == "INFERENCE_ERROR"

    def test_continuous_pipeline_keeps_running_until_manual_stop(self):
        manager = AsyncPipelineManager()
        system = PipelineSystem(status="PASS", delay=0.005)

        manager.start(
            camera=FakeCamera(),
            detection_system=system,
            product="P",
            area="A",
            inference_type="yolo",
            capture_interval=0.01,
            mode="continuous",
        )

        deadline = time.time() + 2.0
        while time.time() < deadline and system.inference_calls < 2:
            time.sleep(0.01)

        assert manager.running is True
        assert system.inference_calls >= 2

        manager.stop(timeout=2.0)
        assert manager.running is False


class TestInferenceWorker:
    def test_model_inference_error_stops_pipeline_and_forwards_error(self):
        """Fatal inference errors stop the worker and emit INFERENCE_ERROR once."""

        class FailingSystem:
            def __init__(self):
                self.calls = 0

            def _run_inference(self, *args, **kwargs):
                self.calls += 1
                raise ModelInferenceError("onnxruntime DLL load failed")

        stop_event = threading.Event()
        inf_q = OverwriteQueue(maxlen=10)
        io_q: queue.Queue[DetectionTask] = queue.Queue()
        system = FailingSystem()
        worker = InferenceWorker(
            inf_q,
            io_q,
            system,
            stop_event=stop_event,
        )

        inf_q.put(_make_task("fatal0"))
        inf_q.put(_make_task("should_drop"))
        worker.start()
        worker.join(timeout=3)

        assert not worker.is_alive()
        assert stop_event.is_set()
        assert system.calls == 1

        outputs = _drain(io_q)
        regular = [task for task in outputs if not task.is_poison_pill]
        pills = [task for task in outputs if task.is_poison_pill]
        assert len(regular) == 1
        assert regular[0].result["status"] == "INFERENCE_ERROR"
        assert "onnxruntime DLL load failed" in regular[0].result["error"]
        assert pills, "Fatal stop should forward a poison pill downstream"

    def test_inference_error_result_stops_pipeline(self):
        """A backend error returned as a result is also fatal for async mode."""

        class ErrorResultSystem:
            def _run_inference(self, *args, **kwargs):
                return {
                    "status": "ERROR",
                    "error": "Model not loaded",
                    "detections": [],
                }

        stop_event = threading.Event()
        inf_q = OverwriteQueue(maxlen=10)
        io_q: queue.Queue[DetectionTask] = queue.Queue()
        worker = InferenceWorker(
            inf_q,
            io_q,
            ErrorResultSystem(),
            stop_event=stop_event,
        )

        inf_q.put(_make_task("error-result"))
        worker.start()
        worker.join(timeout=3)

        outputs = _drain(io_q)
        regular = [task for task in outputs if not task.is_poison_pill]
        assert stop_event.is_set()
        assert regular[0].result["status"] == "INFERENCE_ERROR"
        assert regular[0].result["error"] == "Model not loaded"


class TestStorageWorker:
    def test_single_shot_drops_queued_frames_after_terminal_result(self):
        system = PipelineSystem(status="PASS")
        stop_event = threading.Event()
        inf_q = OverwriteQueue(maxlen=10)
        io_q: queue.Queue[DetectionTask] = queue.Queue()

        inf_q.put(_make_task("queued-inf-1"))
        inf_q.put(_make_task("queued-inf-2"))

        first = _make_task("first")
        first.result = {"status": "PASS", "detections": [], "missing_items": []}
        second = _make_task("queued-storage")
        second.result = {"status": "PASS", "detections": [], "missing_items": []}
        io_q.put(first)
        io_q.put(second)

        worker = StorageWorker(
            io_q,
            system,
            stop_event=stop_event,
            mode="single",
            drop_queue=inf_q,
        )
        worker.start()
        worker.join(timeout=3)

        assert not worker.is_alive()
        assert stop_event.is_set()
        assert worker.saved_count == 1
        assert inf_q.qsize() == 0
        assert io_q.empty()


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
