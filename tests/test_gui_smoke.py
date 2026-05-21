
import pytest
import numpy as np

# Ensure app is importable
# sys.path.append(os.getcwd())

@pytest.mark.gui
def test_gui_smoke(monkeypatch, tmp_path):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication

    from app.gui import main_window as GUI

    # Mock DetectionSystem to match Phase 3 Interface
    class StubDetectionSystem:
        def __init__(self, config_path=None):
            self.config_path = config_path
            self.shutdown_calls = 0
            self.detect_calls = []
            self._pipeline_active = False
            self.stop_pipeline_calls = 0

        @property
        def pipeline_running(self):
            return self._pipeline_active

        def load_model_configs(self, product, area, infer_type):
            pass

        def detect(self, product, area, inference_type, frame=None, cancel_cb=None):
            self.detect_calls.append(frame)
            # Return legacy dict which logic adapts to DetectionResult
            return {
                "status": "PASS",
                "detections": [],
                "product": "test_prod",
                "area": "test_area",
                "inference_type": "yolo",
                "image_path": "test.jpg"
            }

        def stop_pipeline(self, timeout=10.0):
            self.stop_pipeline_calls += 1
            self._pipeline_active = False

        def capture_image(self):
            return None

        def connect_camera(self):
            return True

        def reconnect_camera(self):
            return True

        def disconnect_camera(self):
            pass

        def is_camera_connected(self):
            return True

        def shutdown(self):
            self.shutdown_calls += 1

    # Inject the stub
    # We need to ensure _get_detection_class returns this stub,
    # or Controller uses it. Controller is instantiated in MainWindow using _get_detection_class.
    # We can monkeypatch main_window._get_detection_class.

    monkeypatch.setattr(GUI, "_get_detection_class", lambda: StubDetectionSystem)

    # Also patch DetectionSystem in Controller if lazily imported?
    # No, Controller gets cls from main_window arg.

    app = QApplication.instance() or QApplication([])

    models_dir = tmp_path / "models"
    (models_dir / "Prod1" / "Area1" / "yolo").mkdir(parents=True)
    # create dummy config
    (models_dir / "Prod1" / "Area1" / "yolo" / "config.yaml").touch()

    (models_dir / "Prod1" / "Area1" / "anomalib").mkdir(parents=True)
    (models_dir / "Prod1" / "Area1" / "anomalib" / "config.yaml").touch()

    window = GUI.DetectionSystemGUI()
    # Override models base
    window.controller.catalog.root = models_dir
    window._models_base = models_dir

    # Trigger load
    window.load_available_models()

    # Process events to allow async workers to finish
    # Since they are threads, we might need a small wait or check
    import time
    for _ in range(10):
        app.processEvents()
        if window.product_combo.count() > 0:
            break
        time.sleep(0.1)

    assert window.product_combo.count() >= 1
    # Prod1 should be there

    # Test Signal Emission/Slot
    # Manually trigger on_detection_complete to verify Type handling
    from core.types import DetectionItem, DetectionResult
    res = DetectionResult(
        status="PASS",
        items=[DetectionItem("cat", 0.9, (0,0,10,10))],
        latency=0.1
    )
    window.on_detection_complete(res)

    assert window.info_panel.big_status_label.text() == "PASS"
    assert window.start_btn.objectName() == "primaryAction"
    assert window.stop_btn.objectName() == "dangerAction"
    assert window.info_panel.log_group.isVisible() is False
    assert window.image_panel.result_image.hasScaledContents() is False

    window.control_panel.language_combo.setCurrentIndex(
        window.control_panel.language_combo.findData("zh")
    )
    app.processEvents()
    assert window.current_language == "zh"
    assert window.start_btn.text() == "開始檢測"
    assert window.image_panel.windowTitle() == ""
    assert window.image_panel.title() == "影像檢視"
    assert window.menuBar().actions()[0].text() == "檔案"
    assert window.info_panel.session_stats.title() == "當班統計"

    window.control_panel.language_combo.setCurrentIndex(
        window.control_panel.language_combo.findData("en")
    )
    app.processEvents()
    assert window.current_language == "en"
    assert window.start_btn.text() == "Start Inspection"
    assert window.menuBar().actions()[0].text() == "File"
    assert window.info_panel.result_widget._title.text() == "Result Details"
    assert "Field Decision" in window.info_panel.result_widget._result_text.toPlainText()
    assert window.info_panel.session_stats.title() == "Shift Stats"

    window.close()
    app.processEvents()

    # Controller shutdown calls system shutdown
    if window.controller._system:
        assert window.controller._system.shutdown_calls == 1

    window.deleteLater()


@pytest.mark.gui
def test_single_shot_stop_resets_ui_without_pipeline_shutdown(monkeypatch, tmp_path):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication

    from app.gui import main_window as GUI

    class StubDetectionSystem:
        def __init__(self, config_path=None):
            self.stop_pipeline_calls = 0
            self.shutdown_calls = 0

        @property
        def pipeline_running(self):
            return False

        def is_camera_connected(self):
            return True

        def stop_pipeline(self, timeout=10.0):
            self.stop_pipeline_calls += 1

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(GUI, "_get_detection_class", lambda: StubDetectionSystem)

    app = QApplication.instance() or QApplication([])
    window = GUI.DetectionSystemGUI()
    window.controller._system = StubDetectionSystem()
    window._single_shot_running = True
    window.stop_btn.setEnabled(True)
    window.start_btn.setEnabled(False)

    window.stop_detection()
    app.processEvents()

    assert window._single_shot_running is False
    assert window.start_btn.isEnabled() is True
    assert window.stop_btn.isEnabled() is False
    assert window.controller._system.stop_pipeline_calls == 0

    window.close()
    app.processEvents()
    window.deleteLater()


@pytest.mark.gui
def test_gui_running_state_uses_pipeline_state(monkeypatch):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication

    from app.gui import main_window as GUI

    class StubDetectionSystem:
        def __init__(self, config_path=None):
            self._pipeline_active = True

        @property
        def pipeline_running(self):
            return self._pipeline_active

        def is_camera_connected(self):
            return True

        def shutdown(self):
            pass

    monkeypatch.setattr(GUI, "_get_detection_class", lambda: StubDetectionSystem)

    app = QApplication.instance() or QApplication([])
    window = GUI.DetectionSystemGUI()
    window.controller._system = StubDetectionSystem()
    window.worker = None

    assert window.is_detection_running() is True

    window.controller._system._pipeline_active = False
    window.close()
    app.processEvents()
    window.deleteLater()


@pytest.mark.gui
def test_stop_during_pipeline_startup_waits_for_worker_cancel(monkeypatch):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication

    from app.gui import main_window as GUI

    class StubDetectionSystem:
        @property
        def pipeline_running(self):
            return False

        def is_camera_connected(self):
            return True

        def shutdown(self):
            pass

    class FakeStartupWorker:
        def __init__(self):
            self.running = True
            self.canceled = False

        def isRunning(self):
            return self.running

        def cancel(self):
            self.canceled = True

    monkeypatch.setattr(GUI, "_get_detection_class", lambda: StubDetectionSystem)

    app = QApplication.instance() or QApplication([])
    window = GUI.DetectionSystemGUI()
    window.controller._system = StubDetectionSystem()
    window.worker = FakeStartupWorker()
    window._run_generation = 7
    window.start_btn.setEnabled(False)
    window.stop_btn.setEnabled(True)

    window.stop_detection()
    app.processEvents()

    assert window.worker.canceled is True
    assert window._shutdown_in_progress is True
    assert window.start_btn.isEnabled() is False

    window.worker.running = False
    window._on_start_worker_finished(7)
    app.processEvents()

    assert window._shutdown_in_progress is False
    assert window.start_btn.isEnabled() is True
    assert window.stop_btn.isEnabled() is False

    window.close()
    app.processEvents()
    window.deleteLater()


@pytest.mark.gui
def test_detection_worker_cancel_prevents_late_pipeline_start(monkeypatch):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    import threading
    from PyQt5.QtWidgets import QApplication

    from app.gui.workers import DetectionWorker

    class StartupSystem:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()
            self.pipeline_active = False
            self.stop_calls = 0

        @property
        def pipeline_running(self):
            return self.pipeline_active

        def start_pipeline(self, *args, cancel_cb=None, **kwargs):
            self.entered.set()
            self.release.wait(timeout=2.0)
            if cancel_cb and cancel_cb():
                return
            self.pipeline_active = True

        def stop_pipeline(self, timeout=10.0):
            self.stop_calls += 1
            self.pipeline_active = False

    _ = QApplication.instance() or QApplication([])
    system = StartupSystem()
    worker = DetectionWorker(
        detection_system=system,
        product="P",
        area="A",
        inference_type="yolo",
        run_id=1,
    )
    worker.start()
    assert system.entered.wait(timeout=2.0)

    worker.cancel()
    system.release.set()
    assert worker.wait(3000)

    assert system.pipeline_running is False
    assert system.stop_calls == 1


@pytest.mark.gui
def test_pipeline_bridge_rejects_stale_run_callbacks(monkeypatch):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication

    from app.gui.workers import PipelineBridge
    from core.types import DetectionTask

    _ = QApplication.instance() or QApplication([])
    bridge = PipelineBridge()
    images = []
    results = []
    camera_lost = []
    bridge.image_ready.connect(images.append)
    bridge.result_ready.connect(results.append)
    bridge.camera_disconnected.connect(lambda: camera_lost.append(True))

    task = DetectionTask(
        task_id="t1",
        timestamp=1.0,
        product="P",
        area="A",
        inference_type="yolo",
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        result={"status": "PASS", "detections": []},
    )

    bridge.begin_run(1)
    bridge.on_task_captured(task, run_id=1)
    bridge.on_task_processed(task, run_id=1)
    bridge.on_camera_lost(run_id=1)
    assert len(images) == 1
    assert len(results) == 1
    assert len(camera_lost) == 1

    bridge.begin_run(2)
    bridge.on_task_captured(task, run_id=1)
    bridge.on_task_processed(task, run_id=1)
    bridge.on_camera_lost(run_id=1)
    assert len(images) == 1
    assert len(results) == 1
    assert len(camera_lost) == 1

    bridge.end_run(2)
    bridge.on_task_captured(task, run_id=2)
    bridge.on_task_processed(task, run_id=2)
    bridge.on_camera_lost(run_id=2)
    assert len(images) == 1
    assert len(results) == 1
    assert len(camera_lost) == 1
