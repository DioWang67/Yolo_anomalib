import pytest
import shutil
from unittest.mock import MagicMock
import os
import sys

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
    from core.types import DetectionResult, DetectionItem
    res = DetectionResult(
        status="PASS",
        items=[DetectionItem("cat", 0.9, (0,0,10,10))],
        latency=0.1
    )
    window.on_detection_complete(res)
    
    assert window.info_panel.big_status_label.text() == "PASS"
    
    window.close()
    app.processEvents()
    
    # Controller shutdown calls system shutdown
    if window.controller._system:
        assert window.controller._system.shutdown_calls == 1
    
    window.deleteLater()
