import pytest


@pytest.mark.gui
def test_gui_smoke(monkeypatch, tmp_path):
    _ = pytest.importorskip(
        "PyQt5.QtWidgets", reason="PyQt5 is required for GUI smoke test"
    )
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PyQt5.QtWidgets import QApplication
    import GUI

    class StubDetectionSystem:
        def __init__(self):
            self.detect_calls = []
            self.shutdown_calls = 0

        def detect(self, product, area, inference_type, frame=None, cancel_cb=None):
            self.detect_calls.append((product, area, inference_type))
            return {
                "status": "PASS",
                "product": product,
                "area": area,
                "inference_type": inference_type,
                "ckpt_path": "",
                "anomaly_score": 0.0,
                "detections": [],
                "missing_items": [],
                "original_image_path": "",
                "preprocessed_image_path": "",
                "annotated_path": "",
                "heatmap_path": "",
                "cropped_paths": [],
                "color_check": None,
            }

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(GUI, "DetectionSystem",
                        StubDetectionSystem, raising=True)

    app = QApplication.instance() or QApplication([])

    models_dir = tmp_path / "models"
    (models_dir / "Prod1" / "Area1" / "yolo").mkdir(parents=True)
    (models_dir / "Prod1" / "Area1" / "anomalib").mkdir(parents=True)

    window = GUI.DetectionSystemGUI()
    window._models_base = str(models_dir)
    window.load_available_models()

    assert window.product_combo.count() == 1
    assert window.area_combo.count() == 1
    assert window.inference_combo.count() >= 1

    window.close()
    app.processEvents()
    assert window.detection_system.shutdown_calls == 1
    window.deleteLater()
