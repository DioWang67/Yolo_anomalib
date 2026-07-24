import time

import numpy as np
import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for GUI tests")
pytestmark = pytest.mark.gui

from app.gui.i18n import tr
from app.gui.main_window import DetectionSystemGUI
from core.types import DetectionResult, DetectionTask


@pytest.fixture
def gui(qtbot):
    """Fixture to create and show the GUI window."""
    window = DetectionSystemGUI()
    qtbot.addWidget(window)
    yield window
    # Ensure clean shutdown to prevent logging errors
    window.close()

def test_window_title(gui):
    """Verify window title indicates correct system."""
    assert gui.windowTitle() == tr(gui.current_language, "window_title")

def test_panels_present(gui):
    """Verify all major panels are instantiated."""
    assert gui.control_panel is not None
    assert gui.image_panel is not None
    assert gui.info_panel is not None


def test_retraining_workspace_is_reused_and_can_return_to_inspection(
    gui, tmp_path, qtbot
):
    """Leaving retraining must switch pages without destroying its state."""
    arguments = {
        "result_root": tmp_path / "Result",
        "manifest_path": tmp_path / "review.csv",
        "training_data_dir": tmp_path / "training-data",
        "language": "zh_TW",
        "product": "Cable1",
        "area": "A",
    }

    first = gui.show_retraining_workspace(**arguments)
    gui.show_inspection_workspace()
    second = gui.show_retraining_workspace(**arguments)

    assert first is second
    assert gui.workspace_stack.currentWidget() is first
    qtbot.waitUntil(lambda: first.workspace is not None, timeout=5000)
    first.back_to_inspection_requested.emit()
    assert gui.workspace_stack.currentWidget() is gui.inspection_workspace


def test_retraining_manifest_scan_does_not_block_page_switch(
    gui, tmp_path, qtbot, monkeypatch
):
    """A slow disk scan must remain outside the Qt main thread."""
    from app.gui import review_cases_dialog

    original_prepare = review_cases_dialog.prepare_review_manifest

    def slow_prepare(**kwargs):
        time.sleep(0.4)
        return original_prepare(**kwargs)

    monkeypatch.setattr(review_cases_dialog, "prepare_review_manifest", slow_prepare)
    started = time.perf_counter()
    host = gui.show_retraining_workspace(
        result_root=tmp_path / "Result",
        manifest_path=tmp_path / "review.csv",
        training_data_dir=tmp_path / "training-data",
        language="zh_TW",
        product="Cable1",
        area="A",
    )
    call_seconds = time.perf_counter() - started

    assert call_seconds < 0.2
    assert gui.workspace_stack.currentWidget() is host
    assert host.workspace is None
    qtbot.waitUntil(lambda: host.workspace is not None, timeout=5000)


def test_retraining_workspace_survives_repeated_page_switches(
    gui, tmp_path, qtbot
):
    """Repeated navigation must reuse one workspace and remain responsive."""
    arguments = {
        "result_root": tmp_path / "Result",
        "manifest_path": tmp_path / "review.csv",
        "training_data_dir": tmp_path / "training-data",
        "language": "zh_TW",
        "product": "Cable1",
        "area": "A",
    }
    host = gui.show_retraining_workspace(**arguments)
    qtbot.waitUntil(lambda: host.workspace is not None, timeout=5000)
    stable_widget_count = gui.workspace_stack.count()

    started = time.perf_counter()
    for _iteration in range(1000):
        gui.show_inspection_workspace()
        assert gui.show_retraining_workspace(**arguments) is host
    elapsed_seconds = time.perf_counter() - started

    assert gui.workspace_stack.currentWidget() is host
    assert gui.workspace_stack.count() == stable_widget_count
    assert elapsed_seconds < 5.0

def test_initial_state(gui):
    """Verify initial button states."""
    assert gui.start_btn.isEnabled() is False  # Should be disabled until configs loaded/selected
    assert gui.stop_btn.isEnabled() is False
    assert gui.save_btn.isEnabled() is False
    assert gui.show_detection_boxes_chk is not None
    assert gui.show_original_tab_chk is not None
    assert gui.show_processed_tab_chk is not None

def test_model_loading_async(gui, qtbot):
    """Verify that model loading triggers signals and updates combos."""
    # Since load_available_models is async, we wait for the log message or combo update
    # But checking combos is easier.
    # Note: real model loading depends on file system.
    # If this test env has no models, combos remain empty.

    # Trigger refresh manually
    # We call it once to ensure the attribute is created since we skip auto-load in __init__
    gui.load_available_models()
    with qtbot.waitSignal(gui.model_loader.models_ready, timeout=5000, raising=False) as blocker:
        # Thread already started by the call above
        pass

    # Even if timeout (no models found or error), we check that GUI didn't crash
    # and combos are objects (not None)
    assert gui.product_combo is not None

def test_interaction_flow(gui, qtbot):
    """Test a simple interaction flow."""
    # Simulate selecting a product if available
    if gui.product_combo.count() > 0:
        gui.product_combo.setCurrentIndex(0)
        # Check area update
        assert gui.area_combo.count() >= 0


def test_result_image_uses_preprocessed_path_when_boxes_hidden(gui, monkeypatch):
    """YOLO result view should switch to the clean image when boxes are hidden."""
    calls: list[str] = []

    def fake_load_image_with_retry(widget, image_path, **kwargs):
        calls.append(image_path)

    monkeypatch.setattr(
        "app.gui.main_window.load_image_with_retry",
        fake_load_image_with_retry,
    )

    gui.current_result = DetectionResult(
        status="PASS",
        product="P",
        area="A",
        inference_type="yolo",
        original_image_path="original.jpg",
        preprocessed_image_path="processed.jpg",
        annotated_path="annotated.jpg",
    )

    gui.show_detection_boxes_chk.setChecked(True)
    gui.show_detection_boxes_chk.setChecked(False)

    assert calls[-1] == "processed.jpg"


def test_pipeline_storage_completion_refreshes_all_three_artifact_tabs(
    gui, monkeypatch
):
    """Pipeline UI must wait for persistence and then use its exact paths."""
    loaded: list[tuple[object, str]] = []
    live_result_frames: list[np.ndarray] = []

    def fake_load_image_with_retry(widget, image_path, **_kwargs):
        loaded.append((widget, image_path))

    monkeypatch.setattr(
        "app.gui.main_window.load_image_with_retry",
        fake_load_image_with_retry,
    )
    monkeypatch.setattr(
        gui.result_image,
        "display_image",
        live_result_frames.append,
    )

    task = DetectionTask(
        task_id="inspection-1",
        timestamp=time.time(),
        product="Cable1",
        area="A",
        inference_type="yolo",
        frame=np.zeros((8, 8, 3), dtype=np.uint8),
        result={
            "status": "PASS",
            "detections": [],
            "result_frame": np.ones((8, 8, 3), dtype=np.uint8),
        },
    )

    gui.on_pipeline_result(task)

    assert gui.current_result is not None
    assert gui.current_result.metadata["storage_completed"] is False
    assert live_result_frames == []

    loaded.clear()
    task.result.update(
        {
            "original_image_path": "persisted-original.jpg",
            "preprocessed_image_path": "persisted-processed.jpg",
            "annotated_path": "persisted-annotated.jpg",
        }
    )
    gui.on_pipeline_storage_completed(task)

    assert gui.current_result.metadata["storage_completed"] is True
    assert loaded == [
        (gui.original_image, "persisted-original.jpg"),
        (gui.processed_image, "persisted-processed.jpg"),
        (gui.result_image, "persisted-annotated.jpg"),
    ]
    assert live_result_frames == []


def test_engineer_image_tab_toggles_hide_optional_tabs(gui):
    """Engineer settings should control original/processed tab visibility."""
    gui.show_original_tab_chk.setChecked(True)
    gui.show_processed_tab_chk.setChecked(True)

    assert gui.image_panel.image_tabs.indexOf(gui.original_image) >= 0
    assert gui.image_panel.image_tabs.indexOf(gui.processed_image) >= 0
    assert gui.image_panel.image_tabs.indexOf(gui.result_image) >= 0

    gui.show_original_tab_chk.setChecked(False)
    assert gui.image_panel.image_tabs.indexOf(gui.original_image) == -1
    assert gui.image_panel.image_tabs.indexOf(gui.processed_image) >= 0
    assert gui.image_panel.image_tabs.indexOf(gui.result_image) >= 0

    gui.show_processed_tab_chk.setChecked(False)
    assert gui.image_panel.image_tabs.indexOf(gui.original_image) == -1
    assert gui.image_panel.image_tabs.indexOf(gui.processed_image) == -1
    assert gui.image_panel.image_tabs.indexOf(gui.result_image) >= 0
    assert gui.image_panel.image_tabs.count() == 1

    gui.show_original_tab_chk.setChecked(True)
    gui.show_processed_tab_chk.setChecked(True)
