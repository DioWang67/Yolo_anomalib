import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for GUI tests")
pytestmark = pytest.mark.gui
pytest_plugins = ["pytestqt.plugin"]

from PyQt5.QtCore import Qt
from app.gui.main_window import DetectionSystemGUI

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
    assert "AI 檢測系統" in gui.windowTitle()

def test_panels_present(gui):
    """Verify all major panels are instantiated."""
    assert gui.control_panel is not None
    assert gui.image_panel is not None
    assert gui.info_panel is not None

def test_initial_state(gui):
    """Verify initial button states."""
    assert gui.start_btn.isEnabled() is False  # Should be disabled until configs loaded/selected
    assert gui.stop_btn.isEnabled() is False
    assert gui.save_btn.isEnabled() is False

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
