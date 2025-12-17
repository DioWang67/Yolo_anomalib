import sys
from PyQt5.QtWidgets import QApplication
import pytest
import os

# Set env to avoid full system init which might fail without real camera/hardware
# But wait, we want to test Async Camera Init, so maybe we SHOULD NOT skip system init?
# If I skip system init, I won't test the async workers calls in init_system.
# However, init_system is async now.
# Let's try WITHOUT skipping first. If it fails due to hardware, I'll know.
# But system requires config.yaml.
# Let's point to the one in project root.

from app.gui.main_window import DetectionSystemGUI

def test_gui_init():
    app = QApplication(sys.argv)
    try:
        window = DetectionSystemGUI()
        print("GUI Instantiated successfully")
        
        # Test if panels are there
        assert window.control_panel is not None
        assert window.image_panel is not None
        assert window.info_panel is not None
        
        print("Refactoring verification: Panels found.")
        
        # We can't easily wait for async threads in this script without an event loop running,
        # but instantiation success is a good sign.
        window.close()
        
    except Exception as e:
        print(f"GUI Init Failed: {e}")
        raise e

if __name__ == "__main__":
    test_gui_init()
