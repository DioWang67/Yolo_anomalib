import os
import sys

import pytest

try:
    from PyQt5.QtWidgets import QApplication
except ModuleNotFoundError:
    QApplication = None

pytestmark = pytest.mark.gui

# Ensure app modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

@pytest.fixture(scope="session")
def qapp_args():
    return ["--platform", "offscreen"]

@pytest.fixture(scope="session", autouse=True)
def qapp(qapp_args):
    # pytest-qt provides a qapp fixture, but defining it here ensures
    # we can configure it (e.g. use offscreen platform for headless environments)
    if QApplication is None:
        pytest.skip("PyQt5 is required for GUI tests", allow_module_level=True)
    app = QApplication.instance()
    if app is None:
        app = QApplication(qapp_args + sys.argv)
    return app
