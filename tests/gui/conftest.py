import os
import sys
import pytest
from PyQt5.QtWidgets import QApplication

# Ensure app modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

@pytest.fixture(scope="session")
def qapp_args():
    return ["--platform", "offscreen"]

@pytest.fixture(scope="session", autouse=True)
def qapp(qapp_args):
    # pytest-qt provides a qapp fixture, but defining it here ensures
    # we can configure it (e.g. use offscreen platform for headless environments)
    app = QApplication.instance()
    if app is None:
        app = QApplication(qapp_args + sys.argv)
    return app
