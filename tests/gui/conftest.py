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

@pytest.fixture(autouse=True)
def isolated_gui_preferences(tmp_path, monkeypatch):
    """Give every GUI test its own preferences store.

    ``PreferencesManager`` persists through ``QSettings()``, which writes to the
    OS settings store — the registry on Windows, an INI under ``~/.config`` on
    Linux. A test that toggles a preference therefore changed what every later
    test read back, and what the developer's own installation read back too.

    That leak surfaced as a Linux-only CI failure: an earlier test persisted
    "show detection boxes = off", so a later window rendered the preprocessed
    image instead of the annotated one. Windows happened to hide it because the
    value did not round-trip within the session.
    """
    if QApplication is None:
        return
    from PyQt5.QtCore import QSettings

    import app.gui.main_window as main_window
    from app.gui.preferences import PreferencesManager

    # Replace the manager factory rather than ``QSettings`` itself: the window
    # also reads ``QSettings.IniFormat`` as a class attribute, so swapping the
    # class for a callable would break unrelated construction.
    ini_path = tmp_path / "gui-preferences.ini"
    monkeypatch.setattr(
        main_window,
        "PreferencesManager",
        lambda *args, **kwargs: PreferencesManager(
            QSettings(str(ini_path), QSettings.IniFormat)
        ),
    )


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
