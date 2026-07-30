# ruff: noqa: E402

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI tests")
pytest.importorskip("pytestqt", reason="pytest-qt is required for GUI tests")
pytestmark = pytest.mark.gui

from PyQt5.QtCore import Qt

from app.gui.i18n import tr
from app.gui.widgets import CameraStatusIndicator


def test_indicator_exposes_text_color_state_and_reconnect_action(qtbot):
    indicator = CameraStatusIndicator("zh")
    qtbot.addWidget(indicator)
    indicator.show()

    indicator.set_state("ready")

    assert indicator.state == "ready"
    assert tr("zh", "camera_state_ready") in indicator.state_label.text()
    assert "#dcfce7" in indicator.state_label.styleSheet()
    assert indicator.reconnect_button.isHidden()

    indicator.set_state("unavailable")

    assert tr("zh", "camera_state_unavailable") in indicator.state_label.text()
    assert indicator.reconnect_button.isVisible()
    assert indicator.reconnect_button.isEnabled()
    with qtbot.waitSignal(indicator.reconnect_requested):
        qtbot.mouseClick(indicator.reconnect_button, Qt.LeftButton)


def test_indicator_retranslates_and_blocks_reconnect_while_busy(qtbot):
    indicator = CameraStatusIndicator("zh")
    qtbot.addWidget(indicator)
    indicator.show()
    indicator.set_state("lost")
    indicator.set_reconnect_allowed(False)

    assert indicator.reconnect_button.isVisible()
    assert not indicator.reconnect_button.isEnabled()

    indicator.set_language("en")

    assert tr("en", "camera_state_lost") in indicator.state_label.text()
    assert indicator.reconnect_button.text() == tr(
        "en",
        "camera_status_reconnect",
    )


def test_indicator_rejects_unknown_state(qtbot):
    indicator = CameraStatusIndicator()
    qtbot.addWidget(indicator)

    with pytest.raises(ValueError, match="Unsupported camera indicator state"):
        indicator.set_state("maybe")
