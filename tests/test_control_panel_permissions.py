from __future__ import annotations

from app.gui.panels.control_panel import ControlPanel


def test_model_version_button_is_inside_pin_protected_engineer_panel(qtbot) -> None:
    panel = ControlPanel()
    qtbot.addWidget(panel)

    assert panel.engineering_panel.isAncestorOf(panel.model_versions_btn)
    assert not panel.model_group.isAncestorOf(panel.model_versions_btn)
    assert panel.engineering_panel.isHidden() is True
    assert not panel.engineering_panel.isAncestorOf(panel.model_update_status_btn)
