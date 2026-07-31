from __future__ import annotations

import time

import pytest
from PyQt5.QtCore import QSettings
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QDialog, QMessageBox

from app.gui.panels import control_panel as control_panel_module
from app.gui.panels.control_panel import (
    _PIN_CHANGE_REQUIRED_SETTINGS_KEY,
    ControlPanel,
)


@pytest.fixture
def isolated_panel(qtbot, tmp_path) -> ControlPanel:
    settings = QSettings(
        str(tmp_path / "control-panel.ini"),
        QSettings.IniFormat,
    )
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)
    return panel


def test_engineering_controls_are_not_embedded_in_narrow_operator_layout(
    isolated_panel,
) -> None:
    panel = isolated_panel

    assert panel.engineering_panel.isAncestorOf(panel.model_versions_btn)
    assert panel.engineering_panel.isAncestorOf(panel.inspection_releases_btn)
    assert panel.engineering_panel.isAncestorOf(panel.retraining_workspace_btn)
    assert not panel.model_group.isAncestorOf(panel.model_versions_btn)
    assert not panel.model_group.isAncestorOf(panel.inspection_releases_btn)
    assert not panel.model_group.isAncestorOf(panel.retraining_workspace_btn)
    assert panel.engineering_panel.isAncestorOf(panel.model_update_status_btn)
    assert panel.layout().indexOf(panel.engineering_panel) == -1


def test_engineering_page_navigation_requests_authorization_from_main_window(
    isolated_panel,
) -> None:
    """The panel requests entry; the main window owns the PIN boundary."""
    panel = isolated_panel
    requested = QSignalSpy(panel.engineering_settings_requested)

    panel.engineering_toggle_btn.click()

    assert len(requested) == 1


def test_privileged_controls_only_emit_during_unlocked_session(
    isolated_panel,
    monkeypatch,
) -> None:
    """Disabled controls must not bypass the main window's PIN gate."""
    panel = isolated_panel
    requested = QSignalSpy(panel.retraining_workspace_requested)
    monkeypatch.setattr(panel, "_verify_pin", lambda: True)

    assert panel.engineering_panel.isEnabled() is False
    panel.retraining_workspace_btn.click()
    assert len(requested) == 0

    panel.unlock_engineering_access()
    panel.retraining_workspace_btn.click()
    assert len(requested) == 1

    panel.lock_engineering_access()
    panel.retraining_workspace_btn.click()
    assert len(requested) == 1


def test_legacy_plaintext_pin_is_migrated_to_hashed_credential(
    qtbot,
    tmp_path,
) -> None:
    settings = QSettings(
        str(tmp_path / "engineer-settings.ini"),
        QSettings.IniFormat,
    )
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)
    settings.setValue("engineer_pin", "legacy-pin")

    assert panel._pin_matches("legacy-pin") is True

    credential = str(settings.value("engineer_pin_pbkdf2", ""))
    assert credential.startswith("pbkdf2_sha256$")
    assert "legacy-pin" not in credential
    assert settings.contains("engineer_pin") is False
    assert panel._pin_matches("legacy-pin") is True
    assert panel._pin_matches("wrong-pin") is False


def test_default_pin_requires_site_pin_before_first_engineering_session(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    settings = QSettings(
        str(tmp_path / "engineer-settings.ini"),
        QSettings.IniFormat,
    )
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)

    assert panel._pin_matches("admin") is True
    assert settings.value(
        _PIN_CHANGE_REQUIRED_SETTINGS_KEY,
        False,
        type=bool,
    )

    class _SitePinDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec_(self) -> int:
            return QDialog.Accepted

        def new_pin(self) -> str:
            return "line-a-4821"

    monkeypatch.setattr(control_panel_module, "_ChangePinDialog", _SitePinDialog)
    monkeypatch.setattr(panel, "_verify_pin", lambda: True)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )

    assert panel.unlock_engineering_access() is True
    assert panel.engineering_access_granted is True
    assert panel._pin_matches("line-a-4821") is True
    assert panel._pin_matches("admin") is False
    assert not settings.contains(_PIN_CHANGE_REQUIRED_SETTINGS_KEY)


def test_default_pin_cannot_be_selected_as_replacement(
    isolated_panel,
    monkeypatch,
) -> None:
    panel = isolated_panel
    assert panel._pin_matches("admin") is True

    class _DefaultPinDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec_(self) -> int:
            return QDialog.Accepted

        def new_pin(self) -> str:
            return "admin"

    monkeypatch.setattr(control_panel_module, "_ChangePinDialog", _DefaultPinDialog)
    monkeypatch.setattr(panel, "_verify_pin", lambda: True)
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )

    assert panel.unlock_engineering_access() is False
    assert panel.engineering_access_granted is False
    assert panel._pin_change_required() is True


def test_pin_lockout_survives_new_panel_and_clear(
    qtbot,
    tmp_path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "engineer-settings.ini"
    first_settings = QSettings(str(settings_path), QSettings.IniFormat)
    first = ControlPanel(settings=first_settings)
    qtbot.addWidget(first)
    for _attempt in range(4):
        first._pin_attempt_guard.record_failure()
    first._sync_pin_attempt_snapshot()

    class _WrongPinDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec_(self) -> int:
            return QDialog.Accepted

        def pin_value(self) -> str:
            return "wrong"

        def show_error(self, _message: str) -> None:
            pass

    monkeypatch.setattr(
        control_panel_module,
        "_PinDialog",
        _WrongPinDialog,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )

    assert first._verify_pin() is False
    assert first._pin_locked_until > time.time()

    second_settings = QSettings(str(settings_path), QSettings.IniFormat)
    second = ControlPanel(settings=second_settings)
    qtbot.addWidget(second)

    assert second._pin_locked_until > time.time()
    assert second._verify_pin() is False

    second._clear_pin_attempt_state()
    third_settings = QSettings(str(settings_path), QSettings.IniFormat)
    third = ControlPanel(settings=third_settings)
    qtbot.addWidget(third)

    assert third._pin_locked_until == 0.0
    assert third._pin_failures == 0
    assert third_settings.contains("engineer_pin_attempt_state_v1") is False
