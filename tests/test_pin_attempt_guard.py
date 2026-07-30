from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QDialog, QMessageBox

from app.gui.panels import control_panel as control_panel_module
from app.gui.panels.control_panel import (
    _PIN_ATTEMPT_STATE_SETTINGS_KEY,
    _PIN_CREDENTIAL_SETTINGS_KEY,
    _PIN_LOCKOUT_SECONDS,
    _PIN_MAX_FAILURES,
    ControlPanel,
    _PinAttemptGuard,
)


@dataclass
class _MutableClock:
    wall_seconds: float = 1_000.0
    monotonic_seconds: float = 500.0

    def wall_time(self) -> float:
        return self.wall_seconds

    def monotonic_time(self) -> float:
        return self.monotonic_seconds

    def advance(self, seconds: float) -> None:
        self.wall_seconds += seconds
        self.monotonic_seconds += seconds


def _settings(path: Path) -> QSettings:
    return QSettings(str(path), QSettings.IniFormat)


def _guard(
    settings_path: Path,
    lock_path: Path,
    clock: _MutableClock,
) -> _PinAttemptGuard:
    return _PinAttemptGuard(
        _settings(settings_path),
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        lock_path=lock_path,
    )


def _reach_lockout(guard: _PinAttemptGuard) -> None:
    for _attempt in range(_PIN_MAX_FAILURES):
        guard.record_failure()


def test_pin_failures_accumulate_across_live_guard_instances(
    tmp_path: Path,
) -> None:
    """Two panels must not overwrite each other's persisted failure count."""
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    first = _guard(settings_path, lock_path, clock)
    second = _guard(settings_path, lock_path, clock)

    for attempt in range(_PIN_MAX_FAILURES):
        (first if attempt % 2 == 0 else second).record_failure()

    observer = _guard(settings_path, lock_path, clock)
    assert observer.remaining_lockout() > 0


def test_active_lockout_uses_monotonic_time_during_forward_wall_jump(
    tmp_path: Path,
) -> None:
    """Moving wall time forward must not unlock the current process session."""
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    guard = _guard(settings_path, lock_path, clock)
    _reach_lockout(guard)

    clock.wall_seconds += 3_600.0
    clock.monotonic_seconds += 1.0

    remaining = guard.remaining_lockout()
    assert 0 < remaining <= _PIN_LOCKOUT_SECONDS


def test_forward_wall_jump_stays_locked_after_guard_recreation(
    tmp_path: Path,
) -> None:
    """Persisted boot-monotonic time protects a restarted app in the same boot."""
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    _reach_lockout(_guard(settings_path, lock_path, clock))

    clock.wall_seconds += 3_600.0
    clock.monotonic_seconds += 1.0
    recreated = _guard(settings_path, lock_path, clock)

    remaining = recreated.remaining_lockout()
    assert 0 < remaining <= _PIN_LOCKOUT_SECONDS


def test_wall_clock_rollback_remains_locked_after_guard_recreation(
    tmp_path: Path,
) -> None:
    """A restart after wall-clock rollback must rebase, not clear, lockout."""
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    _reach_lockout(_guard(settings_path, lock_path, clock))

    clock.wall_seconds -= 3_600.0
    clock.monotonic_seconds += 1.0
    recreated = _guard(settings_path, lock_path, clock)

    remaining = recreated.remaining_lockout()
    assert 0 < remaining <= _PIN_LOCKOUT_SECONDS


def test_pin_lockout_expires_only_after_deadline(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    guard = _guard(settings_path, lock_path, clock)
    _reach_lockout(guard)

    clock.advance(_PIN_LOCKOUT_SECONDS - 0.001)
    assert guard.remaining_lockout() > 0

    clock.advance(0.002)
    assert guard.remaining_lockout() == 0


def test_corrupt_persisted_pin_state_fails_closed(
    tmp_path: Path,
) -> None:
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    settings = _settings(settings_path)
    guard = _PinAttemptGuard(
        settings,
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        lock_path=lock_path,
    )
    guard.record_failure()
    settings.sync()
    state_keys = settings.allKeys()
    assert state_keys
    for key in state_keys:
        settings.setValue(key, "{not-valid-lockout-state")
    settings.sync()

    recreated = _guard(settings_path, lock_path, clock)

    assert recreated.remaining_lockout() > 0


def test_semantically_invalid_pin_state_types_fail_closed(
    tmp_path: Path,
) -> None:
    """Parseable JSON must reject invalid types and oversized timestamps."""
    huge_timestamp = "1" + ("0" * 1_000)
    invalid_payloads = (
        (
            '{"version":true,"failures":0,'
            '"lockout_until":false,"last_seen":false}'
        ),
        (
            '{"version":1,"failures":0,'
            '"lockout_until":"0","last_seen":"0"}'
        ),
        (
            '{"version":1,"failures":0,"lockout_until":'
            f"{huge_timestamp},"
            '"last_seen":0,"monotonic_until":0,"last_monotonic":0}'
        ),
    )
    clock = _MutableClock()

    for index, payload in enumerate(invalid_payloads):
        settings_path = tmp_path / f"semantic-{index}.ini"
        lock_path = tmp_path / f"semantic-{index}.lock"
        settings = _settings(settings_path)
        settings.setValue(_PIN_ATTEMPT_STATE_SETTINGS_KEY, payload)
        settings.sync()

        guard = _guard(settings_path, lock_path, clock)

        assert guard.remaining_lockout() > 0


def test_provisioned_credential_loss_does_not_restore_default_admin(
    qtbot,
    tmp_path: Path,
) -> None:
    """A missing provisioned hash must fail closed instead of bootstrapping."""
    settings = _settings(tmp_path / "engineer-settings.ini")
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)
    panel._save_pin("site-pin")
    assert settings.contains(_PIN_CREDENTIAL_SETTINGS_KEY)

    settings.remove(_PIN_CREDENTIAL_SETTINGS_KEY)
    settings.sync()

    assert panel._pin_matches("admin") is False


def test_correct_pin_dialog_cannot_clear_concurrent_lockout(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A lockout created while the dialog is open must win atomically."""
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    settings = _settings(settings_path)
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)
    panel_guard = _PinAttemptGuard(
        settings,
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        lock_path=lock_path,
    )
    panel._pin_attempt_guard = panel_guard
    panel._save_pin("correct-pin")
    concurrent_guard = _guard(settings_path, lock_path, clock)

    class _CorrectPinAfterConcurrentLockoutDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec_(self) -> int:
            _reach_lockout(concurrent_guard)
            return QDialog.Accepted

        def pin_value(self) -> str:
            return "correct-pin"

        def show_error(self, _message: str) -> None:
            raise AssertionError("An active lockout must close the dialog")

    monkeypatch.setattr(
        control_panel_module,
        "_PinDialog",
        _CorrectPinAfterConcurrentLockoutDialog,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )

    assert panel._verify_pin() is False
    assert _guard(settings_path, lock_path, clock).remaining_lockout() > 0


def test_successful_pin_clears_failures_across_guard_recreation(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    settings = _settings(settings_path)
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)
    guard = _PinAttemptGuard(
        settings,
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        lock_path=lock_path,
    )
    panel._pin_attempt_guard = guard
    panel._save_pin("correct-pin")
    for _attempt in range(_PIN_MAX_FAILURES - 1):
        guard.record_failure()

    class _CorrectPinDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec_(self) -> int:
            return QDialog.Accepted

        def pin_value(self) -> str:
            return "correct-pin"

        def show_error(self, _message: str) -> None:
            raise AssertionError("Correct PIN must not show an error")

    monkeypatch.setattr(
        control_panel_module,
        "_PinDialog",
        _CorrectPinDialog,
    )

    assert panel._verify_pin() is True

    recreated = _guard(settings_path, lock_path, clock)
    for _attempt in range(_PIN_MAX_FAILURES - 1):
        recreated.record_failure()
    assert recreated.remaining_lockout() == 0
    recreated.record_failure()
    assert recreated.remaining_lockout() > 0


def test_wrong_current_pin_in_change_dialog_counts_toward_lockout(
    qtbot,
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings_path = tmp_path / "engineer-settings.ini"
    lock_path = tmp_path / "engineer-pin.lock"
    clock = _MutableClock()
    settings = _settings(settings_path)
    panel = ControlPanel(settings=settings)
    qtbot.addWidget(panel)
    guard = _PinAttemptGuard(
        settings,
        wall_clock=clock.wall_time,
        monotonic_clock=clock.monotonic_time,
        lock_path=lock_path,
    )
    panel._pin_attempt_guard = guard
    panel._save_pin("correct-pin")
    panel._engineering_access_granted = True
    panel.engineering_panel.setEnabled(True)

    class _WrongCurrentPinDialog:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def exec_(self) -> int:
            return QDialog.Accepted

        def current_pin(self) -> str:
            return "wrong-pin"

        def new_pin(self) -> str:
            return "replacement-pin"

    monkeypatch.setattr(
        control_panel_module,
        "_ChangePinDialog",
        _WrongCurrentPinDialog,
    )
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.Ok,
    )

    for _attempt in range(_PIN_MAX_FAILURES):
        panel._on_change_pin()

    assert guard.remaining_lockout() > 0
