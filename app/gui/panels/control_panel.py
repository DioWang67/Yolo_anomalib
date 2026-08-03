from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import math
import secrets
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from PyQt5.QtCore import QDir, QLockFile, QSettings, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.gui.i18n import LANGUAGE_LABELS, normalize_language, tr

DEFAULT_PRESETS: dict[str, tuple[str, str, str]] = {
    "Select preset": ("", "", ""),
}

_PIN_SETTINGS_KEY = "engineer_pin"
_PIN_CREDENTIAL_SETTINGS_KEY = "engineer_pin_pbkdf2"
_PIN_PROVISIONED_SETTINGS_KEY = "engineer_pin_provisioned_v1"
_PIN_CHANGE_REQUIRED_SETTINGS_KEY = "engineer_pin_change_required_v1"
_PIN_FAILURES_SETTINGS_KEY = "engineer_pin_failures"
_PIN_LOCKOUT_UNTIL_SETTINGS_KEY = "engineer_pin_lockout_until"
_PIN_ATTEMPT_STATE_SETTINGS_KEY = "engineer_pin_attempt_state_v1"
_PIN_DEFAULT = "admin"
_PIN_HASH_ITERATIONS = 210_000
_PIN_MAX_FAILURES = 5
_PIN_LOCKOUT_SECONDS = 30.0
_PIN_CLOCK_JITTER_SECONDS = 5.0
_PIN_LOCK_WAIT_MS = 250

logger = logging.getLogger(__name__)


class _PinAttemptStateError(RuntimeError):
    """PIN throttling state could not be read or persisted safely."""


@dataclass(frozen=True)
class _PinAttemptState:
    failures: int = 0
    lockout_until: float = 0.0
    last_seen: float = 0.0
    monotonic_until: float = 0.0
    last_monotonic: float = 0.0


class _PinAttemptGuard:
    """Serialize and throttle PIN attempts across panels and app instances.

    The persisted boot-monotonic deadline resists wall-clock jumps while the
    OS stays up. After an OS reboot, expiry necessarily relies on the system
    wall clock; this local UI PIN is a mistake-prevention boundary, not OS
    authentication.
    """

    _VERSION = 1

    def __init__(
        self,
        settings: QSettings,
        *,
        wall_clock: Callable[[], float] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        lock_path: str | None = None,
    ) -> None:
        self._settings = settings
        self._wall_clock = wall_clock or time.time
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._monotonic_deadline = 0.0
        self._state = _PinAttemptState()
        if lock_path is None:
            scope = str(settings.fileName() or "yolo11_inspection/ui_settings")
            digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:20]
            lock_path = f"{QDir.tempPath()}/yolo11-pin-{digest}.lock"
        self._lock = QLockFile(str(lock_path))
        self._lock.setStaleLockTime(10_000)

    @property
    def failures(self) -> int:
        return self._state.failures

    @property
    def locked_until(self) -> float:
        return self._state.lockout_until

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if not self._lock.tryLock(_PIN_LOCK_WAIT_MS):
            raise _PinAttemptStateError(
                "Engineer PIN attempt state is busy in another process."
            )
        try:
            yield
        finally:
            self._lock.unlock()

    def _sync_or_raise(self) -> None:
        self._settings.sync()
        if self._settings.status() != QSettings.NoError:
            raise _PinAttemptStateError(
                "Engineer PIN attempt state storage is unavailable."
            )

    def _fail_closed_state(
        self,
        now: float,
        monotonic_now: float,
    ) -> _PinAttemptState:
        logger.error(
            "Engineer PIN attempt state is corrupt; applying a temporary lockout."
        )
        return _PinAttemptState(
            failures=0,
            lockout_until=now + _PIN_LOCKOUT_SECONDS,
            last_seen=now,
            monotonic_until=monotonic_now + _PIN_LOCKOUT_SECONDS,
            last_monotonic=monotonic_now,
        )

    def _decode_state_locked(
        self,
        now: float,
        monotonic_now: float,
    ) -> tuple[_PinAttemptState, bool]:
        self._sync_or_raise()
        raw = self._settings.value(_PIN_ATTEMPT_STATE_SETTINGS_KEY, "")
        if raw:
            try:
                payload = json.loads(str(raw))
                if not isinstance(payload, dict):
                    raise ValueError("PIN attempt state must be an object.")
                version = payload.get("version")
                if type(version) is not int or version != self._VERSION:
                    raise ValueError("Unsupported PIN attempt state version.")
                failures = payload.get("failures")
                if type(failures) is not int:  # bool must not count as int
                    raise ValueError("Invalid PIN failure count.")
                raw_lockout_until = payload.get("lockout_until")
                raw_last_seen = payload.get("last_seen")
                raw_monotonic_until = payload.get("monotonic_until", 0.0)
                raw_last_monotonic = payload.get("last_monotonic", 0.0)
                if type(raw_lockout_until) not in (int, float) or type(
                    raw_last_seen
                ) not in (int, float):
                    raise ValueError("PIN attempt timestamps have invalid types.")
                if type(raw_monotonic_until) not in (int, float) or type(
                    raw_last_monotonic
                ) not in (int, float):
                    raise ValueError(
                        "PIN monotonic timestamps have invalid types."
                    )
                assert isinstance(raw_lockout_until, (int, float))
                assert isinstance(raw_last_seen, (int, float))
                lockout_until = float(raw_lockout_until)
                last_seen = float(raw_last_seen)
                monotonic_until = float(raw_monotonic_until)
                last_monotonic = float(raw_last_monotonic)
                if not 0 <= failures < _PIN_MAX_FAILURES:
                    raise ValueError("PIN failure count is out of range.")
                if (
                    not math.isfinite(lockout_until)
                    or not math.isfinite(last_seen)
                    or not math.isfinite(monotonic_until)
                    or not math.isfinite(last_monotonic)
                    or lockout_until < 0
                    or last_seen < 0
                    or monotonic_until < 0
                    or last_monotonic < 0
                ):
                    raise ValueError("PIN attempt timestamps are invalid.")
                return (
                    _PinAttemptState(
                        failures=failures,
                        lockout_until=lockout_until,
                        last_seen=last_seen,
                        monotonic_until=monotonic_until,
                        last_monotonic=last_monotonic,
                    ),
                    False,
                )
            except (
                OverflowError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return self._fail_closed_state(now, monotonic_now), True

        legacy_present = self._settings.contains(
            _PIN_FAILURES_SETTINGS_KEY
        ) or self._settings.contains(_PIN_LOCKOUT_UNTIL_SETTINGS_KEY)
        if not legacy_present:
            return _PinAttemptState(), False
        try:
            failures = int(
                self._settings.value(_PIN_FAILURES_SETTINGS_KEY, 0) or 0
            )
            lockout_until = float(
                self._settings.value(
                    _PIN_LOCKOUT_UNTIL_SETTINGS_KEY,
                    0.0,
                )
                or 0.0
            )
            if (
                not 0 <= failures < _PIN_MAX_FAILURES
                or not math.isfinite(lockout_until)
                or lockout_until < 0
            ):
                raise ValueError("Legacy PIN attempt state is invalid.")
        except (OverflowError, TypeError, ValueError):
            return self._fail_closed_state(now, monotonic_now), True
        return (
            _PinAttemptState(
                failures=failures,
                lockout_until=lockout_until,
                last_seen=now,
            ),
            True,
        )

    def _normalize_state(
        self,
        state: _PinAttemptState,
        *,
        now: float,
        monotonic_now: float,
    ) -> tuple[_PinAttemptState, float]:
        if not math.isfinite(now) or not math.isfinite(monotonic_now):
            raise _PinAttemptStateError("PIN throttling clock is unavailable.")
        monotonic_remaining = max(
            0.0,
            self._monotonic_deadline - monotonic_now,
        )
        persisted_monotonic_remaining = 0.0
        same_system_boot = (
            state.last_monotonic <= 0
            or monotonic_now + _PIN_CLOCK_JITTER_SECONDS
            >= state.last_monotonic
        )
        if state.monotonic_until and same_system_boot:
            persisted_monotonic_remaining = min(
                _PIN_LOCKOUT_SECONDS,
                max(0.0, state.monotonic_until - monotonic_now),
            )
        wall_remaining = 0.0
        if state.lockout_until:
            clock_rolled_back = (
                state.last_seen > 0
                and now + _PIN_CLOCK_JITTER_SECONDS < state.last_seen
            )
            if clock_rolled_back:
                wall_remaining = state.lockout_until - state.last_seen
            else:
                wall_remaining = state.lockout_until - now
            wall_remaining = min(
                _PIN_LOCKOUT_SECONDS,
                max(0.0, wall_remaining),
            )
        remaining = min(
            _PIN_LOCKOUT_SECONDS,
            max(
                wall_remaining,
                monotonic_remaining,
                persisted_monotonic_remaining,
            ),
        )
        if remaining > 0:
            self._monotonic_deadline = monotonic_now + remaining
            return (
                _PinAttemptState(
                    failures=0,
                    lockout_until=now + remaining,
                    last_seen=now,
                    monotonic_until=monotonic_now + remaining,
                    last_monotonic=monotonic_now,
                ),
                remaining,
            )
        self._monotonic_deadline = 0.0
        if state.failures:
            return (
                _PinAttemptState(
                    failures=state.failures,
                    last_seen=now,
                ),
                0.0,
            )
        return _PinAttemptState(), 0.0

    def _write_state_locked(self, state: _PinAttemptState) -> None:
        if state.failures or state.lockout_until:
            payload = json.dumps(
                {
                    "version": self._VERSION,
                    "failures": state.failures,
                    "lockout_until": state.lockout_until,
                    "last_seen": state.last_seen,
                    "monotonic_until": state.monotonic_until,
                    "last_monotonic": state.last_monotonic,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            self._settings.setValue(
                _PIN_ATTEMPT_STATE_SETTINGS_KEY,
                payload,
            )
        else:
            self._settings.remove(_PIN_ATTEMPT_STATE_SETTINGS_KEY)
        self._settings.remove(_PIN_FAILURES_SETTINGS_KEY)
        self._settings.remove(_PIN_LOCKOUT_UNTIL_SETTINGS_KEY)
        self._sync_or_raise()

    def remaining_lockout(self) -> float:
        """Return remaining seconds after reloading the shared state."""
        with self._locked():
            now = self._wall_clock()
            monotonic_now = self._monotonic_clock()
            state, migrated_or_corrupt = self._decode_state_locked(
                now,
                monotonic_now,
            )
            normalized, remaining = self._normalize_state(
                state,
                now=now,
                monotonic_now=monotonic_now,
            )
            if migrated_or_corrupt or normalized != state:
                self._write_state_locked(normalized)
            self._state = normalized
            return remaining

    def record_failure(self) -> tuple[int, float]:
        """Atomically record one failure and return attempts-left/lockout."""
        with self._locked():
            now = self._wall_clock()
            monotonic_now = self._monotonic_clock()
            state, _migrated_or_corrupt = self._decode_state_locked(
                now,
                monotonic_now,
            )
            normalized, remaining = self._normalize_state(
                state,
                now=now,
                monotonic_now=monotonic_now,
            )
            if remaining > 0:
                self._write_state_locked(normalized)
                self._state = normalized
                return 0, remaining

            failures = normalized.failures + 1
            if failures >= _PIN_MAX_FAILURES:
                remaining = _PIN_LOCKOUT_SECONDS
                self._monotonic_deadline = monotonic_now + remaining
                updated = _PinAttemptState(
                    failures=0,
                    lockout_until=now + remaining,
                    last_seen=now,
                    monotonic_until=monotonic_now + remaining,
                    last_monotonic=monotonic_now,
                )
                attempts_left = 0
            else:
                updated = _PinAttemptState(
                    failures=failures,
                    last_seen=now,
                )
                attempts_left = _PIN_MAX_FAILURES - failures
            self._write_state_locked(updated)
            self._state = updated
            return attempts_left, remaining

    def clear(self) -> None:
        """Atomically clear failures and lockout state."""
        with self._locked():
            self._monotonic_deadline = 0.0
            self._write_state_locked(_PinAttemptState())
            self._state = _PinAttemptState()

    def clear_if_unlocked(self) -> tuple[bool, float]:
        """Clear failures only if no other panel has established a lockout."""
        with self._locked():
            now = self._wall_clock()
            monotonic_now = self._monotonic_clock()
            state, _migrated_or_corrupt = self._decode_state_locked(
                now,
                monotonic_now,
            )
            normalized, remaining = self._normalize_state(
                state,
                now=now,
                monotonic_now=monotonic_now,
            )
            if remaining > 0:
                self._write_state_locked(normalized)
                self._state = normalized
                return False, remaining
            self._monotonic_deadline = 0.0
            self._write_state_locked(_PinAttemptState())
            self._state = _PinAttemptState()
            return True, 0.0


# ---------------------------------------------------------------------------
# PIN entry dialog
# ---------------------------------------------------------------------------

class _PinDialog(QDialog):
    """Simple PIN entry dialog used to unlock the engineer panel."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        language: str = "en",
    ) -> None:
        super().__init__(parent)
        self._language = normalize_language(language)
        self.setWindowTitle(tr(self._language, "engineer_access_title"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._label = QLabel(tr(self._language, "engineer_pin_prompt"))
        layout.addWidget(self._label)

        self._pin_edit = QLineEdit()
        self._pin_edit.setEchoMode(QLineEdit.Password)
        self._pin_edit.setMaxLength(16)
        self._pin_edit.setPlaceholderText(
            tr(self._language, "engineer_pin_placeholder")
        )
        self._pin_edit.returnPressed.connect(self.accept)
        layout.addWidget(self._pin_edit)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #b42318; font-size: 9pt;")
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def pin_value(self) -> str:
        return self._pin_edit.text()

    def show_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._pin_edit.clear()
        self._pin_edit.setFocus()


class _ChangePinDialog(QDialog):
    """Dialog to change the engineer PIN."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        language: str = "en",
        require_current: bool = True,
    ) -> None:
        super().__init__(parent)
        self._language = normalize_language(language)
        self.setWindowTitle(tr(self._language, "change_pin_title"))
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setFixedWidth(280)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        current_label = QLabel(tr(self._language, "current_pin"))
        layout.addWidget(current_label)
        self._current = QLineEdit()
        self._current.setEchoMode(QLineEdit.Password)
        self._current.setMaxLength(16)
        layout.addWidget(self._current)
        current_label.setVisible(require_current)
        self._current.setVisible(require_current)

        layout.addWidget(QLabel(tr(self._language, "new_pin")))
        self._new = QLineEdit()
        self._new.setEchoMode(QLineEdit.Password)
        self._new.setMaxLength(16)
        layout.addWidget(self._new)

        layout.addWidget(QLabel(tr(self._language, "confirm_pin")))
        self._confirm = QLineEdit()
        self._confirm.setEchoMode(QLineEdit.Password)
        self._confirm.setMaxLength(16)
        layout.addWidget(self._confirm)

        self._msg = QLabel("")
        self._msg.setStyleSheet("color: #b42318; font-size: 9pt;")
        self._msg.setWordWrap(True)
        layout.addWidget(self._msg)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        if not self._new.text():
            self._msg.setText(tr(self._language, "pin_cannot_be_empty"))
            return
        if self._new.text() != self._confirm.text():
            self._msg.setText(tr(self._language, "pin_confirmation_mismatch"))
            return
        self.accept()

    def current_pin(self) -> str:
        return self._current.text()

    def new_pin(self) -> str:
        return self._new.text()


# ---------------------------------------------------------------------------
# Main control panel
# ---------------------------------------------------------------------------

class ControlPanel(QGroupBox):
    """Left-side operator controls plus a PIN-gated engineering-page model.

    All widget *names* are preserved from the previous layout so that
    external code (main_window.py aliases) requires no changes.
    """

    product_changed = pyqtSignal(str)
    area_changed = pyqtSignal(str)
    inference_type_changed = pyqtSignal(str)
    preset_selected = pyqtSignal(str, str, str)
    language_changed = pyqtSignal(str)

    start_requested = pyqtSignal()
    stop_requested = pyqtSignal()
    save_requested = pyqtSignal()
    edit_model_config_requested = pyqtSignal()
    inspection_releases_requested = pyqtSignal()
    model_versions_requested = pyqtSignal()
    acceptance_requested = pyqtSignal()
    retraining_workspace_requested = pyqtSignal()
    model_update_status_requested = pyqtSignal()
    inspection_history_requested = pyqtSignal()
    engineering_settings_requested = pyqtSignal()
    engineering_settings_closed = pyqtSignal()
    auto_mode_toggled = pyqtSignal(bool)

    use_camera_toggled = pyqtSignal(bool)
    reconnect_camera_requested = pyqtSignal()
    disconnect_camera_requested = pyqtSignal()

    pick_image_requested = pyqtSignal()
    clear_image_requested = pyqtSignal()
    show_detection_boxes_toggled = pyqtSignal(bool)
    show_original_tab_toggled = pyqtSignal(bool)
    show_processed_tab_toggled = pyqtSignal(bool)
    calib_sample_empty_requested = pyqtSignal()
    calib_sample_product_requested = pyqtSignal()
    calib_apply_requested = pyqtSignal(int, str, str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: QSettings | None = None,
    ) -> None:
        super().__init__("Inspection Setup", parent)
        self._language = "en"
        self._presets: dict[str, tuple[str, str, str]] = {
            tr(self._language, "select_preset"): ("", "", ""),
        }
        self._output_path = "--"
        self._settings = (
            settings
            if settings is not None
            else QSettings("yolo11_inspection", "ui_settings")
        )
        self._ensure_pin_provisioning_marker()
        self._engineering_access_granted = False
        self._pin_failures = 0
        self._pin_locked_until = 0.0
        self._pin_attempt_guard = _PinAttemptGuard(self._settings)
        try:
            self._load_pin_attempt_state()
        except _PinAttemptStateError as exc:
            logger.error("Cannot load engineer PIN attempt state: %s", exc)
            self._pin_locked_until = time.time() + _PIN_LOCKOUT_SECONDS
        self._calib_empty_area: float | None = None
        self._calib_product_area: float | None = None
        self._calib_target: tuple[str, str] | None = None
        self.setMinimumWidth(260)
        self.setMaximumWidth(340)
        self._setup_ui()

    # ------------------------------------------------------------------
    # PIN helpers
    # ------------------------------------------------------------------

    def _ensure_pin_provisioning_marker(self) -> None:
        """Mark pre-existing hashed credentials before any fallback is possible."""
        if not self._settings.contains(_PIN_CREDENTIAL_SETTINGS_KEY):
            return
        if self._settings.contains(_PIN_PROVISIONED_SETTINGS_KEY):
            return
        self._settings.setValue(_PIN_PROVISIONED_SETTINGS_KEY, True)
        self._settings.sync()
        if self._settings.status() != QSettings.NoError:
            logger.error("Cannot persist engineer PIN provisioning marker.")

    def _clear_pin_attempt_state(self) -> None:
        """Clear persisted throttling state after success or expiry."""
        self._pin_attempt_guard.clear()
        self._sync_pin_attempt_snapshot()

    def _clear_pin_attempt_state_if_unlocked(self) -> tuple[bool, float]:
        """Atomically reject a success raced by another panel's lockout."""
        cleared, remaining = self._pin_attempt_guard.clear_if_unlocked()
        self._sync_pin_attempt_snapshot()
        return cleared, remaining

    def _sync_pin_attempt_snapshot(self) -> None:
        self._pin_failures = self._pin_attempt_guard.failures
        self._pin_locked_until = self._pin_attempt_guard.locked_until

    def _load_pin_attempt_state(self) -> float:
        """Reload and validate shared PIN throttling state from QSettings."""
        remaining = self._pin_attempt_guard.remaining_lockout()
        self._sync_pin_attempt_snapshot()
        return remaining

    def _record_pin_failure(self) -> tuple[int, float]:
        """Atomically record a failed credential check."""
        attempts_left, remaining = self._pin_attempt_guard.record_failure()
        self._sync_pin_attempt_snapshot()
        return attempts_left, remaining

    def _save_pin(self, pin: str, *, require_change: bool = False) -> None:
        """Persist a salted PBKDF2 credential and remove legacy plaintext."""
        normalized = str(pin)
        if not normalized or len(normalized) > 16:
            raise ValueError("Engineer PIN must contain 1 to 16 characters.")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            normalized.encode("utf-8"),
            salt,
            _PIN_HASH_ITERATIONS,
        )
        credential = "$".join(
            (
                "pbkdf2_sha256",
                str(_PIN_HASH_ITERATIONS),
                base64.b64encode(salt).decode("ascii"),
                base64.b64encode(digest).decode("ascii"),
            )
        )
        self._settings.setValue(_PIN_CREDENTIAL_SETTINGS_KEY, credential)
        self._settings.setValue(_PIN_PROVISIONED_SETTINGS_KEY, True)
        if require_change:
            self._settings.setValue(_PIN_CHANGE_REQUIRED_SETTINGS_KEY, True)
        else:
            self._settings.remove(_PIN_CHANGE_REQUIRED_SETTINGS_KEY)
        self._settings.remove(_PIN_SETTINGS_KEY)
        self._settings.sync()
        if self._settings.status() != QSettings.NoError:
            raise OSError("Engineer PIN credential storage is unavailable.")

    def _pin_matches(self, candidate: str) -> bool:
        """Verify a PIN and migrate a matching legacy plaintext value."""
        self._settings.sync()
        if self._settings.status() != QSettings.NoError:
            raise OSError("Engineer PIN credential storage is unavailable.")
        credential_present = self._settings.contains(
            _PIN_CREDENTIAL_SETTINGS_KEY
        )
        credential = str(
            self._settings.value(_PIN_CREDENTIAL_SETTINGS_KEY, "") or ""
        )
        if credential_present:
            if not self._settings.contains(_PIN_PROVISIONED_SETTINGS_KEY):
                self._settings.setValue(_PIN_PROVISIONED_SETTINGS_KEY, True)
                self._settings.sync()
                if self._settings.status() != QSettings.NoError:
                    raise OSError(
                        "Engineer PIN provisioning marker is unavailable."
                    )
            if not credential:
                return False
            try:
                algorithm, raw_iterations, raw_salt, raw_digest = (
                    credential.split("$", 3)
                )
                if algorithm != "pbkdf2_sha256":
                    return False
                iterations = int(raw_iterations)
                if not 100_000 <= iterations <= 1_000_000:
                    return False
                salt = base64.b64decode(raw_salt, validate=True)
                expected = base64.b64decode(raw_digest, validate=True)
                if not 16 <= len(salt) <= 64 or len(expected) != 32:
                    return False
            except (binascii.Error, TypeError, ValueError):
                return False
            actual = hashlib.pbkdf2_hmac(
                "sha256",
                str(candidate).encode("utf-8"),
                salt,
                iterations,
            )
            return hmac.compare_digest(actual, expected)

        if self._settings.contains(_PIN_PROVISIONED_SETTINGS_KEY):
            logger.error(
                "Engineer PIN was provisioned but its credential is missing."
            )
            return False
        has_legacy_pin = self._settings.contains(_PIN_SETTINGS_KEY)
        legacy_pin = (
            str(self._settings.value(_PIN_SETTINGS_KEY))
            if has_legacy_pin
            else _PIN_DEFAULT
        )
        if not legacy_pin or len(legacy_pin) > 16:
            return False
        matches = hmac.compare_digest(str(candidate), legacy_pin)
        if matches:
            self._save_pin(
                str(candidate),
                require_change=not has_legacy_pin,
            )
        return matches

    def _pin_change_required(self) -> bool:
        self._settings.sync()
        if self._settings.status() != QSettings.NoError:
            raise OSError("Engineer PIN credential storage is unavailable.")
        return bool(
            self._settings.value(
                _PIN_CHANGE_REQUIRED_SETTINGS_KEY,
                False,
                type=bool,
            )
        )

    def _force_initial_pin_change(self) -> bool:
        """Require a site PIN before granting the first engineering session."""
        from PyQt5.QtWidgets import QMessageBox

        try:
            required = self._pin_change_required()
        except OSError as exc:
            logger.error("Cannot read engineer PIN change requirement: %s", exc)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "engineer_access_title"),
                tr(self._language, "pin_state_unavailable"),
            )
            return False
        if not required:
            return True
        QMessageBox.information(
            self.window(),
            tr(self._language, "change_pin_title"),
            tr(self._language, "pin_change_required"),
        )
        dialog = _ChangePinDialog(
            self.window(),
            language=self._language,
            require_current=False,
        )
        if dialog.exec_() != QDialog.Accepted:
            return False
        new_pin = dialog.new_pin()
        if hmac.compare_digest(new_pin, _PIN_DEFAULT):
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                tr(self._language, "pin_default_reuse_forbidden"),
            )
            return False
        try:
            self._save_pin(new_pin)
        except (OSError, ValueError) as exc:
            logger.error("Cannot save required engineer PIN change: %s", exc)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                tr(self._language, "pin_state_unavailable"),
            )
            return False
        return True

    def _verify_pin(self) -> bool:
        """Show PIN dialog; return True when the correct PIN is entered."""
        from PyQt5.QtWidgets import QMessageBox

        try:
            lockout_remaining = self._load_pin_attempt_state()
        except _PinAttemptStateError as exc:
            logger.error("Cannot verify engineer PIN attempt state: %s", exc)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "engineer_access_title"),
                tr(self._language, "pin_state_unavailable"),
            )
            return False
        if lockout_remaining > 0:
            QMessageBox.warning(
                self.window(),
                tr(self._language, "engineer_access_title"),
                tr(self._language, "pin_locked_remaining").format(
                    seconds=int(lockout_remaining) + 1
                ),
            )
            return False

        dlg = _PinDialog(self.window(), language=self._language)
        while True:
            if dlg.exec_() != QDialog.Accepted:
                return False
            try:
                pin_matches = self._pin_matches(dlg.pin_value())
            except OSError as exc:
                logger.error("Cannot read engineer PIN credential: %s", exc)
                QMessageBox.warning(
                    self.window(),
                    tr(self._language, "engineer_access_title"),
                    tr(self._language, "pin_state_unavailable"),
                )
                return False
            if pin_matches:
                try:
                    cleared, lockout_remaining = (
                        self._clear_pin_attempt_state_if_unlocked()
                    )
                except _PinAttemptStateError as exc:
                    logger.error(
                        "Cannot clear engineer PIN attempt state: %s",
                        exc,
                    )
                    QMessageBox.warning(
                        self.window(),
                        tr(self._language, "engineer_access_title"),
                        tr(self._language, "pin_state_unavailable"),
                    )
                    return False
                if not cleared:
                    QMessageBox.warning(
                        self.window(),
                        tr(self._language, "engineer_access_title"),
                        tr(self._language, "pin_locked_remaining").format(
                            seconds=int(lockout_remaining) + 1
                        ),
                    )
                    return False
                return True
            try:
                attempts_left, lockout_remaining = (
                    self._record_pin_failure()
                )
            except _PinAttemptStateError as exc:
                logger.error(
                    "Cannot persist engineer PIN failure: %s",
                    exc,
                )
                QMessageBox.warning(
                    self.window(),
                    tr(self._language, "engineer_access_title"),
                    tr(self._language, "pin_state_unavailable"),
                )
                return False
            if lockout_remaining > 0:
                QMessageBox.warning(
                    self.window(),
                    tr(self._language, "engineer_access_title"),
                    tr(self._language, "pin_too_many_attempts").format(
                        seconds=max(1, int(lockout_remaining))
                    ),
                )
                return False
            dlg.show_error(
                tr(self._language, "pin_incorrect_remaining").format(
                    attempts=attempts_left
                )
            )

    def unlock_engineering_access(self) -> bool:
        """Authenticate once and enable the controls for the current page visit."""
        if self._engineering_access_granted:
            return True
        if not self._verify_pin():
            return False
        if not self._force_initial_pin_change():
            return False
        self._engineering_access_granted = True
        self.engineering_panel.setEnabled(True)
        return True

    def lock_engineering_access(self) -> None:
        """Revoke the page session and discard target-sensitive draft data."""
        self._engineering_access_granted = False
        self.engineering_panel.setEnabled(False)
        self.clear_calibration()

    @property
    def engineering_access_granted(self) -> bool:
        return self._engineering_access_granted

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QVBoxLayout()
        root.setSpacing(10)

        # ── Operator section ──────────────────────────────────────────
        self._build_operator_section(root)

        self.inspection_history_btn = QPushButton("Inspection Records >")
        self.inspection_history_btn.setObjectName("secondaryAction")
        self.inspection_history_btn.clicked.connect(
            self.inspection_history_requested.emit
        )
        root.addWidget(self.inspection_history_btn)

        # ── Engineer page navigation (PIN gate) ──────────────────────
        self.engineering_toggle_btn = QPushButton("Engineer Settings >")
        self.engineering_toggle_btn.setObjectName("secondaryAction")
        self.engineering_toggle_btn.setToolTip(
            "PIN required to open the full engineering settings page"
        )
        self.engineering_toggle_btn.clicked.connect(
            self.engineering_settings_requested.emit
        )
        root.addWidget(self.engineering_toggle_btn)

        # The controls are hosted by the main window's full-width engineering
        # page. They are created here to preserve all existing signal wiring.
        self.engineering_panel = QScrollArea()
        self.engineering_panel.setWidgetResizable(True)
        self.engineering_panel.setFrameShape(self.engineering_panel.NoFrame)
        self.engineering_panel.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        _eng_inner = QWidget()
        self._build_engineer_panel(_eng_inner)
        self.engineering_panel.setWidget(_eng_inner)
        self.engineering_panel.setEnabled(False)

        root.addStretch()
        self.setLayout(root)
        self.set_language(self._language)

    def _build_operator_section(self, layout: QVBoxLayout) -> None:
        """Widgets always visible to the operator."""

        # Language
        self.language_label = QLabel()
        self.language_combo = QComboBox()
        for code, label in LANGUAGE_LABELS.items():
            self.language_combo.addItem(label, code)
        self.language_combo.currentIndexChanged.connect(self._on_language_selected)
        layout.addWidget(self.language_label)
        layout.addWidget(self.language_combo)

        # Product / Area / Model selection
        self.model_group = QGroupBox("Product")
        model_layout = QVBoxLayout()
        model_layout.setSpacing(6)

        self.preset_label = QLabel()
        self.preset_combo = QComboBox()
        self.preset_combo.setToolTip("Quick switch product / area / model")
        self._rebuild_preset_combo()
        self.preset_combo.currentTextChanged.connect(self._on_preset_selected)
        model_layout.addWidget(self.preset_label)
        model_layout.addWidget(self.preset_combo)

        self.product_combo = QComboBox()
        self.product_combo.currentTextChanged.connect(
            self._invalidate_calibration_for_selection
        )
        self.product_combo.currentTextChanged.connect(self.product_changed.emit)
        self.product_label = QLabel()
        model_layout.addWidget(self.product_label)
        model_layout.addWidget(self.product_combo)

        self.area_combo = QComboBox()
        self.area_combo.currentTextChanged.connect(
            self._invalidate_calibration_for_selection
        )
        self.area_combo.currentTextChanged.connect(self.area_changed.emit)
        self.area_label = QLabel()
        model_layout.addWidget(self.area_label)
        model_layout.addWidget(self.area_combo)

        self.inference_combo = QComboBox()
        self.inference_combo.currentTextChanged.connect(self.inference_type_changed.emit)
        self.model_label = QLabel()
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.inference_combo)

        self.model_group.setLayout(model_layout)
        layout.addWidget(self.model_group)

        # Auto Mode
        self.auto_mode_chk = QCheckBox("Auto Mode")
        self.auto_mode_chk.setToolTip(
            "Automatically detect and inspect when product is stable in ROI.\n"
            "Requires camera. Button-triggered inspection is disabled while active."
        )
        self.auto_mode_chk.toggled.connect(self._on_auto_mode_changed)
        layout.addWidget(self.auto_mode_chk)

        self.auto_mode_status_label = QLabel("")
        self.auto_mode_status_label.setStyleSheet(
            "color: #0369a1; font-size: 9pt; padding: 2px 4px;"
        )
        self.auto_mode_status_label.setWordWrap(True)
        layout.addWidget(self.auto_mode_status_label)

        # Operation buttons
        self.button_group = QGroupBox("Operation")
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        self.start_btn = QPushButton("Start Inspection")
        self.start_btn.setObjectName("primaryAction")
        self.start_btn.setEnabled(False)
        self.start_btn.setMinimumHeight(48)
        font = self.start_btn.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.start_btn.setFont(font)
        self.start_btn.clicked.connect(self.start_requested.emit)

        # Ghost widgets: kept alive for main_window references but not in layout
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("dangerAction")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.stop_btn.hide()

        self.save_btn = QPushButton("Save Result")
        self.save_btn.setObjectName("secondaryAction")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.save_btn.hide()

        btn_layout.addWidget(self.start_btn)
        self.button_group.setLayout(btn_layout)
        layout.addWidget(self.button_group)

    def _build_engineer_panel(self, panel: QWidget) -> None:
        """Build advanced controls for the full-width engineering page."""
        root_layout = QVBoxLayout(panel)
        root_layout.setContentsMargins(8, 8, 8, 16)
        self.engineering_tabs = QTabWidget()
        self.version_workspace = QWidget()
        self.improvement_workspace = QWidget()
        self.equipment_workspace = QWidget()
        self.engineering_tabs.addTab(self.version_workspace, "版本與上線")
        self.engineering_tabs.addTab(self.improvement_workspace, "資料與補訓")
        self.engineering_tabs.addTab(self.equipment_workspace, "設備與系統")
        root_layout.addWidget(self.engineering_tabs)

        self.version_layout = QVBoxLayout(self.version_workspace)
        self.version_layout.setContentsMargins(14, 14, 14, 14)
        self.version_layout.setSpacing(12)
        improvement_layout = QVBoxLayout(self.improvement_workspace)
        improvement_layout.setContentsMargins(14, 14, 14, 14)
        improvement_layout.setSpacing(12)
        equipment_layout = QGridLayout(self.equipment_workspace)
        equipment_layout.setContentsMargins(14, 14, 14, 14)
        equipment_layout.setHorizontalSpacing(16)
        equipment_layout.setVerticalSpacing(14)
        equipment_layout.setColumnStretch(0, 1)
        equipment_layout.setColumnStretch(1, 1)

        # Camera
        camera_group = QGroupBox("Camera")
        self.camera_group = camera_group
        cam_layout = QVBoxLayout()
        cam_layout.setSpacing(6)

        self.use_camera_chk = QCheckBox("Use camera")
        self.use_camera_chk.toggled.connect(self.use_camera_toggled.emit)
        cam_layout.addWidget(self.use_camera_chk)

        self.reconnect_camera_btn = QPushButton("Reconnect")
        self.reconnect_camera_btn.setObjectName("secondaryAction")
        self.reconnect_camera_btn.clicked.connect(self.reconnect_camera_requested.emit)
        cam_layout.addWidget(self.reconnect_camera_btn)

        self.disconnect_camera_btn = QPushButton("Disconnect")
        self.disconnect_camera_btn.setObjectName("secondaryAction")
        self.disconnect_camera_btn.clicked.connect(self.disconnect_camera_requested.emit)
        cam_layout.addWidget(self.disconnect_camera_btn)

        self.pick_image_btn = QPushButton("Choose Image...")
        self.pick_image_btn.setObjectName("secondaryAction")
        self.pick_image_btn.clicked.connect(self.pick_image_requested.emit)

        self.image_path_label = QLabel("No image selected")
        self.image_path_label.setStyleSheet("color: #6b7280; font-size: 9pt;")
        self.image_path_label.setWordWrap(True)

        # Ghost widget: referenced externally but not shown (pick_image_btn already replaces selection)
        self.clear_image_btn = QPushButton("Clear Image")
        self.clear_image_btn.setObjectName("secondaryAction")
        self.clear_image_btn.setEnabled(False)
        self.clear_image_btn.clicked.connect(self.clear_image_requested.emit)
        self.clear_image_btn.hide()

        cam_layout.addWidget(self.pick_image_btn)
        cam_layout.addWidget(self.image_path_label)
        camera_group.setLayout(cam_layout)
        equipment_layout.addWidget(camera_group, 0, 0)

        # Version and release workflow
        debug_group = QGroupBox("檢測組合管理")
        self.debug_group = debug_group
        debug_layout = QVBoxLayout()
        debug_layout.setSpacing(6)

        self.version_workflow_label = QLabel(
            "① 模型與顏色版本  →  ② 候選組合  →  "
            "③ 組合驗收  →  ④ 上線與回退"
        )
        self.version_workflow_label.setWordWrap(True)
        self.version_workflow_label.setStyleSheet(
            "background:#eef6ff;color:#245b8f;border:1px solid #c7ddf2;"
            "border-radius:6px;padding:10px;font-weight:600;"
        )
        debug_layout.addWidget(self.version_workflow_label)

        self.current_combination_label = QLabel("目前正式組合：—")
        self.current_combination_label.setWordWrap(True)
        self.current_combination_label.setStyleSheet(
            "background:#edf6ed;color:#246b36;border:1px solid #bad8bf;"
            "border-radius:6px;padding:10px;font-weight:600;"
        )
        debug_layout.addWidget(self.current_combination_label)

        self.model_versions_btn = QPushButton("模型與顏色版本")
        self.model_versions_btn.setObjectName("secondaryAction")
        self.model_versions_btn.clicked.connect(self.model_versions_requested.emit)
        debug_layout.addWidget(self.model_versions_btn)

        self.inspection_releases_btn = QPushButton("檢測組合管理")
        self.inspection_releases_btn.setObjectName("secondaryAction")
        self.inspection_releases_btn.clicked.connect(
            self.inspection_releases_requested.emit
        )
        debug_layout.addWidget(self.inspection_releases_btn)

        self.acceptance_btn = QPushButton("組合驗收")
        self.acceptance_btn.setObjectName("primaryAction")
        self.acceptance_btn.clicked.connect(self.acceptance_requested.emit)
        debug_layout.addWidget(self.acceptance_btn)

        self.edit_model_config_btn = QPushButton("編輯目前檢測參數")
        self.edit_model_config_btn.setObjectName("secondaryAction")
        self.edit_model_config_btn.clicked.connect(self.edit_model_config_requested.emit)
        debug_layout.addWidget(self.edit_model_config_btn)

        debug_group.setLayout(debug_layout)
        self.version_layout.addWidget(debug_group)
        self.version_layout.addStretch(1)

        # Display and output controls remain operational settings, not versions.
        self.display_group = QGroupBox("顯示與輸出")
        display_layout = QVBoxLayout()
        display_layout.setSpacing(6)

        self.output_path_label = QLabel("Output: --")
        self.output_path_label.setStyleSheet(
            "color: #6b7280; font-size: 8pt; padding: 2px 4px;"
        )
        self.output_path_label.setWordWrap(True)
        self.output_path_label.setToolTip("Current result output directory")
        display_layout.addWidget(self.output_path_label)

        self.show_detection_boxes_chk = QCheckBox("Show detection boxes")
        self.show_detection_boxes_chk.setChecked(True)
        self.show_detection_boxes_chk.setToolTip("Toggle inspection overlays on result view")
        self.show_detection_boxes_chk.toggled.connect(self.show_detection_boxes_toggled.emit)
        display_layout.addWidget(self.show_detection_boxes_chk)

        self.show_original_tab_chk = QCheckBox("Show original tab")
        self.show_original_tab_chk.setChecked(True)
        self.show_original_tab_chk.setToolTip("Show or hide the original image tab")
        self.show_original_tab_chk.toggled.connect(self.show_original_tab_toggled.emit)
        display_layout.addWidget(self.show_original_tab_chk)

        self.show_processed_tab_chk = QCheckBox("Show processed tab")
        self.show_processed_tab_chk.setChecked(True)
        self.show_processed_tab_chk.setToolTip("Show or hide the processed image tab")
        self.show_processed_tab_chk.toggled.connect(self.show_processed_tab_toggled.emit)
        display_layout.addWidget(self.show_processed_tab_chk)
        self.display_group.setLayout(display_layout)
        equipment_layout.addWidget(self.display_group, 0, 1)

        # Model retraining
        self.retraining_group = QGroupBox("Model Retraining")
        retraining_layout = QVBoxLayout()
        retraining_layout.setSpacing(6)

        self.improvement_workflow_label = QLabel(
            "資料複核  →  模型補訓／顏色校正  →  新模型／顏色版本"
        )
        self.improvement_workflow_label.setWordWrap(True)
        self.improvement_workflow_label.setStyleSheet(
            "background:#fff7e8;color:#8a5a00;border:1px solid #ead3a3;"
            "border-radius:6px;padding:10px;font-weight:600;"
        )
        retraining_layout.addWidget(self.improvement_workflow_label)

        self.retraining_workspace_btn = QPushButton(
            "Open Retraining Data / Submit"
        )
        self.retraining_workspace_btn.setObjectName("secondaryAction")
        self.retraining_workspace_btn.setToolTip(
            "Review selected images, configure retraining, and submit one job"
        )
        self.retraining_workspace_btn.clicked.connect(
            self.retraining_workspace_requested.emit
        )
        retraining_layout.addWidget(self.retraining_workspace_btn)

        self.model_update_status_btn = QPushButton("Retraining Progress")
        self.model_update_status_btn.setObjectName("secondaryAction")
        self.model_update_status_btn.setToolTip(
            "View retraining progress, resume jobs, or request a safe stop"
        )
        self.model_update_status_btn.clicked.connect(
            self.model_update_status_requested.emit
        )
        retraining_layout.addWidget(self.model_update_status_btn)

        self.retraining_group.setLayout(retraining_layout)
        improvement_layout.addWidget(self.retraining_group)
        improvement_layout.addStretch(1)

        # Auto-trigger calibration
        calib_group = QGroupBox("Auto-Trigger Calibration")
        self.calib_group = calib_group
        calib_layout = QVBoxLayout()
        calib_layout.setSpacing(6)

        self._calib_hint_label = QLabel()
        self._calib_hint_label.setStyleSheet("color: #6b7280; font-size: 8pt;")
        self._calib_hint_label.setWordWrap(True)
        calib_layout.addWidget(self._calib_hint_label)

        self._calib_target_label = QLabel()
        self._calib_target_label.setStyleSheet(
            "color:#245b8f;font-size:8pt;font-weight:600;"
        )
        calib_layout.addWidget(self._calib_target_label)

        self._calib_empty_btn = QPushButton("Sample Empty")
        self._calib_empty_btn.setObjectName("secondaryAction")
        self._calib_empty_btn.clicked.connect(self.calib_sample_empty_requested.emit)
        calib_layout.addWidget(self._calib_empty_btn)

        self._calib_empty_val = QLabel("Empty area: --")
        self._calib_empty_val.setStyleSheet("font-size: 8pt; color: #374151;")
        calib_layout.addWidget(self._calib_empty_val)

        self._calib_product_btn = QPushButton("Sample Product")
        self._calib_product_btn.setObjectName("secondaryAction")
        self._calib_product_btn.clicked.connect(self.calib_sample_product_requested.emit)
        calib_layout.addWidget(self._calib_product_btn)

        self._calib_product_val = QLabel("Product area: --")
        self._calib_product_val.setStyleSheet("font-size: 8pt; color: #374151;")
        calib_layout.addWidget(self._calib_product_val)

        self._calib_threshold_val = QLabel("Threshold: --")
        self._calib_threshold_val.setStyleSheet("font-size: 8pt; font-weight: bold; color: #1d4ed8;")
        calib_layout.addWidget(self._calib_threshold_val)

        self._calib_apply_btn = QPushButton("Apply Threshold")
        self._calib_apply_btn.setObjectName("primaryAction")
        self._calib_apply_btn.setEnabled(False)
        self._calib_apply_btn.clicked.connect(self._on_calib_apply)
        calib_layout.addWidget(self._calib_apply_btn)

        calib_group.setLayout(calib_layout)
        equipment_layout.addWidget(calib_group, 1, 0)

        # Security (change PIN / lock)
        sec_group = QGroupBox("Security")
        self.sec_group = sec_group
        sec_layout = QVBoxLayout()
        sec_layout.setSpacing(6)

        self._change_pin_btn = QPushButton("Change PIN...")
        self._change_pin_btn.setObjectName("secondaryAction")
        self._change_pin_btn.clicked.connect(self._on_change_pin)
        sec_layout.addWidget(self._change_pin_btn)

        self._lock_btn = QPushButton("Lock Engineer Mode")
        self._lock_btn.setObjectName("dangerAction")
        self._lock_btn.clicked.connect(self._lock_engineer)
        sec_layout.addWidget(self._lock_btn)

        sec_group.setLayout(sec_layout)
        equipment_layout.addWidget(sec_group, 1, 1)
        equipment_layout.setRowStretch(2, 1)

    # ------------------------------------------------------------------
    # Engineer page navigation (PIN gate)
    # ------------------------------------------------------------------

    def _on_auto_mode_changed(self, enabled: bool) -> None:
        """Lock model-selection combos while Auto Mode is active."""
        for widget in (
            self.preset_combo,
            self.product_combo,
            self.area_combo,
            self.inference_combo,
        ):
            widget.setEnabled(not enabled)
        self.auto_mode_toggled.emit(enabled)

    def _lock_engineer(self) -> None:
        """Leave the engineering page; the next entry requires PIN again."""
        self.lock_engineering_access()
        self.engineering_settings_closed.emit()

    # ------------------------------------------------------------------
    # PIN management
    # ------------------------------------------------------------------

    def _on_change_pin(self) -> None:
        if not self._engineering_access_granted:
            return
        from PyQt5.QtWidgets import QMessageBox

        try:
            lockout_remaining = self._load_pin_attempt_state()
        except _PinAttemptStateError as exc:
            logger.error("Cannot load PIN state before credential change: %s", exc)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                tr(self._language, "pin_state_unavailable"),
            )
            return
        if lockout_remaining > 0:
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                tr(self._language, "pin_locked_remaining").format(
                    seconds=int(lockout_remaining) + 1
                ),
            )
            return
        dlg = _ChangePinDialog(self.window(), language=self._language)
        if dlg.exec_() != QDialog.Accepted:
            return
        try:
            current_pin_matches = self._pin_matches(dlg.current_pin())
        except OSError as exc:
            logger.error("Cannot read current engineer PIN: %s", exc)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                tr(self._language, "pin_state_unavailable"),
            )
            return
        if not current_pin_matches:
            try:
                attempts_left, lockout_remaining = (
                    self._record_pin_failure()
                )
            except _PinAttemptStateError as exc:
                logger.error("Cannot persist change-PIN failure: %s", exc)
                QMessageBox.warning(
                    self.window(),
                    tr(self._language, "change_pin_title"),
                    tr(self._language, "pin_state_unavailable"),
                )
                return
            if lockout_remaining > 0:
                message = tr(
                    self._language,
                    "pin_too_many_attempts",
                ).format(seconds=max(1, int(lockout_remaining)))
            else:
                message = tr(
                    self._language,
                    "pin_incorrect_remaining",
                ).format(attempts=attempts_left)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                message,
            )
            if lockout_remaining > 0:
                self._lock_engineer()
            return
        try:
            cleared, lockout_remaining = (
                self._clear_pin_attempt_state_if_unlocked()
            )
            if not cleared:
                QMessageBox.warning(
                    self.window(),
                    tr(self._language, "change_pin_title"),
                    tr(self._language, "pin_locked_remaining").format(
                        seconds=int(lockout_remaining) + 1
                    ),
                )
                return
            self._save_pin(dlg.new_pin())
        except (OSError, _PinAttemptStateError, ValueError) as exc:
            logger.error("Cannot change engineer PIN: %s", exc)
            QMessageBox.warning(
                self.window(),
                tr(self._language, "change_pin_title"),
                tr(self._language, "pin_state_unavailable"),
            )
            return
        QMessageBox.information(
            self.window(),
            tr(self._language, "change_pin_title"),
            tr(self._language, "pin_changed_successfully"),
        )

    # ------------------------------------------------------------------
    # Preset helpers
    # ------------------------------------------------------------------

    def set_presets(self, presets: dict[str, tuple[str, str, str]]) -> None:
        self._presets = {tr(self._language, "select_preset"): ("", "", ""), **presets}
        self._rebuild_preset_combo()

    def _rebuild_preset_combo(self) -> None:
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(list(self._presets.keys()))
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, name: str) -> None:
        entry = self._presets.get(name)
        if entry and entry != ("", "", ""):
            product, area, inf_type = entry
            self.preset_selected.emit(product, area, inf_type)

    # ------------------------------------------------------------------
    # Public helpers called by main_window
    # ------------------------------------------------------------------

    def engineering_focus_widgets(self) -> tuple[QWidget, ...]:
        """Return engineering controls in their visual keyboard order."""
        return (
            self.model_versions_btn,
            self.inspection_releases_btn,
            self.acceptance_btn,
            self.edit_model_config_btn,
            self.retraining_workspace_btn,
            self.model_update_status_btn,
            self.use_camera_chk,
            self.reconnect_camera_btn,
            self.disconnect_camera_btn,
            self.pick_image_btn,
            self.show_detection_boxes_chk,
            self.show_original_tab_chk,
            self.show_processed_tab_chk,
            self._calib_empty_btn,
            self._calib_product_btn,
            self._calib_apply_btn,
            self._change_pin_btn,
            self._lock_btn,
        )

    def _selected_target(self) -> tuple[str, str]:
        return (
            self.product_combo.currentText().strip(),
            self.area_combo.currentText().strip(),
        )

    def _normalize_calibration_target(
        self,
        target: tuple[str, str],
    ) -> tuple[str, str]:
        normalized = (str(target[0]).strip(), str(target[1]).strip())
        if not all(normalized):
            raise ValueError("Calibration target requires product and area.")
        return normalized

    def _prepare_calibration_target(
        self,
        target: tuple[str, str],
    ) -> tuple[str, str]:
        normalized = self._normalize_calibration_target(target)
        if normalized != self._selected_target():
            raise ValueError(
                "Calibration sample target does not match the current selection."
            )
        if self._calib_target not in (None, normalized):
            self.clear_calibration()
        self._calib_target = normalized
        self._refresh_calibration_target_label()
        return normalized

    def _normalize_calibration_sample(self, value: float) -> float:
        sample = float(value)
        if not math.isfinite(sample) or sample < 0:
            raise ValueError("Calibration sample must be a finite non-negative area.")
        return sample

    def set_calib_empty(
        self,
        area: float,
        *,
        target: tuple[str, str],
    ) -> None:
        sample = self._normalize_calibration_sample(area)
        self._prepare_calibration_target(target)
        self._calib_empty_area = sample
        prefix = tr(self._language, "empty_area_prefix")
        self._calib_empty_val.setText(f"{prefix} {sample:,.0f}")
        self._refresh_calib_threshold()

    def set_calib_product(
        self,
        area: float,
        *,
        target: tuple[str, str],
    ) -> None:
        sample = self._normalize_calibration_sample(area)
        self._prepare_calibration_target(target)
        self._calib_product_area = sample
        prefix = tr(self._language, "product_area_prefix")
        self._calib_product_val.setText(f"{prefix} {sample:,.0f}")
        self._refresh_calib_threshold()

    def clear_calibration(self) -> None:
        """Discard draft samples so they cannot cross target/session boundaries."""
        self._calib_empty_area = None
        self._calib_product_area = None
        self._calib_target = None
        if not hasattr(self, "_calib_apply_btn"):
            return
        self._calib_apply_btn.setEnabled(False)
        self._calib_apply_btn.setToolTip("")
        self._calib_empty_val.setText(
            tr(self._language, "empty_area_label")
        )
        self._calib_product_val.setText(
            tr(self._language, "product_area_label")
        )
        self._calib_threshold_val.setText(
            tr(self._language, "threshold_label")
        )
        self._refresh_calibration_target_label()

    def _invalidate_calibration_for_selection(self, _value: str) -> None:
        if (
            self._calib_target is not None
            and self._selected_target() != self._calib_target
        ):
            self.clear_calibration()

    def _refresh_calibration_target_label(self) -> None:
        if not hasattr(self, "_calib_target_label"):
            return
        if self._calib_target is None:
            text = tr(self._language, "calib_target_missing")
        else:
            product, area = self._calib_target
            text = tr(self._language, "calib_target").format(
                product=product,
                area=area,
            )
        self._calib_target_label.setText(text)

    def _refresh_calib_threshold(self) -> None:
        if (
            self._calib_target is not None
            and self._calib_target == self._selected_target()
            and self._calib_empty_area is not None
            and self._calib_product_area is not None
        ):
            threshold = int((self._calib_empty_area + self._calib_product_area) / 2)
            prefix = tr(self._language, "threshold_prefix")
            self._calib_threshold_val.setText(f"{prefix} {threshold:,}")
            self._calib_apply_btn.setEnabled(True)
            self._calib_apply_btn.setToolTip(
                tr(self._language, "calib_apply_hint").format(
                    threshold=f"{threshold:,}",
                    empty=f"{self._calib_empty_area:,.0f}",
                    product=f"{self._calib_product_area:,.0f}",
                )
            )
            return
        self._calib_apply_btn.setEnabled(False)
        self._calib_apply_btn.setToolTip("")

    def _on_calib_apply(self) -> None:
        if (
            self._calib_target is None
            or self._calib_target != self._selected_target()
            or self._calib_empty_area is None
            or self._calib_product_area is None
        ):
            self.clear_calibration()
            return
        threshold = int((self._calib_empty_area + self._calib_product_area) / 2)
        product, area = self._calib_target
        self.calib_apply_requested.emit(threshold, product, area)

    def set_output_path(self, path: str) -> None:
        self._output_path = path
        self.output_path_label.setText(f"{tr(self._language, 'output')}: {path}")
        self.output_path_label.setToolTip(path)

    def install_version_workspace(self, workspace: QWidget) -> None:
        """Replace legacy version launchers with the embedded workflow."""
        current = getattr(self, "_embedded_version_workspace", None)
        if current is workspace:
            return
        if current is not None:
            self.version_layout.removeWidget(current)
            current.setParent(None)
        self._embedded_version_workspace = workspace
        self.version_layout.insertWidget(0, workspace, 1)
        self.debug_group.hide()

    def set_current_inspection_combination(self, summary: str) -> None:
        """Update the read-only production combination shown in Engineering."""
        self.current_combination_label.setText(summary.strip() or "目前正式組合：—")

    def set_auto_mode_status(self, state_name: str) -> None:
        self.auto_mode_status_label.setText(state_name)

    # ------------------------------------------------------------------
    # Localisation
    # ------------------------------------------------------------------

    def set_language(self, language: str) -> None:
        self._language = normalize_language(language)
        self.setTitle(tr(self._language, "inspection_setup"))

        # Operator section
        self.preset_label.setText(tr(self._language, "preset"))
        self.auto_mode_chk.setText(tr(self._language, "auto_mode"))
        self.button_group.setTitle(tr(self._language, "operation"))
        self.start_btn.setText(tr(self._language, "start"))
        self.stop_btn.setText(tr(self._language, "stop"))
        self.save_btn.setText(tr(self._language, "save_result"))

        # Operator section (continued)
        self.language_label.setText(tr(self._language, "language"))
        self.model_group.setTitle(tr(self._language, "product_group"))
        self.product_label.setText(tr(self._language, "product"))
        self.area_label.setText(tr(self._language, "area"))
        self.model_label.setText(tr(self._language, "model"))
        zh = self._language.lower().startswith("zh")
        self.engineering_tabs.setTabText(
            0, "版本與上線" if zh else "Versions & Deployment"
        )
        self.engineering_tabs.setTabText(
            1, "資料與補訓" if zh else "Data & Retraining"
        )
        self.engineering_tabs.setTabText(
            2, "設備與系統" if zh else "Equipment & System"
        )
        self.debug_group.setTitle(
            "檢測組合管理" if zh else "Inspection Combination Management"
        )
        self.version_workflow_label.setText(
            (
                "① 模型與顏色版本  →  ② 候選組合  →  "
                "③ 組合驗收  →  ④ 上線與回退"
            )
            if zh
            else (
                "1. Model & color versions  →  2. Candidate combination  →  "
                "3. Acceptance  →  4. Deploy or roll back"
            )
        )
        if self.current_combination_label.text() in {
            "目前正式組合：—",
            "Production combination: —",
        }:
            self.current_combination_label.setText(
                "目前正式組合：—" if zh else "Production combination: —"
            )
        self.model_versions_btn.setText(
            "模型與顏色版本" if zh else "Model & Color Versions"
        )
        self.inspection_releases_btn.setText(
            "檢測組合管理" if zh else "Inspection Combinations"
        )
        self.acceptance_btn.setText(
            "組合驗收" if zh else "Combination Acceptance"
        )
        self.edit_model_config_btn.setText(
            "編輯目前檢測參數" if zh else "Edit Active Inspection Parameters"
        )
        self.model_update_status_btn.setText(
            "模型補訓進度"
            if zh
            else "Retraining Progress"
        )
        self.model_update_status_btn.setToolTip(
            tr(self._language, "retraining_progress_hint")
        )

        # Engineer section labels (update even when hidden so they're correct on reveal)
        self.camera_group.setTitle(tr(self._language, "camera_group"))
        self.use_camera_chk.setText(tr(self._language, "use_camera"))
        self.reconnect_camera_btn.setText(tr(self._language, "reconnect"))
        self.disconnect_camera_btn.setText(tr(self._language, "disconnect"))
        self.pick_image_btn.setText(tr(self._language, "choose_image"))
        if self.image_path_label.text() in {"No image selected", "尚未選擇影像"}:
            self.image_path_label.setText(tr(self._language, "no_image"))
        self.clear_image_btn.setText(tr(self._language, "clear_image"))
        self.display_group.setTitle(
            "顯示與輸出" if zh else "Display & Output"
        )
        self.retraining_group.setTitle(
            "資料與補訓工作流" if zh else "Data & Retraining Workflow"
        )
        self.improvement_workflow_label.setText(
            "資料複核  →  模型補訓／顏色校正  →  新模型／顏色版本"
            if zh
            else (
                "Data review  →  Model retraining / color calibration  →  "
                "New component version"
            )
        )
        self.retraining_workspace_btn.setText(
            "資料複核與改善送出" if zh else "Review Data & Submit Improvement"
        )
        self.retraining_workspace_btn.setToolTip(
            tr(self._language, "open_retraining_workspace_hint")
        )
        self.show_detection_boxes_chk.setText(tr(self._language, "show_detection_boxes"))
        self.show_original_tab_chk.setText(tr(self._language, "show_original_tab"))
        self.show_processed_tab_chk.setText(tr(self._language, "show_processed_tab"))
        self.output_path_label.setText(
            f"{tr(self._language, 'output')}: {self._output_path}"
        )
        self.output_path_label.setToolTip(
            self._output_path
            if self._output_path != "--"
            else tr(self._language, "output_path_hint")
        )
        self.show_detection_boxes_chk.setToolTip(
            tr(self._language, "show_detection_boxes_hint")
        )
        self.show_original_tab_chk.setToolTip(
            tr(self._language, "show_original_tab_hint")
        )
        self.show_processed_tab_chk.setToolTip(
            tr(self._language, "show_processed_tab_hint")
        )
        self.calib_group.setTitle(tr(self._language, "auto_trigger_calib"))
        self._calib_hint_label.setText(tr(self._language, "calib_hint"))
        self._refresh_calibration_target_label()
        self._calib_empty_btn.setText(tr(self._language, "sample_empty"))
        self._calib_product_btn.setText(tr(self._language, "sample_product"))
        self._calib_apply_btn.setText(tr(self._language, "apply_threshold"))
        if self._calib_empty_area is None:
            self._calib_empty_val.setText(tr(self._language, "empty_area_label"))
        else:
            self._calib_empty_val.setText(
                f"{tr(self._language, 'empty_area_prefix')} {self._calib_empty_area:,.0f}"
            )
        if self._calib_product_area is None:
            self._calib_product_val.setText(tr(self._language, "product_area_label"))
        else:
            self._calib_product_val.setText(
                f"{tr(self._language, 'product_area_prefix')} {self._calib_product_area:,.0f}"
            )
        if (
            self._calib_target is None
            or self._calib_target != self._selected_target()
            or self._calib_empty_area is None
            or self._calib_product_area is None
        ):
            self._calib_threshold_val.setText(tr(self._language, "threshold_label"))
        else:
            threshold = int((self._calib_empty_area + self._calib_product_area) / 2)
            self._calib_threshold_val.setText(
                f"{tr(self._language, 'threshold_prefix')} {threshold:,}"
            )
        self._refresh_calib_threshold()
        self.sec_group.setTitle(tr(self._language, "security_group"))
        self._change_pin_btn.setText(tr(self._language, "change_pin"))
        self._lock_btn.setText(tr(self._language, "lock_engineer"))

        self.engineering_toggle_btn.setText(
            tr(self._language, "engineer_settings_closed")
        )
        self.engineering_toggle_btn.setToolTip(
            tr(self._language, "engineer_settings_hint")
        )
        self.inspection_history_btn.setText(
            tr(self._language, "inspection_history_open")
        )
        self.inspection_history_btn.setToolTip(
            tr(self._language, "inspection_history_open_hint")
        )

        # Sync language combo
        current_code = self.language_combo.currentData()
        if current_code != self._language:
            index = self.language_combo.findData(self._language)
            if index >= 0:
                self.language_combo.blockSignals(True)
                self.language_combo.setCurrentIndex(index)
                self.language_combo.blockSignals(False)

    def _on_language_selected(self) -> None:
        code = normalize_language(self.language_combo.currentData())
        self.set_language(code)
        self.language_changed.emit(code)
