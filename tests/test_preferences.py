from __future__ import annotations

import pytest

from app.gui.preferences import PreferencesManager


class FakeSettings:
    """Minimal in-memory QSettings replacement for preference unit tests."""

    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})

    def value(self, key: str, default: object = None) -> object:
        """Return a stored setting or its default.

        Args:
            key: Setting key.
            default: Value returned when the key is absent.

        Returns:
            The stored or default value.
        """
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:  # noqa: N802 - Qt API
        """Store a setting using the QSettings-compatible method name.

        Args:
            key: Setting key.
            value: Value to persist.
        """
        self.values[key] = value


def test_missing_language_defaults_to_chinese():
    preferences = PreferencesManager(FakeSettings())  # type: ignore[arg-type]

    assert preferences.restore_language() == "zh"


@pytest.mark.parametrize("language", ["zh", "en"])
def test_existing_supported_language_is_preserved(language: str):
    settings = FakeSettings({"language": language})
    preferences = PreferencesManager(settings)  # type: ignore[arg-type]

    assert preferences.restore_language() == language


@pytest.mark.parametrize("language", [None, "", "unsupported"])
def test_invalid_language_falls_back_to_chinese(language: object):
    settings = FakeSettings({"language": language})
    preferences = PreferencesManager(settings)  # type: ignore[arg-type]

    assert preferences.restore_language() == "zh"


def test_save_language_preserves_explicit_english_selection():
    settings = FakeSettings()
    preferences = PreferencesManager(settings)  # type: ignore[arg-type]

    preferences.save_language("en")

    assert settings.values["language"] == "en"


def test_save_invalid_language_uses_chinese_default():
    settings = FakeSettings()
    preferences = PreferencesManager(settings)  # type: ignore[arg-type]

    preferences.save_language("unsupported")

    assert settings.values["language"] == "zh"
