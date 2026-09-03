from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config_validation import validate_model_cfg


def _baseline(tmp_path: Path, *colors: str) -> Path:
    path = tmp_path / "color_stats.json"
    path.write_text(
        json.dumps({"summary": {color: {"count": 30} for color in colors}}),
        encoding="utf-8",
    )
    return path


def _station_config(*expected_items: str) -> dict:
    return {
        "enable_color_check": True,
        "color_checker_type": "stats",
        "color_model_path": "color_stats.json",
        "expected_items": {"Cable1": {"A": list(expected_items)}},
    }


def test_missing_color_model_fails_when_baseline_enforcement_is_strict(
    tmp_path: Path,
) -> None:
    config = {
        "enable_color_check": True,
        "color_checker_type": "stats",
        "color_model_path": "missing.json",
        "color_baseline_algorithm_enforcement": "strict",
    }

    with pytest.raises(ValueError, match="strict baseline enforcement"):
        validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is True


def test_missing_color_model_keeps_legacy_warn_mode_behavior(tmp_path: Path) -> None:
    config = {
        "enable_color_check": True,
        "color_checker_type": "stats",
        "color_model_path": "missing.json",
        "color_baseline_algorithm_enforcement": "warn",
    }

    validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is False


def test_a_color_the_baseline_cannot_score_is_a_startup_failure(
    tmp_path: Path,
) -> None:
    """Adding a color the baseline lacks used to fail every board instead.

    The runtime folds an unscoreable candidate into every item's verdict, so
    the station kept running and rejected everything. Naming the color at
    startup is the difference between a five-minute fix and a lost shift.
    """
    _baseline(tmp_path, "Red", "Green", "Orange", "Yellow", "Black")
    config = _station_config("Red", "Green", "Orange", "Yellow", "Black", "Blue")

    with pytest.raises(ValueError, match="Blue") as excinfo:
        validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    # The message has to carry the fix, not just the complaint.
    assert "rebuild the baseline" in str(excinfo.value)


def test_a_misspelled_color_is_reported_rather_than_quietly_dropped(
    tmp_path: Path,
) -> None:
    _baseline(tmp_path, "Red", "Black")
    config = _station_config("Red", "Blakc")

    with pytest.raises(ValueError, match="Blakc"):
        validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)


def test_configured_colors_present_in_the_baseline_validate(tmp_path: Path) -> None:
    _baseline(tmp_path, "Red", "Green", "Orange", "Yellow", "Black")
    # Duplicates and casing differences are how real station configs are
    # written, and neither is a missing color.
    config = _station_config("red", "Green", "Orange", "Yellow", "Black", "Black")

    validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is True


def test_a_generic_detector_class_is_not_expected_in_the_baseline(
    tmp_path: Path,
) -> None:
    """``LED`` names a part, not a color, and the runtime excludes it too.

    Rejecting it here would take down a station that is configured correctly.
    """
    _baseline(tmp_path, "Red", "Black")
    config = _station_config("LED", "Red", "Black")

    validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is True


def test_a_declared_generic_class_is_honored(tmp_path: Path) -> None:
    _baseline(tmp_path, "Red")
    config = _station_config("Lamp", "Red")
    config["steps"] = {"color_check": {"generic_classes": ["Lamp"]}}

    validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is True


def test_an_unreadable_baseline_is_left_to_the_loader(tmp_path: Path) -> None:
    """A corrupt file must not be reported as every color being wrong."""
    (tmp_path / "color_stats.json").write_text("{ not json", encoding="utf-8")
    config = _station_config("Red", "Blue")

    validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is True


def test_the_legacy_color_qc_checker_is_not_held_to_the_stats_vocabulary(
    tmp_path: Path,
) -> None:
    """``color_qc`` artifacts keep their colors under a different key."""
    _baseline(tmp_path, "Red")
    config = _station_config("Red", "Blue")
    config["color_checker_type"] = "color_qc"

    validate_model_cfg(config, "Cable1", "A", model_cfg_dir=tmp_path)

    assert config["enable_color_check"] is True
