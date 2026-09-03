from __future__ import annotations

from pathlib import Path

import pytest

from core.services.station_color_settings import (
    StationColorSettingsError,
    expected_color_names,
    expected_color_positions,
    station_color_decision_tuning,
    station_color_preflight,
    station_color_roi_policy,
    station_config_path,
    station_expected_color_names,
    station_expected_color_positions,
)


def test_expected_colors_collapse_repeated_positions() -> None:
    """A station lists a color once per position, not once per color.

    Cable1/A carries two black wires, so its ``expected_items`` names Black
    twice. Passing that through as a color list makes the rebuilder reject the
    request as a duplicate.
    """
    config = {
        "expected_items": {
            "Cable1": {"A": ["Red", "Green", "Black", "Black"]},
        }
    }

    assert expected_color_names(config, "Cable1", "A") == (
        "Red",
        "Green",
        "Black",
    )


def test_positions_keep_the_repeats_that_names_collapse() -> None:
    """Two accessors because two callers need opposite things.

    The rebuilder rejects a repeated colour request, so it wants the
    vocabulary; the pre-shift check counts what it read against what the board
    carries, so it wants the layout. Sharing one accessor made the second black
    wire read as a surplus.
    """
    config = {
        "expected_items": {
            "Cable1": {"A": ["Red", "Green", "Orange", "Yellow", "Black", "Black"]},
        }
    }

    assert expected_color_positions(config, "Cable1", "A") == (
        "Red",
        "Green",
        "Orange",
        "Yellow",
        "Black",
        "Black",
    )
    assert expected_color_names(config, "Cable1", "A") == (
        "Red",
        "Green",
        "Orange",
        "Yellow",
        "Black",
    )


def test_positions_also_drop_generic_detector_classes() -> None:
    config = {"expected_items": {"Cable1": {"A": ["LED", "Red", "Red"]}}}

    assert expected_color_positions(config, "Cable1", "A") == ("Red", "Red")


def test_expected_colors_drop_generic_detector_classes() -> None:
    """``LED`` names a part, and no baseline carries statistics for it."""
    config = {"expected_items": {"Cable1": {"A": ["LED", "Red"]}}}

    assert expected_color_names(config, "Cable1", "A") == ("Red",)


def test_declared_generic_classes_replace_the_default() -> None:
    config = {"expected_items": {"Cable1": {"A": ["Lamp", "LED", "Red"]}}}

    assert expected_color_names(
        config, "Cable1", "A", generic_classes=["Lamp"]
    ) == ("LED", "Red")


@pytest.mark.parametrize(
    "config",
    (
        {},
        {"expected_items": None},
        {"expected_items": {"Cable1": None}},
        {"expected_items": {"Cable1": {"A": None}}},
        # A bare string is not a one-item list of colors.
        {"expected_items": {"Cable1": {"A": "Red"}}},
        {"expected_items": {"Other": {"A": ["Red"]}}},
    ),
)
def test_a_scope_that_names_nothing_yields_nothing(config: dict) -> None:
    assert expected_color_names(config, "Cable1", "A") == ()


def test_settings_come_from_the_live_station_config(tmp_path: Path) -> None:
    models_root = tmp_path / "models"
    station = models_root / "Cable1" / "A" / "yolo"
    station.mkdir(parents=True)
    (station / "config.yaml").write_text(
        "color_roi_policy:\n"
        "  inset_x_ratio: 0.2\n"
        "color_decision_tuning:\n"
        "  center_margin_ratio: 0.25\n"
        "expected_items:\n"
        "  Cable1:\n"
        "    A:\n"
        "    - Red\n"
        "    - Red\n",
        encoding="utf-8",
    )

    config_path = station_config_path(models_root, "Cable1", "A", "yolo")

    assert config_path == station / "config.yaml"
    assert station_color_roi_policy(config_path).inset_x_ratio == pytest.approx(0.2)
    assert station_color_decision_tuning(
        config_path
    ).center_margin_ratio == pytest.approx(0.25)
    assert station_expected_color_names(config_path, "Cable1", "A") == ("Red",)
    assert station_expected_color_positions(config_path, "Cable1", "A") == (
        "Red",
        "Red",
    )
    # No pre-shift reference recorded yet, and the accessor says so rather than
    # inventing one.
    assert station_color_preflight(config_path).has_reference is False


def test_a_missing_station_config_is_an_error_not_a_default(tmp_path: Path) -> None:
    """Silently defaulting would build a baseline in a geometry nobody chose."""
    missing = station_config_path(tmp_path, "Cable1", "A", "yolo")

    with pytest.raises(StationColorSettingsError):
        station_color_roi_policy(missing)


def test_a_non_mapping_decision_tuning_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("color_decision_tuning: 0.15\n", encoding="utf-8")

    with pytest.raises(StationColorSettingsError, match="mapping"):
        station_color_decision_tuning(config_path)
