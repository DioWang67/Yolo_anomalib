from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import core.services.color_override_loader as color_override_loader
from core.services.color_override_loader import ColorOverrideLoader


def test_color_override_loader_returns_global_fallback_when_file_is_missing(tmp_path):
    config = SimpleNamespace(
        color_threshold_overrides={"global": 0.8},
        color_rules_overrides={"global": {"mode": "rgb"}},
    )
    loader = ColorOverrideLoader(tmp_path)

    overrides, rules, tuning = loader.load(config, "LED", "A", "yolo", MagicMock())

    assert overrides == {"global": 0.8}
    assert rules == {"global": {"mode": "rgb"}}


def test_color_override_loader_reads_model_config_overrides(tmp_path):
    cfg_dir = tmp_path / "LED" / "A" / "yolo"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        """
color_threshold_overrides:
  red: 0.91
color_rules_overrides:
  red:
    min_area: 3
""",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        color_threshold_overrides={"global": 0.8},
        color_rules_overrides={"global": {"mode": "rgb"}},
    )
    loader = ColorOverrideLoader(tmp_path)

    overrides, rules, tuning = loader.load(config, "LED", "A", "yolo", MagicMock())

    assert overrides == {"red": 0.91}
    assert rules == {"red": {"min_area": 3}}


def test_color_override_loader_uses_fallback_for_non_mapping_config(tmp_path):
    cfg_dir = tmp_path / "LED" / "A" / "yolo"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text("- not-a-mapping\n", encoding="utf-8")
    logger = MagicMock()
    config = SimpleNamespace(
        color_threshold_overrides={"global": 0.8},
        color_rules_overrides=None,
    )
    loader = ColorOverrideLoader(tmp_path)

    overrides, rules, tuning = loader.load(config, "LED", "A", "yolo", logger)

    assert overrides == {"global": 0.8}
    assert rules is None
    logger.warning.assert_called_once()


def test_color_override_loader_uses_fallback_when_pyyaml_is_missing(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "LED" / "A" / "yolo"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        "color_threshold_overrides:\n  red: 0.91\n",
        encoding="utf-8",
    )
    logger = MagicMock()
    config = SimpleNamespace(
        color_threshold_overrides={"global": 0.8},
        color_rules_overrides={"global": {"mode": "rgb"}},
    )
    loader = ColorOverrideLoader(tmp_path)
    monkeypatch.setattr(color_override_loader, "_import_yaml", lambda: None)

    overrides, rules, tuning = loader.load(config, "LED", "A", "yolo", logger)

    assert overrides == {"global": 0.8}
    assert rules == {"global": {"mode": "rgb"}}
    logger.warning.assert_called_once()

def test_color_override_loader_reads_decision_tuning(tmp_path):
    cfg_dir = tmp_path / "Cable1" / "A" / "yolo"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        """
color_decision_tuning:
  yellow_h_min: 18
  black_s_threshold: 55
""",
        encoding="utf-8",
    )
    config = SimpleNamespace(
        color_threshold_overrides=None,
        color_rules_overrides=None,
        color_decision_tuning={"yellow_h_min": 25},
    )
    loader = ColorOverrideLoader(tmp_path)

    overrides, rules, tuning = loader.load(config, "Cable1", "A", "yolo", MagicMock())

    assert tuning == {"yellow_h_min": 18, "black_s_threshold": 55}

    # Cached path returns the same values
    overrides, rules, tuning = loader.load(config, "Cable1", "A", "yolo", MagicMock())
    assert tuning == {"yellow_h_min": 18, "black_s_threshold": 55}


def test_color_override_loader_falls_back_to_global_decision_tuning(tmp_path):
    config = SimpleNamespace(
        color_threshold_overrides=None,
        color_rules_overrides=None,
        color_decision_tuning={"yellow_h_min": 25},
    )
    loader = ColorOverrideLoader(tmp_path)

    overrides, rules, tuning = loader.load(config, "LED", "A", "yolo", MagicMock())

    assert tuning == {"yellow_h_min": 25}
