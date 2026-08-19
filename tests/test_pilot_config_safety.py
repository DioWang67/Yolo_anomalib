from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_cable1_duplicate_filter_stays_report_only_until_pilot_approval() -> None:
    config_path = ROOT / "models" / "Cable1" / "A" / "yolo" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    policy = config["steps"]["cross_class_duplicate_filter"]

    assert policy["enabled"] is True
    assert policy["mode"] == "report_only"
