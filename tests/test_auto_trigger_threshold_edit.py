"""Calibration rewrites only the threshold value in auto_trigger.yaml."""

import yaml

from core.auto_trigger import with_product_area_threshold

DOCUMENTED = """\
# Auto-trigger tuning for Cable1/A.
# DO NOT DROP: the empty background already measures ~7900.
product_area_threshold: 46712

# 3 -> 2: presence has a large calibrated margin.
appear_frames: 2
stable_frames: 3
"""


def test_replaces_the_value_and_keeps_every_comment():
    updated = with_product_area_threshold(DOCUMENTED, 76944)

    assert updated == DOCUMENTED.replace("46712", "76944")
    assert yaml.safe_load(updated) == {
        "product_area_threshold": 76944,
        "appear_frames": 2,
        "stable_frames": 3,
    }


def test_keeps_an_inline_comment_after_the_value():
    updated = with_product_area_threshold(
        "product_area_threshold: 5000  # station default\n", 120
    )

    assert updated == "product_area_threshold: 120  # station default\n"


def test_appends_the_key_when_the_file_lacks_it():
    assert with_product_area_threshold("# notes\nappear_frames: 2", 300) == (
        "# notes\nappear_frames: 2\nproduct_area_threshold: 300\n"
    )
    assert with_product_area_threshold("", 300) == "product_area_threshold: 300\n"


def test_leaves_a_nested_key_of_the_same_name_alone():
    text = "legacy:\n  product_area_threshold: 1\n"

    updated = with_product_area_threshold(text, 300)

    assert yaml.safe_load(updated) == {
        "legacy": {"product_area_threshold": 1},
        "product_area_threshold": 300,
    }
