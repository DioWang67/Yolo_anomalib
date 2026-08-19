"""The one-line fail-reason banner must describe the board, not the detector.

This banner was the fourth surface to report per-box color failures and the last
one to keep naming a box the duplicate filter had removed: the operator card,
the detail panel, and the annotated overlay had all stopped, so the same
inspection read differently depending on where you looked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PyQt5", reason="PyQt5 is required for GUI widgets")

from test_color_result_display import suppressed_duplicate_result  # noqa: E402

from app.gui.widgets import FailReasonLabel  # noqa: E402


def test_banner_omits_a_suppressed_duplicates_color_failure(qtbot):
    label = FailReasonLabel()
    qtbot.addWidget(label)
    label.set_language("zh")

    label.update_from_result(suppressed_duplicate_result())

    text = label.text()
    # The surviving Black still needs operator attention ...
    assert "Black 顏色信心不足" in text
    # ... but #5 was removed, so the operator cannot find it in the image.
    assert "Red → Orange" not in text
    # The board's actual defect stays.
    assert "排列順序錯誤" in text


def test_banner_still_reports_a_failure_that_was_only_proposed(qtbot):
    """Report-only leaves the box on the board, so the banner keeps it."""
    result = suppressed_duplicate_result()
    result.metadata["duplicate_filter"] = {
        "status": "reported",
        "suppressions": [],
        "proposed_suppressions": [{"suppressed_index": 5, "kept_index": 6}],
    }

    label = FailReasonLabel()
    qtbot.addWidget(label)
    label.set_language("zh")

    label.update_from_result(result)

    assert "Red → Orange" in label.text()
