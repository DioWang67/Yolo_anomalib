from core.services.results.position_summary import (
    format_fixture_shift_hint,
    summarize_position_records,
)


def test_summarize_position_records_detects_likely_fixture_shift():
    summary = summarize_position_records(
        [
            {
                "class": "a",
                "position_status": "WRONG",
                "position_error": 10.0,
                "position_offset": {"dx": 9.0, "dy": 4.0},
            },
            {
                "class": "b",
                "position_status": "WRONG",
                "position_error": 11.0,
                "position_offset": {"dx": 8.5, "dy": 4.2},
            },
            {
                "class": "c",
                "position_status": "WRONG",
                "position_error": 10.5,
                "position_offset": {"dx": 8.8, "dy": 3.9},
            },
        ]
    )

    assert summary.fail_count == 3
    assert summary.correct_count == 0
    assert summary.likely_fixture_shift is True
    assert format_fixture_shift_hint(summary) is not None


def test_summarize_position_records_keeps_localized_failures_as_non_fixture_shift():
    summary = summarize_position_records(
        [
            {
                "label": "a",
                "position_status": "CORRECT",
                "position_offset": {"dx": 0.5, "dy": 0.2},
            },
            {
                "label": "b",
                "position_status": "WRONG",
                "position_error": 7.0,
                "position_offset": {"dx": 7.0, "dy": -1.0},
            },
            {
                "label": "c",
                "position_status": "CORRECT",
                "position_offset": {"dx": 0.3, "dy": 0.1},
            },
        ]
    )

    assert summary.correct_count == 2
    assert summary.fail_count == 1
    assert summary.likely_fixture_shift is False
    assert format_fixture_shift_hint(summary) is None
