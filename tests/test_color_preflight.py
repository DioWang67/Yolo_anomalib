from __future__ import annotations

import pytest

from core.models import ColorCheckItemResult
from core.services.color_preflight import (
    STATE_BELOW_THRESHOLD,
    STATE_MARGIN_LOW,
    STATE_MISSING,
    STATE_NO_REFERENCE,
    STATE_OK,
    STATE_REFERENCE_STALE,
    STATE_UNEXPECTED,
    STATE_UNMEASURED,
    STATUS_NG,
    STATUS_OK,
    STATUS_WARN,
    evaluate_color_preflight,
    item_margin,
)

CABLE1_A = ("Red", "Green", "Orange", "Yellow", "Black", "Black")


def _item(best_color: str, *, score: float, threshold: float) -> ColorCheckItemResult:
    """Build an item the way the checker reports one.

    ``StatsColorChecker`` publishes both figures inverted -- ``diff`` is
    ``1 - score`` and ``threshold`` is ``1 - threshold`` -- so the fixture has
    to invert them too, or the tests would agree with a sign error.
    """
    return ColorCheckItemResult(
        index=0,
        class_name=best_color,
        bbox=[0, 0, 10, 10],
        best_color=best_color,
        diff=max(0.0, 1.0 - score),
        threshold=max(0.0, 1.0 - threshold),
        is_ok=score >= threshold,
        measurement_is_ok=score >= threshold,
    )


def _board(**margins: float) -> list[ColorCheckItemResult]:
    """One item per expected position, each at its colour's given margin.

    Both black positions get the same margin; the test that needs them to
    differ builds its items directly.
    """
    return [
        _item(color, score=0.5 + margins[color.casefold()], threshold=0.5)
        for color in CABLE1_A
    ]


def _reference(**margins: float) -> dict[str, float]:
    return {name.title(): value for name, value in margins.items()}


def test_margin_is_the_score_distance_above_its_threshold() -> None:
    """The whole feature reads backwards if this sign is wrong.

    Both reported figures are inverted, and subtracting one from the other
    cancels the inversion -- so a margin can be taken off a live result or a
    persisted one without knowing the convention.
    """
    item = _item("Black", score=0.478, threshold=0.450)

    assert item_margin(item) == pytest.approx(0.028)
    # A saved inspection carries the same numbers under the same keys.
    assert item_margin(item.to_dict()) == pytest.approx(0.028)


def test_a_board_holding_its_recorded_headroom_passes() -> None:
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.13),
    )

    assert report.status == STATUS_OK
    assert {item.state for item in report.colors} == {STATE_OK}


def test_an_eroded_margin_warns_before_anything_misjudges() -> None:
    """Black still reads black, and has 22% of the headroom it was given.

    Nothing is failing yet -- that is the point. An OK/NG check on this board
    reports OK and the station learns nothing until it starts rejecting good
    product.
    """
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.028),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    assert report.status == STATUS_WARN
    black = next(item for item in report.colors if item.color == "Black")
    assert black.state == STATE_MARGIN_LOW
    assert black.retention == pytest.approx(0.028 / 0.126, rel=1e-3)


def test_a_relative_criterion_is_what_makes_that_visible() -> None:
    """An absolute tolerance cannot serve colours an order of magnitude apart.

    Red loses 0.09 of its 0.57 and is fine; black loses 0.10 of its 0.126 and
    is nearly gone. Any single absolute tolerance either misses the second or
    fires on the first.
    """
    report = evaluate_color_preflight(
        _board(red=0.48, green=0.47, orange=0.43, yellow=0.39, black=0.026),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    states = {item.color: item.state for item in report.colors}
    assert states["Red"] == STATE_OK
    assert states["Black"] == STATE_MARGIN_LOW


def test_a_better_board_than_the_reference_is_not_a_finding() -> None:
    """Only degradation matters; a cleaner board than the reference is good news."""
    report = evaluate_color_preflight(
        _board(red=0.70, green=0.60, orange=0.55, yellow=0.50, black=0.30),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    assert report.status == STATUS_OK


def test_a_colour_that_no_longer_clears_its_threshold_is_ng_not_a_warning() -> None:
    """It read the right name and failed the bar: the line rejects this board now."""
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=-0.01),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    assert report.status == STATUS_NG
    black = next(item for item in report.colors if item.color == "Black")
    assert black.state == STATE_BELOW_THRESHOLD


def test_a_misread_shows_as_a_missing_colour_and_an_unexpected_one() -> None:
    """The expected board comes from config, so a swap is visible from both sides."""
    items = _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12)
    # The position that should read Orange reads Red instead.
    items[2] = _item("Red", score=0.9, threshold=0.5)

    report = evaluate_color_preflight(
        items,
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    assert report.status == STATUS_NG
    states = {item.color: item.state for item in report.colors}
    assert states["Orange"] == STATE_MISSING
    assert states["Red"] == STATE_UNEXPECTED


def test_repeated_positions_are_judged_by_their_worst_measurement() -> None:
    """A board with one good black and one marginal black is a marginal station."""
    items = [
        _item("Red", score=1.06, threshold=0.5),
        _item("Green", score=0.97, threshold=0.5),
        _item("Orange", score=0.93, threshold=0.5),
        _item("Yellow", score=0.89, threshold=0.5),
        _item("Black", score=0.62, threshold=0.5),
        _item("Black", score=0.53, threshold=0.5),
    ]

    report = evaluate_color_preflight(
        items,
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    black = next(item for item in report.colors if item.color == "Black")
    assert black.expected_count == 2
    assert black.observed_count == 2
    assert black.margin == pytest.approx(0.03)
    assert black.state == STATE_MARGIN_LOW


def test_an_unmeasurable_detection_is_not_reported_as_a_collapsed_margin() -> None:
    """A degenerate box says nothing about colour, and its numbers look terrible.

    The checker reports an unmeasurable ROI as ``diff`` 1.0 against threshold
    0.0, which is a margin of -1.0. Folding that into a colour would report a
    collapse that never happened.
    """
    items = _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12)
    items.append(
        ColorCheckItemResult(
            index=6,
            class_name="Black",
            bbox=[0, 0, 0, 0],
            best_color="",
            diff=1.0,
            threshold=0.0,
            is_ok=False,
            measurement_is_ok=False,
        )
    )

    report = evaluate_color_preflight(
        items,
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
    )

    assert report.status == STATUS_NG
    unmeasured = report.colors_in_state(STATE_UNMEASURED)
    assert len(unmeasured) == 1
    assert unmeasured[0].observed_count == 1
    black = next(item for item in report.colors if item.color == "Black")
    assert black.state == STATE_OK


def test_without_a_reference_the_run_reports_rather_than_passes() -> None:
    """A first run can only say what it measured, and must say so."""
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12),
        CABLE1_A,
        None,
    )

    assert report.reference_is_missing is True
    assert report.status == STATUS_WARN
    assert {item.state for item in report.colors} == {STATE_NO_REFERENCE}


def test_a_reference_margin_too_thin_to_divide_by_is_not_a_verdict() -> None:
    """A ratio against ~0.01 swings on noise, so it is not used as evidence."""
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.008),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.01),
    )

    black = next(item for item in report.colors if item.color == "Black")
    assert black.state == STATE_NO_REFERENCE
    assert black.retention is None


def test_a_baseline_that_is_not_the_approved_one_warns_without_failing() -> None:
    """Reported, but not as NG, and not as a reason to refuse the check.

    It is a standing station condition that stays true every shift until the
    baseline migration finishes, so reporting it as NG daily would teach
    operators that NG means nothing. NG is reserved for a board that read
    wrongly, which is something the shift can act on. The enforcement that
    should stop a line over an unverifiable baseline is
    ``color_baseline_algorithm_enforcement: strict``, which refuses to load it.
    """
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
        baseline_provenance_failure="顏色 ROI policy 不一致",
    )

    assert report.status == STATUS_WARN
    assert "ROI policy" in report.baseline_provenance_failure


def test_a_reference_from_another_baseline_file_is_set_aside() -> None:
    """A margin is a distance above a threshold the baseline defines.

    Rebuild the baseline and the recorded margins stop being comparable, so the
    reference expires with the file it was measured on instead of quietly
    outliving it -- which is also what lets a station use this check while its
    baseline still predates the current contract.
    """
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
        reference_baseline_id="a" * 64,
        current_baseline_id="b" * 64,
    )

    assert report.reference_is_stale is True
    assert report.status == STATUS_WARN
    assert {item.state for item in report.colors} == {STATE_REFERENCE_STALE}
    # The reference is set aside, not used with a warning beside it.
    assert all(item.reference_margin is None for item in report.colors)


def test_the_same_baseline_file_keeps_the_reference_usable() -> None:
    report = evaluate_color_preflight(
        _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12),
        CABLE1_A,
        _reference(red=0.57, green=0.49, orange=0.45, yellow=0.40, black=0.126),
        reference_baseline_id="a" * 64,
        current_baseline_id="a" * 64,
    )

    assert report.reference_is_stale is False
    assert report.status == STATUS_OK


def test_recordable_margins_exclude_colours_that_did_not_read_themselves() -> None:
    """Recording a faulty board would enshrine the fault as the target."""
    items = _board(red=0.56, green=0.47, orange=0.43, yellow=0.39, black=0.12)
    items[2] = _item("Red", score=0.9, threshold=0.5)

    report = evaluate_color_preflight(items, CABLE1_A, None)

    recorded = report.measured_margins()
    assert "Orange" not in recorded
    assert "Red" not in recorded
    assert set(recorded) == {"Green", "Yellow", "Black"}
