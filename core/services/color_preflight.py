"""Pre-shift color verification against a known reference board.

The station already has a start-of-shift ritual: put the golden sample in the
fixture and run the illumination calibration until mean luma is back inside its
recorded tolerance. That closes the loop on brightness, which is the cheapest
axis to control and the one that drifts most -- but the color check itself runs
against statistics measured on another day, and nothing measures *color* before
production starts. ``docs/operations/MISJUDGE_TRIAGE_SOP.md`` says so outright:
brightness calibration does not correct a color cast, and a suspected cast has
to be chased through the light hardware. Today the first evidence of a cast is a
misjudged board.

This module is the second step of that ritual. It reads the golden sample
through the production color path and answers two questions the line cannot
answer for itself:

  * did every expected color read as itself, and
  * how much threshold headroom is left compared with the day the reference was
    recorded.

The second question is the point. A check that only reports "still passing"
says nothing about how close the station is to the cliff, and black's headroom
on real crops has been as thin as a single percent.

**It never adjusts anything.** A baseline derived from one board at shift start
carries no provenance -- no evidence set, no holdout, no named approval -- and
would be exactly what the deployed-baseline contract exists to refuse. When
this check fails the remedies are the illumination calibration, the light
hardware, or an escalation to a rebuild with sign-off. Never a quiet write.

Concurrency: pure and single-threaded by contract. Nothing here captures
frames, touches the camera, or writes a file; the caller supplies already
measured items.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from core.models import ColorCheckItemResult

#: Fraction of the recorded margin a color must retain to pass without comment.
#: Margins differ by an order of magnitude between colors -- red sits far above
#: its threshold while black can sit a percent above its own -- so a single
#: absolute tolerance would let black collapse to zero unreported while flagging
#: ordinary variation in red. A retention ratio scales with each color, and is
#: the same shape as the rebuilder's chroma-retention floor.
DEFAULT_MINIMUM_MARGIN_RETENTION = 0.6

#: A reference margin at or below this is too thin to compute a ratio against:
#: ordinary noise swings the ratio wildly. Such a color is reported as having no
#: usable reference rather than passing or failing on a meaningless number.
MINIMUM_USABLE_REFERENCE_MARGIN = 0.02

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_NG = "NG"

STATE_OK = "OK"
#: Read as itself, but the headroom recorded for it has largely gone.
STATE_MARGIN_LOW = "MARGIN_LOW"
#: Read as itself and did not clear its own threshold -- the line fails this
#: board now, not later.
STATE_BELOW_THRESHOLD = "BELOW_THRESHOLD"
#: Read as itself, but there is nothing recorded to compare the headroom with.
STATE_NO_REFERENCE = "NO_REFERENCE"
#: Read as itself, but the reference was measured against a different baseline
#: file, so the two numbers are not comparable.
STATE_REFERENCE_STALE = "REFERENCE_STALE"
#: Expected on this board and not read at all.
STATE_MISSING = "MISSING"
#: Read more often than the board carries it, or not expected at all.
STATE_UNEXPECTED = "UNEXPECTED"
#: A detection produced no measurable pixels, so this position says nothing.
STATE_UNMEASURED = "UNMEASURED"

_DEGRADED_STATES = frozenset(
    {STATE_MARGIN_LOW, STATE_NO_REFERENCE, STATE_REFERENCE_STALE}
)
_FAILED_STATES = frozenset(
    {
        STATE_BELOW_THRESHOLD,
        STATE_MISSING,
        STATE_UNEXPECTED,
        STATE_UNMEASURED,
    }
)


@dataclass(frozen=True)
class ColorPreflightColor:
    """One color's standing on the reference board."""

    color: str
    expected_count: int
    observed_count: int
    #: Worst headroom among this color's measurements: ``threshold - diff``,
    #: which is the measured score minus the threshold it had to clear. ``None``
    #: when the color was not measured at all.
    margin: float | None
    reference_margin: float | None
    #: ``margin / reference_margin``, or ``None`` when either is unavailable.
    retention: float | None
    state: str

    @property
    def is_ok(self) -> bool:
        return self.state == STATE_OK


@dataclass(frozen=True)
class ColorPreflightReport:
    """What the reference board says about the color system right now."""

    status: str
    colors: tuple[ColorPreflightColor, ...]
    #: Empty when the deployed baseline is the approved current one; otherwise
    #: the reason it is not, carried here because a preflight that passed
    #: against statistics measured elsewhere has proven nothing.
    baseline_provenance_failure: str
    minimum_margin_retention: float
    #: True when a reference has never been recorded for this station, so the
    #: run can only report what it measured.
    reference_is_missing: bool
    #: True when a reference exists but was measured against a different
    #: baseline file. Its numbers are not comparable with today's, so it has to
    #: be recorded again -- which is what binds a reference to its baseline
    #: instead of letting it quietly outlive one.
    reference_is_stale: bool = False

    @property
    def is_ok(self) -> bool:
        return self.status == STATUS_OK

    def colors_in_state(self, state: str) -> tuple[ColorPreflightColor, ...]:
        return tuple(item for item in self.colors if item.state == state)

    def measured_margins(self) -> dict[str, float]:
        """Margins suitable for recording as a new reference.

        Only colors that read as themselves contribute. Recording a margin for
        a color that misread would enshrine the fault as the target.
        """
        return {
            item.color: item.margin
            for item in self.colors
            if item.margin is not None
            and item.state
            in {
                STATE_OK,
                STATE_MARGIN_LOW,
                STATE_NO_REFERENCE,
                STATE_REFERENCE_STALE,
            }
        }


#: Why a reading needs attention, in the order an operator should act. Stable
#: identifiers rather than sentences: the dialog renders them through the
#: translation table and the CLI prints its own text, but which causes apply is
#: one decision made here. Two front ends deciding it separately is how one of
#: them ends up telling an operator to check the lighting when the lighting is
#: fine.
REMEDY_MISREAD = "misread"
REMEDY_MARGIN = "margin"
REMEDY_STALE_REFERENCE = "stale_reference"
REMEDY_NO_REFERENCE = "no_reference"
REMEDY_BASELINE = "baseline"


def remedy_causes(report: "ColorPreflightReport") -> tuple[str, ...]:
    """Return what to do about this reading, most actionable first.

    The baseline cause is appended rather than competing for a single slot:
    when there is also no reference yet, the actionable step is to record one,
    and an operator shown only a baseline warning would reasonably wonder
    whether recording was safe. It is -- the reference binds to that baseline.
    """
    states = {item.state for item in report.colors}
    causes: list[str] = []
    if states & {
        STATE_MISSING,
        STATE_UNEXPECTED,
        STATE_UNMEASURED,
        STATE_BELOW_THRESHOLD,
    }:
        causes.append(REMEDY_MISREAD)
    elif STATE_MARGIN_LOW in states:
        causes.append(REMEDY_MARGIN)
    elif report.reference_is_stale:
        causes.append(REMEDY_STALE_REFERENCE)
    elif report.reference_is_missing:
        causes.append(REMEDY_NO_REFERENCE)
    if report.baseline_provenance_failure:
        causes.append(REMEDY_BASELINE)
    return tuple(causes) or (REMEDY_MISREAD,)


def item_margin(item: ColorCheckItemResult | Mapping[str, object]) -> float:
    """Threshold headroom for one measured item.

    ``ColorCheckItemResult`` reports both figures inverted -- ``diff`` is
    ``1 - score`` and ``threshold`` is ``1 - threshold`` -- so the difference
    between them is the score's own distance above the threshold it had to
    clear, and needs no knowledge of the inversion.
    """
    threshold, diff = _item_field(item, "threshold"), _item_field(item, "diff")
    return float(threshold) - float(diff)


def _item_field(
    item: ColorCheckItemResult | Mapping[str, object], name: str
) -> object:
    """Read one field from a live result or a persisted one.

    A saved inspection carries the same numbers as a dict, so accepting both
    lets this run over `Result/` records -- which is how a station sees whether
    its headroom has been eroding for a fortnight rather than only today.
    """
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _normalize(value: object) -> str:
    return str(value or "").strip()


def _counted(names: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name in names:
        key = name.casefold()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _reference_lookup(
    reference: Mapping[str, object] | None,
) -> dict[str, float]:
    if not isinstance(reference, Mapping):
        return {}
    resolved: dict[str, float] = {}
    for name, value in reference.items():
        key = _normalize(name).casefold()
        if not key:
            continue
        try:
            margin = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        resolved[key] = margin
    return resolved


def _state_for_measured(
    margin: float,
    reference_margin: float | None,
    retention_floor: float,
    missing_reference_state: str = STATE_NO_REFERENCE,
) -> tuple[str, float | None]:
    """Classify a color that read as itself, and its retention ratio."""
    if margin <= 0.0:
        # The name is right and the score did not clear the bar: this board
        # fails the color check as it stands, whatever the reference says.
        return STATE_BELOW_THRESHOLD, None
    if (
        reference_margin is None
        or reference_margin <= MINIMUM_USABLE_REFERENCE_MARGIN
    ):
        return missing_reference_state, None
    retention = margin / reference_margin
    if retention < retention_floor:
        return STATE_MARGIN_LOW, retention
    return STATE_OK, retention


def evaluate_color_preflight(
    items: Sequence[ColorCheckItemResult | Mapping[str, object]],
    expected_colors: Sequence[str],
    reference_margins: Mapping[str, object] | None = None,
    *,
    minimum_margin_retention: float = DEFAULT_MINIMUM_MARGIN_RETENTION,
    baseline_provenance_failure: str = "",
    reference_baseline_id: str = "",
    current_baseline_id: str = "",
) -> ColorPreflightReport:
    """Judge one reference-board reading against what the station recorded.

    ``expected_colors`` is the station's declared board -- repeated positions
    included, so a board carrying two black wires expects black twice. It comes
    from the station config rather than from the detector's own answers: taking
    the expected color from the detector would make this check agree with the
    detector by construction, which is the circularity the color verifier was
    fixed to avoid.
    """
    retention_floor = float(minimum_margin_retention)
    reference = _reference_lookup(reference_margins)
    # A reference is a self-comparison: "on a good day, against *this*
    # baseline, black had this much room". Change the baseline and the two
    # numbers stop being comparable, so the reference is set aside rather than
    # quietly outliving the file it was measured on. Binding it this way is
    # also what lets a station use this check today, while its baseline still
    # predates the current contract -- refusing to run until the migration is
    # done would have made the check useless exactly where drift goes unseen.
    reference_is_stale = bool(
        reference
        and reference_baseline_id
        and current_baseline_id
        and reference_baseline_id != current_baseline_id
    )
    if reference_is_stale:
        reference = {}

    expected_counts = _counted(
        name for color in expected_colors if (name := _normalize(color))
    )
    display_names: dict[str, str] = {}
    for color in expected_colors:
        name = _normalize(color)
        if name:
            display_names.setdefault(name.casefold(), name)

    # Group measurements by the color each one actually read. An item that
    # produced no color at all is held aside: it is not evidence about any
    # color, and folding its margin in would report a collapse that never
    # happened.
    margins_by_color: dict[str, list[float]] = {}
    unmeasured = 0
    for item in items:
        observed = _normalize(_item_field(item, "best_color"))
        if not observed:
            unmeasured += 1
            continue
        key = observed.casefold()
        display_names.setdefault(key, observed)
        margins_by_color.setdefault(key, []).append(item_margin(item))

    colors: list[ColorPreflightColor] = []
    for key in sorted(set(expected_counts) | set(margins_by_color)):
        expected_count = expected_counts.get(key, 0)
        observed = margins_by_color.get(key, [])
        observed_count = len(observed)
        reference_margin = reference.get(key)
        # The worst measurement decides: a board with one good black and one
        # marginal black is a station with a marginal black.
        margin = min(observed) if observed else None

        if observed_count < expected_count:
            state = STATE_MISSING
            retention = None
        elif observed_count > expected_count:
            state = STATE_UNEXPECTED
            retention = None
        elif margin is None:
            # Expected zero and observed zero: nothing to say about this color.
            continue
        else:
            state, retention = _state_for_measured(
                margin,
                reference_margin,
                retention_floor,
                STATE_REFERENCE_STALE
                if reference_is_stale
                else STATE_NO_REFERENCE,
            )

        colors.append(
            ColorPreflightColor(
                color=display_names.get(key, key),
                expected_count=expected_count,
                observed_count=observed_count,
                margin=margin,
                reference_margin=reference_margin,
                retention=retention,
                state=state,
            )
        )

    if unmeasured:
        colors.append(
            ColorPreflightColor(
                color="",
                expected_count=0,
                observed_count=unmeasured,
                margin=None,
                reference_margin=None,
                retention=None,
                state=STATE_UNMEASURED,
            )
        )

    states = {item.state for item in colors}
    provenance = baseline_provenance_failure.strip()
    if states & _FAILED_STATES:
        # NG means this board read wrongly, which is a thing the shift can act
        # on now.
        status = STATUS_NG
    elif states & _DEGRADED_STATES or provenance:
        # A baseline that cannot be shown to be current is a standing station
        # condition, not this board's fault, and it stays true every shift
        # until the migration completes. Reporting it as NG daily would teach
        # operators that NG means nothing; the enforcement that should stop a
        # line for it is `color_baseline_algorithm_enforcement: strict`, which
        # refuses to load the baseline at all.
        status = STATUS_WARN
    else:
        status = STATUS_OK

    return ColorPreflightReport(
        status=status,
        colors=tuple(colors),
        baseline_provenance_failure=provenance,
        minimum_margin_retention=retention_floor,
        reference_is_missing=not reference,
        reference_is_stale=reference_is_stale,
    )
