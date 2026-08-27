"""``tools/color_verifier.py`` must report the verdict the line reaches.

The tool is a standalone CLI, so nothing forces it to agree with
``core/stats_color_checker.py``. It used to carry its own scoring and the two
had drifted into different policies -- the tool matched every color against
the recorded ``hsv_min``/``hsv_max`` envelope, the runtime uses hand-tuned
open-ended gates -- and they agreed on 8 of 15 conformance cases. A plain
bright red came back ``Unknown``; a red region catching an orange edge came
back ``Orange``. For a tool named "verifier" that is a trap.

It now calls the runtime for its verdict. These tests hold it there, over the
same cases the shared cross-repository fixture uses, so the scoring cannot
quietly grow back.

The envelope check survives as its own reported signal, and is checked here
too: it is a real question ("does this sit inside the baseline's recorded
range?") that simply must not stand in for the other one.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.stats_color_checker import StatsColorChecker
from tools import color_verifier

FIXTURE = Path(__file__).parent / "fixtures" / "color_conformance.json"
_PAYLOAD = json.loads(FIXTURE.read_text(encoding="utf-8"))
_ALL_CASES = _PAYLOAD["cases"] + _PAYLOAD["known_divergences"]


@pytest.fixture(scope="module")
def color_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("verifier") / "color_stats.json"
    path.write_text(json.dumps(_PAYLOAD["color_model"]), encoding="utf-8")
    return path


def _render(spec: dict) -> np.ndarray:
    size = int(spec.get("size", 96))
    hsv = np.zeros((size, size, 3), np.uint8)
    bands = spec["bands"]
    edges = np.linspace(0, size, len(bands) + 1).astype(int)
    for index, band in enumerate(bands):
        low, high = edges[index], edges[index + 1]
        hsv[low:high, :, 0] = band[0]
        hsv[low:high, :, 1] = band[1]
        hsv[low:high, :, 2] = band[2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "-")


@pytest.fixture(scope="module")
def tool_results(color_model: Path, tmp_path_factory: pytest.TempPathFactory) -> dict:
    image_dir = tmp_path_factory.mktemp("verifier-images")
    for case in _ALL_CASES:
        cv2.imwrite(str(image_dir / f"{_slug(case['name'])}.png"), _render(case["spec"]))
    _summary, decisions = color_verifier.verify_directory(
        image_dir, color_model, infer_expected_from_name=False
    )
    return {Path(str(item.image)).stem: item for item in decisions}


@pytest.mark.parametrize(
    "case", _ALL_CASES, ids=[case["name"] for case in _ALL_CASES]
)
def test_tool_reports_the_runtime_verdict(
    case: dict, color_model: Path, tool_results: dict
) -> None:
    runtime = StatsColorChecker.from_json(color_model).check(_render(case["spec"]))
    decision = tool_results[_slug(case["name"])]

    has_evidence = bool(runtime.metrics.get("has_evidence", True))
    expected_color = runtime.best_color if has_evidence else "Unknown"

    assert decision.predicted_color == expected_color, (
        f"{case['name']}: the tool no longer reports the runtime's verdict"
    )
    accepted = decision.status != "low_confidence"
    assert accepted is bool(runtime.is_ok), (
        f"{case['name']}: the tool and the runtime disagree on acceptance"
    )


def test_a_region_outside_the_recorded_envelope_is_still_reported_as_such(
    tool_results: dict,
) -> None:
    """The envelope check is kept, beside the verdict rather than instead of it.

    A bright red whose V sits past the baseline's recorded 99th percentile is
    genuinely outside the envelope. That is worth saying -- it used to be said
    by answering ``Unknown``, which is not.
    """
    decision = tool_results[_slug("bright red just past the seam")]

    assert decision.predicted_color.casefold() == "red"
    assert decision.debug_info["envelope_match"] == pytest.approx(0.0)


def test_a_clean_region_matches_both_the_verdict_and_the_envelope(
    tool_results: dict,
) -> None:
    decision = tool_results[_slug("pure green")]

    assert decision.predicted_color.casefold() == "green"
    assert decision.debug_info["envelope_match"] > 0.5
