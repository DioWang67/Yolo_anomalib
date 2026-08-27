"""``tools/color_verifier.py`` decides differently from the production runtime.

The tool is a standalone CLI, so nothing forces it to agree with
``core/stats_color_checker.py``. It does not: it matches every color against
the recorded ``hsv_min``/``hsv_max`` envelope and zeroes any color whose match
ratio falls below ``MIN_HSV_MATCH_RATIO``, while the runtime uses hand-tuned
open-ended gates for red, orange and green.

That is a defensible policy for a calibration tool and a dangerous one for
anything named "verifier": someone asking why the line called a part red gets
``Unknown`` back. These tests pin the difference on the same cases the shared
conformance fixture uses, so it stays a documented choice. If either side
moves, this fails and someone has to decide which behavior was intended
rather than discovering the gap from a confusing tool run.
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


@pytest.fixture(scope="module")
def color_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("divergence") / "color_stats.json"
    path.write_text(json.dumps(_PAYLOAD["color_model"]), encoding="utf-8")
    return path


def _render(bands: list[list[int]], size: int = 96) -> np.ndarray:
    hsv = np.zeros((size, size, 3), np.uint8)
    edges = np.linspace(0, size, len(bands) + 1).astype(int)
    for index, band in enumerate(bands):
        low, high = edges[index], edges[index + 1]
        hsv[low:high, :, 0] = band[0]
        hsv[low:high, :, 1] = band[1]
        hsv[low:high, :, 2] = band[2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _tool_verdict(model: Path, image_dir: Path) -> dict[str, str]:
    _summary, results = color_verifier.verify_directory(
        image_dir, model, infer_expected_from_name=False
    )
    return {
        Path(str(item.image)).stem: str(item.predicted_color) for item in results
    }


#: Cases where the runtime and the tool are known to disagree, with the value
#: each one produces today. Written out rather than computed so that a change
#: on either side has to be an edit here.
DIVERGENT = {
    "pure_red": {"bands": [[4, 210, 190]], "runtime": "red", "tool": "Unknown"},
    "bright_red_past_seam": {
        "bands": [[176, 210, 190]],
        "runtime": "red",
        "tool": "Unknown",
    },
    "red_with_orange_edge": {
        "bands": [[4, 210, 190], [4, 210, 190], [12, 200, 210]],
        "runtime": "red",
        "tool": "Orange",
    },
}

AGREEING = {
    "pure_orange": {"bands": [[12, 210, 210]], "color": "orange"},
    "pure_yellow": {"bands": [[28, 190, 220]], "color": "yellow"},
    "pure_green": {"bands": [[85, 120, 60]], "color": "green"},
}


@pytest.fixture(scope="module")
def tool_verdicts(color_model: Path, tmp_path_factory: pytest.TempPathFactory):
    image_dir = tmp_path_factory.mktemp("divergence-images")
    for name, case in {**DIVERGENT, **AGREEING}.items():
        cv2.imwrite(str(image_dir / f"{name}.png"), _render(case["bands"]))
    return _tool_verdict(color_model, image_dir)


@pytest.mark.parametrize("name", sorted(DIVERGENT))
def test_tool_disagrees_with_the_runtime_the_way_it_is_documented(
    name: str, color_model: Path, tool_verdicts: dict[str, str]
) -> None:
    case = DIVERGENT[name]
    runtime = StatsColorChecker.from_json(color_model).check(_render(case["bands"]))

    assert runtime.best_color.casefold() == case["runtime"]
    assert tool_verdicts[name] == case["tool"], (
        f"{name}: tools/color_verifier no longer behaves as its docstring "
        "describes. Decide which behavior was intended and update both."
    )


@pytest.mark.parametrize("name", sorted(AGREEING))
def test_tool_still_agrees_with_the_runtime_where_it_used_to(
    name: str, color_model: Path, tool_verdicts: dict[str, str]
) -> None:
    """The divergence is bounded, not total; this pins the part that holds."""
    case = AGREEING[name]
    runtime = StatsColorChecker.from_json(color_model).check(_render(case["bands"]))

    assert runtime.best_color.casefold() == case["color"]
    assert tool_verdicts[name].casefold() == case["color"]
