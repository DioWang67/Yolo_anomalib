import json
from types import SimpleNamespace

import numpy as np
import pytest

from core.color_qc_enhanced import ColorQCEnhanced
from core.services.color_checker import ColorCheckerService


class StubColorChecker:
    def __init__(
        self,
        *,
        best_color: str,
        is_ok: bool = True,
        supported_colors=("Black", "Orange"),
    ) -> None:
        self.supported_colors = supported_colors
        self.allowed_colors: list[list[str] | None] = []
        self._result = SimpleNamespace(
            best_color=best_color,
            diff=0.60,
            threshold=0.75,
            is_ok=is_ok,
        )

    def check(self, _image, allowed_colors=None):
        self.allowed_colors.append(allowed_colors)
        return self._result


def _check_detection(
    *,
    detected_class: str,
    best_color: str,
    is_ok: bool = True,
    candidates=("Black", "Orange"),
    supported_colors=("Black", "Orange"),
    generic_classes=None,
):
    service = ColorCheckerService()
    checker = StubColorChecker(
        best_color=best_color,
        is_ok=is_ok,
        supported_colors=supported_colors,
    )
    service._checker = checker
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    result = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[{"class": detected_class, "bbox": [0, 0, 10, 10]}],
        candidates=candidates,
        generic_classes=generic_classes,
    )
    return result, checker


def test_color_label_mismatch_fails_even_when_distance_is_within_threshold():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="Orange",
    )

    assert result.is_ok is False
    assert result.items[0].is_ok is False
    assert result.items[0].class_name == "Black"
    assert result.items[0].best_color == "Orange"


def test_matching_color_label_passes_when_distance_is_within_threshold():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="black",
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True


def test_generic_detector_class_uses_color_threshold_result():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True
    assert checker.allowed_colors == [["Black", "Orange"]]


def test_generic_detector_class_is_not_used_as_candidate_when_candidates_are_none():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
        candidates=None,
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True
    assert checker.allowed_colors == [None]


def test_generic_detector_candidate_is_removed_before_checker_call():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
        candidates=["LED"],
    )

    assert result.is_ok is True
    assert result.items[0].is_ok is True
    assert checker.allowed_colors == [None]


def test_generic_detector_with_only_unsupported_color_candidates_fails_closed():
    result, checker = _check_detection(
        detected_class="LED",
        best_color="Orange",
        candidates=["Purple"],
        supported_colors=("Orange",),
    )

    assert checker.allowed_colors == [None]
    assert result.is_ok is False
    assert result.items[0].is_ok is False


def test_unknown_detector_class_is_not_implicitly_treated_as_generic():
    result, checker = _check_detection(
        detected_class="Lamp",
        best_color="Orange",
        candidates=None,
        supported_colors=("Orange",),
    )

    assert checker.allowed_colors == [None]
    assert result.is_ok is False
    assert result.items[0].is_ok is False


@pytest.mark.parametrize(
    ("candidates", "expected_allowed"),
    ((["Green"], None), (["Green", "Orange"], ["Orange"])),
)
def test_configured_color_missing_from_model_fails_closed(
    candidates,
    expected_allowed,
):
    result, checker = _check_detection(
        detected_class="Green",
        best_color="Orange",
        candidates=candidates,
        supported_colors=("Orange",),
    )

    assert checker.allowed_colors == [expected_allowed]
    assert result.is_ok is False
    assert result.items[0].is_ok is False


def test_custom_generic_detector_class_can_be_declared_explicitly():
    result, checker = _check_detection(
        detected_class="Lamp",
        best_color="Orange",
        candidates=["Lamp"],
        supported_colors=("Orange",),
        generic_classes=["Lamp"],
    )

    assert checker.allowed_colors == [None]
    assert result.is_ok is True


@pytest.mark.parametrize("candidates", (None, ["LED"]))
def test_generic_detector_uses_all_colors_with_production_checker(
    tmp_path,
    candidates,
):
    model_path = tmp_path / "color.json"
    model_path.write_text(
        json.dumps(
            {
                "config": {
                    "hist_bins": [1, 1, 1],
                    "default_hist_thr": 0.25,
                },
                "colors": {
                    "Orange": {
                        "avg_color_hist": [1.0],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    service = ColorCheckerService()
    service._checker = ColorQCEnhanced.from_json(model_path)
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)

    result = service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[{"class": "LED", "bbox": [0, 0, 10, 10]}],
        candidates=candidates,
    )

    assert service._checker.supported_colors == ("Orange",)
    assert result.is_ok is True
    assert result.items[0].best_color == "Orange"


def test_matching_color_label_still_fails_when_distance_exceeds_threshold():
    result, _checker = _check_detection(
        detected_class="Black",
        best_color="Black",
        is_ok=False,
    )

    assert result.is_ok is False
    assert result.items[0].is_ok is False
