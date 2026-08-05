from types import SimpleNamespace

import numpy as np

from core.services.color_checker import ColorCheckerService


class StubColorChecker:
    def __init__(self, *, best_color: str, is_ok: bool = True) -> None:
        self._result = SimpleNamespace(
            best_color=best_color,
            diff=0.60,
            threshold=0.75,
            is_ok=is_ok,
        )

    def check(self, _image, allowed_colors=None):
        return self._result


def _check_detection(*, detected_class: str, best_color: str, is_ok: bool = True):
    service = ColorCheckerService()
    service._checker = StubColorChecker(best_color=best_color, is_ok=is_ok)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)

    return service.check_items(
        frame=frame,
        processed_image=frame,
        detections=[{"class": detected_class, "bbox": [0, 0, 10, 10]}],
        candidates=["Black", "Orange"],
    )


def test_color_label_mismatch_fails_even_when_distance_is_within_threshold():
    result = _check_detection(detected_class="Black", best_color="Orange")

    assert result.is_ok is False
    assert result.items[0].is_ok is False
    assert result.items[0].class_name == "Black"
    assert result.items[0].best_color == "Orange"


def test_matching_color_label_passes_when_distance_is_within_threshold():
    result = _check_detection(detected_class="Black", best_color="black")

    assert result.is_ok is True
    assert result.items[0].is_ok is True


def test_generic_detector_class_uses_color_threshold_result():
    result = _check_detection(detected_class="LED", best_color="Orange")

    assert result.is_ok is True
    assert result.items[0].is_ok is True


def test_matching_color_label_still_fails_when_distance_exceeds_threshold():
    result = _check_detection(
        detected_class="Black",
        best_color="Black",
        is_ok=False,
    )

    assert result.is_ok is False
    assert result.items[0].is_ok is False
