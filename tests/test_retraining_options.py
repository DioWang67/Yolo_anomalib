import pytest

from core.retraining_options import RetrainingOptions


def test_retraining_options_round_trip_and_estimate():
    options = RetrainingOptions.from_mapping(
        {
            "epochs": 80,
            "augmentations_per_image": 6,
            "batch": 4,
            "imgsz": 960,
        }
    )

    assert options.to_dict() == {
        "epochs": 80,
        "augmentations_per_image": 6,
        "batch": 4,
        "imgsz": 960,
    }
    assert options.estimated_maximum_images(10) == 70


@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 19},
        {"augmentations_per_image": 51},
        {"batch": 0},
        {"imgsz": 641},
        {"unknown": 1},
        {"batch": "8"},
        {"epochs": 20.5},
        {"augmentations_per_image": False},
    ],
)
def test_retraining_options_reject_unsafe_values(values):
    with pytest.raises(ValueError):
        RetrainingOptions.from_mapping(values)
