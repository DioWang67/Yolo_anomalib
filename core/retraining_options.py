"""Validated operator-controlled settings for one retraining job."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

MIN_EPOCHS = 20
MAX_EPOCHS = 300
MIN_AUGMENTATIONS_PER_IMAGE = 0
MAX_AUGMENTATIONS_PER_IMAGE = 50
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 64
MIN_IMAGE_SIZE = 320
MAX_IMAGE_SIZE = 1280
IMAGE_SIZE_STEP = 32
POSITION_MODE_AUTO = "auto"
POSITION_MODE_YOLO_ONLY = "yolo_only"
POSITION_MODE_CALIBRATE_VALIDATE = "calibrate_validate"
POSITION_TRAINING_MODES = frozenset(
    {
        POSITION_MODE_AUTO,
        POSITION_MODE_YOLO_ONLY,
        POSITION_MODE_CALIBRATE_VALIDATE,
    }
)
POSITION_ACTIVATION_PRESERVE = "preserve"
POSITION_ACTIVATION_ENABLE_AFTER_GATE = "enable_after_gate"
POSITION_ACTIVATION_MODES = frozenset(
    {
        POSITION_ACTIVATION_PRESERVE,
        POSITION_ACTIVATION_ENABLE_AFTER_GATE,
    }
)


@dataclass(frozen=True)
class RetrainingOptions:
    """Immutable, serializable settings accepted by both projects."""

    epochs: int = 20
    augmentations_per_image: int = 20
    batch: int = 8
    imgsz: int = 640
    # New operator jobs are explicit opt-in. ``auto`` remains accepted only
    # for reading handoffs created by older GUI versions.
    position_training_mode: str = POSITION_MODE_YOLO_ONLY
    position_activation: str = POSITION_ACTIVATION_PRESERVE

    def __post_init__(self) -> None:
        _validate_range("epochs", self.epochs, MIN_EPOCHS, MAX_EPOCHS)
        _validate_range(
            "augmentations_per_image",
            self.augmentations_per_image,
            MIN_AUGMENTATIONS_PER_IMAGE,
            MAX_AUGMENTATIONS_PER_IMAGE,
        )
        _validate_range("batch", self.batch, MIN_BATCH_SIZE, MAX_BATCH_SIZE)
        _validate_range("imgsz", self.imgsz, MIN_IMAGE_SIZE, MAX_IMAGE_SIZE)
        if self.imgsz % IMAGE_SIZE_STEP != 0:
            raise ValueError(f"imgsz must be a multiple of {IMAGE_SIZE_STEP}.")
        if (
            not isinstance(self.position_training_mode, str)
            or self.position_training_mode not in POSITION_TRAINING_MODES
        ):
            raise ValueError(
                "position_training_mode must be one of: "
                + ", ".join(sorted(POSITION_TRAINING_MODES))
            )
        if (
            not isinstance(self.position_activation, str)
            or self.position_activation not in POSITION_ACTIVATION_MODES
        ):
            raise ValueError(
                "position_activation must be one of: "
                + ", ".join(sorted(POSITION_ACTIVATION_MODES))
            )
        if (
            self.position_training_mode == POSITION_MODE_YOLO_ONLY
            and self.position_activation == POSITION_ACTIVATION_ENABLE_AFTER_GATE
        ):
            raise ValueError(
                "YOLO-only retraining cannot enable position detection because "
                "no position gate will run."
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RetrainingOptions:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("training_options must be a mapping.")
        integer_options = {"epochs", "augmentations_per_image", "batch", "imgsz"}
        string_options = {"position_training_mode", "position_activation"}
        allowed = integer_options | string_options
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(
                "Unsupported training option(s): " + ", ".join(sorted(unexpected))
            )
        invalid_types = [
            key for key in value if key in integer_options and type(value[key]) is not int
        ]
        if invalid_types:
            raise ValueError(
                "Training option(s) must be integers: "
                + ", ".join(sorted(invalid_types))
            )
        invalid_string_types = [
            key
            for key in value
            if key in string_options and not isinstance(value[key], str)
        ]
        if invalid_string_types:
            raise ValueError(
                "Training option(s) must be strings: "
                + ", ".join(sorted(invalid_string_types))
            )
        defaults = cls()
        return cls(
            epochs=value.get("epochs", defaults.epochs),
            augmentations_per_image=value.get(
                "augmentations_per_image", defaults.augmentations_per_image
            ),
            batch=value.get("batch", defaults.batch),
            imgsz=value.get("imgsz", defaults.imgsz),
            position_training_mode=value.get(
                "position_training_mode", defaults.position_training_mode
            ),
            position_activation=value.get(
                "position_activation", defaults.position_activation
            ),
        )

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def estimated_maximum_images(self, source_count: int) -> int:
        """Return originals plus the maximum requested generated variants."""
        return max(int(source_count), 0) * (self.augmentations_per_image + 1)


def _validate_range(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
