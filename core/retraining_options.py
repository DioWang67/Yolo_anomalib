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


@dataclass(frozen=True)
class RetrainingOptions:
    """Immutable, serializable settings accepted by both projects."""

    epochs: int = 20
    augmentations_per_image: int = 20
    batch: int = 8
    imgsz: int = 640

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

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RetrainingOptions:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("training_options must be a mapping.")
        allowed = {"epochs", "augmentations_per_image", "batch", "imgsz"}
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(
                "Unsupported training option(s): " + ", ".join(sorted(unexpected))
            )
        invalid_types = [key for key in value if type(value[key]) is not int]
        if invalid_types:
            raise ValueError(
                "Training option(s) must be integers: "
                + ", ".join(sorted(invalid_types))
            )
        defaults = cls()
        return cls(
            epochs=value.get("epochs", defaults.epochs),
            augmentations_per_image=value.get(
                "augmentations_per_image", defaults.augmentations_per_image
            ),
            batch=value.get("batch", defaults.batch),
            imgsz=value.get("imgsz", defaults.imgsz),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    def estimated_maximum_images(self, source_count: int) -> int:
        """Return originals plus the maximum requested generated variants."""
        return max(int(source_count), 0) * (self.augmentations_per_image + 1)


def _validate_range(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
