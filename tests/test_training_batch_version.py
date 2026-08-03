from __future__ import annotations

import pytest

from core.training_batch_version import (
    TrainingBatchVersionError,
    format_training_batch_version,
    parse_training_batch_semver,
    training_batch_prefix,
    validate_training_batch_version,
)


def test_training_batch_version_is_target_scoped_and_canonical() -> None:
    assert training_batch_prefix("Cable 1", "Top/A") == "Cable_1_Top_A"
    assert (
        validate_training_batch_version(
            "cable_1_top_a_V0.0.2",
            product="Cable 1",
            area="Top/A",
        )
        == "Cable_1_Top_A_v0.0.2"
    )


@pytest.mark.parametrize(
    "value",
    [
        "Cable1_B_v0.0.1",
        "Cable1_A_0.0.1",
        "Cable1_A_v00.0.1",
        "Cable1_A_v0.1",
        "",
    ],
)
def test_training_batch_version_rejects_wrong_target_or_format(value: str) -> None:
    with pytest.raises(TrainingBatchVersionError):
        validate_training_batch_version(value, product="Cable1", area="A")


def test_training_batch_version_formats_and_parses_semver() -> None:
    version = format_training_batch_version("Cable1", "A", (1, 2, 3))

    assert version == "Cable1_A_v1.2.3"
    assert parse_training_batch_semver(version) == (1, 2, 3)
    assert parse_training_batch_semver("legacy") is None
