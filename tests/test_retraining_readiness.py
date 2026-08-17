import csv
from pathlib import Path

from tools.retraining_readiness import (
    MINIMUM_TEST_SOURCE_GROUPS,
    MINIMUM_VAL_SOURCE_GROUPS,
    count_accumulated_training_images,
    evaluate_retraining_readiness,
    required_historical_count,
)


def _write_ready_manifest(
    training_data_dir: Path, *, product: str, area: str, sample_ids: list[str]
) -> None:
    manifest = (
        training_data_dir / product / area / "metadata" / "review_dataset_manifest.csv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "output_image"])
        writer.writeheader()
        for sample_id in sample_ids:
            writer.writerow(
                {"sample_id": sample_id, "output_image": f"{sample_id}.jpg"}
            )


def test_required_historical_count_matches_split_minimums():
    assert required_historical_count() == (
        MINIMUM_VAL_SOURCE_GROUPS + MINIMUM_TEST_SOURCE_GROUPS
    )
    assert required_historical_count() == 15


def test_position_goldens_reduce_the_historical_requirement():
    # Goldens are forced into test, so the dynamic test split needs fewer.
    assert required_historical_count(position_golden_count=10) == 5
    assert required_historical_count(position_golden_count=4) == 11
    # More goldens than the test minimum never drops below the val minimum.
    assert required_historical_count(position_golden_count=99) == 5


def test_new_station_is_reported_as_not_ready(tmp_path):
    readiness = evaluate_retraining_readiness(
        tmp_path, product="Cable1", area="A", submitted_count=6
    )

    assert readiness.historical_count == 0
    assert readiness.is_ready is False
    assert readiness.shortfall == 15
    assert "還需要約 15 張" in readiness.to_operator_text()


def test_station_with_enough_history_is_ready(tmp_path):
    _write_ready_manifest(
        tmp_path,
        product="Cable1",
        area="A",
        sample_ids=[f"sample{index}" for index in range(20)],
    )

    readiness = evaluate_retraining_readiness(
        tmp_path, product="Cable1", area="A", submitted_count=3
    )

    assert readiness.historical_count == 17
    assert readiness.is_ready is True
    assert "已達安全切分下限" in readiness.to_operator_text()


def test_resubmitted_images_are_not_double_counted_as_history(tmp_path):
    _write_ready_manifest(
        tmp_path,
        product="Cable1",
        area="A",
        sample_ids=[f"sample{index}" for index in range(16)],
    )

    readiness = evaluate_retraining_readiness(
        tmp_path, product="Cable1", area="A", submitted_count=16
    )

    assert readiness.historical_count == 0
    assert readiness.is_ready is False


def test_tiny_station_is_blocked_by_the_absolute_group_minimum(tmp_path):
    readiness = evaluate_retraining_readiness(
        tmp_path, product="Cable1", area="A", submitted_count=1, position_golden_count=99
    )

    # Goldens relax the historical requirement but three groups are still the
    # hard floor inside dataset_splitter.
    assert readiness.total_count == 1
    assert readiness.total_shortfall == 2
    assert readiness.is_ready is False


def test_unreadable_manifest_never_blocks_a_submission(tmp_path):
    manifest = tmp_path / "Cable1" / "A" / "metadata" / "review_dataset_manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"\xff\xfe\x00broken")

    assert count_accumulated_training_images(
        tmp_path, product="Cable1", area="A"
    ) == 0


def test_folder_names_match_the_exporter_sanitizer(tmp_path):
    _write_ready_manifest(
        tmp_path,
        product="Cable_1",
        area="A_B",
        sample_ids=["sample1", "sample2"],
    )

    assert (
        count_accumulated_training_images(tmp_path, product="Cable/1", area="A B")
        == 2
    )
