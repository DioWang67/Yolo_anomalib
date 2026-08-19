from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.services.acceptance_artifacts import build_acceptance_artifact_bundle
from core.services.acceptance_runs import (
    AcceptanceRunError,
    AcceptanceRunRepository,
    AcceptanceRunState,
)
from core.services.model_acceptance import (
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceRepository,
)


def _write_image(path: Path, value: int) -> None:
    assert cv2.imwrite(
        str(path),
        np.full((12, 16, 3), value, dtype=np.uint8),
    )


def _bundle(tmp_path: Path, *, version: str = "v1"):
    """Build one pinned combination; a different ``version`` is a different bundle."""
    models_root = tmp_path / "models"
    station_root = models_root / "Cable1" / "A" / "yolo"
    station_root.mkdir(parents=True, exist_ok=True)
    weight = station_root / "model.onnx"
    config = station_root / "config.yaml"
    global_config = tmp_path / "config.yaml"
    weight.write_bytes(b"model")
    config.write_text("weights: model.onnx\n", encoding="utf-8")
    global_config.write_text("device: cpu\n", encoding="utf-8")
    return build_acceptance_artifact_bundle(
        product="Cable1",
        area="A",
        inference_type="yolo",
        version=version,
        global_config_path=global_config,
        model_config_path=config,
        models_root=models_root,
        model_weight_path=weight,
    )


def _repository(tmp_path: Path) -> AcceptanceRepository:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    _write_image(first, 10)
    _write_image(second, 20)
    return AcceptanceRepository(tmp_path / "acceptance")


def _records(repository: AcceptanceRepository, tmp_path: Path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    return repository.import_images(
        (first, second),
        product="Cable1",
        area="A",
    )


def _outcome(record, bundle, status: str) -> AcceptanceInferenceOutcome:
    return AcceptanceInferenceOutcome(
        sample_id=record.sample_id,
        machine_status=status,
        machine_reasons=(),
        model_version=bundle.version,
        model_sha256=bundle.model_weight.sha256,
        runtime_config_sha256=bundle.model_config.sha256,
        color_model_sha256="",
        inference_at="2026-08-13T12:00:00+00:00",
        latency_ms=1.0,
        error="",
    )


def test_completed_run_is_staged_then_committed_as_one_manifest(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    bundle = _bundle(tmp_path)
    run_repository = AcceptanceRunRepository(repository.root)
    source_sha = repository.manifest_sha256()
    run = run_repository.begin(
        artifact_bundle=bundle,
        sample_ids=tuple(record.sample_id for record in records),
        source_manifest_sha256=source_sha,
    )
    outcomes = tuple(
        _outcome(record, bundle, status)
        for record, status in zip(records, ("OK", "NG"), strict=True)
    )

    for outcome in outcomes:
        run_repository.append_outcome(run, outcome)
    assert repository.manifest_sha256() == source_sha
    assert all(not record.machine_status for record in repository.records())

    _updated, committed_sha = repository.save_inference_batch(
        outcomes,
        run_id=run.run_id,
        artifact_bundle=bundle,
        expected_manifest_sha256=source_sha,
    )
    run_repository.complete(run, committed_manifest_sha256=committed_sha)

    committed = repository.records()
    assert {record.acceptance_run_id for record in committed} == {run.run_id}
    assert {record.artifact_bundle_sha256 for record in committed} == {
        bundle.bundle_sha256
    }
    assert run_repository.state(run) is AcceptanceRunState.COMPLETED
    state = json.loads((run.root / "state.json").read_text(encoding="utf-8"))
    assert state["details"]["committed_manifest_sha256"] == committed_sha
    snapshot = repository.create_snapshot(
        label="completed-run",
        require_completed_run=True,
    )
    assert snapshot.manifest_path.is_file()


def test_cancelled_run_never_changes_formal_manifest(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    bundle = _bundle(tmp_path)
    run_repository = AcceptanceRunRepository(repository.root)
    source_sha = repository.manifest_sha256()
    run = run_repository.begin(
        artifact_bundle=bundle,
        sample_ids=tuple(record.sample_id for record in records),
        source_manifest_sha256=source_sha,
    )
    run_repository.append_outcome(run, _outcome(records[0], bundle, "OK"))
    run_repository.cancel(run)

    assert run_repository.state(run) is AcceptanceRunState.CANCELLED
    assert repository.manifest_sha256() == source_sha


def test_batch_commit_rejects_manifest_compare_and_swap_conflict(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    bundle = _bundle(tmp_path)
    source_sha = repository.manifest_sha256()
    repository.confirm(
        records[0].sample_id,
        verdict="OK",
        reviewed_by="other-process",
    )

    with pytest.raises(AcceptanceDataError, match="changed during inference"):
        repository.save_inference_batch(
            (_outcome(records[0], bundle, "OK"),),
            run_id="run-conflict",
            artifact_bundle=bundle,
            expected_manifest_sha256=source_sha,
        )


def test_partial_batch_clears_stale_results_outside_the_new_run(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    bundle = _bundle(tmp_path)
    repository.save_inference(_outcome(records[0], bundle, "NG"))
    source_sha = repository.manifest_sha256()

    repository.save_inference_batch(
        (_outcome(records[1], bundle, "OK"),),
        run_id="run-partial",
        artifact_bundle=bundle,
        expected_manifest_sha256=source_sha,
    )

    first, second = repository.records()
    assert first.machine_status == ""
    assert first.model_version == ""
    assert first.acceptance_run_id == ""
    assert second.machine_status == "OK"
    assert second.acceptance_run_id == "run-partial"


def test_same_bundle_partial_runs_accumulate_into_one_formal_snapshot(
    tmp_path: Path,
) -> None:
    """Two runs of one identical bundle are as comparable as one run would be.

    This is the case the previous run-ID invariant forbade: re-running the
    samples that failed transiently discarded the ones that had succeeded under
    the very same pinned artifacts, so a partial run could never converge.
    """
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    bundle = _bundle(tmp_path)
    run_repository = AcceptanceRunRepository(repository.root)

    for record, status in zip(records, ("OK", "NG"), strict=True):
        run = run_repository.begin(
            artifact_bundle=bundle,
            sample_ids=(record.sample_id,),
            source_manifest_sha256=repository.manifest_sha256(),
        )
        outcome = _outcome(record, bundle, status)
        run_repository.append_outcome(run, outcome)
        _committed, committed_sha = repository.save_inference_batch(
            (outcome,),
            run_id=run.run_id,
            artifact_bundle=bundle,
            expected_manifest_sha256=run.source_manifest_sha256,
        )
        run_repository.complete(run, committed_manifest_sha256=committed_sha)

    first, second = repository.records()
    assert (first.machine_status, second.machine_status) == ("OK", "NG")
    assert first.acceptance_run_id != second.acceptance_run_id
    assert first.artifact_bundle_sha256 == second.artifact_bundle_sha256
    assert repository.create_snapshot(
        label="two-runs-one-bundle",
        require_completed_run=True,
    ).manifest_path.is_file()


def test_batch_under_a_different_bundle_clears_the_earlier_results(
    tmp_path: Path,
) -> None:
    """Changing the pinned combination does invalidate every earlier result."""
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    first_bundle = _bundle(tmp_path)
    second_bundle = _bundle(tmp_path, version="v2")
    assert first_bundle.bundle_sha256 != second_bundle.bundle_sha256

    repository.save_inference_batch(
        (_outcome(records[0], first_bundle, "OK"),),
        run_id="run-first-bundle",
        artifact_bundle=first_bundle,
        expected_manifest_sha256=repository.manifest_sha256(),
    )
    repository.save_inference_batch(
        (_outcome(records[1], second_bundle, "NG"),),
        run_id="run-second-bundle",
        artifact_bundle=second_bundle,
        expected_manifest_sha256=repository.manifest_sha256(),
    )

    first, second = repository.records()
    assert first.machine_status == ""
    assert first.artifact_bundle_sha256 == ""
    assert second.machine_status == "NG"
    assert second.artifact_bundle_sha256 == second_bundle.bundle_sha256


def test_formal_snapshot_rejects_a_run_that_never_completed(tmp_path: Path) -> None:
    """A committed manifest is not evidence until its run reached COMPLETED."""
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    bundle = _bundle(tmp_path)
    run_repository = AcceptanceRunRepository(repository.root)
    run = run_repository.begin(
        artifact_bundle=bundle,
        sample_ids=tuple(record.sample_id for record in records),
        source_manifest_sha256=repository.manifest_sha256(),
    )
    outcomes = tuple(
        _outcome(record, bundle, status)
        for record, status in zip(records, ("OK", "NG"), strict=True)
    )
    repository.save_inference_batch(
        outcomes,
        run_id=run.run_id,
        artifact_bundle=bundle,
        expected_manifest_sha256=run.source_manifest_sha256,
    )

    with pytest.raises(AcceptanceDataError, match="completed"):
        repository.create_snapshot(label="still-running", require_completed_run=True)


def test_formal_snapshot_rejects_partial_or_untracked_results(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    records = _records(repository, tmp_path)
    repository.save_inference(_outcome(records[0], _bundle(tmp_path), "OK"))

    with pytest.raises(AcceptanceDataError, match="identical artifact bundle"):
        repository.create_snapshot(
            label="unsafe",
            require_completed_run=True,
        )


def test_run_rejects_sample_id_that_cannot_be_round_tripped(tmp_path: Path) -> None:
    run_repository = AcceptanceRunRepository(tmp_path / "acceptance")

    with pytest.raises(AcceptanceRunError, match="unique, non-empty"):
        run_repository.begin(
            artifact_bundle=_bundle(tmp_path),
            sample_ids=("sample.with-dot",),
            source_manifest_sha256="a" * 64,
        )
