"""Fail-closed boundary contracts for acceptance command-line entry points."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import app.acceptance.color_revision_contract as contract_cli
import app.acceptance.headless as headless
from app.acceptance.matrix_dialog import AcceptanceMatrixWorker
from core.services.acceptance_matrix import AcceptanceMatrixCancelled


def test_color_revision_cli_rejects_unsafe_or_incomplete_reports(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    revisions_root = tmp_path / ".color_revisions"
    report_path = tmp_path / "acceptance-report.json"
    args = ["--revisions-root", str(revisions_root), "--report", str(report_path)]

    assert contract_cli.main(args) == 2
    assert "missing or unsafe" in capsys.readouterr().err

    report_path.write_text("[]", encoding="utf-8")
    assert contract_cli.main(args) == 2
    assert "must contain an object" in capsys.readouterr().err

    report_path.write_text("{}", encoding="utf-8")
    assert contract_cli.main(args) == 2
    assert "has no color revision contract" in capsys.readouterr().err

    monkeypatch.setattr(Path, "is_symlink", lambda path: path == report_path)
    assert contract_cli.main(args) == 2
    assert "cannot be a symbolic link" in capsys.readouterr().err


def test_headless_runner_must_execute_live_color_revision_validator(
    monkeypatch,
    tmp_path: Path,
) -> None:
    weight_path = tmp_path / "candidate.pt"
    config_path = tmp_path / "config.yaml"
    weight_path.write_bytes(b"candidate-weight")
    config_path.write_text("weights: candidate.pt\n", encoding="utf-8")
    revisions_root = tmp_path / ".color_revisions"
    contract = {"enabled": False, "entries": [], "identity_sha256": "a" * 64}
    verify_contract = Mock(return_value=contract)

    monkeypatch.setattr(
        headless,
        "capture_candidate_color_revision_contract",
        Mock(return_value=contract),
    )
    monkeypatch.setattr(headless, "color_revision_overrides", Mock(return_value={}))
    monkeypatch.setattr(
        headless,
        "verify_active_color_revision_contract",
        verify_contract,
    )

    def run_candidate_acceptance(**kwargs):
        assert kwargs["color_revision_contract_validator"]() == ()
        return SimpleNamespace(
            passed=True,
            report_path=tmp_path / "report.json",
            failures=(),
        )

    monkeypatch.setattr(headless, "run_candidate_acceptance", run_candidate_acceptance)

    exit_code = headless.main(
        [
            "--project-root",
            str(tmp_path),
            "--models-root",
            str(tmp_path / "models"),
            "--global-config",
            str(config_path),
            "--color-revisions-root",
            str(revisions_root),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--snapshot-manifest",
            str(tmp_path / "snapshot.json"),
            "--report",
            str(tmp_path / "report.json"),
            "--product",
            "Cable1",
            "--area",
            "A",
            "--candidate-weight",
            str(weight_path),
            "--candidate-config",
            str(config_path),
            "--min-confirmed",
            "1",
            "--max-false-positives",
            "0",
            "--max-false-negatives",
            "0",
        ]
    )

    assert exit_code == 0
    verify_contract.assert_called_once_with(contract, revisions_root=revisions_root)


def test_matrix_worker_forwards_progress_and_completed_result() -> None:
    request = object()
    result = object()
    progress: list[tuple[int, int, str, str]] = []
    completed: list[object] = []

    def runner(received_request, *, progress_callback, cancel_callback):
        assert received_request is request
        assert not cancel_callback()
        progress_callback(2, 5, "model/config", "sample-2")
        return result

    worker = AcceptanceMatrixWorker(request, runner=runner)
    worker.progress_changed.connect(lambda *args: progress.append(args))
    worker.completed.connect(completed.append)

    worker.run()

    assert progress == [(2, 5, "model/config", "sample-2")]
    assert completed == [result]


def test_matrix_worker_maps_cancellation_and_expected_failures_to_signals() -> None:
    cancellations: list[bool] = []
    cancelled_worker = AcceptanceMatrixWorker(
        object(),
        runner=Mock(side_effect=AcceptanceMatrixCancelled),
    )
    cancelled_worker.cancelled.connect(lambda: cancellations.append(True))
    cancelled_worker.run()
    assert cancellations == [True]

    failures: list[str] = []
    failed_worker = AcceptanceMatrixWorker(
        object(),
        runner=Mock(side_effect=RuntimeError("GPU unavailable")),
    )
    failed_worker.failed.connect(failures.append)
    failed_worker.run()
    assert failures == ["GPU unavailable"]
