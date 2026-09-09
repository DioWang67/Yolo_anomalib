"""Guards on how the retraining worker is started.

The point of this launcher is not that it starts a process -- the old code did
that too -- but *how*: windowless, so nothing appears over the inspection
screen, and attached, so a failure is visible instead of silent. Both are easy
to lose in a later edit that "simplifies" the call, and neither is visible in a
screenshot of the happy path, so they are pinned here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from app.gui.training_launcher import (
    CREATE_NO_WINDOW,
    LAUNCHER_NAME,
    TrainingLaunchError,
    launch_retraining_worker,
    launch_training_workbench,
)


@pytest.fixture()
def training_root(tmp_path: Path) -> Path:
    root = tmp_path / "Yolo11_auto_train"
    root.mkdir()
    (root / LAUNCHER_NAME).write_text("@echo off\n", encoding="utf-8")
    return root


@pytest.fixture()
def handoff(tmp_path: Path) -> Path:
    path = tmp_path / "handoff.json"
    path.write_text("{}", encoding="utf-8")
    return path


def test_worker_starts_without_a_console_window(
    training_root: Path, handoff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launch must carry CREATE_NO_WINDOW on Windows.

    This is the whole reason the launcher uses :mod:`subprocess` rather than
    ``QProcess``: PyQt5 cannot pass the flag, and without it a console window
    opens over the inspection screen every time an operator starts a retrain.
    """
    captured: dict[str, object] = {}

    class _Recorded:
        pid = 4321
        stdout = None

        def wait(self) -> int:
            return 0

    def _popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return _Recorded()

    monkeypatch.setattr(subprocess, "Popen", _popen)

    launch_retraining_worker(training_root=training_root, handoff_path=handoff)

    if sys.platform == "win32":
        assert CREATE_NO_WINDOW != 0
    assert captured["creationflags"] == CREATE_NO_WINDOW


def test_worker_is_attached_so_output_can_be_read(
    training_root: Path, handoff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Output must come back through a pipe rather than being detached.

    A detached worker reported a failure to start as nothing happening, which
    is what made a broken retrain look like an ignored button press.
    """
    captured: dict[str, object] = {}

    class _Recorded:
        pid = 4321
        stdout = None

        def wait(self) -> int:
            return 0

    def _popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return _Recorded()

    monkeypatch.setattr(subprocess, "Popen", _popen)

    launch_retraining_worker(training_root=training_root, handoff_path=handoff)

    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.STDOUT
    # stdin is closed rather than inherited: a headless worker that blocks on
    # a prompt would hang with nowhere to show the prompt.
    assert captured["stdin"] is subprocess.DEVNULL


def test_worker_runs_the_training_projects_own_launcher(
    training_root: Path, handoff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interpreter choice stays with the training project's launcher.

    Resolving a Python here would give the two projects two answers to the
    same question, and the launcher also runs the training runtime doctor
    first.
    """
    captured: dict[str, object] = {}

    class _Recorded:
        pid = 1
        stdout = None

        def wait(self) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda args, **kwargs: (captured.update(args=args, **kwargs), _Recorded())[1],
    )

    launch_retraining_worker(training_root=training_root, handoff_path=handoff)

    assert captured["args"] == [
        str(training_root / LAUNCHER_NAME),
        str(handoff.resolve()),
        "--background",
    ]
    assert captured["cwd"] == str(training_root)


def test_missing_launcher_is_reported_rather_than_started(
    tmp_path: Path, handoff: Path
) -> None:
    with pytest.raises(TrainingLaunchError, match="launcher not found"):
        launch_retraining_worker(
            training_root=tmp_path / "absent", handoff_path=handoff
        )


def test_missing_job_data_is_reported_rather_than_started(
    training_root: Path, tmp_path: Path
) -> None:
    with pytest.raises(TrainingLaunchError, match="job data not found"):
        launch_retraining_worker(
            training_root=training_root,
            handoff_path=tmp_path / "absent.json",
        )


def test_start_failure_is_raised_not_swallowed(
    training_root: Path, handoff: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError from the OS must reach the caller.

    The page turns this into a message on the screen the operator started
    from; swallowing it would leave a page that looks like it began work.
    """

    def _refuse(args, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(subprocess, "Popen", _refuse)

    with pytest.raises(TrainingLaunchError, match="Failed to start retraining"):
        launch_retraining_worker(
            training_root=training_root, handoff_path=handoff
        )


def test_workbench_opens_with_no_job_preselected(
    training_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full training tool launches windowed, with ``--open``.

    Unlike the retraining worker, this is the tool an engineer looks at, so
    it must not carry ``--background`` -- that flag is what makes the other
    entry point skip ``window.show()``. It also must not fall back to the
    launcher's bare no-argument form, which resolves to ``--resume-latest``
    and raises once the latest job is already deployed -- the first shape
    this bug took was exactly that: a silent no-op with no resumable job.
    """
    captured: dict[str, object] = {}

    class _Recorded:
        pid = 99
        stdout = None

        def wait(self) -> int:
            return 0

    def _popen(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return _Recorded()

    monkeypatch.setattr(subprocess, "Popen", _popen)

    launch_training_workbench(training_root=training_root)

    assert captured["args"] == [str(training_root / LAUNCHER_NAME), "--open"]
    assert captured["creationflags"] == CREATE_NO_WINDOW
    assert captured["stdout"] is subprocess.PIPE
    assert captured["stderr"] is subprocess.STDOUT
    assert captured["stdin"] is subprocess.DEVNULL


def test_workbench_missing_launcher_is_reported(tmp_path: Path) -> None:
    with pytest.raises(TrainingLaunchError, match="Training launcher not found"):
        launch_training_workbench(training_root=tmp_path / "absent")


def test_workbench_start_failure_is_raised_not_swallowed(
    training_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _refuse(args, **kwargs):
        raise OSError("no such interpreter")

    monkeypatch.setattr(subprocess, "Popen", _refuse)

    with pytest.raises(TrainingLaunchError, match="Failed to start training tool"):
        launch_training_workbench(training_root=training_root)
