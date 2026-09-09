"""Start the retraining worker without putting a window in front of the line.

The operator-facing complaint this exists to fix is small and concrete: every
way of starting a retrain went through ``QProcess.startDetached("cmd.exe", …)``,
so a console window appeared over the inspection screen and the work then ran
somewhere the GUI could not see. Detaching also meant a failure to start showed
up as nothing happening.

Two things follow from that, and they are why this is a module rather than two
lines at each call site:

* **No console.** ``CREATE_NO_WINDOW`` is a ``CreateProcess`` flag, and PyQt5
  exposes no way to pass it -- ``QProcess.setCreateProcessArgumentsModifier``
  is absent from these bindings -- so the launch goes through
  :mod:`subprocess` instead of ``QProcess``.
* **Not detached.** The caller keeps the handle, so output can be read and a
  non-zero exit can be reported where the operator is looking.

What is *not* here is any decision about which interpreter runs the training.
That belongs to the training project's own launcher, which resolves
``PICTURE_TOOL_PYTHON``, then a local virtualenv, then the conda environments,
and runs its runtime doctor first. Re-deriving that here would give the two
projects two answers to the same question.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)

#: The training project's operator launcher, relative to its root.
LAUNCHER_NAME = "open_operator_training.bat"

#: Windows-only; zero elsewhere, where the flag has no meaning and passing a
#: non-zero value would be an error rather than a no-op.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class TrainingLaunchError(RuntimeError):
    """The retraining worker could not be started."""


def launch_retraining_worker(
    *,
    training_root: str | Path,
    handoff_path: str | Path,
) -> subprocess.Popen[str]:
    """Start the retraining worker for ``handoff_path``, windowless.

    ``--background`` is what makes the worker headless rather than a second
    application: in that mode the training entry point applies its
    below-normal resource policy and never shows its window.

    Raises:
        TrainingLaunchError: If the launcher or the job data is missing, or
            the process could not be started.
    """
    root = Path(training_root).expanduser().resolve()
    launcher = root / LAUNCHER_NAME
    if not launcher.is_file():
        raise TrainingLaunchError(f"Retraining launcher not found: {launcher}")
    handoff = Path(handoff_path).expanduser().resolve()
    if not handoff.is_file():
        raise TrainingLaunchError(f"Retraining job data not found: {handoff}")

    try:
        process = subprocess.Popen(
            [str(launcher), str(handoff), "--background"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise TrainingLaunchError(f"Failed to start retraining: {exc}") from exc

    logger.info("Retraining worker started (pid %s) for %s", process.pid, handoff)
    return process


def launch_training_workbench(*, training_root: str | Path) -> subprocess.Popen[str]:
    """Open the training project's own full editing GUI, windowed.

    This is the annotation/config/color-baseline tool, not the headless
    retraining worker: it calls the launcher with ``--open``, which shows the
    window with no job pre-selected. The launcher's own console is still
    suppressed -- the operator asked for the training tool's window, not an
    extra console box behind it -- so this goes through the same
    ``CREATE_NO_WINDOW`` path.

    ``--open`` matters, not just ``--resume-latest`` with the console hidden:
    resuming requires an actual resumable job, and the first shape this bug
    took was a silent no-op once the latest job had already been deployed --
    the launcher exited almost instantly, and its own failure ``pause``
    returned immediately too, because a windowless child has no console for
    a person to press a key in. That silence is also why the output is
    captured here rather than discarded: a future failure this launcher
    cannot anticipate should show up in this process's log, not disappear
    the same way.

    Raises:
        TrainingLaunchError: If the launcher is missing or the process could
            not be started.
    """
    root = Path(training_root).expanduser().resolve()
    launcher = root / LAUNCHER_NAME
    if not launcher.is_file():
        raise TrainingLaunchError(f"Training launcher not found: {launcher}")

    try:
        process = subprocess.Popen(
            [str(launcher), "--open"],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
    except OSError as exc:
        raise TrainingLaunchError(f"Failed to start training tool: {exc}") from exc

    logger.info("Training workbench opened (pid %s)", process.pid)
    return process


class WorkerOutputReader(QThread):
    """Stream a child process's merged output without blocking the GUI.

    Kept beside the launcher because the two are only correct together: the
    launcher asks for a pipe, and a pipe nobody drains will eventually block
    the child once its buffer fills -- turning "runs quietly in the
    background" into "stops halfway through, for no visible reason".
    """

    line_ready = pyqtSignal(str)
    finished_with_code = pyqtSignal(int)

    def __init__(self, process: subprocess.Popen[str]) -> None:
        super().__init__()
        self._process = process

    def run(self) -> None:  # pragma: no cover - exercised through the GUI
        stream = self._process.stdout
        if stream is not None:
            for line in stream:
                if self.isInterruptionRequested():
                    break
                text = line.rstrip()
                if text:
                    self.line_ready.emit(text)
        self.finished_with_code.emit(self._process.wait())
