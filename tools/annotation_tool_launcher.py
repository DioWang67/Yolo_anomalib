"""Optional, non-blocking annotation-tool launch abstraction for Phase 3C2."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from tools.annotation_packages import AnnotationWorkPackage


class AnnotationToolLaunchStatus(str, Enum):
    MANUAL = "MANUAL"
    LAUNCHED = "LAUNCHED"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AnnotationToolLaunchResult:
    status: AnnotationToolLaunchStatus
    message: str
    argv: tuple[str, ...] = ()
    process_id: int | None = None


class AnnotationToolLauncher(ABC):
    @abstractmethod
    def launch(self, package: AnnotationWorkPackage) -> AnnotationToolLaunchResult:
        """Launch without waiting for process exit or modifying package content."""


class ManualAnnotationToolLauncher(AnnotationToolLauncher):
    def launch(self, package: AnnotationWorkPackage) -> AnnotationToolLaunchResult:
        return AnnotationToolLaunchResult(
            AnnotationToolLaunchStatus.MANUAL,
            f"Open the package with an annotation tool: {package.root}",
        )


class SubprocessAnnotationToolLauncher(AnnotationToolLauncher):
    """Launch a controlled executable using argv and shell=False."""

    def __init__(
        self,
        executable: str | Path,
        *,
        argument_builder: Callable[[AnnotationWorkPackage], Sequence[str]] | None = None,
        popen: Callable[..., Any] | None = None,
    ) -> None:
        self._executable = Path(executable).resolve()
        self._argument_builder = argument_builder or (
            lambda package: (str(package.root / "source_images"), str(package.root / "class_mapping.json"))
        )
        self._popen = popen or subprocess.Popen

    def launch(self, package: AnnotationWorkPackage) -> AnnotationToolLaunchResult:
        if not self._executable.is_file():
            return AnnotationToolLaunchResult(
                AnnotationToolLaunchStatus.UNAVAILABLE,
                f"Annotation tool is unavailable: {self._executable}",
            )
        argv = (str(self._executable), *(str(value) for value in self._argument_builder(package)))
        try:
            process = self._popen(argv, shell=False, cwd=str(package.root))
        except (OSError, ValueError) as exc:
            return AnnotationToolLaunchResult(
                AnnotationToolLaunchStatus.FAILED,
                f"Could not launch annotation tool: {exc}",
                argv=argv,
            )
        process_id = getattr(process, "pid", None)
        return AnnotationToolLaunchResult(
            AnnotationToolLaunchStatus.LAUNCHED,
            "Annotation tool launched; processing execution did not wait for it.",
            argv=argv,
            process_id=int(process_id) if process_id is not None else None,
        )
