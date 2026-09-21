"""Individual preflight checks.

Each module exposes ``check(context) -> list[CheckResult]`` and is registered
in :data:`CHECK_REGISTRY`. The orchestrator wraps every entry in
``run_check``, so a module may raise freely without endangering the run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tools.system_check.checks import (
    camera_check,
    cpu_check,
    dependency_check,
    disk_check,
    gpu_check,
    memory_check,
    network_check,
    os_check,
    runtime_check,
)
from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult

CheckFunc = Callable[[AppContext], Sequence[CheckResult]]

#: ``(check_id, title, function)`` in report order.
CHECK_REGISTRY: tuple[tuple[str, str, CheckFunc], ...] = (
    ("os", "Operating system", os_check.check),
    ("cpu", "Processor", cpu_check.check),
    ("memory", "Memory", memory_check.check),
    ("disk", "Storage", disk_check.check),
    ("gpu", "Graphics", gpu_check.check),
    ("dependency", "Python packages", dependency_check.check),
    ("runtime", "Native runtime", runtime_check.check),
    ("camera", "Camera", camera_check.check),
    ("network", "Network", network_check.check),
)

__all__ = ["CHECK_REGISTRY", "CheckFunc"]
