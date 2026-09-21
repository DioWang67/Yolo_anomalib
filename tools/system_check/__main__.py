"""Allow ``python -m tools.system_check`` and serve as the frozen entry point.

Adds the repository root to ``sys.path`` when run from a source checkout, so
the package's absolute ``tools.system_check.*`` imports resolve without an
install step.
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

# Must run before anything that may spawn a worker. ONNX Runtime and the BLAS
# library behind numpy both start child processes on Windows, and in a frozen
# build each child re-executes this executable with ``--multiprocessing-fork``
# on its command line. Without this call those children reach argparse, print a
# usage error, and litter the operator's console mid-report. GUI.py carries the
# same guard for the same reason.
multiprocessing.freeze_support()

if __package__ in {None, ""}:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.system_check.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
