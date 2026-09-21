"""Standalone hardware / environment / performance preflight for yolo11_inference.

This package is deliberately decoupled from ``core``: nothing here imports the
production application, so the checker can run from a small standalone
executable on a station that has neither the source checkout nor a Python
installation. The one piece of production logic it must reproduce — letterbox
preprocessing — lives in :mod:`tools.system_check.benchmark.preprocess` and is
pinned to ``core.utils.ImageUtils.letterbox`` by a drift test.

The checker is read-only. It never writes outside its report directory and
never installs, downloads, or reconfigures anything.
"""

from __future__ import annotations

__all__ = ["REPORT_SCHEMA_VERSION"]

# Bump when the JSON report layout changes in a way consumers must notice.
REPORT_SCHEMA_VERSION = 1
