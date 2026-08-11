from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "relative_script",
    (
        "tools/collect_review_cases.py",
        "tools/review_training_data.py",
        "tools/inspection_sync_admin.py",
        "tools/maintain_inspection_data.py",
        "tools/pcba_pilot.py",
        "tools/production_preflight.py",
        "tools/rebuild_inspection_database.py",
        "tools/restore_inspection_database.py",
    ),
)
def test_maintenance_cli_supports_direct_help(
    relative_script: str,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(project_root / relative_script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
