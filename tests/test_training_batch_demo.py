import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from app.gui.training_batch_dialog import TrainingBatchDialog
from core.station_data import load_station_data_paths
from tools.open_training_batch_demo import (
    _active_entries,
    _apply_demo_result,
    load_demo_rows,
)


def _demo_root() -> Path:
    return Path(__file__).resolve().parents[1] / "demo" / "operator_queue"


def _load_demo_rows_or_skip() -> list[dict[str, str]]:
    try:
        return load_demo_rows(_demo_root())
    except FileNotFoundError:
        pytest.skip("operator demo images are station-local and not in source control")


def test_demo_contains_ten_isolated_images_in_three_routes():
    demo_root = _demo_root()
    rows = _load_demo_rows_or_skip()

    assert len(rows) == 10
    assert all(
        Path(row["annotated_path"]).is_relative_to(demo_root.resolve())
        for row in rows
    )
    assert sum(row["review_label"] == "confirmed_ng" for row in rows) == 3
    assert sum(row["action_route"] == "color" for row in rows) == 2
    project_root = demo_root.parents[1]
    station_paths = load_station_data_paths(project_root)
    for row in rows:
        demo_image = Path(row["original_path"])
        source_original = station_paths.relocate_legacy_result_path(
            project_root / row["source_original"]
        )
        with Image.open(demo_image) as image:
            assert image.size == (3072, 2048)
        assert source_original.is_file()
        assert _sha256(demo_image) == _sha256(source_original)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_demo_script_imports_when_started_outside_project_root(tmp_path):
    script = _demo_root().parents[1] / "tools" / "open_training_batch_demo.py"
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import runpy; runpy.run_path({str(script)!r}, run_name='import_test')",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_demo_direct_submission_only_removes_direct_cases(qtbot):
    rows = _load_demo_rows_or_skip()
    dialog = TrainingBatchDialog(
        _active_entries(rows),
        language="zh_TW",
        queue_mode=True,
    )
    qtbot.addWidget(dialog)

    dialog._confirm_queue_action("direct")
    action_label, affected_count = _apply_demo_result(rows, dialog)

    assert action_label == "直接訓練"
    assert affected_count == 3
    assert len(_active_entries(rows)) == 7
