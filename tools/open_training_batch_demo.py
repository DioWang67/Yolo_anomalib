"""Open a safe ten-image demo of the operator retraining queue."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox

# A file-path launch sets ``sys.path[0]`` to ``tools`` instead of the project
# root. Resolve it from this file so desktop shortcuts and batch files behave
# the same as ``python -m tools.open_training_batch_demo``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gui.retraining_settings_dialog import (  # noqa: E402
    RetrainingSettingsDialog,
)
from app.gui.training_batch_dialog import TrainingBatchDialog  # noqa: E402


def load_demo_rows(demo_root: Path) -> list[dict[str, str]]:
    """Load and validate the isolated demo manifest."""
    manifest_path = demo_root / "demo_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if len(rows) != 10:
        raise ValueError(f"Demo manifest must contain exactly 10 rows: {manifest_path}")
    for row in rows:
        image_path = (demo_root / "images" / str(row.get("file") or "")).resolve()
        if not image_path.is_file() or not image_path.is_relative_to(demo_root.resolve()):
            raise FileNotFoundError(f"Demo image not found: {image_path}")
        row["product"] = "Cable1"
        row["area"] = "A"
        row["annotated_path"] = str(image_path)
        row["preprocessed_path"] = str(image_path)
        row["original_path"] = str(image_path)
    return rows


def _active_entries(
    rows: list[dict[str, str]],
) -> list[tuple[int, dict[str, str]]]:
    return [
        (index, row)
        for index, row in enumerate(rows)
        if str(row.get("training_selected") or "1") != "0"
    ]


def _apply_demo_result(
    rows: list[dict[str, str]],
    dialog: TrainingBatchDialog,
) -> tuple[str, int]:
    selected_indices = dialog.selected_indices()
    action_indices = dialog.action_selected_indices()
    action = dialog.selected_action
    current_indices = {index for index, _row in dialog.entries}
    for index in current_indices:
        rows[index]["training_selected"] = "1" if index in selected_indices else "0"
    for index in action_indices:
        rows[index]["training_selected"] = "0"
    action_labels = {
        "direct": "直接訓練",
        "annotation": "補標後訓練",
        "color": "顏色校正",
        None: "更新待送清單",
    }
    affected_count = len(action_indices) if action else len(current_indices - selected_indices)
    return action_labels[action], affected_count


def main() -> int:
    demo_root = PROJECT_ROOT / "demo" / "operator_queue"
    rows = load_demo_rows(demo_root)
    _application = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.information(
        None,
        "安全示範模式",
        "這 10 張是隔離的舊圖片副本。\n\n"
        "所有送出、移除與勾選都只在本次示範有效，不會啟動補訓或修改正式資料。",
    )
    while True:
        entries = _active_entries(rows)
        if not entries:
            QMessageBox.information(
                None,
                "安全示範模式",
                "示範待送清單已清空。重新開啟 open_demo.bat 可重置 10 張圖片。",
            )
            return 0
        dialog = TrainingBatchDialog(
            entries,
            language="zh_TW",
            queue_mode=True,
        )
        dialog.setWindowTitle(f"安全示範｜{dialog.windowTitle()}")
        if dialog.exec_() != QDialog.Accepted:
            return 0
        settings_summary = ""
        if dialog.selected_action in {"direct", "annotation"}:
            settings_dialog = RetrainingSettingsDialog(
                len(dialog.action_selected_indices()),
                parent=dialog,
            )
            if settings_dialog.exec_() != QDialog.Accepted:
                continue
            options = settings_dialog.options()
            settings_summary = (
                f"\n設定：{options.epochs} Epochs／增強 {options.augmentations_per_image}／"
                f"Batch {options.batch}／{options.imgsz}px"
            )
        action_label, affected_count = _apply_demo_result(rows, dialog)
        QMessageBox.information(
            None,
            "安全示範結果",
            f"動作：{action_label}\n影響：{affected_count} 張{settings_summary}\n\n"
            "這是示範模式，沒有啟動補訓，也沒有修改正式資料。",
        )


if __name__ == "__main__":
    raise SystemExit(main())
