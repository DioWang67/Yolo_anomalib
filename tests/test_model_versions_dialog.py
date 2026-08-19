from __future__ import annotations

from datetime import datetime
from unittest.mock import Mock

from PyQt5.QtWidgets import QMessageBox

from app.gui.model_versions_dialog import ModelVersionsDialog
from core.services.model_version_registry import ModelVersionRecord


def _record(tmp_path, filename: str, *, current: bool, date: str):
    path = tmp_path / filename
    path.write_bytes(filename.encode())
    return ModelVersionRecord(
        product="PCBA1",
        area="A",
        model_type="yolo",
        version="1.0.1" if current else "1.0.0",
        weight_path=path,
        is_current=current,
        trained_at=datetime.fromisoformat(date),
        deployed_at=datetime.fromisoformat(date),
        activated_at=None,
        training_time_inferred=False,
        file_size=path.stat().st_size,
    )


def test_dialog_filters_versions_by_training_date(tmp_path, qtbot) -> None:
    records = [
        _record(tmp_path, "new.onnx", current=True, date="2026-07-15T09:00:00+08:00"),
        _record(tmp_path, "old.onnx", current=False, date="2026-07-01T09:00:00+08:00"),
    ]
    registry = Mock()
    registry.list_versions.return_value = records
    dialog = ModelVersionsDialog(registry, language="zh_TW")
    qtbot.addWidget(dialog)

    dialog.date_filter.setCurrentIndex(
        dialog.date_filter.findData("2026-07-01")
    )

    assert dialog.table.rowCount() == 1
    assert dialog.table.item(0, 9).text() == "old.onnx"


def test_dialog_blocks_switch_while_inspection_is_running(
    tmp_path, qtbot, monkeypatch
) -> None:
    historical = _record(
        tmp_path, "old.onnx", current=False, date="2026-07-01T09:00:00+08:00"
    )
    registry = Mock()
    registry.list_versions.return_value = [historical]
    warning = Mock()
    monkeypatch.setattr(
        "app.gui.model_versions_dialog.QMessageBox.warning", warning
    )
    dialog = ModelVersionsDialog(
        registry,
        is_inspection_running=lambda: True,
    )
    qtbot.addWidget(dialog)
    dialog.table.selectRow(0)

    dialog._activate_selected()

    registry.activate.assert_not_called()
    warning.assert_called_once()


def test_dialog_activates_selected_legacy_version_after_confirmation(
    tmp_path, qtbot, monkeypatch
) -> None:
    historical = _record(
        tmp_path, "old.onnx", current=False, date="2026-07-01T09:00:00+08:00"
    )
    activated = ModelVersionRecord(
        **{**historical.__dict__, "is_current": True}
    )
    registry = Mock()
    registry.list_versions.return_value = [historical]
    registry.activate.return_value = activated
    on_activated = Mock()
    monkeypatch.setattr(
        "app.gui.model_versions_dialog.QMessageBox.question",
        Mock(return_value=QMessageBox.Yes),
    )
    monkeypatch.setattr(
        "app.gui.model_versions_dialog.QMessageBox.information", Mock()
    )
    dialog = ModelVersionsDialog(
        registry,
        is_inspection_running=lambda: False,
        on_activated=on_activated,
    )
    qtbot.addWidget(dialog)
    dialog.table.selectRow(0)

    dialog._activate_selected()

    registry.activate.assert_called_once_with(historical, allow_incomplete=True)
    on_activated.assert_called_once_with(activated)
