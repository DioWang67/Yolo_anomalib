from unittest.mock import MagicMock

from PyQt5.QtCore import Qt

from app.gui.training_batch_dialog import (
    CATEGORY_ROLE,
    TrainingBatchDialog,
)


def _entries():
    return [
        (
            3,
            {
                "review_label": "confirmed_ng",
                "timestamp": "2026-07-15T10:00:00",
                "training_selected": "1",
            },
        ),
        (
            7,
            {
                "review_label": "false_negative",
                "timestamp": "2026-07-15T10:01:00",
                "training_selected": "0",
            },
        ),
        (
            9,
            {
                "review_label": "uncertain",
                "timestamp": "2026-07-15T10:02:00",
                "training_selected": "1",
            },
        ),
    ]


def test_batch_overview_restores_selection_and_counts(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW")
    qtbot.addWidget(dialog)

    assert dialog.selected_indices() == {3}
    assert "本次補訓 1 張" in dialog.summary_label.text()
    assert "已排除 1 張" in dialog.summary_label.text()
    assert "暫不送訓 1 張" in dialog.summary_label.text()


def test_batch_overview_filters_direct_and_annotation_rows(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW")
    qtbot.addWidget(dialog)

    dialog.filter_combo.setCurrentIndex(dialog.filter_combo.findData("annotation"))

    visible_categories = {
        dialog.thumbnail_list.item(index).data(CATEGORY_ROLE)
        for index in range(dialog.thumbnail_list.count())
        if not dialog.thumbnail_list.item(index).isHidden()
    }
    assert visible_categories == {"annotation"}


def test_batch_overview_can_exclude_all_visible_rows(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW")
    qtbot.addWidget(dialog)

    dialog.filter_combo.setCurrentIndex(dialog.filter_combo.findData("direct"))
    dialog._set_visible_check_state(Qt.Unchecked)

    assert dialog.selected_indices() == set()


def test_batch_overview_never_selects_on_hold_rows(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW")
    qtbot.addWidget(dialog)
    hold_item = dialog.thumbnail_list.item(2)

    assert hold_item.data(CATEGORY_ROLE) == "hold"
    assert hold_item.checkState() == Qt.Unchecked
    assert not bool(hold_item.flags() & Qt.ItemIsUserCheckable)


def test_submit_mode_rejects_empty_batch(qtbot, monkeypatch):
    entries = _entries()
    for _index, row in entries:
        row["training_selected"] = "0"
    dialog = TrainingBatchDialog(entries, language="zh_TW", submit_mode=True)
    qtbot.addWidget(dialog)
    warning = MagicMock()
    monkeypatch.setattr(
        "app.gui.training_batch_dialog.QMessageBox.warning",
        warning,
    )

    dialog._confirm()

    warning.assert_called_once()
    assert dialog.result() == 0


def test_queue_mode_direct_action_returns_only_direct_rows(qtbot):
    entries = _entries()
    entries[1][1]["training_selected"] = "1"
    dialog = TrainingBatchDialog(entries, language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.direct_action_button, Qt.LeftButton)

    assert dialog.selected_action == "direct"
    assert dialog.action_selected_indices() == {3}


def test_queue_mode_annotation_action_returns_only_annotation_rows(qtbot):
    entries = _entries()
    entries[1][1]["training_selected"] = "1"
    dialog = TrainingBatchDialog(entries, language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.annotation_action_button, Qt.LeftButton)

    assert dialog.selected_action == "annotation"
    assert dialog.action_selected_indices() == {7}


def test_queue_mode_can_save_selection_without_submitting(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    dialog._set_visible_check_state(Qt.Unchecked)
    save_button = next(
        button
        for button in dialog.findChildren(type(dialog.confirm_button))
        if button.text() == "儲存清單"
    )
    qtbot.mouseClick(save_button, Qt.LeftButton)

    assert dialog.selected_action is None
    assert dialog.selected_indices() == set()
