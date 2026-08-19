from unittest.mock import MagicMock

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QLabel, QScrollArea

from app.gui.hover_help import HoverHelpBadge
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
    selected_item = dialog.thumbnail_list.item(0)
    excluded_item = dialog.thumbnail_list.item(1)
    assert selected_item.text().startswith("✓ 已加入待送")
    assert selected_item.background().color().name() == "#9fe0b2"
    assert selected_item.font().bold() is True
    assert excluded_item.text().startswith("○ 未加入待送")
    assert "已選擇 1 張" in dialog.summary_label.text()
    assert "已排除 1 張" in dialog.summary_label.text()
    assert "暫不送訓 1 張" in dialog.summary_label.text()


def test_queue_guidance_is_available_as_hover_help(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    title_help = dialog.findChild(HoverHelpBadge, "batchTitleHelp")
    selection_help = dialog.findChild(HoverHelpBadge, "queueSelectionHelp")
    route_help = dialog.findChild(HoverHelpBadge, "routeGuidanceHelp")
    visible_copy = {
        label.text()
        for label in dialog.findChildren(QLabel)
        if label.isVisibleTo(dialog)
    }

    assert title_help is not None
    assert "先勾選要處理的影像" in title_help.toolTip()
    assert selection_help is not None
    assert "雙擊照片可放大" in selection_help.toolTip()
    assert route_help is not None
    assert "不會混送" in route_help.toolTip()
    assert all("系統已依判定結果分好類" not in text for text in visible_copy)


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


def test_queue_mode_keeps_long_action_panel_scrollable(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    route_scroll = dialog.findChild(QScrollArea, "RouteActionScroll")

    assert route_scroll is not None
    assert route_scroll.widgetResizable() is True
    assert route_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff


def test_queue_mode_portable_action_includes_direct_and_annotation(qtbot):
    entries = _entries()
    entries[1][1]["training_selected"] = "1"
    dialog = TrainingBatchDialog(entries, language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.portable_action_button, Qt.LeftButton)

    assert dialog.selected_action == "portable"
    assert dialog.action_selected_indices() == {3, 7}


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
        if button.text() == "儲存待送清單並返回"
    )
    qtbot.mouseClick(save_button, Qt.LeftButton)

    assert dialog.selected_action is None
    assert dialog.selected_indices() == set()


def test_queue_mode_can_explicitly_remove_multiple_selected_cases(qtbot):
    entries = _entries()
    entries[1][1]["training_selected"] = "1"
    dialog = TrainingBatchDialog(entries, language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)
    first = dialog.thumbnail_list.item(0)
    second = dialog.thumbnail_list.item(1)
    first.setSelected(True)
    second.setSelected(True)

    assert dialog.remove_selected_button is not None
    assert dialog.remove_selected_button.isEnabled() is True

    dialog._remove_selected_from_queue()

    assert dialog.result() == QDialog.Accepted
    assert dialog.selected_action is None
    assert dialog.selected_indices() == set()


def test_queue_mode_routes_color_only_feedback_to_calibration(qtbot):
    entries = _entries()
    entries.append(
        (
            12,
            {
                "review_label": "color_false_reject",
                "action_route": "color",
                "timestamp": "2026-07-15T10:03:00",
                "training_selected": "1",
            },
        )
    )
    dialog = TrainingBatchDialog(entries, language="zh_TW", queue_mode=True)
    qtbot.addWidget(dialog)

    qtbot.mouseClick(dialog.color_action_button, Qt.LeftButton)

    assert dialog.selected_action == "color"
    assert dialog.action_selected_indices() == {12}


def test_history_mode_is_read_only_and_keeps_category_summary(qtbot):
    dialog = TrainingBatchDialog(_entries(), language="zh_TW", history_mode=True)
    qtbot.addWidget(dialog)

    assert dialog.windowTitle() == "已送訓照片（唯讀）"
    assert dialog.confirm_button is None
    assert dialog.direct_action_button is None
    assert dialog.selected_indices() == set()
    assert all(
        not bool(dialog.thumbnail_list.item(index).flags() & Qt.ItemIsUserCheckable)
        for index in range(dialog.thumbnail_list.count())
    )
    assert "本批共 3 張" in dialog.summary_label.text()
