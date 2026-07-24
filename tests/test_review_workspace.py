from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QComboBox, QLineEdit, QWidget

from app.gui.review_workspace import (
    ROW_INDEX_ROLE,
    ReviewImageViewer,
    ReviewThumbnailPanel,
    build_review_details,
    build_review_list_item,
    is_text_input_focus,
)


def _reviewed_record(**overrides):
    return {
        "status": "FAIL",
        "review_selected": "1",
        "review_outcome": "pass",
        "review_label": "false_positive",
        "product_verdict": "ok",
        "detection_verdict": "false_positive",
        "color_verdict": "not_applicable",
        "action_route": "yolo",
        "training_selected": "1",
        **overrides,
    }


def test_thumbnail_panel_filters_and_emits_clicked_row(qtbot):
    panel = ReviewThumbnailPanel(language="en")
    qtbot.addWidget(panel)
    items = [
        build_review_list_item(0, {"status": "FAIL"}, language="en"),
        build_review_list_item(1, _reviewed_record(), language="en"),
        build_review_list_item(
            2,
            {
                "status": "FAIL",
                "review_selected": "1",
                "review_outcome": "skip",
                "skip_reason": "image_quality_issue",
                "training_selected": "1",
            },
            language="en",
        ),
    ]
    selected = []
    panel.current_changed.connect(selected.append)
    panel.set_items(items, current_index=0)

    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("completed"))
    assert panel.displayed_indices() == {1}
    assert selected[-1] == 1
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("error"))
    assert panel.displayed_indices() == {2}
    panel.filter_combo.setCurrentIndex(panel.filter_combo.findData("all"))
    target = next(
        panel.list_widget.item(index)
        for index in range(panel.list_widget.count())
        if panel.list_widget.item(index).data(ROW_INDEX_ROLE) == 1
    )
    panel.list_widget.setCurrentItem(target)
    assert selected[-1] == 1


def test_thumbnail_panel_ignores_queued_timer_after_list_is_deleted(qtbot):
    panel = ReviewThumbnailPanel(language="en")
    qtbot.addWidget(panel)
    panel.set_items(
        [build_review_list_item(0, {"status": "FAIL"}, language="en")],
        current_index=0,
    )

    panel.list_widget.deleteLater()
    qtbot.wait(10)


def test_image_viewer_distinguishes_missing_corrupt_and_loaded(tmp_path, qtbot):
    viewer = ReviewImageViewer(language="en")
    qtbot.addWidget(viewer)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not-an-image")
    valid = tmp_path / "valid.png"
    image = QImage(24, 16, QImage.Format_RGB32)
    image.fill(Qt.white)
    assert image.save(str(valid))

    viewer.set_images(original_path=tmp_path / "missing.png", overlay_path=corrupt)
    viewer.set_mode("original")
    assert viewer.image_state == "loading"
    qtbot.waitUntil(lambda: viewer.image_state != "loading")
    assert viewer.image_state == "missing"
    assert "does not exist" in viewer.image_label.text()
    viewer.set_mode("overlay")
    qtbot.waitUntil(lambda: viewer.image_state != "loading")
    assert viewer.image_state == "corrupt"
    assert "corrupt" in viewer.image_label.text()
    viewer.set_images(original_path=valid, overlay_path=valid)
    qtbot.waitUntil(lambda: viewer.image_state != "loading")
    assert viewer.image_state == "loaded"
    viewer.zoom_in()
    assert viewer.image_label.pixmap() is not None
    viewer.toggle_mode()
    assert viewer.mode == "original"


def test_review_details_exposes_ai_metadata_and_typed_semantics():
    details = build_review_details(
        _reviewed_record(
            product="Cable1",
            area="A",
            machine_id="M01",
            camera_id="CAM01",
            model_version="yolo11_v8",
            detections_json=(
                '[{"class":"scratch","confidence":0.91,"bbox":[1,2,3,4]}]'
            ),
        ),
        language="en",
    )

    assert "scratch" in details.ai_summary
    assert "0.910" in details.ai_summary
    assert "yolo11_v8" in details.ai_summary
    assert "M01" in details.metadata_summary
    assert "AI correctness: false_positive" in details.semantics_summary


def test_text_focus_guard_only_matches_editing_widgets(qtbot):
    line = QLineEdit()
    combo = QComboBox()
    plain = QWidget()
    for widget in (line, combo, plain):
        qtbot.addWidget(widget)

    assert is_text_input_focus(line)
    assert is_text_input_focus(combo)
    assert not is_text_input_focus(plain)
