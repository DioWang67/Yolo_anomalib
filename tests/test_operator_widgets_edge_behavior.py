from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtGui import QImage

import app.gui.widgets as widgets
from core.types import DetectionItem, DetectionResult


@pytest.mark.parametrize(
    ("status", "expected"),
    (
        ("PASS", "PASS"),
        ("FAIL", "FAIL"),
        ("DETECTION_FAIL", "DETECTION FAIL"),
        ("ERROR", "ERROR"),
        ("INFERENCE_ERROR", "INFERENCE ERROR"),
        ("", "READY"),
        ("custom", "CUSTOM"),
    ),
)
def test_big_status_label_renders_all_operator_states(qtbot, status: str, expected: str) -> None:
    label = widgets.BigStatusLabel()
    qtbot.addWidget(label)
    label.set_status(status)
    assert label.text() == expected


def test_auto_phase_banner_preserves_results_and_controls_pulse(qtbot) -> None:
    banner = widgets.AutoPhaseBanner()
    qtbot.addWidget(banner)

    banner.activate("en")
    assert not banner.isHidden()
    waiting = banner.text()
    banner.set_phase("SHOW_RESULT", "en")
    banner.set_phase("unknown", "en")
    assert banner.text() == waiting

    banner.set_phase("INSPECTING", "en")
    assert banner._pulse_timer.isActive()
    banner._tick_pulse()
    assert banner._pulse_on
    banner.set_result("DETECTION_FAIL", "en")
    assert "DETECTION FAIL" in banner.text()
    assert not banner._pulse_timer.isActive()
    banner.set_result("", "en")
    assert banner.text()

    banner._pulse_color = None
    banner._pulse_timer.start()
    banner._tick_pulse()
    assert not banner._pulse_timer.isActive()
    banner.deactivate()
    assert banner.isHidden()


def _position_item(label: str, status: str, dx: float, dy: float) -> DetectionItem:
    return DetectionItem(
        label=label,
        confidence=0.9,
        bbox_xyxy=(1, 2, 3, 4),
        metadata={
            "source_index": 0 if label == "A" else 1,
            "position_status": status,
            "position_error": 8.0,
            "position_offset": {"dx": dx, "dy": dy},
        },
    )


def _rich_failure() -> DetectionResult:
    return DetectionResult(
        status="DETECTION_FAIL",
        product="Cable1",
        area="A",
        inference_type="yolo",
        ckpt_path="models/v9/best.pt",
        latency=0.012,
        anomaly_score=0.75,
        error="inspection rule failed",
        missing_items=["M1", "M2", "M3", "M4"],
        over_items=["O1"],
        unexpected_items=["U1"],
        items=[
            _position_item("A", "WRONG", 8.0, 4.0),
            _position_item("B", "WRONG", 8.5, 4.2),
        ],
        sequence_check={
            "is_ok": False,
            "reason": "order_mismatch",
            "expected": ["A", "B"],
            "observed": ["B", "A"],
        },
        color_check={
            "is_ok": False,
            "items": [
                {
                    "index": 0,
                    "class_name": "LED-A",
                    "best_color": "Red",
                    "diff": 0.8,
                    "threshold": 0.2,
                    "is_ok": False,
                },
                {
                    "index": "invalid",
                    "class_name": "ignored",
                    "is_ok": False,
                },
            ],
        },
        metadata={
            "decision": {"reasons": ["BOARD_ALIGNMENT", "COLOR_MISMATCH"]},
            "alignment_quality": {
                "enabled": True,
                "is_ok": False,
                "issues": ["alignment_dx_out_of_range", "custom_issue"],
                "dx": "8.5",
                "dy": "bad",
                "observed_source_count": 2,
                "required_source_count": 3,
            },
            "duplicate_filter": {
                "status": "suppressed",
                "proposed_suppressions": [],
                "suppressions": [
                    {
                        "suppressed_index": 1,
                        "kept_index": 0,
                        "iou": 0.91,
                        "verified_class": "LED-A",
                    },
                    {"suppressed_index": "bad"},
                ],
            },
        },
    )


def test_fail_reason_label_explains_backend_and_detection_failures(qtbot) -> None:
    label = widgets.FailReasonLabel()
    qtbot.addWidget(label)
    label.set_language("zh_TW")

    label.update_from_result(DetectionResult(status="PASS"))
    assert label.text() == ""
    label.update_from_result(DetectionResult(status="INFERENCE_ERROR", error="ORT unavailable"))
    assert label.text() == "ORT unavailable"

    label.update_from_result(_rich_failure())
    assert "inspection rule failed" in label.text()
    assert "M1" in label.text()
    assert "LED-A" in label.text()
    assert "avg dx" in label.text()

    unknown = DetectionResult(status="FAIL")
    label.update_from_result(unknown)
    assert label.text()
    label.clear_reason()
    assert label.text() == ""


def test_guidance_card_renders_language_and_status_specific_actions(qtbot) -> None:
    card = widgets.OperatorGuidanceCard()
    qtbot.addWidget(card)
    card.set_language("en")
    card.update_from_result(DetectionResult(status="PASS"))
    assert card._headline.text() == "PASS"
    card.update_from_result(DetectionResult(status="INFERENCE_ERROR", error="backend offline"))
    assert "backend offline" in card._details.text()
    card.update_from_result(
        DetectionResult(
            status="DETECTION_FAIL",
            missing_items=["bolt"],
            unexpected_items=["wire"],
        )
    )
    assert "bolt" in card._details.text()
    assert "wire" in card._details.text()

    card.set_language("zh_TW")
    assert card._headline.text()
    card.show_message("custom", "retry", "unsupported", ["detail"])
    assert card._headline.text() == "custom"
    assert "detail" in card._details.text()


def test_session_stats_alerts_resets_and_ignores_non_verdicts(qtbot) -> None:
    stats = widgets.SessionStatsWidget()
    qtbot.addWidget(stats)
    alerts: list[int] = []
    stats.consecutive_fail_reached.connect(alerts.append)

    stats.record_result("UNKNOWN")
    stats.record_result("FAIL")
    stats.record_result("DETECTION_FAIL")
    stats.record_result("FAIL")
    assert stats.consecutive_fails == 3
    assert alerts == [3]
    assert stats._yield_lbl.text() == "0.0%"

    stats.record_result("PASS")
    assert stats.consecutive_fails == 0
    assert stats._yield_lbl.text() == "25.0%"
    stats.set_language("zh_TW")
    stats.reset_session()
    assert stats._pass_lbl.text() == "0"
    assert stats._fail_lbl.text() == "0"


def test_image_viewer_handles_disk_live_and_stale_images(
    qtbot,
    monkeypatch,
    tmp_path: Path,
) -> None:
    viewer = widgets.ImageViewer("fixture")
    qtbot.addWidget(viewer)
    viewer.set_language("zh_TW")
    viewer.set_title("board")
    viewer.set_image(str(tmp_path / "missing.png"))
    assert "Unable" in viewer.text()

    image_path = tmp_path / "valid.png"
    image = QImage(8, 6, QImage.Format_RGB888)
    image.fill(0x00FF00)
    assert image.save(str(image_path))
    viewer.set_image(str(image_path))
    token = viewer._load_token
    viewer._load_and_display(str(image_path), token - 1)
    viewer._load_and_display(str(image_path), token)
    assert viewer.pixmap() is not None and not viewer.pixmap().isNull()

    invalid_path = tmp_path / "invalid.png"
    invalid_path.write_text("not an image", encoding="utf-8")
    viewer._load_and_display(str(invalid_path), viewer._load_token)
    assert "Unable" in viewer.text()

    viewer.display_image(QImage(4, 4, QImage.Format_RGB888))
    viewer.display_image(np.zeros((4, 5, 3), dtype=np.uint8))
    assert viewer.pixmap() is not None
    viewer.display_image(np.array([], dtype=np.uint8))
    assert "Unable" in viewer.text()

    monkeypatch.setattr(widgets.cv2, "cvtColor", lambda *_args: (_ for _ in ()).throw(ValueError("bad frame")))
    viewer.display_image(np.zeros((2, 2, 3), dtype=np.uint8))
    assert "Unable" in viewer.text()
    viewer.clear()
    assert viewer._last_image_path is None


def test_result_display_renders_all_evidence_sections(qtbot) -> None:
    display = widgets.ResultDisplayWidget()
    qtbot.addWidget(display)
    result = _rich_failure()

    display.update_result(result)
    text = display._result_text.toPlainText()
    assert "Cross-class Duplicates" in text
    assert "Alignment Quality" in text
    assert "Decision Reasons" in text
    assert "LED-A" in text
    assert "A | conf=0.900" in text

    display.set_language("zh_TW")
    assert display._result_text.toPlainText()

    for status in ("reported", "blocked_policy"):
        result.metadata["duplicate_filter"] = {
            "status": status,
            "proposed_suppressions": [
                {
                    "suppressed_index": 1,
                    "kept_index": 0,
                    "iou": 0.8,
                    "verified_class": "LED-A",
                }
            ],
        }
        display.update_result(result)

    result.metadata["duplicate_filter"] = {"status": "clean"}
    result.sequence_check = {
        "is_ok": False,
        "reason": "length_mismatch",
        "expected": ["A"],
        "observed": [],
    }
    result.color_check = {"is_ok": True, "items": []}
    result.metadata["alignment_quality"] = {
        "enabled": True,
        "is_ok": True,
        "issues": [],
        "dx": 0,
        "dy": 0,
    }
    display.update_result(result)
    assert "PASS" in display._result_text.toPlainText()

    result.sequence_check = {"is_ok": False, "reason": "custom"}
    result.color_check = None
    result.metadata = {}
    result.items = [
        DetectionItem(
            label="OK-item",
            confidence=0.8,
            bbox_xyxy=(0, 0, 1, 1),
            metadata={"position_status": "CORRECT"},
        )
    ]
    display.update_result(result)
    assert "OK-item" in display._result_text.toPlainText()

    for status in ("PASS", "INFERENCE_ERROR"):
        display.set_language("en")
        display.update_result(DetectionResult(status=status))
        assert display._result_text.toPlainText()


@pytest.mark.parametrize(
    ("item", "expected"),
    (
        ({"class": "Green"}, "Green"),
        ({"index": -1}, "frame"),
        ({"index": 2}, "item"),
    ),
)
def test_color_item_labels_have_stable_fallbacks(item: dict[str, object], expected: str) -> None:
    assert expected.lower() in widgets._color_item_label(item, "en").lower()


def test_alignment_issue_labels_localize_known_codes_and_preserve_unknown() -> None:
    assert "Horizontal" in widgets._alignment_issue_label("alignment_dx_out_of_range", "en")
    assert widgets._alignment_issue_label("custom", "en") == "custom"
    assert widgets._alignment_issue_label("custom", "zh_TW") == "custom"
