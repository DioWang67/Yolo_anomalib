from __future__ import annotations

import copy

from PyQt5.QtWidgets import QLabel
from test_golden_sample_dialog import build_baseline, dialog, publish_samples, worker

from app.gui.golden_sample_view import CropView, GoldenEvidenceView
from core.services.golden_sample import evaluate_readings, write_json_atomic
from core.services.golden_sample_preview import build_preview, preview_or_empty


def actual_reference(tmp_path):
    capture = worker(tmp_path)
    publish_samples(capture, 5)
    return build_baseline(capture, delta_e=3, repeatability=1)


def test_preview_pairs_real_crops_and_fixed_roi(tmp_path, qtbot):
    reference = actual_reference(tmp_path)
    result = evaluate_readings(reference["readings"][:3], reference, reference["identity"])
    previews = build_preview(reference, result)
    assert len(previews) == 1
    preview = previews[0]
    assert preview["roi"] == reference["positions"][0]["measurement_bbox"]
    assert preview["reference_image"].shape == (32, 32, 3)
    assert preview["current_image"].shape == (32, 32, 3)
    assert preview["heatmap"] == [[0.0] * 4 for _ in range(4)]
    view = GoldenEvidenceView()
    qtbot.addWidget(view)
    view.resize(950, 500)
    view.show_previews(previews)
    view.show()
    assert not view.grab().isNull()
    assert len(view.findChildren(CropView)) == 2
    assert any("穩定" in label.text() for label in view.findChildren(QLabel))
    previews[0]["row"]["reasons"] = ["顏色偏移超限"]
    previews[0]["heatmap"][0][0] = 10
    view.show_previews(previews)
    assert not view.grab().isNull()
    assert any("需處理" in label.text() for label in view.findChildren(QLabel))


def test_missing_preview_is_explicit_and_does_not_change_report(tmp_path, qtbot):
    reference = actual_reference(tmp_path)
    original = copy.deepcopy(reference)
    reference["readings"][0]["positions"][0]["crop_path"] = str(tmp_path / "missing.png")
    previews = build_preview(reference)
    assert previews[0]["reference_image"] is None
    assert previews[0]["current_image"] is None
    view = GoldenEvidenceView()
    qtbot.addWidget(view)
    view.show_previews(previews)
    view.show()
    assert not view.grab().isNull()
    assert preview_or_empty({}, {}) == []
    assert preview_or_empty(reference, {"readings": [{"positions": []}]}) == []
    assert reference["positions"] == original["positions"]


def test_saved_settings_autofill_and_maintenance_is_collapsed(tmp_path, qtbot):
    reference = actual_reference(tmp_path)
    reference["repeatability_limit"] = 2.5
    view = dialog(tmp_path, qtbot)
    write_json_atomic(view._reference_path, reference)
    view.show()
    qtbot.waitUntil(lambda: not view._preview_workers)
    assert view._sample_id.text() == "G1"
    assert view._repeatability.value() == 2.5
    assert view._delta.value() == 3
    # The sample id is no longer guarded by a read-only flag on the operator
    # surface -- it lives inside the engineering panel, which stays collapsed.
    assert not view._maintenance.isVisible()
    assert not view._sample_id.isVisible()
    # The operator reads it off the control they must actually tick.
    assert "G1" in view._confirm.text()
    assert view._start_button.isEnabled()
    assert not view._confirm.isChecked()
    assert len(view._evidence.findChildren(CropView)) == 2
    view._maintenance_toggle.setChecked(True)
    assert view._maintenance.isVisible()
    assert view._sample_id.isVisible()
    view._show_failure("位置 1 基準量測不穩定")
    assert view._verdict.text().startswith("取樣不穩定")
    view._show_failure("裁切圖未涵蓋固定取樣區域")
    assert view._verdict.text().startswith("取樣位置")
    view._show_failure("相機離線")
    assert view._verdict.text().startswith("檢查未完成")
    assert view.prepare_shutdown()
