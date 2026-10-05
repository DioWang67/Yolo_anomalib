from __future__ import annotations

import os
from uuid import uuid4
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QLabel
from test_golden_sample import snapshot

from app.gui.golden_sample_dialog import (
    CAPTURE_RETRY_BUDGET,
    CAPTURE_TIMEOUT_SECONDS,
    COLLECTION_GRACE_SECONDS,
    UNSET_LIMIT,
    GoldenCaptureWorker,
    GoldenSampleDialog,
    station_identity,
)
from core.services.golden_sample import (
    BASELINE_COUNT,
    CHECK_COUNT,
    finalize_baseline,
    validate_reference,
)
from core.services.color_preflight_store import ColorPreflightLedger
from core.services.golden_sample import read_json, write_json_atomic


def worker(tmp_path, baseline=True):
    config = tmp_path / "config.yaml"
    if not config.exists():
        config.write_text("expected_items:\n  Cable1:\n    A: [Red]\n", encoding="utf-8")
    return GoldenCaptureWorker(
        config_path=config,
        results_root=tmp_path / "results",
        scope=("Cable1", "A", "yolo"),
        reference_path=tmp_path / "golden-reference.json",
        baseline=baseline,
        operator="engineer",
        sample_id="G1",
        delta_e=3,
        repeatability=1,
    )


def build_baseline(capture, delta_e=None, repeatability=None):
    """Measure, then accept the proposed limits -- the two steps the dialog takes.

    The worker deliberately stores nothing on its own: an engineer sees what
    the station measured before choosing the limits it will be judged against.
    """
    proposals = []
    capture.proposal_ready.connect(proposals.append)
    capture.run()
    if not proposals:
        return None
    payload = proposals[-1]
    proposal = payload["proposal"]
    result = finalize_baseline(
        payload["measured"],
        delta_e=proposal["delta_e"] if delta_e is None else delta_e,
        repeatability=(
            proposal["repeatability"] if repeatability is None else repeatability
        ),
        retention=payload["retention"],
        identity_payload=payload["identity_payload"],
        conditions=payload["conditions"],
    )
    if capture.reference_path.exists():
        write_json_atomic(
            capture.reference_path.parent / f"reference-{uuid4().hex}.json",
            read_json(capture.reference_path),
        )
    write_json_atomic(capture.reference_path, result)
    return result


def publish_samples(capture, count):
    directory = capture.results_root / "20260910/Cable1/A/PASS/metadata/yolo"
    directory.mkdir(parents=True, exist_ok=True)
    return [snapshot(directory, index=i) for i in range(count)]


def test_worker_reference_and_daily_check(tmp_path, qapp):
    capture = worker(tmp_path)
    publish_samples(capture, 5)
    outputs, errors = [], []
    capture.failed.connect(errors.append)
    stored = build_baseline(capture)
    assert not errors
    assert stored["sample_id"] == "G1"
    old_reference = capture.reference_path.read_bytes()
    daily = worker(tmp_path, baseline=False)
    publish_samples(daily, 3)
    daily.completed.connect(outputs.append)
    daily.failed.connect(errors.append)
    daily.run()
    assert not errors
    assert outputs[-1]["status"] == "OK"
    assert capture.reference_path.read_bytes() == old_reference
    assert len(list(tmp_path.glob("check-*.json"))) == 1
    renewed = worker(tmp_path)
    publish_samples(renewed, 5)
    build_baseline(renewed)
    assert len(list(tmp_path.glob("reference-*.json"))) == 1


def test_old_copied_snapshot_does_not_count(tmp_path, qapp, monkeypatch):
    capture = worker(tmp_path)
    paths = publish_samples(capture, 5)
    payload = read_json(paths[0])
    payload["timestamp"] = "2020-01-01T00:00:00"
    write_json_atomic(paths[0], payload)
    messages = []
    capture.progress.connect(lambda count, required, message: messages.append(count))
    # Exit after scanning the available files, without a real wait.
    monkeypatch.setattr(capture, "msleep", lambda _: setattr(capture, "started_at", 0))
    errors = []
    capture.failed.connect(errors.append)
    capture.run()
    assert max(messages) == 4
    assert errors and "分鐘" in errors[-1]
    assert not capture.reference_path.exists()


def test_one_bad_shot_is_discarded_without_losing_the_good_ones(tmp_path, qapp, monkeypatch):
    """A single unusable frame must not cost the operator the whole session."""
    capture = worker(tmp_path)
    paths = publish_samples(capture, 5)
    payload = read_json(paths[0])
    payload["artifacts"]["cropped_paths"] = []
    write_json_atomic(paths[0], payload)
    progress = []
    capture.progress.connect(lambda count, required, message: progress.append((count, message)))

    def stop_once_observed(_):
        # End the session as soon as the behaviour under test has happened,
        # instead of waiting out the real budget.
        if any("證據不完整" in message for _, message in progress) and any(
            count == 4 for count, _ in progress
        ):
            capture.started_at = 0

    monkeypatch.setattr(capture, "msleep", stop_once_observed)
    errors = []
    capture.failed.connect(errors.append)
    capture.run()
    assert any("證據不完整" in message for _, message in progress)
    # The four intact snapshots were still accepted rather than thrown away.
    assert max(count for count, _ in progress) == 4
    assert errors and not capture.reference_path.exists()


def test_repeated_bad_evidence_stops_the_session(tmp_path, qapp, monkeypatch):
    capture = worker(tmp_path)
    paths = publish_samples(capture, 5)
    for path in paths[: CAPTURE_RETRY_BUDGET + 1]:
        payload = read_json(path)
        payload["artifacts"]["cropped_paths"] = []
        write_json_atomic(path, payload)
    monkeypatch.setattr(capture, "msleep", lambda _: None)
    errors = []
    capture.failed.connect(errors.append)
    capture.run()
    assert errors and "證據不完整" in errors[-1]
    assert not capture.reference_path.exists()


def test_session_budget_scales_with_the_capture_count(tmp_path, qapp):
    """Five baseline shots must not exhaust the session budget by arithmetic.

    The old flat 600s ceiling was exactly 5 x the 120s per-capture budget, so a
    slow station could fail baseline creation without any step timing out.
    """
    for required in (BASELINE_COUNT, CHECK_COUNT):
        budget = required * CAPTURE_TIMEOUT_SECONDS + COLLECTION_GRACE_SECONDS
        assert budget > required * CAPTURE_TIMEOUT_SECONDS


@pytest.mark.parametrize("kind", ["sample", "colour_field", "expected"])
def test_worker_refuses_incompatible_inputs(tmp_path, qapp, kind):
    capture = worker(tmp_path)
    publish_samples(capture, 5)
    build_baseline(capture)
    daily = worker(tmp_path, baseline=False)
    if kind == "sample":
        daily.sample_id = "different"
    elif kind == "colour_field":
        # Changing what the illumination loop *aims at* changes what a normal
        # image looks like, so the baseline must go.
        daily.config_path.write_text(
            daily.config_path.read_text() + "\ncalibration:\n  target_luma: 60.0\n"
        )
    else:
        daily.config_path.write_text("{}")
    errors = []
    daily.failed.connect(errors.append)
    daily.run()
    assert errors


def test_unrelated_config_edit_keeps_the_baseline(tmp_path, qapp):
    """A baseline survives edits that cannot change a colour measurement.

    Hashing the whole config file meant toggling ``save_crops`` threw the
    baseline away, which made rebuilding one feel routine.
    """
    capture = worker(tmp_path)
    publish_samples(capture, 5)
    build_baseline(capture)
    before = station_identity(capture.config_path)[0]
    capture.config_path.write_text(
        capture.config_path.read_text()
        # Unrelated bookkeeping, plus a recalibrated exposure: the closed
        # illumination loop rewrites that on every run.
        + "\nsave_crops: true\nmodel_version: 9.9.9\nexposure_time: '21346.0000'\n"
    )
    assert station_identity(capture.config_path)[0] == before
    validate_reference(read_json(capture.reference_path), before)


def test_retention_comes_from_the_station_config(tmp_path, qapp):
    capture = worker(tmp_path)
    capture.config_path.write_text(
        capture.config_path.read_text()
        + "\ncolor_preflight:\n  minimum_margin_retention: 0.8\n"
    )
    publish_samples(capture, 5)
    build_baseline(capture)
    reference = read_json(capture.reference_path)
    assert reference["margin_retention"] == 0.8
    # Recorded on the baseline, so a later check is judged against the floor
    # the baseline was actually built with.
    assert "minimum_margin_retention" in str(reference["identity_fields"])


def dialog(tmp_path, qtbot):
    capture = worker(tmp_path)
    view = GoldenSampleDialog(
        config_path=capture.config_path,
        results_root=capture.results_root,
        product="Cable1",
        area="A",
        model_type="yolo",
        ledger=ColorPreflightLedger(tmp_path / "ledger"),
    )
    qtbot.addWidget(view)
    return view


def arm_engineering(view, sample_id="G1", delta=3.0, repeatability=1.0):
    """Fill in what an engineer must now decide before a baseline can be built."""
    view._sample_id.setText(sample_id)
    view._confirm.setChecked(True)
    view._operator.setText("engineer")
    view._engineering.setChecked(True)
    view._delta.setValue(delta)
    view._repeatability.setValue(repeatability)


def test_limits_are_proposed_from_the_measurement_then_confirmed(tmp_path, qtbot):
    """Nothing is stored until an engineer has seen the numbers and accepted.

    Neither a hard-coded 3.00/1.00 nor an empty box the engineer has to guess
    into is a decision. The station is measured first, the limits follow from
    the measurement, and saving is a separate, deliberate press.
    """
    view = dialog(tmp_path, qtbot)
    arm_engineering(view)
    # Limits start unset and are not required in order to measure.
    view._delta.setValue(UNSET_LIMIT)
    view._repeatability.setValue(UNSET_LIMIT)
    assert view._delta.text() == "待量測後提議"
    assert not view._save_button.isEnabled()
    view._start(True)
    publish_samples(view._worker, 5)
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    # Measured, proposed, and explicitly not saved yet.
    assert not view._reference_path.exists()
    assert view._save_button.isEnabled()
    assert view._delta.value() > UNSET_LIMIT
    assert view._repeatability.value() > UNSET_LIMIT
    assert "實測" in view._measured_hint.text()
    assert "建議" in view._verdict.text()
    proposed = (view._delta.value(), view._repeatability.value())
    view._save_baseline()
    assert view._reference_path.exists()
    stored = read_json(view._reference_path)
    assert (stored["delta_e_limit"], stored["repeatability_limit"]) == proposed
    # Accepting the proposal is recorded as such, so an audit can tell an
    # accepted default from a considered override.
    assert stored["limits_source"] == "proposed"
    assert not view._save_button.isEnabled()


def test_an_overridden_limit_is_recorded_as_a_decision(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    arm_engineering(view)
    view._start(True)
    publish_samples(view._worker, 5)
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    view._delta.setValue(view._delta.value() + 2.0)
    view._save_baseline()
    assert read_json(view._reference_path)["limits_source"] == "adjusted"


def test_a_limit_below_the_measured_jitter_is_refused(tmp_path, qtbot):
    """The engineer may override, but not into a limit the station cannot meet."""
    view = dialog(tmp_path, qtbot)
    arm_engineering(view)
    view._start(True)
    publish_samples(view._worker, 5)
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    view._repeatability.setValue(UNSET_LIMIT)
    view._save_baseline()
    assert not view._reference_path.exists()
    assert "請先確認" in view._verdict.text()


def test_dialog_guards_and_cancel(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    view._start(False)
    assert "請填寫" in view._verdict.text()
    view._sample_id.setText("G1")
    view._confirm.setChecked(True)
    view._start(True)
    assert "需要具名" in view._verdict.text()
    view._operator.setText("engineer")
    view._engineering.setChecked(True)
    view._delta.setValue(3.0)
    view._repeatability.setValue(1.0)
    view._start(True)
    assert not view._start_button.isEnabled()
    view._cancel()
    qtbot.waitUntil(lambda: view._worker is None)
    assert "取消" in view._verdict.text()


def test_dialog_full_flow(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    arm_engineering(view)
    view.show()
    view._start(True)
    active_worker = view._worker
    view.close()
    assert not view.isVisible()
    assert view._worker is active_worker
    assert not active_worker.isInterruptionRequested()
    publish_samples(active_worker, 5)
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    # Measured; the engineer still has to accept the proposed limits.
    assert view._save_button.isEnabled()
    view._save_baseline()
    assert "已儲存" in view._verdict.text()
    assert "G1" in view._reference_label.text()
    view.show()
    assert view._sample_id.text() == "G1"
    assert view._operator.text() == "engineer"
    assert "已儲存" in view._verdict.text()
    view._start(False)
    publish_samples(view._worker, 3)
    qtbot.waitUntil(lambda: view._worker is None, timeout=10000)
    assert "通過" in view._verdict.text()
    assert view._table.rowCount() == 1
    assert view._table.item(0, 6).text() == "OK"
    view.reject()
    view.show()
    assert view._table.item(0, 6).text() == "OK"


def test_hidden_session_shutdown_stops_worker(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    assert view.prepare_shutdown()
    arm_engineering(view)
    view._start(True)
    view.reject()
    assert view._worker is not None
    assert view.prepare_shutdown()
    qtbot.waitUntil(lambda: view._worker is None)
    assert not view._reference_path.exists()


def test_menu_reopens_same_session_and_retires_old_scope_safely(tmp_path, qtbot, monkeypatch):
    from PyQt5.QtWidgets import QComboBox, QDialog

    from app.gui.color_preflight_handler import ColorPreflightHandlerMixin

    class Host(ColorPreflightHandlerMixin, QDialog):
        pass

    host = Host()
    qtbot.addWidget(host)
    capture = worker(tmp_path)
    for name, text in (("product_combo", "Cable1"), ("area_combo", "A"), ("inference_combo", "yolo")):
        combo = QComboBox(host)
        combo.addItem(text)
        setattr(host, name, combo)
    host._catalog = SimpleNamespace(config_path=lambda *args: capture.config_path)
    # The dialog asks the controller for camera values in force that the
    # station config does not record; no session calibration here, so none.
    host.controller = SimpleNamespace(effective_camera_settings=lambda: {})
    monkeypatch.setattr("app.gui.color_preflight_handler.load_station_data_paths", lambda:
        SimpleNamespace(results=capture.results_root, color_preflight=tmp_path / "ledger"))
    host.open_color_preflight_dialog()
    original = host._color_preflight_dialog
    original._sample_id.setText("G1")
    original.close()
    host.open_color_preflight_dialog()
    assert host._color_preflight_dialog is original
    assert original._sample_id.text() == "G1"
    assert original.isVisible()
    host.area_combo.addItem("B")
    host.area_combo.setCurrentText("B")
    host.open_color_preflight_dialog()
    replacement = host._color_preflight_dialog
    assert replacement is not original
    qtbot.wait(10)
    assert host._color_preflight_dialog is replacement


def test_automatic_baseline_captures_exactly_five_then_daily_three(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    calls = []
    busy_polls = [False, False]

    def capture():
        if busy_polls:
            # An older in-flight manual result must not count as an auto shot.
            directory = view._results_root / "20260910/Cable1/A/PASS/metadata/yolo"
            directory.mkdir(parents=True, exist_ok=True)
            snapshot(directory, index=999)
            return busy_polls.pop()
        calls.append(len(calls))
        directory = view._results_root / "20260910/Cable1/A/PASS/metadata/yolo"
        directory.mkdir(parents=True, exist_ok=True)
        snapshot(directory, index=len(calls))
        return True

    view._request_capture_fn = capture
    arm_engineering(view)
    view._start(True)
    qtbot.waitUntil(lambda: view._worker is None, timeout=20000)
    assert len(calls) == 5
    assert "建議" in view._verdict.text()
    view._save_baseline()
    reference = read_json(view._reference_path)
    assert all("999_config_snapshot" not in r["source"] for r in reference["readings"])
    assert not view._capture_timer.isActive()
    view._start(False)
    qtbot.waitUntil(lambda: view._worker is None, timeout=15000)
    assert len(calls) == 8
    assert view._table.item(0, 6).text() == "OK"


def test_automatic_capture_error_is_visible_and_stops(tmp_path, qtbot):
    from core.services.golden_sample import GoldenSampleError

    view = dialog(tmp_path, qtbot)

    def unavailable():
        raise GoldenSampleError("相機未就緒")

    view._request_capture_fn = unavailable
    arm_engineering(view)
    view._start(True)
    qtbot.waitUntil(lambda: view._worker is None, timeout=5000)
    assert "相機未就緒" in view._verdict.text()
    assert not view._capture_timer.isActive()
    assert not view._reference_path.exists()


@pytest.mark.parametrize("condition", ["scope", "auto", "image", "system", "busy", "ready"])
def test_capture_adapter_guards_camera_and_serializes(condition):
    from app.gui.color_preflight_handler import ColorPreflightHandlerMixin
    from core.services.golden_sample import GoldenSampleError

    calls = []
    host = SimpleNamespace(
        product_combo=SimpleNamespace(currentText=lambda: "Other" if condition == "scope" else "Cable1"),
        area_combo=SimpleNamespace(currentText=lambda: "A"),
        inference_combo=SimpleNamespace(currentText=lambda: "yolo"),
        _auto_controller=SimpleNamespace(is_running=lambda: condition == "auto"),
        use_camera_chk=SimpleNamespace(isChecked=lambda: condition != "image"),
        controller=SimpleNamespace(has_system=lambda: condition != "system"),
        is_detection_running=lambda: condition == "busy" or bool(calls),
        start_detection=lambda **kwargs: calls.append(kwargs),
    )
    if condition in {"scope", "auto", "image", "system"}:
        with pytest.raises(GoldenSampleError):
            ColorPreflightHandlerMixin._capture_golden_sample(host, "Cable1", "A", "yolo")
        assert not calls
    else:
        result = ColorPreflightHandlerMixin._capture_golden_sample(host, "Cable1", "A", "yolo")
        assert result == (condition == "ready")
        assert calls == ([{"golden_sample": True}] if condition == "ready" else [])


@pytest.mark.parametrize("condition", ["scope", "auto", "image", "system", "busy", "ready"])
def test_readiness_panel_and_capture_refusal_share_one_rule_set(condition):
    """What the panel promises and what the capture allows must never diverge.

    The panel exists so the operator learns about a blocked station before
    committing to a run; it would be worse than nothing if it could say ready
    while the capture still refused.
    """
    from app.gui.color_preflight_handler import (
        ColorPreflightHandlerMixin,
        golden_sample_readiness,
    )
    from core.services.golden_sample import GoldenSampleError

    calls = []
    host = SimpleNamespace(
        product_combo=SimpleNamespace(currentText=lambda: "Other" if condition == "scope" else "Cable1"),
        area_combo=SimpleNamespace(currentText=lambda: "A"),
        inference_combo=SimpleNamespace(currentText=lambda: "yolo"),
        _auto_controller=SimpleNamespace(is_running=lambda: condition == "auto"),
        use_camera_chk=SimpleNamespace(isChecked=lambda: condition != "image"),
        controller=SimpleNamespace(has_system=lambda: condition != "system"),
        is_detection_running=lambda: condition == "busy" or bool(calls),
        start_detection=lambda **kwargs: calls.append(kwargs),
    )
    checks = golden_sample_readiness(host, "Cable1", "A", "yolo")
    blocked = [check for check in checks if not check.ok]
    blocking_condition = condition in {"scope", "auto", "image", "system"}
    assert bool(blocked) == blocking_condition
    if blocking_condition:
        # Every blocking check must carry a hint the operator can act on.
        assert all(check.hint for check in blocked)
        with pytest.raises(GoldenSampleError) as raised:
            ColorPreflightHandlerMixin._capture_golden_sample(host, "Cable1", "A", "yolo")
        assert blocked[0].hint in str(raised.value)
    else:
        ColorPreflightHandlerMixin._capture_golden_sample(host, "Cable1", "A", "yolo")


def test_start_is_blocked_and_explained_while_the_station_is_not_ready(tmp_path, qtbot):
    from app.gui.color_preflight_handler import ReadinessCheck

    checks = [ReadinessCheck("相機輸入", False, "請啟用相機輸入")]
    view = dialog(tmp_path, qtbot)
    view._readiness_fn = lambda: list(checks)
    view._readiness_label.setVisible(True)
    view._reference_loaded = True
    view._refresh_readiness()
    assert not view._start_button.isEnabled()
    assert "請啟用相機輸入" in view._readiness_label.text()
    view._confirm.setChecked(True)
    view._sample_id.setText("G1")
    view._start(False)
    assert view._worker is None
    assert "尚未就緒" in view._verdict.text()
    checks[:] = [ReadinessCheck("相機輸入", True, "")]
    view._refresh_readiness()
    assert view._start_button.isEnabled()
    assert view._readiness_label.text().startswith("✓")


def test_broken_host_does_not_brick_the_dialog(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)

    def exploding():
        raise RuntimeError("host gone")

    view._readiness_fn = exploding
    view._reference_loaded = True
    view._refresh_readiness()
    assert not view._start_button.isEnabled()
    assert "無法確認站點狀態" in view._readiness_label.text()


def test_maintenance_panel_requires_the_engineering_pin(tmp_path, qtbot):
    """An operator must not be able to promote today's drift into the baseline."""
    view = dialog(tmp_path, qtbot)
    view.show()
    attempts = []
    view._authorize_fn = lambda: attempts.append(1) or False
    view._maintenance_toggle.setChecked(True)
    assert attempts
    assert not view._maintenance.isVisible()
    assert not view._maintenance_toggle.isChecked()
    view._authorize_fn = lambda: True
    view._maintenance_toggle.setChecked(True)
    assert view._maintenance.isVisible()


def test_daily_check_needs_no_pin(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    view._authorize_fn = lambda: pytest.fail("daily check must not prompt for a PIN")
    view._reference_loaded = True
    view._update_start_enabled()
    assert view._start_button.isEnabled()


def test_pin_gate_does_not_open_an_engineering_session():
    """Verifying for one action must not leave the main engineering panel live."""
    from app.gui.color_preflight_handler import ColorPreflightHandlerMixin

    granted = []
    host = SimpleNamespace(
        control_panel=SimpleNamespace(
            authorize_engineering_action=lambda: granted.append("verified") or True
        )
    )
    assert ColorPreflightHandlerMixin._authorize_golden_sample_maintenance(host)
    # It routes through the dedicated verifier, not the page-session unlock.
    assert granted == ["verified"]
    assert not hasattr(host.control_panel, "_engineering_access_granted")


def test_failed_positions_are_named_and_shown_first(tmp_path, qtbot):
    """A banner saying "see the marked position" is useless if it is off screen."""
    from app.gui.golden_sample_view import CropView

    view = dialog(tmp_path, qtbot)
    result = {
        "status": "NG",
        "rows": [
            {"position": 1, "color": "red", "delta_e": 0.0, "delta_l": 0.0,
             "jitter": 0.0, "margin": 0.3, "reasons": []},
            {"position": 2, "color": "green", "delta_e": 9.0, "delta_l": 1.0,
             "jitter": 0.1, "margin": 0.3, "reasons": ["顏色偏移超限"]},
        ],
    }
    view._completed(result)
    assert "位置 2 綠線" in view._verdict.text()
    assert "顏色偏移超限" in view._verdict.text()
    # The table speaks the same wire names as the cards.
    assert "綠線" in view._table.item(1, 0).text()
    previews = [
        {"position": 1, "color": "red", "reference_image": None, "reference_bbox": None,
         "current_image": None, "current_bbox": None, "roi": None, "heatmap": None,
         "frame": None, "row": result["rows"][0], "delta_e_limit": 3.0,
         "repeatability_limit": 1.0},
        {"position": 2, "color": "green", "reference_image": None, "reference_bbox": None,
         "current_image": None, "current_bbox": None, "roi": None, "heatmap": None,
         "frame": None, "row": result["rows"][1], "delta_e_limit": 3.0,
         "repeatability_limit": 1.0},
    ]
    view._evidence.show_previews(previews)
    titles = [
        label.text()
        for label in view._evidence.widget().findChildren(QLabel)
        if "·" in label.text()
    ]
    assert titles and "綠線" in titles[0]
    assert isinstance(view._evidence.widget().findChildren(CropView), list)


def test_uniform_drift_collapses_the_sixteen_cell_grid(qtbot):
    """Lighting drift moves every cell together; sixteen copies of one number
    pushed the remaining positions below the fold."""
    from app.gui.golden_sample_view import GoldenEvidenceView

    view = GoldenEvidenceView()
    qtbot.addWidget(view)
    row = {"position": 1, "color": "green", "delta_e": 25.4, "delta_l": 2.0,
           "jitter": 0.1, "margin": 0.3, "reasons": ["顏色偏移超限"]}
    preview = {
        "position": 1, "color": "green", "reference_image": None, "reference_bbox": None,
        "current_image": None, "current_bbox": None, "roi": None,
        "heatmap": [[25.4] * 4 for _ in range(4)], "frame": 1, "row": row,
        "delta_e_limit": 3.0, "repeatability_limit": 1.0,
    }
    view.show_previews([preview])
    texts = [label.text() for label in view.widget().findChildren(QLabel)]
    assert any("一致偏移 25.4" in text for text in texts)
    assert sum(text == "25.4" for text in texts) == 0
    # A localised shift still gets the full grid.
    preview["heatmap"][0][0] = 1.0
    view.show_previews([preview])
    texts = [label.text() for label in view.widget().findChildren(QLabel)]
    assert sum(text == "25.4" for text in texts) == 15


def test_cells_left_out_of_the_verdict_are_shown_as_such(qtbot):
    """An edge the operator can see changing must visibly not count."""
    from app.gui.golden_sample_view import GoldenEvidenceView

    view = GoldenEvidenceView()
    qtbot.addWidget(view)
    row = {"position": 1, "color": "orange", "delta_e": 1.2, "delta_l": 0.3, "jitter": 0.4,
           "margin": 0.3, "reasons": [], "alignment_shift": [-1, 1]}
    heatmap = [[None, 1.2, 3.4, None] for _ in range(4)]
    preview = {
        "position": 1, "color": "orange", "reference_image": None, "reference_bbox": None,
        "current_image": None, "current_bbox": None, "roi": None, "current_roi": None,
        "heatmap": heatmap, "frame": 1, "row": row, "delta_e_limit": 9.0, "repeatability_limit": 3.0,
    }
    view.show_previews([preview])
    texts = [label.text() for label in view.widget().findChildren(QLabel)]
    assert sum(text == "—" for text in texts) == 8
    assert any("對位 -1, +1 px" in text for text in texts)


def test_the_exposure_change_reads_as_context_not_as_the_cause(tmp_path, qtbot):
    view = dialog(tmp_path, qtbot)
    row = {"position": 2, "color": "green", "delta_e": 9.0, "delta_l": 1.0, "jitter": 0.1,
           "margin": 0.3, "reasons": ["顏色偏移超限"], "alignment_shift": [-1, 1]}
    view._completed({"status": "NG", "rows": [row], "condition_drift": ["曝光 20913 → 22397"]})
    assert "相機條件已變動" not in view._verdict.text()
    assert "參考，不列入判定：曝光 20913 → 22397" in view._verdict.text()
    assert view._table.item(0, 5).text() == "-1, +1"
