from __future__ import annotations

import copy
import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from core.services.golden_sample import (
    MAX_ACCEPTABLE_JITTER,
    SCHEMA,
    GoldenSampleError,
    build_reference,
    configuration_identity,
    encode_template,
    evaluate_readings,
    judged_cells,
    lab_grid,
    locate,
    measure_baseline,
    measure_snapshot,
    normal_sessions,
    propose_limits,
    read_json,
    replay_sessions,
    stable_pixels,
    validate_reference,
    write_json_atomic,
)

BOARD = (60, 130, 70)
RED_WIRE = (30, 40, 180)
SCENE = 72


def scene(wire=RED_WIRE):
    """A wire on a board, as the camera sees one position.

    Piecewise uniform on purpose: a crimp terminal, a print mark and the wire
    edges give alignment something to lock on to, and the flat interiors are
    what the measurement is meant to read. The wire's left edge falls just
    past the middle of a grid cell, as Cable1/A's did: that cell is 54% wire.
    """
    image = np.empty((SCENE, SCENE, 3), np.uint8)
    image[:] = BOARD
    image[:, 30:43] = wire
    image[:12, 20:52] = (225, 225, 225)
    image[30:34, 33:40] = (20, 20, 20)
    return image


def detection(tmp_path, name, *, image=None, dx=0, dy=0, box_dx=0, box_dy=0,
              gain=1.0, origin=(20, 20), color="red"):
    """One detection: the crop a detector box cut out of a camera frame.

    ``dx``/``dy`` move the board (an operator re-seating it); ``box_dx``/
    ``box_dy`` move only the detector's box.
    """
    content = scene() if image is None else image
    canvas = np.empty((200, 200, 3), np.uint8)
    canvas[:] = BOARD
    left, top = origin[0] + dx, origin[1] + dy
    canvas[top : top + SCENE, left : left + SCENE] = content
    canvas = np.clip(canvas.astype(float) * gain, 0, 255).astype(np.uint8)
    x1, y1 = origin[0] + box_dx, origin[1] + box_dy
    path = tmp_path / f"{name}.png"
    cv2.imwrite(str(path), canvas[y1 : y1 + SCENE, x1 : x1 + SCENE])
    return {
        "color": color,
        "bbox": [x1, y1, x1 + SCENE, y1 + SCENE],
        "crop_path": str(path),
        "margin": 0.3,
        "accepted": True,
    }


def readings(tmp_path, count=5, tag="r", **kwargs):
    return [
        {"source": f"{tag}-{index}", "positions": [detection(tmp_path, f"{tag}_{index}", **kwargs)]}
        for index in range(count)
    ]


def reference(tmp_path, **kwargs):
    return build_reference(
        readings(tmp_path, tag="base"), identity="config", operator="engineer", sample_id="G1",
        delta_e=3, repeatability=1, **kwargs,
    )


def snapshot(tmp_path, index=0, timestamp=None):
    crop = tmp_path / f"capture_Red_{index}.png"
    cv2.imwrite(str(crop), np.full((32, 32, 3), (30, 40, 180), dtype=np.uint8))
    path = tmp_path / f"{index}_config_snapshot.json"
    write_json_atomic(
        path,
        {
            "timestamp": timestamp or datetime.now().isoformat(),
            "artifacts": {"cropped_paths": [str(crop)]},
            "color_result": {
                "status": "evaluated",
                "items": [
                    {
                        "index": index,
                        "class_name": "Red",
                        "best_color": "Red",
                        "bbox": [0, 0, 32, 32],
                        "threshold": 0.5,
                        "diff": 0.2,
                        "is_ok": True,
                    }
                ],
            },
        },
    )
    return path


def test_a_re_seated_board_is_not_a_colour_shift(tmp_path):
    """Regression: Cable1/A failed almost every daily check on placement alone.

    The sampling window used to sit at fixed image coordinates. Re-seating the
    golden board moved the wire a pixel or two under it, cells on the wire
    edge and the crimp terminal flipped between materials, and 10-05's check
    read ΔE 35.6 on a board whose wire colour had not changed.
    """
    baseline = reference(tmp_path)
    moved = readings(tmp_path, 3, tag="moved", dx=2, dy=-1)

    # What the fixed window now sees: the same board, measured as a new colour.
    roi = baseline["positions"][0]["measurement_bbox"]
    fixed = cv2.imread(moved[0]["positions"][0]["crop_path"])
    box = moved[0]["positions"][0]["bbox"]
    stale = np.asarray(lab_grid(fixed[roi[1] - box[1] : roi[3] - box[1], roi[0] - box[0] : roi[2] - box[0]]))
    first = cv2.imread(baseline["readings"][0]["positions"][0]["crop_path"])
    original = np.asarray(lab_grid(first[roi[1] - 20 : roi[3] - 20, roi[0] - 20 : roi[2] - 20]))
    assert np.max(np.linalg.norm(stale - original, axis=1)) > 3

    result = evaluate_readings(moved, baseline, "config")
    assert result["status"] == "OK"
    row = result["rows"][0]
    assert row["delta_e"] < 0.5
    assert row["alignment_shift"] == [2, -1]
    assert row["alignment_score"] > 0.99


def test_the_box_following_the_wire_reports_the_same_shift(tmp_path):
    baseline = reference(tmp_path)
    result = evaluate_readings(readings(tmp_path, 3, tag="follow", dx=3, box_dx=3), baseline, "config")
    assert result["status"] == "OK"
    assert result["rows"][0]["alignment_shift"] == [3, 0]


def test_a_brightness_drift_is_still_caught_after_alignment(tmp_path):
    """Alignment is gain-invariant, so it must not absorb the drift it serves."""
    baseline = reference(tmp_path)
    original = copy.deepcopy(baseline)
    result = evaluate_readings(readings(tmp_path, 3, tag="bright", dx=1, gain=1.15), baseline, "config")
    assert result["status"] == "NG"
    row = result["rows"][0]
    assert row["reasons"] == ["顏色偏移超限"]
    assert row["delta_l"] > 3
    assert row["alignment_score"] > 0.99
    assert baseline == original  # Daily checks never adapt the normal reference.


def test_a_frame_showing_another_scene_fails_alignment(tmp_path):
    baseline = reference(tmp_path)
    noise = np.random.default_rng(3).integers(0, 256, (SCENE, SCENE, 3), dtype=np.uint8)
    result = evaluate_readings(readings(tmp_path, 3, tag="other", image=noise), baseline, "config")
    assert result["status"] == "NG"
    assert result["rows"][0]["reasons"][0] == "取樣對位失敗"


def test_one_bad_frame_cannot_hide_in_the_average(tmp_path):
    baseline = reference(tmp_path)
    measured = readings(tmp_path, 3, tag="steady")
    measured[2]["positions"][0] = detection(tmp_path, "odd", image=scene(wire=(40, 60, 205)))
    row = evaluate_readings(measured, baseline, "config")["rows"][0]
    assert row["worst_frame"] == 2
    assert row["delta_e"] > 3
    assert {"顏色偏移超限", "連拍不穩定"} <= set(row["reasons"])


def test_only_cells_with_stable_pixels_are_judged(tmp_path):
    flat = np.full((40, 40, 3), BOARD, np.uint8)
    assert all(judged_cells(stable_pixels(flat)))
    edge = scene()[11:61, 11:61]
    mask = stable_pixels(edge)
    # The wire edge (scene x=30, window x=19) is a material boundary, the
    # wire's interior is not.
    assert not mask[30, 19] and mask[30, 25]
    textured = np.random.default_rng(5).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    assert not any(judged_cells(stable_pixels(textured)))
    row = evaluate_readings(readings(tmp_path, 3, tag="cells"), reference(tmp_path), "config")["rows"][0]
    assert len(row["cells"]) == 16


def test_a_window_with_nothing_stable_cannot_be_a_baseline(tmp_path):
    noise = np.random.default_rng(9).integers(0, 256, (SCENE, SCENE, 3), dtype=np.uint8)
    with pytest.raises(GoldenSampleError, match="可穩定量測"):
        build_reference(readings(tmp_path, image=noise), identity="c", operator="e",
                        sample_id="g", delta_e=3, repeatability=1)


def test_a_flat_window_is_read_in_place_and_must_be_covered(tmp_path):
    """Nothing to align on, and nothing a small misplacement could change."""
    flat = np.full((SCENE, SCENE, 3), RED_WIRE, np.uint8)
    baseline = build_reference(readings(tmp_path, image=flat), identity="config", operator="e",
                               sample_id="G1", delta_e=3, repeatability=1)
    result = evaluate_readings(readings(tmp_path, 3, tag="flat", image=flat), baseline, "config")
    assert result["status"] == "OK"
    assert result["rows"][0]["alignment_shift"] == [0, 0]
    # Read in place, the window must still lie inside the crop. Boxes that
    # align_positions accepts always contain it (the inset equals that
    # tolerance), so the guard is exercised directly.
    with pytest.raises(GoldenSampleError, match="未涵蓋"):
        locate(flat[:40, :40], [0, 0, 40, 40], flat[:20, :20], [30, 0, 50, 20], 0)
    with pytest.raises(GoldenSampleError, match="小於"):
        locate(flat[:10, :10], [0, 0, 10, 10], flat[:20, :20], [0, 0, 20, 20], 0)


def test_repeated_colors_match_geometry_when_detector_order_changes(tmp_path):
    def two_wires(tag, count, box_dx=0):
        frames = []
        for index in range(count):
            red = detection(tmp_path, f"{tag}_red_{index}", box_dx=box_dx)
            black = detection(tmp_path, f"{tag}_black_{index}", image=scene(wire=(25, 25, 25)),
                              origin=(110, 20), color="black")
            frames.append({"source": f"{tag}-{index}", "positions": [red, black]})
        return frames

    baseline = build_reference(two_wires("base", 5), identity="config", operator="e",
                               sample_id="G1", delta_e=3, repeatability=1)
    daily = two_wires("daily", 3, box_dx=1)
    daily[1]["positions"].reverse()
    assert evaluate_readings(daily, baseline, "config")["status"] == "OK"


@pytest.mark.parametrize("field,value", [("margin", 0), ("margin", 0.1), ("accepted", False)])
def test_production_gates(tmp_path, field, value):
    measured = readings(tmp_path, 3, tag="gate")
    measured[0]["positions"][0][field] = value
    assert evaluate_readings(measured, reference(tmp_path), "config")["status"] == "NG"


@pytest.mark.parametrize("mutation", ["duplicate", "few", "empty", "shift", "color", "count", "crop"])
def test_invalid_session(tmp_path, mutation):
    measured = readings(tmp_path, 3, tag="bad")
    if mutation == "duplicate":
        measured[1]["source"] = measured[0]["source"]
    elif mutation == "few":
        measured.pop()
    elif mutation == "empty":
        for item in measured:
            item["positions"] = []
    elif mutation == "shift":
        measured[1]["positions"][0]["bbox"][0] += 20
    elif mutation == "color":
        measured[1]["positions"][0]["color"] = "green"
    elif mutation == "crop":
        del measured[1]["positions"][0]["crop_path"]
    else:
        measured[1]["positions"].append(copy.deepcopy(measured[1]["positions"][0]))
    with pytest.raises(GoldenSampleError):
        evaluate_readings(measured, reference(tmp_path), "config")


@pytest.mark.parametrize("delta,jitter", [(float("nan"), 1), (3, float("inf")), (0, 1), (1, 2)])
def test_bad_limits(tmp_path, delta, jitter):
    with pytest.raises(GoldenSampleError):
        build_reference(readings(tmp_path), identity="c", operator="e", sample_id="g",
                        delta_e=delta, repeatability=jitter)


def test_reject_unstable_or_bad_baseline(tmp_path):
    measured = readings(tmp_path)
    measured[0]["positions"][0]["accepted"] = False
    with pytest.raises(GoldenSampleError):
        build_reference(measured, identity="c", operator="e", sample_id="g", delta_e=3, repeatability=1)
    measured = readings(tmp_path, tag="wobble")
    measured[0]["positions"][0] = detection(tmp_path, "wobble_odd", image=scene(wire=(34, 46, 192)))
    with pytest.raises(GoldenSampleError, match="不穩定"):
        build_reference(measured, identity="c", operator="e", sample_id="g", delta_e=3, repeatability=1)
    with pytest.raises(GoldenSampleError, match="填寫"):
        build_reference(readings(tmp_path), identity="c", operator="", sample_id="g", delta_e=3, repeatability=1)


@pytest.mark.parametrize(
    "mutation", ["schema", "nan", "template", "template_size", "bbox", "margin", "missing", "empty"]
)
def test_corrupt_reference(tmp_path, mutation):
    baseline = reference(tmp_path)
    position = baseline["positions"][0]
    if mutation == "schema":
        baseline["schema"] = "old"
    elif mutation == "nan":
        position["measurement_bbox"][0] = float("nan")
    elif mutation == "template":
        position["template"] = "not-an-image"
    elif mutation == "template_size":
        position["template"] = encode_template(np.zeros((9, 9, 3), np.uint8))
    elif mutation == "bbox":
        position["bbox"] = [0, 0, 0, 0]
    elif mutation == "margin":
        position["margin"] = float("nan")
    elif mutation == "missing":
        del baseline["created_at"]
    else:
        baseline["positions"] = []
    with pytest.raises(GoldenSampleError):
        validate_reference(baseline, "config")


def test_color_measurement_and_unfiltered_local_change(tmp_path):
    path = snapshot(tmp_path)
    measured = measure_snapshot(path, ("Red",))
    assert len(measured["positions"][0]["lab"]) == 16
    image = np.full((32, 32, 3), (30, 40, 180), dtype=np.uint8)
    before = np.asarray(lab_grid(image))
    image[:8, :8] = (0, 255, 0)
    after = np.asarray(lab_grid(image))
    assert np.linalg.norm(before[0] - after[0]) > 50
    np.testing.assert_allclose(before[1:], after[1:])


@pytest.mark.parametrize(
    "mutation", ["status", "items", "item", "index", "missing_crop", "class", "box", "nan", "color", "paths"]
)
def test_invalid_snapshot(tmp_path, mutation):
    path = snapshot(tmp_path)
    payload = read_json(path)
    item = payload["color_result"]["items"][0]
    if mutation == "status":
        payload["color_result"]["status"] = "skipped"
    elif mutation == "items":
        payload["color_result"]["items"] = []
    elif mutation == "item":
        payload["color_result"]["items"] = [None]
    elif mutation == "index":
        item["index"] = None
    elif mutation == "missing_crop":
        payload["artifacts"]["cropped_paths"] = []
    elif mutation == "class":
        item["class_name"] = "Black"
    elif mutation == "box":
        item["bbox"] = [0, 0, 0, 0]
    elif mutation == "nan":
        item["diff"] = float("nan")
    elif mutation == "color":
        item["best_color"] = "Black"
    else:
        payload["artifacts"]["cropped_paths"] = None
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GoldenSampleError):
        measure_snapshot(path, ("Red",))


def test_files_and_identity(tmp_path):
    path = tmp_path / "profile.json"
    baseline = reference(tmp_path)
    write_json_atomic(path, baseline)
    assert read_json(path)["schema"] == baseline["schema"] == SCHEMA
    first = configuration_identity(path, None)
    assert configuration_identity(path, path) != first
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(GoldenSampleError):
        read_json(path)
    path.write_text("bad", encoding="utf-8")
    with pytest.raises(GoldenSampleError):
        read_json(path)
    assert not list(tmp_path.glob("*.tmp"))


def test_tiny_image_and_missing_reference(tmp_path):
    with pytest.raises(GoldenSampleError):
        lab_grid(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(GoldenSampleError):
        validate_reference(reference(tmp_path), "changed")


def past_session(tmp_path, tag, **kwargs):
    return {
        "identity": "config",
        "sample_id": "G1",
        "checked_at": f"2026-10-0{len(tag)}T00:00:00+00:00",
        "readings": readings(tmp_path, 3, tag=tag, **kwargs),
    }


def test_a_proposal_re_measures_past_sessions_against_the_new_baseline(tmp_path):
    """Stored delta-E values were taken against other baselines, and before v5
    by a measurement that mostly recorded where the board had been placed."""
    measured = measure_baseline(readings(tmp_path, tag="base"), identity="config",
                                operator="e", sample_id="G1")
    sessions = [
        past_session(tmp_path, "a", dx=2, gain=1.02),
        past_session(tmp_path, "bb", dx=-1, gain=1.05),
    ]
    samples = replay_sessions(sessions, measured, "config")
    assert len(samples) == 2
    # Newest first; the brighter session sits further from the baseline.
    assert 0 < samples[1]["delta_e"] < samples[0]["delta_e"]
    proposal = propose_limits(measured["measured_jitter"], samples)
    assert proposal["history_count"] == 2
    assert proposal["delta_e"] == pytest.approx(round(samples[0]["delta_e"] * 1.5, 2))
    assert "重算過往 2 次" in proposal["basis"]


def test_sessions_that_cannot_stand_for_this_board_are_skipped(tmp_path):
    measured = measure_baseline(readings(tmp_path, tag="base"), identity="config",
                                operator="e", sample_id="G1")
    noise = np.random.default_rng(4).integers(0, 256, (SCENE, SCENE, 3), dtype=np.uint8)
    missing = past_session(tmp_path, "gone")
    missing["readings"][0]["positions"][0]["crop_path"] = str(tmp_path / "deleted.png")
    sessions = [
        {**past_session(tmp_path, "other"), "identity": "another-setup"},
        {**past_session(tmp_path, "sample"), "sample_id": "G2"},
        past_session(tmp_path, "scene", image=noise),
        missing,
        {"identity": "config", "sample_id": "G1", "readings": []},
    ]
    assert replay_sessions(sessions, measured, "config") == []
    # With nothing to replay the proposal says so instead of inventing history.
    assert "尚無可重算" in propose_limits(measured["measured_jitter"], [])["basis"]


def test_an_excursion_or_an_unstable_session_does_not_widen_the_limit():
    ordinary = [{"delta_e": value, "jitter": 1.0} for value in (1.0, 1.4, 1.2)]
    drifted = {"delta_e": 30.0, "jitter": 1.0}
    shaky = {"delta_e": 1.3, "jitter": MAX_ACCEPTABLE_JITTER + 1}
    assert normal_sessions(ordinary + [drifted, shaky]) == ordinary
    assert propose_limits(1.0, ordinary + [drifted])["delta_e"] == propose_limits(1.0, ordinary)["delta_e"]


def test_run_to_run_noise_seen_on_other_days_sets_the_repeatability_floor():
    proposal = propose_limits(1.0, [{"delta_e": 2.0, "jitter": 2.0}])
    assert proposal["repeatability"] == 3.0
    assert "過往取樣最大 2.00" in proposal["basis"]
    assert proposal["delta_e"] >= proposal["repeatability"]


def test_the_conditions_report_shows_the_exposure_actually_in_use():
    """A session calibration moves exposure without touching config.yaml.

    Reading only the file, this report would stay silent through a day in
    which the exposure moved nearly ten percent -- quietest exactly when it
    had the most to say.
    """
    from core.services.golden_sample import condition_drift, observed_conditions

    config = {"exposure_time": "20134.0000", "gain": "23.0", "light_brightness": 0}

    assert observed_conditions(config)["exposure_time"] == "20134.0000"

    live = observed_conditions(config, {"exposure_time": 21980.0})
    assert live["exposure_time"] == 21980.0
    # Only the overridden field moves; the rest still come from the file.
    assert live["gain"] == "23.0"
    assert condition_drift(observed_conditions(config), live) == [
        "曝光 20134.0000 → 21980.0"
    ]


def test_a_session_exposure_cannot_expire_a_baseline(tmp_path):
    """Exposure is an observed condition, not part of colour identity."""
    from core.services.golden_sample import color_identity_payload

    config = {"enable_color_check": True, "exposure_time": "20134.0000"}
    moved = {"enable_color_check": True, "exposure_time": "21980.0"}

    assert color_identity_payload(config) == color_identity_payload(moved)


def test_colour_identity_field_list_is_locked():
    """The field list is a correctness boundary, not an implementation detail.

    Identity no longer hashes the whole config file, so a colour-relevant
    setting missing from this list is drift the check will not notice. Any
    edit here should be a deliberate, reviewed decision -- hence the exact
    comparison rather than a membership spot-check.
    """
    from core.services.golden_sample import (
        COLOR_IDENTITY_FIELDS,
        COLOR_IDENTITY_NESTED_FIELDS,
        OBSERVED_CONDITION_FIELDS,
    )

    assert set(COLOR_IDENTITY_FIELDS) == {
        "enable_color_check",
        "color_checker_type",
        "color_model_path",
        "color_roi_policy",
        "color_decision_tuning",
        "calibration",
        "weights",
        "conf_thres",
        "iou_thres",
        "imgsz",
    }
    # Calibration-loop outputs must stay out of identity; see the regression
    # test below for why.
    assert not set(COLOR_IDENTITY_FIELDS) & set(OBSERVED_CONDITION_FIELDS)
    assert set(COLOR_IDENTITY_NESTED_FIELDS) == {
        "color_preflight.minimum_margin_retention",
    }


def test_identity_tracks_colour_settings_and_ignores_the_rest():
    from core.services.golden_sample import color_identity_payload

    base = {
        "exposure_time": "9604.0",
        "gain": "23.0",
        "color_roi_policy": {"inset_x_ratio": 0.2},
        "color_preflight": {"minimum_margin_retention": 0.6, "recorded_by": "1"},
        "save_crops": True,
        "model_version": "1.0.6",
    }
    payload = color_identity_payload(base)
    assert payload["color_roi_policy"] == {"inset_x_ratio": 0.2}
    assert payload["color_preflight.minimum_margin_retention"] == 0.6
    # Bookkeeping inside color_preflight must not invalidate a baseline.
    assert "save_crops" not in payload and "model_version" not in payload
    assert "recorded_by" not in str(payload)
    # Nor may the values the calibration loop reads back from the camera.
    assert "exposure_time" not in payload and "gain" not in payload
    moved_on = dict(base, save_crops=False, model_version="9.9.9", exposure_time="1.0")
    assert color_identity_payload(moved_on) == payload


def test_margin_retention_reads_config_and_rejects_nonsense():
    from core.services.golden_sample import FALLBACK_MARGIN_RETENTION, margin_retention

    assert margin_retention({}) == FALLBACK_MARGIN_RETENTION
    assert margin_retention({"color_preflight": {"minimum_margin_retention": 0.8}}) == 0.8
    for bad in (0, -0.1, 1.5, "abc", float("nan")):
        with pytest.raises(GoldenSampleError):
            margin_retention({"color_preflight": {"minimum_margin_retention": bad}})


def test_retention_floor_comes_from_the_baseline_not_a_constant(tmp_path):
    """Hard-coding 0.6 kept a tuned station judged against the untuned value."""
    baseline = build_reference(
        readings(tmp_path), identity="config", operator="e", sample_id="G1",
        delta_e=3, repeatability=1, retention=0.9,
    )
    assert baseline["margin_retention"] == 0.9
    # Baseline margin is 0.3, so a 0.9 floor trips at anything under 0.27.
    weak = readings(tmp_path, 3, tag="weak")
    for reading in weak:
        reading["positions"][0]["margin"] = 0.26
    result = evaluate_readings(weak, baseline, "config")
    assert result["status"] == "NG"
    assert any("90%" in reason for row in result["rows"] for reason in row["reasons"])
    assert result["margin_retention"] == 0.9
    # The same measurement passes under the default floor.
    lenient = build_reference(
        readings(tmp_path, tag="lenient"), identity="config", operator="e", sample_id="G1",
        delta_e=3, repeatability=1, retention=0.6,
    )
    assert evaluate_readings(weak, lenient, "config")["status"] == "OK"


def test_recalibrated_exposure_does_not_expire_the_baseline():
    """Regression: the baseline had to be rebuilt after every restart.

    ``calibration_session.record_current`` reads exposure/gain back from the
    camera and writes them into the config, so the closed loop that holds
    ``target_luma`` steady lands on a different exposure every run. With those
    values in the identity, no baseline could ever survive a calibration --
    and rebuilding is precisely how a real drift gets absorbed into the new
    normal. The delta-E measurement is what should judge a bad exposure.
    """
    from core.services.golden_sample import (
        color_identity_payload,
        condition_drift,
        observed_conditions,
    )

    before = {
        "color_checker_type": "stats",
        "calibration": {"target_luma": 45.0, "tolerance": 2.0},
        "exposure_time": "35559.0000",
        "gain": "23.0",
        "light_brightness": 0,
    }
    after = dict(before, exposure_time="21346.0000")
    assert color_identity_payload(before) == color_identity_payload(after)
    # The change is recorded and reportable, just not fatal.
    assert observed_conditions(after)["exposure_time"] == "21346.0000"
    drift = condition_drift(observed_conditions(before), observed_conditions(after))
    assert drift == ["曝光 35559.0000 → 21346.0000"]
    # The calibration *target* is still identity: changing what the loop aims
    # at does change what a normal image looks like.
    retargeted = dict(before, calibration={"target_luma": 60.0, "tolerance": 2.0})
    assert color_identity_payload(retargeted) != color_identity_payload(before)


def test_baseline_survives_recalibration_end_to_end(tmp_path):
    baseline = build_reference(
        readings(tmp_path), identity="config", operator="e", sample_id="G1",
        delta_e=3, repeatability=1,
        conditions={"exposure_time": "35559.0000", "gain": "23.0"},
    )
    assert baseline["observed_conditions"]["exposure_time"] == "35559.0000"
    # Identity is unchanged by a recalibration, so the stored baseline still
    # validates and the daily check actually runs.
    validate_reference(baseline, "config")
    assert evaluate_readings(readings(tmp_path, 3, tag="daily"), baseline, "config")["status"] == "OK"


def test_condition_drift_is_quiet_when_nothing_moved():
    from core.services.golden_sample import condition_drift

    same = {"exposure_time": "35559.0000", "gain": "23.0"}
    assert condition_drift(same, dict(same)) == []
    assert condition_drift(None, same) == []
    assert condition_drift(same, None) == []
    # Numeric equality, not string equality: the writer formats to 4 places.
    assert condition_drift({"gain": "23.0"}, {"gain": "23.0000"}) == []


def test_schema_bump_and_identity_mismatch_say_different_things(tmp_path):
    """An operator's next action differs, so the message must too."""
    baseline = reference(tmp_path)
    stale = dict(baseline, schema="golden-lab-fixed-grid-v4")
    with pytest.raises(GoldenSampleError) as raised:
        validate_reference(stale, baseline["identity"])
    assert "舊版格式" in str(raised.value)
    with pytest.raises(GoldenSampleError) as raised:
        validate_reference(baseline, "a-different-station-setup")
    assert "不一致" in str(raised.value)
