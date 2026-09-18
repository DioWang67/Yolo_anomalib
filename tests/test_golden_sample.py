from __future__ import annotations

import copy
import json
from datetime import datetime

import cv2
import numpy as np
import pytest

from core.services.golden_sample import (
    GoldenSampleError,
    build_reference,
    configuration_identity,
    evaluate_readings,
    fixed_grid_measurements,
    lab_grid,
    measure_snapshot,
    read_json,
    validate_reference,
    write_json_atomic,
)


def readings(count=5, offset=0):
    return [
        {
            "source": str(i),
            "positions": [
                {
                    "color": "red",
                    "bbox": [0, 0, 32, 32],
                    "lab": [[50 + offset, 30, 20] for _ in range(16)],
                    "margin": 0.3,
                    "accepted": True,
                }
            ],
        }
        for i in range(count)
    ]


def reference():
    return build_reference(
        readings(), identity="config", operator="engineer", sample_id="G1", delta_e=3, repeatability=1
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


def test_stable_and_shifted_same_production_score():
    baseline = reference()
    original = copy.deepcopy(baseline)
    assert evaluate_readings(readings(3), baseline, "config")["status"] == "OK"
    result = evaluate_readings(readings(3, 4), baseline, "config")
    assert result["status"] == "NG"
    assert result["rows"][0]["delta_l"] == 4
    assert result["rows"][0]["reasons"] == ["顏色偏移超限"]
    assert baseline == original  # Daily checks never adapt the normal reference.


def test_one_bad_frame_and_cell_cannot_hide_in_average():
    measured = readings(3)
    measured[2]["positions"][0]["lab"][15][1] += 6
    result = evaluate_readings(measured, reference(), "config")
    assert result["rows"][0]["delta_e"] == 6
    assert "連拍不穩定" in result["rows"][0]["reasons"]


def test_repeated_colors_match_geometry_when_detector_order_changes():
    measured = readings()
    for reading in measured:
        second = copy.deepcopy(reading["positions"][0])
        second["bbox"] = [0, 50, 32, 82]
        second["lab"] = [[30, 10, 20] for _ in range(16)]
        reading["positions"].append(second)
    baseline = build_reference(measured, identity="config", operator="e", sample_id="G1", delta_e=3, repeatability=1)
    daily = copy.deepcopy(measured[:3])
    daily[1]["positions"].reverse()
    daily[1]["positions"][0]["bbox"] = [1, 50, 33, 82]
    assert evaluate_readings(daily, baseline, "config")["status"] == "OK"


@pytest.mark.parametrize("field,value", [("margin", 0), ("margin", 0.1), ("accepted", False)])
def test_production_gates(field, value):
    measured = readings(3)
    measured[0]["positions"][0][field] = value
    assert evaluate_readings(measured, reference(), "config")["status"] == "NG"


@pytest.mark.parametrize("mutation", ["duplicate", "few", "empty", "shift", "color", "count"])
def test_invalid_session(mutation):
    measured = readings(3)
    if mutation == "duplicate":
        measured[1]["source"] = measured[0]["source"]
    elif mutation == "few":
        measured.pop()
    elif mutation == "empty":
        for item in measured:
            item["positions"] = []
    elif mutation == "shift":
        measured[1]["positions"][0]["bbox"][0] += 10
    elif mutation == "color":
        measured[1]["positions"][0]["color"] = "green"
    else:
        measured[1]["positions"].append(copy.deepcopy(measured[1]["positions"][0]))
    with pytest.raises(GoldenSampleError):
        evaluate_readings(measured, reference(), "config")


@pytest.mark.parametrize("delta,jitter", [(float("nan"), 1), (3, float("inf")), (0, 1), (1, 2)])
def test_bad_limits(delta, jitter):
    with pytest.raises(GoldenSampleError):
        build_reference(readings(), identity="c", operator="e", sample_id="g", delta_e=delta, repeatability=jitter)


def test_reject_unstable_or_bad_baseline():
    for measured in (readings(), readings(), readings()):
        measured[0]["positions"][0]["accepted"] = False
        with pytest.raises(GoldenSampleError):
            build_reference(measured, identity="c", operator="e", sample_id="g", delta_e=3, repeatability=1)
    measured = readings()
    measured[0]["positions"][0]["lab"][0][0] += 5
    with pytest.raises(GoldenSampleError, match="不穩定"):
        build_reference(measured, identity="c", operator="e", sample_id="g", delta_e=3, repeatability=1)
    with pytest.raises(GoldenSampleError, match="填寫"):
        build_reference(readings(), identity="c", operator="", sample_id="g", delta_e=3, repeatability=1)


@pytest.mark.parametrize("mutation", ["schema", "nan", "shape", "bbox", "margin", "missing", "empty"])
def test_corrupt_reference(mutation):
    baseline = reference()
    if mutation == "schema":
        baseline["schema"] = "old"
    elif mutation == "nan":
        baseline["positions"][0]["lab"][0][0] = float("nan")
    elif mutation == "shape":
        baseline["positions"][0]["lab"] = []
    elif mutation == "bbox":
        baseline["positions"][0]["bbox"] = [0, 0, 0, 0]
    elif mutation == "margin":
        baseline["positions"][0]["margin"] = float("nan")
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
    write_json_atomic(path, reference())
    assert read_json(path) == reference() or read_json(path)["schema"] == reference()["schema"]
    first = configuration_identity(path, None)
    assert configuration_identity(path, path) != first
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(GoldenSampleError):
        read_json(path)
    path.write_text("bad", encoding="utf-8")
    with pytest.raises(GoldenSampleError):
        read_json(path)
    assert not list(tmp_path.glob("*.tmp"))


def test_tiny_image_and_missing_reference():
    with pytest.raises(GoldenSampleError):
        lab_grid(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(GoldenSampleError):
        validate_reference(reference(), "changed")


def test_fixed_grid_ignores_one_pixel_detection_box_jitter(tmp_path):
    scene = np.random.default_rng(17).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    frames = readings()
    for index, frame in enumerate(frames):
        left = index % 2
        crop = scene[:, left:]
        path = tmp_path / f"crop_{index}.png"
        cv2.imwrite(str(path), crop)
        frame["positions"][0].update(bbox=[left, 0, 40, 40], lab=lab_grid(crop), crop_path=str(path))
    raw = np.asarray([f["positions"][0]["lab"] for f in frames])
    assert np.max(np.linalg.norm(raw - raw.mean(0), axis=2)) > 1
    baseline = build_reference(frames, identity="config", operator="e", sample_id="G1",
                               delta_e=3, repeatability=1)
    assert baseline["positions"][0]["jitter"] == 0
    # Inset by the drift align_positions tolerates (15% of 40px), not by a
    # token two pixels: an ROI tighter than the matcher's own tolerance fails
    # containment on a shift the matcher has just called the same position.
    assert baseline["positions"][0]["measurement_bbox"] == [7, 6, 34, 34]
    result = evaluate_readings(frames[:3], baseline, "config")
    assert result["status"] == "OK"
    assert result["rows"][0]["delta_e"] == 0


def test_fixed_grid_survives_the_drift_alignment_accepts(tmp_path):
    """Five pixels on a 40px box: inside tolerance, so it must measure.

    The two rules used to disagree -- align_positions accepted the box and
    fixed_grid_measurements then aborted the whole pre-shift check over it,
    naming the fixture for what was really a too-tight ROI.
    """
    scene = np.random.default_rng(17).integers(0, 256, (40, 40, 3), dtype=np.uint8)

    def framed(count, left):
        frames = readings(count)
        for index, frame in enumerate(frames):
            crop = scene[:, left:]
            path = tmp_path / f"crop_{left}_{index}.png"
            cv2.imwrite(str(path), crop)
            frame["positions"][0].update(
                bbox=[left, 0, 40, 40], lab=lab_grid(crop), crop_path=str(path)
            )
        return frames

    baseline = build_reference(framed(5, 0), identity="config", operator="e",
                               sample_id="G1", delta_e=3, repeatability=1)

    result = evaluate_readings(framed(3, 5), baseline, "config")

    # Same physical pixels despite the shift, so no colour difference at all.
    assert result["status"] == "OK"
    assert result["rows"][0]["delta_e"] == 0


def test_a_stored_roi_outside_the_crop_is_still_refused(tmp_path):
    """The guard stays for baselines written before the inset was widened."""
    scene = np.random.default_rng(17).integers(0, 256, (40, 40, 3), dtype=np.uint8)
    crop = scene[:, 1:]
    path = tmp_path / "crop.png"
    cv2.imwrite(str(path), crop)
    position = dict(readings(1)[0]["positions"][0])
    position.update(bbox=[1, 0, 40, 40], lab=lab_grid(crop), crop_path=str(path))
    legacy = [{**position, "measurement_bbox": [0, 0, 39, 40]}]

    with pytest.raises(GoldenSampleError, match="未涵蓋"):
        fixed_grid_measurements([[position]], legacy)


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


def test_retention_floor_comes_from_the_baseline_not_a_constant():
    """Hard-coding 0.6 kept a tuned station judged against the untuned value."""
    baseline = build_reference(
        readings(), identity="config", operator="e", sample_id="G1",
        delta_e=3, repeatability=1, retention=0.9,
    )
    assert baseline["margin_retention"] == 0.9
    # Baseline margin is 0.3, so a 0.9 floor trips at anything under 0.27.
    weak = readings(3)
    for reading in weak:
        reading["positions"][0]["margin"] = 0.26
    result = evaluate_readings(weak, baseline, "config")
    assert result["status"] == "NG"
    assert any("90%" in reason for row in result["rows"] for reason in row["reasons"])
    assert result["margin_retention"] == 0.9
    # The same measurement passes under the default floor.
    lenient = build_reference(
        readings(), identity="config", operator="e", sample_id="G1",
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
        readings(), identity="config", operator="e", sample_id="G1",
        delta_e=3, repeatability=1,
        conditions={"exposure_time": "35559.0000", "gain": "23.0"},
    )
    assert baseline["observed_conditions"]["exposure_time"] == "35559.0000"
    # Identity is unchanged by a recalibration, so the stored baseline still
    # validates and the daily check actually runs.
    validate_reference(baseline, "config")
    assert evaluate_readings(readings(3), baseline, "config")["status"] == "OK"


def test_condition_drift_is_quiet_when_nothing_moved():
    from core.services.golden_sample import condition_drift

    same = {"exposure_time": "35559.0000", "gain": "23.0"}
    assert condition_drift(same, dict(same)) == []
    assert condition_drift(None, same) == []
    assert condition_drift(same, None) == []
    # Numeric equality, not string equality: the writer formats to 4 places.
    assert condition_drift({"gain": "23.0"}, {"gain": "23.0000"}) == []


def test_schema_bump_and_identity_mismatch_say_different_things():
    """An operator's next action differs, so the message must too."""
    baseline = reference()
    stale = dict(baseline, schema="golden-lab-fixed-grid-v3")
    with pytest.raises(GoldenSampleError) as raised:
        validate_reference(stale, baseline["identity"])
    assert "舊版格式" in str(raised.value)
    with pytest.raises(GoldenSampleError) as raised:
        validate_reference(baseline, "a-different-station-setup")
    assert "不一致" in str(raised.value)
