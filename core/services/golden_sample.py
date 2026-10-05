"""Golden-board repeatability measurements, independent of color acceptance masks.

Each position is compared pixel for pixel against an image of the golden board
taken when the baseline was built. The frame is first aligned to that image,
because an operator re-seating the board moves the wire a pixel or two, and on
a 20-pixel-wide sampling window a pixel is the whole difference between wire
and board. Pixels that sit on an edge, a print mark or the crimp terminal are
left out by where they are, never by their colour: rejecting pixels by their
expected color would hide the environmental drift being measured.
Pure calculations are separated from snapshot and profile persistence.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

SCHEMA = "golden-lab-registered-v5"
BASELINE_COUNT = 5
CHECK_COUNT = 3

#: Station settings that change what a colour measurement *means*, and so must
#: invalidate a stored baseline. Hashing the whole config file instead made an
#: unrelated edit (``save_crops``, ``model_version``) throw the baseline away,
#: which trained everyone to rebuild baselines casually -- the opposite of the
#: control this check exists to provide.
#:
#: This tuple is a correctness boundary: a colour-relevant field missing from it
#: is a drift the check will not notice. ``test_golden_sample`` locks it against
#: the station config schema, so widening the schema without revisiting this
#: list fails there rather than silently on the line.
COLOR_IDENTITY_FIELDS = (
    # What is measured, and where in the frame.
    "enable_color_check",
    "color_checker_type",
    "color_model_path",
    "color_roi_policy",
    "color_decision_tuning",
    # The illumination *intent*. Not the exposure/gain the calibration loop
    # lands on to reach it -- see OBSERVED_CONDITION_FIELDS.
    "calibration",
    # Detector geometry: the crop boxes the fixed sampling grid is cut from.
    "weights",
    "conf_thres",
    "iou_thres",
    "imgsz",
)

#: Values the calibration loop *reads back from the hardware* and writes into
#: the config (``calibration_session.record_current``), rather than settings a
#: person chose. They land on a different number every calibration.
#:
#: Deliberately NOT part of identity. Expiring the baseline whenever exposure
#: moved disabled the only check that can catch a bad exposure: the operator
#: would rebuild, and the drift would be absorbed into the new normal --
#: exactly what this check exists to prevent. Left in place, a bad calibration
#: surfaces as 顏色偏移超限, which is actionable. Recorded on the baseline and
#: reported as context so the change is visible without being silently fatal.
OBSERVED_CONDITION_FIELDS = ("exposure_time", "gain", "light_brightness")

#: Nested keys kept for identity, as dotted paths. The rest of ``color_preflight``
#: (recorded margins, timestamps, operator) describes the legacy score check and
#: moves on every recalibration; only the retention floor changes this verdict.
COLOR_IDENTITY_NESTED_FIELDS = ("color_preflight.minimum_margin_retention",)

#: Absolute repeatability ceiling. The proposed repeatability limit is derived
#: from the measurement it will judge, so without a fixed ceiling a station
#: could pass a limit it set for itself. A board this unstable is not a usable
#: reference regardless of what limit is chosen.
MAX_ACCEPTABLE_JITTER = 6.0

#: Proposed repeatability limit = worst measured jitter x this. Headroom for
#: ordinary run-to-run variation without accepting a drifting fixture.
JITTER_SAFETY_FACTOR = 1.5

#: How far a detection box may sit from the baseline's, as a fraction of its
#: own size, and still be taken as the same physical position.
#:
#: Read by two rules that have to agree. ``align_positions`` decides what
#: counts as the same position, and ``shared_roi`` insets the sampling window
#: by the same fraction, which is also the room the window has to be found in
#: when the board was re-seated. When the inset was a flat two pixels the two
#: disagreed: a shift the matcher had just accepted then failed containment,
#: and a whole pre-shift check aborted with a message about the fixture over
#: three pixels.
POSITION_DRIFT_TOLERANCE = 0.15

#: Floor for that inset, so a box small enough for the proportional inset to
#: round to nothing still gets room for ordinary box rounding.
MIN_ROI_INSET = 2

#: Proposed colour limit = worst *normal* day-to-day delta-E x this. Cross-day
#: drift is larger than within-session jitter, so the colour limit cannot be
#: read off a single session.
DELTA_E_SAFETY_FACTOR = 1.5

#: With no history, the colour limit falls back to a multiple of the measured
#: jitter: a day-to-day shift has to clear the station's own noise to be
#: detectable at all.
DELTA_E_JITTER_MULTIPLE = 3.0

#: History entries further than this multiple of the median are treated as real
#: excursions, not as the normal spread the limit should tolerate.
OUTLIER_MEDIAN_MULTIPLE = 3.0

#: Mirrors ``core.services.color_preflight.DEFAULT_MINIMUM_MARGIN_RETENTION``.
#: Imported lazily in :func:`margin_retention` to keep this module free of the
#: heavier preflight import chain.
FALLBACK_MARGIN_RETENTION = 0.6

#: Normalised cross-correlation below which a frame is not taken to show the
#: baseline's scene at all. Measured over Cable1/A's stored checks: re-seating
#: the board by up to three pixels scored 0.89-1.00, while the 2026-09-21
#: frames in which the whole scene had changed scored 0.36-0.45.
MIN_REGISTRATION_SCORE = 0.7

#: A baseline pixel is measured only when none of its eight neighbours differs
#: from it by more than this ΔE76. Alignment is to the whole pixel and a wire
#: bends a little between placements, so a pixel can still be read one pixel
#: off; next to an edge, a print mark or the crimp terminal that one pixel is
#: a different material, and the 4 x 4 fixed grid used to measure exactly
#: that. On Cable1/A, 4 and 8 separated placement from real faults as well as
#: 6 does; 6 keeps the interior of every chromatic wire.
STABLE_PIXEL_DELTA_E = 6.0

#: Stable pixels a cell needs before its median stands for anything.
MIN_STABLE_PIXELS = 8

#: Cells a position must be able to judge, or its baseline is refused.
MIN_JUDGED_CELLS = 2

#: A template flatter than this grey-level spread has nothing to align on. It
#: also has nothing a small misplacement could change, so it is read in place.
MIN_TEMPLATE_CONTRAST = 1.0

#: Past sessions re-measured for a limit proposal, newest first.
REPLAY_SESSION_LIMIT = 60


class GoldenSampleError(ValueError):
    """Incomplete evidence or an incompatible golden-board reference."""


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GoldenSampleError(f"無法讀取 {path.name}：{exc}") from exc
    if not isinstance(value, dict):
        raise GoldenSampleError(f"{path.name} 格式錯誤")
    return value


def _nested(config: Mapping, dotted: str):
    value = config
    for segment in dotted.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            return None
        value = value[segment]
    return value


def read_config_mapping(config_path: Path) -> dict:
    """Parse a station config, treating an unreadable one as a hard stop.

    A silently-empty config would make every station share one identity, so a
    parse failure must never degrade into ``{}``.
    """
    try:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise GoldenSampleError(f"無法讀取站點設定：{exc}") from exc
    if not isinstance(payload, dict):
        raise GoldenSampleError("站點設定格式錯誤，無法判斷顏色量測條件")
    return payload


def color_identity_payload(config: Mapping) -> dict:
    """The colour-relevant station settings, in a stable, auditable shape.

    Stored alongside the baseline so an operator asking "why did my baseline
    expire" gets the changed field, not just a mismatched hash.
    """
    payload = {
        field: config.get(field)
        for field in COLOR_IDENTITY_FIELDS
        if config.get(field) is not None
    }
    for dotted in COLOR_IDENTITY_NESTED_FIELDS:
        value = _nested(config, dotted)
        if value is not None:
            payload[dotted] = value
    return payload


def observed_conditions(config: Mapping, overrides: Mapping | None = None) -> dict:
    """Hardware values the calibration loop landed on, for the record.

    Reported alongside a result so a drifting exposure is visible, without
    being allowed to invalidate the baseline that would detect its effect.

    ``overrides`` carries values in force that the config file does not
    record --- today, an exposure a session auto-calibration landed on.
    Without it this reads the file, sees a number nobody has changed, and
    reports no drift through a day in which the exposure moved by nearly ten
    percent: the report would be at its quietest exactly when it had the most
    to say. The fields are still only the observed ones, so feeding a live
    value here can never expire a baseline.
    """
    overrides = overrides or {}
    conditions = {}
    for field in OBSERVED_CONDITION_FIELDS:
        value = overrides.get(field)
        if value is None:
            value = config.get(field)
        if value is not None:
            conditions[field] = value
    return conditions


def condition_drift(baseline: Mapping | None, current: Mapping | None) -> list[str]:
    """Human-readable differences between recorded and current conditions."""
    baseline, current = baseline or {}, current or {}
    labels = {"exposure_time": "曝光", "gain": "增益", "light_brightness": "光源亮度"}
    drift = []
    for field in OBSERVED_CONDITION_FIELDS:
        was, now = baseline.get(field), current.get(field)
        if was is None or now is None:
            continue
        try:
            changed = abs(float(was) - float(now)) > 1e-9
        except (TypeError, ValueError):
            changed = str(was) != str(now)
        if changed:
            drift.append(f"{labels[field]} {was} → {now}")
    return drift


def margin_retention(config: Mapping) -> float:
    """Retention floor the station scores production colour against.

    Reads the same ``color_preflight.minimum_margin_retention`` the legacy
    check uses; hard-coding 0.6 here meant a tuned station kept being judged
    against the untuned value.
    """
    raw = _nested(config, "color_preflight.minimum_margin_retention")
    if raw is None:
        return FALLBACK_MARGIN_RETENTION
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise GoldenSampleError("minimum_margin_retention 不是有效數值") from exc
    if not np.isfinite(value) or not 0 < value <= 1:
        raise GoldenSampleError("minimum_margin_retention 必須介於 0 與 1 之間")
    return value


def configuration_identity(
    config_path: Path, baseline_path: Path | None, *, config: Mapping | None = None
) -> str:
    """Identity over the colour decision inputs, not the whole config file."""
    if config is None:
        config = read_config_mapping(config_path)
    digest = hashlib.sha256(
        json.dumps(color_identity_payload(config), sort_keys=True, default=str).encode("utf-8")
    )
    if baseline_path is not None:
        digest.update(baseline_path.read_bytes())
    return digest.hexdigest()


def lab_grid(image: np.ndarray) -> list:
    if image is None or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise GoldenSampleError("裁切影像缺失或小於 8 × 8，無法量測")
    lab = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return [
        np.median(cell.reshape(-1, 3), axis=0).tolist()
        for row in np.array_split(lab, 4, axis=0)
        for cell in np.array_split(row, 4, axis=1)
    ]


def measure_snapshot(path: Path, expected_colors: tuple[str, ...]) -> dict:
    try:
        return _measure_snapshot(path, expected_colors)
    except (TypeError, KeyError, IndexError, OverflowError) as exc:
        raise GoldenSampleError("檢測快照欄位格式錯誤") from exc


def _measure_snapshot(path: Path, expected_colors: tuple[str, ...]) -> dict:
    payload = read_json(path)
    result = payload.get("color_result")
    if not isinstance(result, dict) or result.get("status") != "evaluated":
        raise GoldenSampleError("本次沒有完整的顏色檢測結果")
    items = result.get("items")
    artifacts = payload.get("artifacts")
    if not isinstance(items, list) or not items or not isinstance(artifacts, dict):
        raise GoldenSampleError("缺少顏色量測或影像")
    crops = {}
    for raw in artifacts.get("cropped_paths", []):
        crop = Path(raw)
        try:
            index = int(crop.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if index in crops:
            raise GoldenSampleError("裁切影像索引重複")
        crops[index] = crop
    positions = []
    indices = set()
    for item in items:
        if not isinstance(item, dict):
            raise GoldenSampleError("顏色量測格式錯誤")
        index = item.get("index")
        if not isinstance(index, int) or index in indices:
            raise GoldenSampleError("顏色量測索引缺失或重複")
        indices.add(index)
        crop = crops.get(index)
        if crop is None or not crop.is_file():
            raise GoldenSampleError(f"位置 {index + 1} 缺少裁切影像")
        declared = str(item.get("class_name") or item.get("class") or "")
        if crop.stem.rsplit("_", 2)[-2].casefold() != declared.casefold():
            raise GoldenSampleError("裁切影像與檢測類別不一致")
        try:
            box = np.asarray(item.get("bbox"), dtype=float)
            margin = float(item["threshold"]) - float(item["diff"])
        except (TypeError, ValueError, KeyError) as exc:
            raise GoldenSampleError("位置或分數格式錯誤") from exc
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(margin):
            raise GoldenSampleError("位置或分數不是有效數值")
        if np.any(box[2:] <= box[:2]):
            raise GoldenSampleError("取樣位置範圍無效")
        image = cv2.imdecode(np.frombuffer(crop.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        positions.append(
            {
                "color": str(item.get("best_color") or "").casefold(),
            "bbox": box.tolist(),
            "crop_path": str(crop.resolve()),
                "lab": lab_grid(image),
                "margin": margin,
                "accepted": item.get("is_ok") is True and item.get("measurement_is_ok", True) is True,
            }
        )
    if Counter(p["color"] for p in positions) != Counter(c.casefold() for c in expected_colors):
        raise GoldenSampleError("golden sample 色別／數量不符，請檢查樣品與辨識結果")
    # Detection order is not stable between frames; fixture coordinates are.
    positions.sort(key=lambda p: ((p["bbox"][0] + p["bbox"][2]) / 2, (p["bbox"][1] + p["bbox"][3]) / 2))
    return {"source": str(path.resolve()), "timestamp": str(payload.get("timestamp", "")), "positions": positions}


def validate_readings(readings: list[dict], count: int) -> None:
    if len(readings) < count or len({r["source"] for r in readings}) != len(readings):
        raise GoldenSampleError(f"需要至少 {count} 次不同的新檢測")
    anchor = readings[0]["positions"]
    if not anchor:
        raise GoldenSampleError("沒有可量測的位置")
    for reading in readings:
        align_positions(reading["positions"], anchor)


def align_positions(positions: list[dict], anchor: list[dict]) -> list[dict]:
    if len(positions) != len(anchor):
        raise GoldenSampleError("位置數量與參考不符")
    aligned = []
    used = set()
    for reference in anchor:
        original = np.asarray(reference["bbox"])
        scale = np.tile(np.maximum(original[2:] - original[:2], 1), 2)
        candidates = [
            index
            for index, current in enumerate(positions)
            if current["color"] == reference["color"]
            and np.max(np.abs(np.asarray(current["bbox"]) - original) / scale)
            <= POSITION_DRIFT_TOLERANCE
        ]
        if len(candidates) != 1 or candidates[0] in used:
            raise GoldenSampleError("取樣位置／色別與參考不一致，請固定 golden sample 與治具")
        used.add(candidates[0])
        aligned.append(positions[candidates[0]])
    return aligned


def validate_limits(delta_e: float, repeatability: float) -> None:
    if not all(np.isfinite(v) and 0 < v <= 100 for v in (delta_e, repeatability)):
        raise GoldenSampleError("色差及波動上限必須為 0 到 100 之間的有限正數")
    if repeatability > delta_e:
        raise GoldenSampleError("連拍波動上限不可大於色差上限")


def to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def grid_cells(height: int, width: int) -> list[tuple[slice, slice]]:
    """The 4 x 4 split :func:`lab_grid` uses, as index slices, row by row."""

    def bounds(size: int) -> list[slice]:
        edges = np.cumsum([0] + [len(part) for part in np.array_split(np.arange(size), 4)])
        return [slice(int(start), int(stop)) for start, stop in zip(edges[:-1], edges[1:], strict=True)]

    return [(rows, columns) for rows in bounds(height) for columns in bounds(width)]


def stable_pixels(template: np.ndarray) -> np.ndarray:
    """Pixels a one-pixel misreading cannot move by more than the tolerance.

    Decided from the baseline image's own structure -- how much each pixel
    differs from its neighbours -- and never from what colour it is, so it
    cannot hide a colour that has drifted.
    """
    lab = to_lab(template)
    height, width = lab.shape[:2]
    padded = np.pad(lab, ((1, 1), (1, 1), (0, 0)), mode="edge")
    worst = np.zeros((height, width), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbour = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            worst = np.maximum(worst, np.linalg.norm(neighbour - lab, axis=2))
    return worst <= STABLE_PIXEL_DELTA_E


def judged_cells(mask: np.ndarray) -> list[bool]:
    return [
        int(np.count_nonzero(mask[rows, columns])) >= MIN_STABLE_PIXELS
        for rows, columns in grid_cells(*mask.shape)
    ]


def encode_template(patch: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".png", patch)
    if not ok:
        raise GoldenSampleError("無法保存基準取樣影像")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def decode_template(text, roi) -> np.ndarray:
    """The stored baseline image for one position, checked against its window."""
    try:
        raw = base64.b64decode(str(text), validate=True)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    except (binascii.Error, ValueError, cv2.error) as exc:
        raise GoldenSampleError("正常基準取樣影像損壞，請重新建立") from exc
    width, height = int(roi[2]) - int(roi[0]), int(roi[3]) - int(roi[1])
    if image is None or image.shape != (height, width, 3):
        raise GoldenSampleError("正常基準取樣影像損壞，請重新建立")
    return image


def shared_roi(boxes) -> np.ndarray:
    """The sampling window every baseline box contains, inset by the drift tolerance.

    The inset is what lets ``align_positions`` accept a box anywhere inside
    its tolerance and still find the whole window inside that box's crop.
    """
    boxes = np.asarray(boxes, dtype=int)
    inset = np.maximum(
        np.ceil(POSITION_DRIFT_TOLERANCE * (boxes[:, 2:] - boxes[:, :2]).min(axis=0)),
        MIN_ROI_INSET,
    ).astype(int)
    return np.concatenate((boxes[:, :2].max(axis=0) + inset, boxes[:, 2:].min(axis=0) - inset))


def _load_crop(position: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    raw = position.get("crop_path")
    if not raw:
        raise GoldenSampleError(f"位置 {index + 1} 缺少裁切影像")
    box = np.asarray(position["bbox"], dtype=int)
    image = cv2.imdecode(np.frombuffer(Path(raw).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (box[3] - box[1], box[2] - box[0]):
        raise GoldenSampleError(f"位置 {index + 1} 裁切影像與座標尺寸不符")
    return image, box


def locate(image: np.ndarray, box, template: np.ndarray, roi, index: int) -> tuple[int, int, float]:
    """Where ``template`` sits inside this crop, and how well it matched.

    Normalised cross-correlation on grey levels does not change under a gain
    or an offset, so aligning on it cannot absorb a brightness drift: it only
    removes where the board happens to sit today.
    """
    height, width = template.shape[:2]
    if image.shape[0] < height or image.shape[1] < width:
        raise GoldenSampleError(f"位置 {index + 1} 裁切圖小於取樣區域，請檢查治具與檢測框")
    grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if float(grey.std()) < MIN_TEMPLATE_CONTRAST:
        x, y = (int(value) for value in np.asarray(roi[:2], dtype=int) - np.asarray(box[:2], dtype=int))
        if x < 0 or y < 0 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise GoldenSampleError(f"位置 {index + 1} 裁切圖未涵蓋固定取樣區域，請檢查治具與檢測框")
        return x, y, 1.0
    scores = cv2.matchTemplate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), grey, cv2.TM_CCOEFF_NORMED)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, best, _, (x, y) = cv2.minMaxLoc(scores)
    return int(x), int(y), float(best)


def measure_frames(frames: list[dict], roi, template: np.ndarray, mask: np.ndarray, index: int) -> dict:
    """Paired Lab differences per cell and frame, read where the board actually is.

    Every pixel is compared with the same pixel of the baseline image, so a
    cell holding both wire and board cannot flip between the two the way a
    median of raw colours did. Unjudged cells are NaN.
    """
    height, width = template.shape[:2]
    reference = to_lab(template)
    judged = judged_cells(mask)
    cells = grid_cells(height, width)
    differences, scores, placements = [], [], []
    for position in frames:
        image, box = _load_crop(position, index)
        x, y, score = locate(image, box, template, roi, index)
        difference = to_lab(image[y : y + height, x : x + width]) - reference
        differences.append(
            [
                np.median(difference[rows, columns][mask[rows, columns]], axis=0)
                if keep
                else np.full(3, np.nan)
                for (rows, columns), keep in zip(cells, judged, strict=True)
            ]
        )
        scores.append(score)
        placements.append(
            [int(box[0] + x), int(box[1] + y), int(box[0] + x + width), int(box[1] + y + height)]
        )
    return {
        "differences": np.asarray(differences, dtype=float),
        "scores": scores,
        "placements": placements,
    }


def summarise_frames(differences: np.ndarray) -> dict:
    """Worst cell and worst frame: averaging must not hide a bad frame."""
    judged = ~np.isnan(differences[0, :, 0])
    kept = differences[:, judged]
    distances = np.linalg.norm(kept, axis=2)
    spread = np.linalg.norm(kept - kept.mean(axis=0), axis=2)
    cells: list[float | None] = [None] * differences.shape[1]
    for cell, value in zip(np.flatnonzero(judged), distances.max(axis=0), strict=True):
        cells[int(cell)] = float(value)
    return {
        "delta_e": float(distances.max()),
        "jitter": float(spread.max()),
        "delta_l": float(kept[:, :, 0].mean()),
        "cells": cells,
        "worst_frame": int(np.argmax(distances.max(axis=1))),
    }


def prepare_position(position: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roi = np.asarray(position["measurement_bbox"], dtype=int)
    template = decode_template(position["template"], roi)
    return roi, template, stable_pixels(template)


def _baseline_template(frames: list[dict], roi: np.ndarray, index: int) -> np.ndarray:
    """Median of the baseline frames, each aligned to the first one.

    Aligning before the median means a board that settled by a pixel during
    the five shots sharpens the template instead of smearing its edges.
    """
    first, box = _load_crop(frames[0], index)
    x, y = (int(value) for value in roi[:2] - box[:2])
    width, height = int(roi[2] - roi[0]), int(roi[3] - roi[1])
    anchor = first[y : y + height, x : x + width]
    patches = []
    for position in frames:
        image, frame_box = _load_crop(position, index)
        px, py, _ = locate(image, frame_box, anchor, roi, index)
        patches.append(image[py : py + height, px : px + width])
    return np.median(np.stack(patches), axis=0).round().astype(np.uint8)


def measure_baseline(
    readings: list[dict], *, identity: str, operator: str, sample_id: str
) -> dict:
    """Measure the golden board without judging it against any limit.

    Split from :func:`finalize_baseline` so the engineer can see what this
    station actually does before choosing the limits it will be held to.
    Asking for the numbers first meant guessing, and a guess typed into a spin
    box is indistinguishable from a decision.
    """
    validate_readings(readings, BASELINE_COUNT)
    if not operator.strip() or not sample_id.strip() or not identity:
        raise GoldenSampleError("請填寫建立人員與 golden sample 編號")
    ordered = [align_positions(r["positions"], readings[0]["positions"]) for r in readings]
    positions = []
    for index, anchor in enumerate(readings[0]["positions"]):
        frames = [frame[index] for frame in ordered]
        if any(not p["accepted"] or p["margin"] <= 0 for p in frames):
            raise GoldenSampleError("基準樣本必須全部通過生產顏色檢測")
        roi = shared_roi([p["bbox"] for p in frames])
        if np.any(roi[2:] - roi[:2] < 8):
            raise GoldenSampleError(f"位置 {index + 1} 缺少足夠的固定取樣區域，請重建基準")
        template = _baseline_template(frames, roi, index)
        mask = stable_pixels(template)
        if sum(judged_cells(mask)) < MIN_JUDGED_CELLS:
            raise GoldenSampleError(
                f"位置 {index + 1} 取樣區域內可穩定量測的部分不足（多為線緣、印字或端子），"
                "請確認檢測框與治具"
            )
        measured = measure_frames(frames, roi, template, mask, index)
        if min(measured["scores"]) < MIN_REGISTRATION_SCORE:
            raise GoldenSampleError(
                f"位置 {index + 1} 基準取樣期間影像不一致，請固定樣品與光源後重新量測"
            )
        positions.append(
            {
                "color": anchor["color"],
                "bbox": anchor["bbox"],
                "measurement_bbox": roi.tolist(),
                "template": encode_template(template),
                "jitter": summarise_frames(measured["differences"])["jitter"],
                "margin": min(p["margin"] for p in frames),
            }
        )
    worst = max((p["jitter"] for p in positions), default=0.0)
    # An absolute ceiling, because the repeatability limit may be derived from
    # this very measurement -- a station could otherwise pass a limit it just
    # set for itself, and the repeatability gate would mean nothing.
    if worst > MAX_ACCEPTABLE_JITTER:
        raise GoldenSampleError(
            f"連拍波動 {worst:.2f} 超過可接受上限 {MAX_ACCEPTABLE_JITTER:.2f}，"
            "請先穩定光源／治具再建立基準"
        )
    return {
        "identity": identity,
        "operator": operator.strip(),
        "sample_id": sample_id.strip(),
        "positions": positions,
        "measured_jitter": worst,
        "readings": readings,
    }


def finalize_baseline(
    measured: dict,
    *,
    delta_e: float,
    repeatability: float,
    retention: float = FALLBACK_MARGIN_RETENTION,
    identity_payload: Mapping | None = None,
    conditions: Mapping | None = None,
    limits_source: str = "manual",
) -> dict:
    """Apply the chosen limits to an accepted measurement and make it storable."""
    validate_limits(delta_e, repeatability)
    if not np.isfinite(retention) or not 0 < retention <= 1:
        raise GoldenSampleError("餘裕保留率必須介於 0 與 1 之間")
    positions = measured["positions"]
    unstable = [
        f"位置 {index + 1} {p['color']}：{p['jitter']:.2f}"
        for index, p in enumerate(positions)
        if p["jitter"] > repeatability
    ]
    if unstable:
        raise GoldenSampleError(f"基準量測不穩定（上限 {repeatability:.2f}）：" + "；".join(unstable))
    identity, operator = measured["identity"], measured["operator"]
    sample_id, readings = measured["sample_id"], measured["readings"]
    return {
        "schema": SCHEMA,
        "identity": identity,
        "operator": operator.strip(),
        "sample_id": sample_id.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "delta_e_limit": delta_e,
        "repeatability_limit": repeatability,
        "margin_retention": retention,
        # What the identity was computed over, so an expired baseline can be
        # diffed against the current config instead of only reported as stale.
        "identity_fields": dict(identity_payload or {}),
        # Not part of identity: recorded so a later exposure change can be
        # reported as context rather than expiring the baseline.
        "observed_conditions": dict(conditions or {}),
        # What the sample actually did, so the next baseline can be anchored on
        # measurement rather than on a prefill.
        "measured_jitter": measured["measured_jitter"],
        # Whether these limits were proposed from measurement or typed in, so an
        # audit can tell a decision from an accepted default.
        "limits_source": limits_source,
        "positions": positions,
        "readings": readings,
    }


def build_reference(
    readings: list[dict],
    *,
    identity: str,
    operator: str,
    sample_id: str,
    delta_e: float,
    repeatability: float,
    **kwargs,
) -> dict:
    """Measure and finalize in one step, for callers that already have limits."""
    measured = measure_baseline(
        readings, identity=identity, operator=operator, sample_id=sample_id
    )
    return finalize_baseline(measured, delta_e=delta_e, repeatability=repeatability, **kwargs)



def _session_stamp(session: Mapping) -> str:
    return str(session.get("checked_at") or session.get("created_at") or "")


def replay_sessions(
    sessions: list[dict], measured: dict, identity: str, *, limit: int = REPLAY_SESSION_LIMIT
) -> list[dict]:
    """Earlier sessions of this golden board, re-measured against a new baseline.

    The colour limit has to cover how far an ordinary day sits from *this*
    baseline. Re-measuring the stored images answers that directly, where the
    delta-E numbers saved with past checks were taken against other baselines
    and, before ``golden-lab-registered-v5``, by a measurement that mostly
    recorded where the board had been placed.

    No exposure filter. The calibration loop moves exposure to hold the image
    steady, so ordinary days land on different exposures; requiring an exact
    match left the history empty after every calibration. What the old filter
    guarded against -- Cable1/A's limit widened to 12.92 by a check taken while
    a release override held the camera at three times its exposure -- is
    caught here by the images themselves: a session whose frames do not align
    with the baseline is not a sample of this board, and a gross excursion is
    removed by :func:`normal_sessions`.

    A session whose images are gone, or that shows another sample, setting or
    scene, is skipped.
    """
    prepared = [prepare_position(position) for position in measured["positions"]]
    eligible = sorted(
        (
            session
            for session in sessions
            if session.get("identity") == identity
            and session.get("sample_id") == measured["sample_id"]
            and isinstance(session.get("readings"), list)
            and session["readings"]
        ),
        key=_session_stamp,
        reverse=True,
    )[:limit]
    samples = []
    for session in eligible:
        try:
            samples.append(_replay_one(session["readings"], measured["positions"], prepared))
        except (GoldenSampleError, OSError, cv2.error, KeyError, TypeError, ValueError, IndexError):
            continue
    return samples


def _replay_one(readings: list[dict], anchors: list[dict], prepared: list) -> dict:
    ordered = [align_positions(reading["positions"], anchors) for reading in readings]
    worst_delta = worst_jitter = 0.0
    for index, (roi, template, mask) in enumerate(prepared):
        measured = measure_frames([frame[index] for frame in ordered], roi, template, mask, index)
        if min(measured["scores"]) < MIN_REGISTRATION_SCORE:
            raise GoldenSampleError("影像與基準無法對位")
        summary = summarise_frames(measured["differences"])
        worst_delta = max(worst_delta, summary["delta_e"])
        worst_jitter = max(worst_jitter, summary["jitter"])
    return {"delta_e": worst_delta, "jitter": worst_jitter}


def normal_sessions(samples: list[dict]) -> list[dict]:
    """Replayed sessions that describe the ordinary spread, worst-first removed.

    A session that was unstable, or that sat far from the baseline because
    the station really had drifted, says nothing about the *normal* spread, so
    neither may widen a limit. Excursions are found by distance from the
    median rather than by any stored OK/NG verdict -- that verdict came from
    the limit being replaced.
    """
    steady = [sample for sample in samples if sample["jitter"] <= MAX_ACCEPTABLE_JITTER]
    if not steady:
        return []
    median = float(np.median([sample["delta_e"] for sample in steady]))
    if median <= 0:
        return steady
    return [sample for sample in steady if sample["delta_e"] <= median * OUTLIER_MEDIAN_MULTIPLE]


def propose_limits(measured_jitter: float, history: list[dict] | None = None) -> dict:
    """Suggest limits from what this station measurably does.

    ``history`` holds :func:`replay_sessions` samples. Returns the two limits
    plus the reasoning to put on screen: an unexplained number is just a
    different prefill.
    """
    jitter = float(measured_jitter or 0.0)
    normal = normal_sessions(history or [])
    # Run-to-run noise seen on other days is just as ordinary as today's.
    seen_jitter = max([jitter] + [sample["jitter"] for sample in normal])
    repeatability = max(round(seen_jitter * JITTER_SAFETY_FACTOR, 2), 0.5)
    if normal:
        worst_normal = max(sample["delta_e"] for sample in normal)
        delta_e = round(worst_normal * DELTA_E_SAFETY_FACTOR, 2)
        basis = (
            f"以本次基準重算過往 {len(normal)} 次取樣（已排除無法對位及離群者），"
            f"最大色差 {worst_normal:.2f} × {DELTA_E_SAFETY_FACTOR} 建議 {delta_e:.2f}"
        )
    else:
        delta_e = round(max(jitter * DELTA_E_JITTER_MULTIPLE, repeatability * 2), 2)
        basis = (
            f"尚無可重算的跨日影像，暫以本次實測波動 {jitter:.2f} × {DELTA_E_JITTER_MULTIPLE:.0f} "
            f"建議 {delta_e:.2f}；累積數日紀錄後應重新檢討"
        )
    # The colour limit must leave room above the station's own noise, or every
    # ordinary run reads as a colour shift.
    delta_e = max(delta_e, round(repeatability + jitter, 2), repeatability)
    seen = f"（過往取樣最大 {seen_jitter:.2f}）" if seen_jitter > jitter else ""
    return {
        "delta_e": delta_e,
        "repeatability": repeatability,
        "measured_jitter": jitter,
        "history_count": len(normal),
        "basis": (
            f"本次實測最大連拍波動 {jitter:.2f}{seen}，"
            f"建議波動上限 {repeatability:.2f}（× {JITTER_SAFETY_FACTOR}）。{basis}"
        ),
    }


def validate_reference(reference: dict, identity: str) -> None:
    # Two different causes, two different actions for the operator: a schema
    # bump is a one-off migration, a mismatched identity means something about
    # this station's colour setup actually changed.
    if reference.get("schema") != SCHEMA:
        raise GoldenSampleError(
            f"正常基準為舊版格式（{reference.get('schema') or '未知'}），"
            f"目前版本為 {SCHEMA}，需重新建立一次基準"
        )
    if reference.get("identity") != identity:
        raise GoldenSampleError("正常基準與目前設定／顏色模型不一致，需重新建立基準")
    try:
        validate_limits(reference["delta_e_limit"], reference["repeatability_limit"])
        positions = reference["positions"]
        if not isinstance(positions, list) or not positions:
            raise ValueError("positions")
        for position in positions:
            box = np.asarray(position["bbox"], dtype=float)
            roi = np.asarray(position["measurement_bbox"], dtype=float)
            if box.shape != (4,) or roi.shape != (4,):
                raise ValueError("geometry")
            if not np.isfinite(box).all() or not np.isfinite(roi).all():
                raise ValueError("geometry")
            if np.any(box[2:] <= box[:2]) or np.any(roi[2:] - roi[:2] < 8):
                raise ValueError("geometry")
            if not position["color"] or not 0 < float(position["margin"]) <= 1:
                raise ValueError("position")
            decode_template(position["template"], roi.astype(int))
        if not reference["sample_id"] or not reference["created_at"]:
            raise ValueError("sample_id")
        retention = float(reference["margin_retention"])
        if not np.isfinite(retention) or not 0 < retention <= 1:
            raise ValueError("margin_retention")
    except (GoldenSampleError, KeyError, TypeError, ValueError) as exc:
        raise GoldenSampleError("正常基準資料損壞或不完整，請重新建立") from exc


def evaluate_readings(readings: list[dict], reference: dict, identity: str) -> dict:
    validate_reference(reference, identity)
    validate_readings(readings, CHECK_COUNT)
    ordered = [align_positions(r["positions"], reference["positions"]) for r in readings]
    # Recorded when the baseline was built, from the station's configured
    # retention floor. Identity covers that config key, so a tuned station
    # rebuilds its baseline rather than being judged against a stale floor.
    retention = float(reference["margin_retention"])
    rows = []
    for index, anchor in enumerate(reference["positions"]):
        frames = [positions[index] for positions in ordered]
        roi, template, mask = prepare_position(anchor)
        measured = measure_frames(frames, roi, template, mask, index)
        summary = summarise_frames(measured["differences"])
        placement = measured["placements"][summary["worst_frame"]]
        alignment = min(measured["scores"])
        margin = min(p["margin"] for p in frames)
        reasons = []
        # First, because nothing after it is meaningful when it fails: the
        # frame does not show the scene the baseline was built from.
        if alignment < MIN_REGISTRATION_SCORE:
            reasons.append("取樣對位失敗")
        if summary["delta_e"] > reference["delta_e_limit"]:
            reasons.append("顏色偏移超限")
        if summary["jitter"] > reference["repeatability_limit"]:
            reasons.append("連拍不穩定")
        if any(not p["accepted"] for p in frames) or margin <= 0:
            reasons.append("生產顏色檢測未通過")
        elif margin < retention * anchor["margin"]:
            reasons.append(f"辨識餘裕低於基準 {retention:.0%}")
        rows.append(
            {
                "position": index + 1,
                "color": anchor["color"],
                "delta_e": summary["delta_e"],
                "delta_l": summary["delta_l"],
                "jitter": summary["jitter"],
                "margin": margin,
                "reasons": reasons,
                # Evidence for the cards, so the picture shown is the one judged.
                "cells": summary["cells"],
                "worst_frame": summary["worst_frame"],
                "measured_roi": placement,
                "alignment_shift": [placement[0] - int(roi[0]), placement[1] - int(roi[1])],
                "alignment_score": alignment,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "NG" if any(r["reasons"] for r in rows) else "OK",
        "rows": rows,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "reference_created_at": reference["created_at"],
        "sample_id": reference["sample_id"],
        "delta_e_limit": reference["delta_e_limit"],
        "repeatability_limit": reference["repeatability_limit"],
        "margin_retention": retention,
        "readings": readings,
    }


def write_json_atomic(path: Path, payload: dict) -> None:
    """Single writer (dialog session); atomic replacement for power-loss safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
