from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import core.services.color_baseline_recalibration as recalibration
from core.services.color_baseline_recalibration import (
    DEFAULT_COLORS,
    ColorBaselineCandidateStore,
    ColorBaselineError,
    ColorCropEvidence,
    StatsColorBaselineRebuilder,
    collect_confirmed_ok_evidence,
)
from core.types import DetectionItem, DetectionResult

HSV_BY_COLOR = {
    "Black": (0, 20, 30),
    "Green": (85, 120, 60),
    "Orange": (12, 210, 210),
    "Red": (2, 210, 190),
    "Yellow": (28, 190, 220),
}


def _crop(color: str, offset: int = 0) -> np.ndarray:
    h, s, v = HSV_BY_COLOR[color]
    hsv = np.full((32, 32, 3), (h, s, max(1, min(255, v + offset))), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _stats(color: str) -> dict:
    hsv = np.array(HSV_BY_COLOR[color], dtype=float)
    lab = cv2.cvtColor(
        np.uint8([[HSV_BY_COLOR[color]]]),
        cv2.COLOR_HSV2BGR,
    )
    lab = cv2.cvtColor(lab, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return {
        "count": 6,
        "hsv_mean": hsv.tolist(),
        "hsv_min": (hsv - [2, 20, 20]).clip(0, 255).tolist(),
        "hsv_max": (hsv + [2, 20, 20]).clip(0, 255).tolist(),
        "hsv_p10": hsv.tolist(),
        "hsv_p90": hsv.tolist(),
        "lab_mean": lab.tolist(),
        "lab_min": (lab - 10).clip(0, 255).tolist(),
        "lab_max": (lab + 10).clip(0, 255).tolist(),
        "lab_p10": lab.tolist(),
        "lab_p90": lab.tolist(),
        "coverage_mean": 1.0,
    }


def _base_model(tmp_path: Path) -> Path:
    path = tmp_path / "color_stats.json"
    path.write_text(
        json.dumps({"summary": {color: _stats(color) for color in DEFAULT_COLORS}}),
        encoding="utf-8",
    )
    return path


def _evidence(colors=DEFAULT_COLORS, count: int = 35):
    return tuple(
        ColorCropEvidence(
            sample_id=f"ACC-{index:03d}",
            color=color,
            image_bgr=_crop(color, (index % 5) - 2),
            source_sha256=f"{index:064x}",
        )
        for color in colors
        for index in range(count)
    )


def test_rebuilds_all_five_colors_with_holdout(tmp_path: Path) -> None:
    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=_evidence(),
    )

    assert build.status == "READY"
    assert set(build.model_payload["summary"]) == set(DEFAULT_COLORS)
    assert all(item.state == "REBUILT" for item in build.color_reports)
    assert all(item.training_crops + item.holdout_crops == 35 for item in build.color_reports)
    assert all(item.holdout_crops >= 5 for item in build.color_reports)
    assert build.report_payload["limitations"]


def test_isolated_color_outlier_photo_is_excluded_from_all_colors(
    tmp_path: Path,
) -> None:
    evidence = list(_evidence())
    for color in DEFAULT_COLORS:
        evidence.append(
            ColorCropEvidence(
                sample_id="ACC-OUTLIER",
                color=color,
                image_bgr=(
                    _crop("Green") if color == "Yellow" else _crop(color)
                ),
                source_sha256="f" * 64,
            )
        )

    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=tuple(evidence),
    )

    assert build.outlier_filter.status == "AUTO_EXCLUDED"
    assert build.outlier_filter.excluded_sample_ids == ("ACC-OUTLIER",)
    assert all(report.total_crops == 35 for report in build.color_reports)
    persisted = build.report_payload["statistical_outlier_filter"]
    assert persisted["excluded_sample_ids"] == ["ACC-OUTLIER"]
    assert build.model_payload["recalibration"][
        "statistical_outlier_filter"
    ] == persisted


def test_batch_wide_shift_is_not_mass_excluded(tmp_path: Path) -> None:
    evidence = list(_evidence())
    for index in range(4):
        evidence.append(
            ColorCropEvidence(
                sample_id=f"ACC-SHIFT-{index}",
                color="Yellow",
                image_bgr=_crop("Green"),
                source_sha256=f"{1000 + index:064x}",
            )
        )

    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=tuple(evidence),
    )

    assert build.outlier_filter.status == "SYSTEMATIC_SHIFT_NOT_FILTERED"
    assert build.outlier_filter.excluded_sample_ids == ()
    yellow_finding = next(
        finding
        for finding in build.outlier_filter.findings
        if finding.color == "Yellow"
    )
    assert yellow_finding.status == "SYSTEMATIC_SHIFT_NOT_FILTERED"
    assert len(yellow_finding.candidate_sample_ids) == 4
    yellow_report = next(
        report for report in build.color_reports if report.color == "Yellow"
    )
    assert yellow_report.total_crops == 39


def test_unsafe_color_update_is_rejected_and_old_baseline_is_preserved(
    tmp_path: Path,
) -> None:
    base = _base_model(tmp_path)
    original = json.loads(base.read_text(encoding="utf-8"))
    shifted_evidence = tuple(
        ColorCropEvidence(
            sample_id=item.sample_id,
            color=item.color,
            image_bgr=(
                _crop("Green") if item.color == "Yellow" else item.image_bgr
            ),
            source_sha256=item.source_sha256,
        )
        for item in _evidence()
    )

    build = StatsColorBaselineRebuilder().build(
        base_model_path=base,
        evidence=shifted_evidence,
    )

    yellow = next(item for item in build.color_reports if item.color == "Yellow")
    assert build.status == "READY"
    assert yellow.state == "PRESERVED_SAFETY_REJECTED"
    assert build.model_payload["summary"]["Yellow"] == original["summary"]["Yellow"]
    assert yellow.hue_drift == pytest.approx(0.0)
    assert yellow.lab_drift == pytest.approx(0.0)
    assert "Hue" not in yellow.note
    assert "Lab" not in yellow.note
    assert yellow.rejection_reasons
    persisted = next(
        item
        for item in build.report_payload["color_reports"]
        if item["color"] == "Yellow"
    )
    assert persisted["rejected_proposal"]["reasons"]
    assert "Yellow" in build.report_payload["preserved_by_safety"]


def test_holdout_regression_falls_back_until_final_checker_is_safe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base = _base_model(tmp_path)
    original = json.loads(base.read_text(encoding="utf-8"))
    old_checker = recalibration.StatsColorChecker.from_json(base)

    class _GreenRegressionChecker:
        def check(self, image):
            result = old_checker.check(image)
            if result.best_color.casefold() == "green":
                return SimpleNamespace(best_color="Black", is_ok=True)
            return result

    def checker_from_payload(payload):
        if payload["summary"] == original["summary"]:
            return old_checker
        return _GreenRegressionChecker()

    monkeypatch.setattr(
        recalibration,
        "_checker_from_payload",
        checker_from_payload,
    )
    build = StatsColorBaselineRebuilder(
        maximum_hue_drift=180.0,
        maximum_lab_drift=1000.0,
    ).build(
        base_model_path=base,
        evidence=_evidence(),
    )

    assert build.status == "READY"
    assert build.model_payload["summary"] == original["summary"]
    assert set(build.report_payload["preserved_by_safety"]) == set(DEFAULT_COLORS)
    assert all(
        report.candidate_holdout_correct == report.previous_holdout_correct
        for report in build.color_reports
    )
    green = next(report for report in build.color_reports if report.color == "Green")
    assert green.rejection_reasons == ("HOLDOUT_ACCURACY_REGRESSION",)


def test_preserves_colors_that_do_not_reach_minimum(tmp_path: Path) -> None:
    base = _base_model(tmp_path)
    original = json.loads(base.read_text(encoding="utf-8"))

    build = StatsColorBaselineRebuilder().build(
        base_model_path=base,
        evidence=_evidence(colors=("Black",), count=35),
    )

    assert build.status == "INCOMPLETE"
    green = next(item for item in build.color_reports if item.color == "Green")
    assert green.state == "PRESERVED_INSUFFICIENT"
    assert build.model_payload["summary"]["Green"] == original["summary"]["Green"]
    assert build.model_payload["summary"]["Black"]["count"] > 6


def test_candidate_report_preserves_evidence_source_lineage(tmp_path: Path) -> None:
    metadata = {
        "schema_version": 1,
        "counts": {
            "selected_total": 7,
            "selected_acceptance_ok": 5,
            "selected_color_review_ok": 2,
        },
        "samples": [
            {
                "sample_id": "color-review-1",
                "image_sha256": "a" * 64,
                "source_kind": "color_review",
            }
        ],
    }

    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=_evidence(),
        evidence_metadata=metadata,
    )

    assert build.report_payload["evidence_sources"] == metadata
    recalibration = build.model_payload["recalibration"]
    assert recalibration["evidence_source_counts"]["selected_total"] == 7
    assert len(recalibration["evidence_lineage_sha256"]) == 64


def test_candidate_store_is_deterministic_and_detects_tampering(
    tmp_path: Path,
) -> None:
    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=_evidence(),
    )
    store = ColorBaselineCandidateStore(tmp_path / ".color_baselines")

    first = store.commit(
        product="Cable1",
        area="A",
        model_type="yolo",
        build=build,
    )
    second = store.commit(
        product="Cable1",
        area="A",
        model_type="yolo",
        build=build,
    )

    assert first == second
    assert first.status == "READY"
    assert store.list_candidates(product="Cable1") == (first,)
    first.color_model_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ColorBaselineError, match="完整性"):
        store.load(first.manifest_path)


@dataclass(frozen=True)
class _Record:
    sample_id: str = "ACC-1"
    image_sha256: str = "a" * 64
    review_status: str = "confirmed"
    expected_verdict: str = "OK"


class _Repository:
    def image_file(self, _record):
        return Path("image.png")


class _Service:
    def detect_raw(self, _record, _path, *, inference_type, cancel_cb):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        processed = np.zeros((80, 100, 3), dtype=np.uint8)
        processed[20:60, 30:70] = _crop("Red", 0)[0, 0]
        return frame, DetectionResult(
            status="PASS",
            items=[
                DetectionItem("Red", 0.9, (30, 20, 70, 60)),
                DetectionItem("Other", 0.9, (0, 0, 10, 10)),
            ],
            processed_image=processed,
        )


def test_collects_only_known_component_crops_from_confirmed_ok() -> None:
    evidence = collect_confirmed_ok_evidence(
        repository=_Repository(),
        inference_service=_Service(),
        records=(_Record(),),
        inference_type="yolo",
    )

    assert len(evidence) == 1
    assert evidence[0].color == "Red"
    assert evidence[0].sample_id == "ACC-1"
    assert evidence[0].image_bgr.shape == (40, 40, 3)
    assert np.mean(evidence[0].image_bgr[:, :, 2]) > 100


def test_correct_predictions_requires_threshold_acceptance_and_color_match() -> None:
    evidence = tuple(
        ColorCropEvidence(
            sample_id=f"sample-{index}",
            color="Black",
            image_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
            source_sha256=f"{index:064x}",
        )
        for index in range(3)
    )

    class _Checker:
        def __init__(self) -> None:
            self._results = iter(
                (
                    SimpleNamespace(best_color="Black", is_ok=False),
                    SimpleNamespace(best_color="Black", is_ok=True),
                    SimpleNamespace(best_color="Orange", is_ok=True),
                )
            )

        def check(self, _image):
            return next(self._results)

    assert recalibration._correct_predictions(_Checker(), evidence, "Black") == 1



# ---------------------------------------------------------------------------
# Regressions from the color-baseline audit
# ---------------------------------------------------------------------------


def _mostly_background_crop(color: str) -> np.ndarray:
    """A crop where almost nothing carries the color's own kind of pixel."""
    hsv = np.zeros((64, 64, 3), np.uint8)
    h, s, v = HSV_BY_COLOR[color]
    hsv[:2, :2] = (h, s, v)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_crop_without_color_evidence_is_rejected_not_sampled_wholesale():
    """The mask used to be discarded and *every* pixel sampled instead.

    A crop that is mostly unsaturated background then produced a "color
    baseline" built out of that background, while the recorded coverage
    reported 0.00.
    """
    with pytest.raises(recalibration._InsufficientColorPixels) as excinfo:
        recalibration._sample_color_pixels(
            _mostly_background_crop("Red"), "Red", sample_size=64
        )

    assert excinfo.value.color == "Red"
    assert excinfo.value.coverage < 0.01


def test_clean_crop_sampling_is_unchanged():
    hsv_values, lab_values, coverage = recalibration._sample_color_pixels(
        _crop("Red"), "Red", sample_size=64
    )

    assert coverage == pytest.approx(1.0)
    assert len(hsv_values) == len(lab_values) == 64 * 64


def test_statistics_skip_unusable_crops_and_report_the_count_actually_used():
    rebuilder = StatsColorBaselineRebuilder()
    evidence = [
        ColorCropEvidence(
            sample_id=f"good-{index}",
            color="Red",
            image_bgr=_crop("Red", offset=index),
            source_sha256=f"sha-good-{index}",
        )
        for index in range(3)
    ] + [
        ColorCropEvidence(
            sample_id="empty-1",
            color="Red",
            image_bgr=_mostly_background_crop("Red"),
            source_sha256="sha-empty-1",
        )
    ]

    stats = rebuilder._calculate_stats(evidence)

    # ``count`` must describe the evidence the numbers came from, not the
    # evidence that was offered.
    assert stats["count"] == 3
    assert stats["skipped_crops"] == ["empty-1"]
    assert stats["coverage_mean"] == pytest.approx(1.0)


def test_statistics_fail_closed_when_no_crop_carries_color_evidence():
    rebuilder = StatsColorBaselineRebuilder()
    evidence = [
        ColorCropEvidence(
            sample_id=f"empty-{index}",
            color="Red",
            image_bgr=_mostly_background_crop("Red"),
            source_sha256=f"sha-empty-{index}",
        )
        for index in range(3)
    ]

    with pytest.raises(ColorBaselineError):
        rebuilder._calculate_stats(evidence)


def _two_tone_crop(color: str, spread: int) -> np.ndarray:
    """A crop whose hue sits ``spread`` either side of the color's own hue.

    The circular mean stays on the color, so the drift check sees nothing
    moving, while the distribution is far too wide to be one color.
    """
    h, s, v = HSV_BY_COLOR[color]
    hsv = np.full((32, 32, 3), (h, s, v), np.uint8)
    hsv[:16, :, 0] = (h - spread) % 180
    hsv[16:, :, 0] = (h + spread) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _washed_out_crop(color: str) -> np.ndarray:
    h, _s, v = HSV_BY_COLOR[color]
    hsv = np.full((32, 32, 3), (h, 30, v), np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_hue_spread_is_measured_on_the_circle():
    """Red evidence either side of the 0/179 seam is narrow, not circle-wide."""
    seam_red = {"hsv_p10": [2.0, 200.0, 150.0], "hsv_p90": [178.0, 200.0, 150.0],
                "hsv_mean": [0.0, 200.0, 150.0]}
    assert recalibration._hue_spread(seam_red) == pytest.approx(4.0)

    polluted = {"hsv_p10": [4.0, 150.0, 150.0], "hsv_p90": [90.0, 150.0, 150.0],
                "hsv_mean": [40.0, 150.0, 150.0]}
    assert recalibration._hue_spread(polluted) == pytest.approx(86.0)


def test_hue_spread_is_undefined_for_an_achromatic_color():
    """Black has no hue, so its spread must not raise a flag on every rebuild."""
    black = {"hsv_p10": [0.0, 20.0, 30.0], "hsv_p90": [170.0, 25.0, 40.0],
             "hsv_mean": [90.0, 22.0, 35.0]}
    assert recalibration._hue_spread(black) is None


def test_chroma_retention_is_undefined_when_the_baseline_is_achromatic():
    black = {"hsv_mean": [90.0, 22.0, 35.0]}
    assert recalibration._chroma_retention(black, {"hsv_mean": [90.0, 5.0, 35.0]}) is None

    red = {"hsv_mean": [2.0, 200.0, 190.0]}
    assert recalibration._chroma_retention(
        red, {"hsv_mean": [2.0, 50.0, 190.0]}
    ) == pytest.approx(0.25)


def test_widely_spread_evidence_asks_for_review_without_rejecting(tmp_path: Path) -> None:
    """Drift alone misses pollution that leaves the center in place.

    In the stored evidence one Green proposal drifted 13.7 -- inside the 18.0
    limit -- while its hue ran from 5 to 173.
    """
    evidence = tuple(
        ColorCropEvidence(
            sample_id=item.sample_id,
            color=item.color,
            image_bgr=(
                _two_tone_crop("Green", 25) if item.color == "Green" else item.image_bgr
            ),
            source_sha256=item.source_sha256,
        )
        for item in _evidence()
    )

    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=evidence,
    )

    green = next(item for item in build.color_reports if item.color == "Green")
    assert green.hue_spread is not None and green.hue_spread > 30.0
    assert recalibration.HUE_SPREAD_REVIEW_REASON in green.review_reasons
    # Asking for review must not double as a rejection.
    assert not green.rejection_reasons
    assert build.status == "REVIEW_REQUIRED"
    assert build.report_payload["review_required_colors"] == ["Green"]


def test_a_clean_rebuild_is_not_flagged_for_review(tmp_path: Path) -> None:
    """The guard against the check firing on good evidence."""
    build = StatsColorBaselineRebuilder().build(
        base_model_path=_base_model(tmp_path),
        evidence=_evidence(),
    )

    assert build.status == "READY"
    assert build.report_payload["review_required_colors"] == []
    assert all(not item.review_reasons for item in build.color_reports)
