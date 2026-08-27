"""Rebuild immutable Stats Color baselines from verified human-reviewed OK evidence.

The module intentionally separates numerical rebuilding from append-only
persistence.  It never changes a model config, an active color revision, or an
inspection release pointer.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from core.services.inspection_release_store import sha256_file
from core.services.slot_roi import ColorRoiPolicy, extract_bbox_roi
from core.stats_color_checker import StatsColorChecker, circular_hue_mean

logger = logging.getLogger(__name__)

COLOR_BASELINE_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "stats-robust-v4"
OUTLIER_FILTER_ALGORITHM = "per-color-sample-lab-mad-v1"
DEFAULT_COLORS = ("Black", "Green", "Orange", "Red", "Yellow")

#: Widest hue distribution (circular p10..p90, in OpenCV hue degrees) a single
#: color's evidence may show before a human is asked to look at it.
#:
#: Chosen from the stored baselines rather than guessed. The one approved
#: candidate spans 1.6 (Red), 2.3 (Orange), 7.5 (Green) and 9.6 (Yellow); every
#: candidate whose crops demonstrably picked up neighbouring wires spans 34 to
#: 89. Anything in between separates the two populations, and 30 leaves room
#: for a legitimately larger sample to be wider than the small approved one.
MAXIMUM_HUE_SPREAD = 30.0

#: Below this mean saturation a color has no meaningful hue, so its spread says
#: nothing. Black measures 21-29 across every stored baseline and would
#: otherwise raise the flag on every rebuild, which teaches reviewers to ignore
#: it.
ACHROMATIC_SATURATION = 40.0

#: A color's evidence is spread too widely across the hue circle to be one
#: color. Reported for human review; never an automatic rejection.
#:
#: Since the sampler narrows a crop to its dominant hue, a baseline rebuilt
#: here can no longer widen this way, and the reason now mainly guards
#: baselines that arrive from elsewhere -- the annotation tool, an older
#: algorithm version, or a hand-edited file.
HUE_SPREAD_REVIEW_REASON = "HUE_SPREAD_LIMIT_EXCEEDED"

#: Least of the approved baseline's saturation a proposal must retain before a
#: human is asked to look. Guards the blind spot in the spread check: hue
#: spread is meaningless below ``ACHROMATIC_SATURATION``, so evidence polluted
#: badly enough to wash a color out to grey would slip past it. In the stored
#: baselines exactly that happened -- one candidate's Red, Orange, Yellow and
#: Green all measure a mean saturation of 33-38 against an approved Red of 201,
#: and the four became statistically indistinguishable from each other.
MINIMUM_CHROMA_RETENTION = 0.5

#: Half-width, in OpenCV hue degrees, of the window kept around a crop's
#: dominant hue when sampling a chromatic color.
#:
#: A detection box drawn around one wire routinely catches part of the next
#: one, and the sampler used to feed every saturated pixel in the box into that
#: color's baseline. In the stored evidence this is the norm rather than an
#: edge case: the approved baseline's colors span 1.6 to 9.6 hue degrees, while
#: candidates built from boxes that caught a neighbour span 34 to 89 and
#: overlap each other. 15 comfortably contains a real color -- the widest in
#: the approved baseline needs 5 either side of its center -- while excluding a
#: neighbouring wire, whose hues sit tens of degrees away.
DOMINANT_HUE_WINDOW = 15.0

#: Bin count for locating the dominant hue. 36 bins is 5 degrees each, fine
#: enough to separate red from orange (their approved centers are 5.4 apart)
#: without splitting one color's own spread across many bins.
DOMINANT_HUE_BINS = 36

#: Least of a color's chromatic evidence that may belong to its dominant hue
#: before a human is asked to look.
#:
#: The sampler now keeps only the dominant-hue window, so a box that caught a
#: neighbouring wire no longer poisons the baseline -- but it still means the
#: detection boxes are wrong, and the reviewer should know. Deliberately set at
#: a level that needs no calibration to defend: below half, most of what the
#: box contained was not the color it names. The exact figure for milder
#: pollution is recorded as ``dominant_fraction_mean`` rather than guessed at
#: with a threshold there is no data to place.
MINIMUM_DOMINANT_FRACTION = 0.5

#: Most of what the detection boxes contained was not the color they name.
#: Reported for human review; never an automatic rejection.
DOMINANT_FRACTION_REVIEW_REASON = "DOMINANT_FRACTION_LOW"

#: A color that the approved baseline says is chromatic has washed out. Like
#: the spread reason, this asks for review rather than rejecting.
CHROMA_COLLAPSE_REVIEW_REASON = "CHROMA_COLLAPSE"

#: A candidate below this measured holdout accuracy is never READY, even when
#: it improves on an already-bad deployed baseline. Relative-only safety can
#: otherwise preserve or approve a baseline that remains operationally useless.
ABSOLUTE_HOLDOUT_ACCURACY_REASON = "ABSOLUTE_HOLDOUT_ACCURACY_BELOW_MINIMUM"
DEFAULT_MINIMUM_HOLDOUT_ACCURACY = 0.90


def _mean_saturation(stats: Mapping[str, Any]) -> float | None:
    try:
        return float(stats["hsv_mean"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _chroma_retention(
    base_stats: Mapping[str, Any], candidate_stats: Mapping[str, Any]
) -> float | None:
    """Fraction of the approved baseline's saturation a proposal still has.

    None when the approved baseline is itself achromatic -- Black has no
    saturation to lose, so the ratio would be noise.
    """
    base = _mean_saturation(base_stats)
    candidate = _mean_saturation(candidate_stats)
    if base is None or candidate is None or base < ACHROMATIC_SATURATION:
        return None
    return float(candidate / base) if base > 0 else None


def _hue_spread(stats: Mapping[str, Any]) -> float | None:
    """Circular width of a color's hue distribution, or None if undefined.

    Measured between the recorded 10th and 90th hue percentiles, so a handful
    of stray pixels cannot widen it, and measured *circularly*, so genuinely
    red evidence sitting either side of the 0/179 seam reads as narrow rather
    than as spanning the whole circle.

    This complements the drift check rather than duplicating it. Drift asks how
    far the new center moved from the approved one, and misses pollution that
    happens to leave the center in place: in the stored evidence, one Green
    proposal drifted only 13.7 -- inside the 18.0 limit -- while its hue ran
    from 5 to 173.
    """
    try:
        low = float(stats["hsv_p10"][0])
        high = float(stats["hsv_p90"][0])
        saturation = float(stats["hsv_mean"][1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if saturation < ACHROMATIC_SATURATION:
        return None
    delta = (high - low) % 180.0
    return float(min(delta, 180.0 - delta))


class ColorBaselineError(ValueError):
    """Raised when baseline evidence or an immutable candidate is invalid."""


class ColorBaselineCancelled(RuntimeError):
    """Raised when an operator cancels baseline collection or rebuilding."""


@dataclass(frozen=True)
class ColorBaselineImageSample:
    """One verified OK image selected from an auditable evidence source."""

    sample_id: str
    image_path: Path
    image_sha256: str
    product: str
    area: str
    source_kind: str
    source_manifest: str = ""


@dataclass(frozen=True)
class ColorCropEvidence:
    """One component crop tied to a verified OK image."""

    sample_id: str
    color: str
    image_bgr: np.ndarray
    source_sha256: str = ""


@dataclass(frozen=True)
class ColorBaselineColorReport:
    """Safety and evidence summary for one color."""

    color: str
    state: str
    total_crops: int
    training_crops: int
    holdout_crops: int
    previous_holdout_correct: int
    candidate_holdout_correct: int
    hue_drift: float | None
    lab_drift: float | None
    note: str
    rejected_proposal_holdout_correct: int | None = None
    rejected_proposal_hue_drift: float | None = None
    rejected_proposal_lab_drift: float | None = None
    rejection_reasons: tuple[str, ...] = ()
    #: Circular width of this color's hue evidence, or None when hue carries no
    #: meaning for it.
    hue_spread: float | None = None
    #: Saturation retained relative to the approved baseline, or None when that
    #: baseline is achromatic.
    chroma_retention: float | None = None
    #: How much of the evidence's chromatic content belonged to its dominant
    #: hue. Below 1.0 means detection boxes were catching more than the one
    #: wire they name.
    dominant_fraction: float | None = None
    #: Why this color needs a human to look at it. Distinct from
    #: ``rejection_reasons``: nothing was rejected, the evidence just does not
    #: look like a single color.
    review_reasons: tuple[str, ...] = ()

    @property
    def previous_accuracy(self) -> float | None:
        return _ratio(self.previous_holdout_correct, self.holdout_crops)

    @property
    def candidate_accuracy(self) -> float | None:
        return _ratio(self.candidate_holdout_correct, self.holdout_crops)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "color": self.color,
            "state": self.state,
            "total_crops": self.total_crops,
            "training_crops": self.training_crops,
            "holdout_crops": self.holdout_crops,
            "previous_holdout_correct": self.previous_holdout_correct,
            "candidate_holdout_correct": self.candidate_holdout_correct,
            "previous_accuracy": self.previous_accuracy,
            "candidate_accuracy": self.candidate_accuracy,
            "hue_drift": self.hue_drift,
            "lab_drift": self.lab_drift,
            "hue_spread": self.hue_spread,
            "chroma_retention": self.chroma_retention,
            "dominant_fraction": self.dominant_fraction,
            "note": self.note,
        }
        if self.review_reasons:
            payload["review_reasons"] = list(self.review_reasons)
        if self.rejection_reasons:
            payload["rejected_proposal"] = {
                "holdout_correct": self.rejected_proposal_holdout_correct,
                "hue_drift": self.rejected_proposal_hue_drift,
                "lab_drift": self.rejected_proposal_lab_drift,
                "reasons": list(self.rejection_reasons),
            }
        return payload


@dataclass(frozen=True)
class ColorBaselineOutlierColorFinding:
    """Robust outlier result for one expected color."""

    color: str
    sample_count: int
    status: str
    candidate_sample_ids: tuple[str, ...]
    excluded_sample_ids: tuple[str, ...]
    scores: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "color": self.color,
            "sample_count": self.sample_count,
            "status": self.status,
            "candidate_sample_ids": list(self.candidate_sample_ids),
            "excluded_sample_ids": list(self.excluded_sample_ids),
            "scores": [
                {"sample_id": sample_id, "robust_distance": score}
                for sample_id, score in self.scores
            ],
        }


@dataclass(frozen=True)
class ColorBaselineOutlierFilterReport:
    """Immutable audit record for photos omitted as isolated batch outliers."""

    status: str
    total_sample_count: int
    z_score_threshold: float
    maximum_auto_exclusion_fraction: float
    candidate_sample_ids: tuple[str, ...]
    excluded_sample_ids: tuple[str, ...]
    findings: tuple[ColorBaselineOutlierColorFinding, ...]

    @property
    def excluded_count(self) -> int:
        return len(self.excluded_sample_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": OUTLIER_FILTER_ALGORITHM,
            "status": self.status,
            "total_sample_count": self.total_sample_count,
            "z_score_threshold": self.z_score_threshold,
            "maximum_auto_exclusion_fraction": (
                self.maximum_auto_exclusion_fraction
            ),
            "candidate_sample_ids": list(self.candidate_sample_ids),
            "excluded_sample_ids": list(self.excluded_sample_ids),
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True)
class _RejectedColorProposal:
    holdout_correct: int
    hue_drift: float | None
    lab_drift: float | None
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ColorBaselineBuild:
    """Pure numerical output ready for immutable persistence."""

    status: str
    model_payload: dict[str, Any]
    report_payload: dict[str, Any]
    evidence_sha256: str
    color_reports: tuple[ColorBaselineColorReport, ...]
    outlier_filter: ColorBaselineOutlierFilterReport


@dataclass(frozen=True)
class ColorBaselineCandidate:
    """One verified append-only baseline candidate package."""

    candidate_id: str
    display_version: str
    product: str
    area: str
    model_type: str
    status: str
    algorithm: str
    created_at: str
    color_model_path: Path
    report_path: Path
    manifest_path: Path
    color_model_sha256: str
    report_sha256: str
    evidence_sha256: str
    colors: tuple[str, ...]


class StatsColorBaselineRebuilder:
    """Build robust five-color statistics while retaining a holdout split."""

    def __init__(
        self,
        *,
        minimum_crops_per_color: int = 30,
        minimum_holdout_crops: int = 5,
        holdout_fraction: float = 0.2,
        maximum_hue_drift: float = 18.0,
        maximum_hue_spread: float = MAXIMUM_HUE_SPREAD,
        minimum_chroma_retention: float = MINIMUM_CHROMA_RETENTION,
        minimum_dominant_fraction: float = MINIMUM_DOMINANT_FRACTION,
        maximum_lab_drift: float = 35.0,
        maximum_accuracy_regression: float = 0.02,
        minimum_holdout_accuracy: float = DEFAULT_MINIMUM_HOLDOUT_ACCURACY,
        outlier_z_score_threshold: float = 6.0,
        maximum_outlier_fraction: float = 0.1,
        minimum_outlier_sample_count: int = 20,
        sample_size: int = 64,
    ) -> None:
        if minimum_crops_per_color < 2:
            raise ColorBaselineError("每色最低裁切數必須至少為 2。")
        if minimum_holdout_crops < 1:
            raise ColorBaselineError("每色 holdout 數必須至少為 1。")
        if not 0.05 <= holdout_fraction <= 0.5:
            raise ColorBaselineError("holdout_fraction 必須介於 0.05 與 0.5。")
        if sample_size < 16:
            raise ColorBaselineError("sample_size 必須至少為 16。")
        if not np.isfinite(outlier_z_score_threshold) or outlier_z_score_threshold < 3.0:
            raise ColorBaselineError("outlier_z_score_threshold 必須至少為 3。")
        if not 0.0 <= maximum_outlier_fraction <= 0.25:
            raise ColorBaselineError("maximum_outlier_fraction 必須介於 0 與 0.25。")
        if minimum_outlier_sample_count < 5:
            raise ColorBaselineError("minimum_outlier_sample_count 必須至少為 5。")
        if not 0.0 <= minimum_holdout_accuracy <= 1.0:
            raise ColorBaselineError("minimum_holdout_accuracy 必須介於 0 與 1。")
        self.minimum_crops_per_color = minimum_crops_per_color
        self.minimum_holdout_crops = minimum_holdout_crops
        self.holdout_fraction = holdout_fraction
        self.maximum_hue_drift = maximum_hue_drift
        self.maximum_hue_spread = maximum_hue_spread
        self.minimum_chroma_retention = minimum_chroma_retention
        self.minimum_dominant_fraction = minimum_dominant_fraction
        self.maximum_lab_drift = maximum_lab_drift
        self.maximum_accuracy_regression = maximum_accuracy_regression
        self.minimum_holdout_accuracy = minimum_holdout_accuracy
        self.outlier_z_score_threshold = outlier_z_score_threshold
        self.maximum_outlier_fraction = maximum_outlier_fraction
        self.minimum_outlier_sample_count = minimum_outlier_sample_count
        self.sample_size = sample_size

    def build(
        self,
        *,
        base_model_path: str | Path,
        evidence: Sequence[ColorCropEvidence],
        expected_colors: Sequence[str] = DEFAULT_COLORS,
        evidence_metadata: Mapping[str, Any] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> ColorBaselineBuild:
        base_path = Path(base_model_path).expanduser().resolve()
        base_payload = _read_stats_payload(base_path)
        base_summary = base_payload["summary"]
        canonical_colors = _resolve_expected_colors(
            expected_colors,
            base_summary,
        )
        grouped = _validate_and_group_evidence(evidence, canonical_colors)
        normalized_metadata = _normalized_evidence_metadata(evidence_metadata)
        outlier_filter = _build_outlier_filter_report(
            grouped,
            z_score_threshold=self.outlier_z_score_threshold,
            maximum_auto_exclusion_fraction=self.maximum_outlier_fraction,
            minimum_sample_count=self.minimum_outlier_sample_count,
            sample_size=min(self.sample_size, 32),
            cancel_callback=cancel_callback,
        )
        excluded_sample_ids = set(outlier_filter.excluded_sample_ids)
        filtered_evidence = tuple(
            item for item in evidence if item.sample_id not in excluded_sample_ids
        )
        grouped = _validate_and_group_evidence(filtered_evidence, canonical_colors)
        digest_metadata = {
            **normalized_metadata,
            "statistical_outlier_filter": outlier_filter.to_dict(),
        }
        evidence_sha256 = _evidence_digest(filtered_evidence, digest_metadata)
        candidate_summary = json.loads(json.dumps(base_summary))
        split_by_color: dict[str, tuple[tuple[ColorCropEvidence, ...], tuple[ColorCropEvidence, ...]]] = {}
        preliminary_states: dict[str, tuple[str, str]] = {}
        proposal_dominant_by_color: dict[str, float | None] = {}

        for color in canonical_colors:
            _raise_if_cancelled(cancel_callback)
            color_evidence = grouped[color.casefold()]
            training, holdout = self._split(color_evidence)
            split_by_color[color] = (training, holdout)
            if len(color_evidence) < self.minimum_crops_per_color or len(holdout) < self.minimum_holdout_crops:
                preliminary_states[color] = (
                    "PRESERVED_INSUFFICIENT",
                    (
                        f"需至少 {self.minimum_crops_per_color} 個裁切，"
                        f"其中 holdout 至少 {self.minimum_holdout_crops} 個；"
                        "此色沿用原基準。"
                    ),
                )
                continue
            candidate_summary[color] = self._calculate_stats(training)
            # Keep the *proposal's* value: a color preserved by the safety
            # check has its summary replaced by the old baseline, and the
            # reviewer still needs to know what the rejected evidence looked
            # like -- that is the whole point of showing it to them.
            proposal_dominant_by_color[color] = candidate_summary[color].get(
                "dominant_fraction_mean"
            )
            preliminary_states[color] = ("REBUILT", "")

        model_payload = json.loads(json.dumps(base_payload))
        model_payload["summary"] = candidate_summary
        model_payload["recalibration"] = {
            "schema_version": COLOR_BASELINE_SCHEMA_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "base_model_sha256": sha256_file(base_path),
            "evidence_sha256": evidence_sha256,
            "minimum_crops_per_color": self.minimum_crops_per_color,
            "minimum_holdout_crops": self.minimum_holdout_crops,
            "holdout_fraction": self.holdout_fraction,
            "statistical_outlier_filter": outlier_filter.to_dict(),
        }
        if normalized_metadata:
            model_payload["recalibration"]["evidence_lineage_sha256"] = (
                hashlib.sha256(_canonical_json(normalized_metadata)).hexdigest()
            )
            model_payload["recalibration"]["evidence_source_counts"] = dict(
                normalized_metadata.get("counts") or {}
            )

        previous_checker = StatsColorChecker.from_json(base_path)
        proposal_checker = _checker_from_payload(model_payload)
        previous_correct_by_color: dict[str, int] = {}
        proposal_drift_by_color: dict[
            str, tuple[float | None, float | None]
        ] = {}
        rejected_proposals: dict[str, _RejectedColorProposal] = {}

        for color in canonical_colors:
            _raise_if_cancelled(cancel_callback)
            _training, holdout = split_by_color[color]
            previous_correct = _correct_predictions(
                previous_checker,
                holdout,
                color,
            )
            proposal_correct = _correct_predictions(
                proposal_checker,
                holdout,
                color,
            )
            hue_drift, lab_drift = _center_drift(
                base_summary[color],
                candidate_summary[color],
            )
            if color.casefold() == "black":
                # Hue is undefined for achromatic evidence. Using its numeric
                # OpenCV placeholder rejected Black proposals for moving 38
                # hue units even though their S/V and LAB evidence improved.
                hue_drift = None
            previous_correct_by_color[color] = previous_correct
            proposal_drift_by_color[color] = (hue_drift, lab_drift)
            if preliminary_states[color][0] != "REBUILT":
                continue
            rejection_reasons: list[str] = []
            if hue_drift is not None and hue_drift > self.maximum_hue_drift:
                rejection_reasons.append("HUE_DRIFT_LIMIT_EXCEEDED")
            if lab_drift is not None and lab_drift > self.maximum_lab_drift:
                rejection_reasons.append("LAB_DRIFT_LIMIT_EXCEEDED")
            if rejection_reasons:
                rejected_proposals[color] = _RejectedColorProposal(
                    holdout_correct=proposal_correct,
                    hue_drift=hue_drift,
                    lab_drift=lab_drift,
                    reasons=tuple(rejection_reasons),
                )
                candidate_summary[color] = json.loads(
                    json.dumps(base_summary[color])
                )

        while True:
            _raise_if_cancelled(cancel_callback)
            current_checker = _checker_from_payload(model_payload)
            current_correct_by_color = {
                color: _correct_predictions(
                    current_checker,
                    split_by_color[color][1],
                    color,
                )
                for color in canonical_colors
            }
            regressed_rebuilt_colors: list[str] = []
            preserved_color_regressed = False
            for color in canonical_colors:
                holdout_count = len(split_by_color[color][1])
                previous_accuracy = _ratio(
                    previous_correct_by_color[color],
                    holdout_count,
                )
                current_accuracy = _ratio(
                    current_correct_by_color[color],
                    holdout_count,
                )
                has_regression = (
                    previous_accuracy is not None
                    and current_accuracy is not None
                    and current_accuracy + self.maximum_accuracy_regression
                    < previous_accuracy
                )
                if not has_regression:
                    continue
                is_active_proposal = (
                    preliminary_states[color][0] == "REBUILT"
                    and color not in rejected_proposals
                )
                if is_active_proposal:
                    regressed_rebuilt_colors.append(color)
                else:
                    preserved_color_regressed = True

            if preserved_color_regressed:
                newly_rejected = [
                    color
                    for color in canonical_colors
                    if preliminary_states[color][0] == "REBUILT"
                    and color not in rejected_proposals
                ]
                rejection_reason = "CROSS_COLOR_HOLDOUT_REGRESSION"
            else:
                newly_rejected = regressed_rebuilt_colors
                rejection_reason = "HOLDOUT_ACCURACY_REGRESSION"
            if not newly_rejected:
                if preserved_color_regressed:
                    raise ColorBaselineError(
                        "最終顏色基準仍造成保留驗證退步，已停止建立候選。"
                    )
                final_correct_by_color = current_correct_by_color
                break
            for color in newly_rejected:
                hue_drift, lab_drift = proposal_drift_by_color[color]
                rejected_proposals[color] = _RejectedColorProposal(
                    holdout_correct=current_correct_by_color[color],
                    hue_drift=hue_drift,
                    lab_drift=lab_drift,
                    reasons=(rejection_reason,),
                )
                candidate_summary[color] = json.loads(
                    json.dumps(base_summary[color])
                )

        color_reports: list[ColorBaselineColorReport] = []
        incomplete = False
        review_required = False
        for color in canonical_colors:
            training, holdout = split_by_color[color]
            state, note = preliminary_states[color]
            rejected = rejected_proposals.get(color)
            if state == "PRESERVED_INSUFFICIENT":
                incomplete = True
            elif rejected is not None:
                state = "PRESERVED_SAFETY_REJECTED"
                note = "自動安全檢查未通過，已沿用舊基準。"
            final_hue_drift, final_lab_drift = _center_drift(
                base_summary[color],
                candidate_summary[color],
            )
            if color.casefold() == "black":
                final_hue_drift = None
            # Spread is judged on the statistics that will actually ship for
            # this color, so a proposal already preserved by the drift check is
            # measured on the preserved baseline rather than on the discarded
            # proposal.
            spread = _hue_spread(candidate_summary[color])
            retention = _chroma_retention(
                base_summary[color], candidate_summary[color]
            )
            dominant = proposal_dominant_by_color.get(color)
            dominant = float(dominant) if dominant is not None else None
            reasons: list[str] = []
            final_accuracy = _ratio(
                final_correct_by_color[color],
                len(holdout),
            )
            if (
                final_accuracy is None
                or final_accuracy < self.minimum_holdout_accuracy
            ):
                reasons.append(ABSOLUTE_HOLDOUT_ACCURACY_REASON)
                incomplete = True
            if spread is not None and spread > self.maximum_hue_spread:
                reasons.append(HUE_SPREAD_REVIEW_REASON)
            if (
                retention is not None
                and retention < self.minimum_chroma_retention
            ):
                reasons.append(CHROMA_COLLAPSE_REVIEW_REASON)
            if (
                dominant is not None
                and dominant < self.minimum_dominant_fraction
            ):
                reasons.append(DOMINANT_FRACTION_REVIEW_REASON)
            review_reasons = tuple(reasons)
            if review_reasons:
                review_required = True
            color_reports.append(
                ColorBaselineColorReport(
                    color=color,
                    state=state,
                    total_crops=len(training) + len(holdout),
                    training_crops=len(training),
                    holdout_crops=len(holdout),
                    previous_holdout_correct=previous_correct_by_color[color],
                    candidate_holdout_correct=final_correct_by_color[color],
                    hue_drift=final_hue_drift,
                    lab_drift=final_lab_drift,
                    note=note,
                    rejected_proposal_holdout_correct=(
                        rejected.holdout_correct if rejected is not None else None
                    ),
                    rejected_proposal_hue_drift=(
                        rejected.hue_drift if rejected is not None else None
                    ),
                    rejected_proposal_lab_drift=(
                        rejected.lab_drift if rejected is not None else None
                    ),
                    rejection_reasons=(
                        rejected.reasons if rejected is not None else ()
                    ),
                    hue_spread=spread,
                    chroma_retention=retention,
                    dominant_fraction=dominant,
                    review_reasons=review_reasons,
                )
            )

        model_payload["recalibration"]["preserved_by_safety"] = sorted(
            rejected_proposals
        )
        # A wide spread never rejects a rebuild on its own -- the threshold is
        # calibrated on a handful of baselines, and wrongly blocking a good one
        # is a production problem too. It routes the candidate to the human
        # gate instead, which is the control that has actually been catching
        # these.
        if incomplete:
            status = "INCOMPLETE"
        elif review_required:
            status = "REVIEW_REQUIRED"
        else:
            status = "READY"
        report_payload = {
            "schema_version": COLOR_BASELINE_SCHEMA_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "status": status,
            "evidence_sha256": evidence_sha256,
            "base_model_path": str(base_path),
            "base_model_sha256": sha256_file(base_path),
            "statistical_outlier_filter": outlier_filter.to_dict(),
            "safety_limits": {
                "maximum_hue_drift": self.maximum_hue_drift,
                "maximum_hue_spread": self.maximum_hue_spread,
                "minimum_chroma_retention": self.minimum_chroma_retention,
                "minimum_dominant_fraction": self.minimum_dominant_fraction,
                "maximum_lab_drift": self.maximum_lab_drift,
                "maximum_accuracy_regression": self.maximum_accuracy_regression,
                "minimum_holdout_accuracy": self.minimum_holdout_accuracy,
            },
            "preserved_by_safety": sorted(rejected_proposals),
            "review_required_colors": sorted(
                item.color for item in color_reports if item.review_reasons
            ),
            "absolute_accuracy_failures": sorted(
                item.color
                for item in color_reports
                if ABSOLUTE_HOLDOUT_ACCURACY_REASON in item.review_reasons
            ),
            "color_reports": [item.to_dict() for item in color_reports],
            "limitations": [
                "資料只取人工確認為 OK 的元件裁切；不代表已有真實顏色缺陷 NG。",
                "候選仍須用既有驗收矩陣測試，通過後才能建立與啟用發布組合。",
            ],
        }
        if normalized_metadata:
            report_payload["evidence_sources"] = normalized_metadata
        return ColorBaselineBuild(
            status=status,
            model_payload=model_payload,
            report_payload=report_payload,
            evidence_sha256=evidence_sha256,
            color_reports=tuple(color_reports),
            outlier_filter=outlier_filter,
        )

    def _split(
        self,
        evidence: Sequence[ColorCropEvidence],
    ) -> tuple[tuple[ColorCropEvidence, ...], tuple[ColorCropEvidence, ...]]:
        grouped: dict[str, list[ColorCropEvidence]] = defaultdict(list)
        for item in evidence:
            grouped[item.sample_id].append(item)
        ordered_groups = sorted(
            grouped.values(),
            key=lambda items: hashlib.sha256(items[0].sample_id.encode("utf-8")).hexdigest(),
        )
        if len(ordered_groups) < 2:
            return tuple(evidence), ()
        target_holdout_count = max(
            self.minimum_holdout_crops,
            int(round(len(evidence) * self.holdout_fraction)),
        )
        holdout_groups: list[list[ColorCropEvidence]] = []
        holdout_count = 0
        for group in ordered_groups[:-1]:
            if holdout_count >= target_holdout_count:
                break
            holdout_groups.append(group)
            holdout_count += len(group)
        holdout_ids = {item.sample_id for group in holdout_groups for item in group}
        training = tuple(item for item in evidence if item.sample_id not in holdout_ids)
        holdout = tuple(item for item in evidence if item.sample_id in holdout_ids)
        return training, holdout

    def _calculate_stats(
        self,
        evidence: Sequence[ColorCropEvidence],
    ) -> dict[str, Any]:
        hsv_rows: list[np.ndarray] = []
        lab_rows: list[np.ndarray] = []
        coverages: list[float] = []
        dominant_fractions: list[float] = []
        skipped: list[str] = []
        for item in evidence:
            try:
                sampled = _sample_color_pixels(
                    item.image_bgr,
                    item.color,
                    sample_size=self.sample_size,
                )
            except _InsufficientColorPixels as exc:
                # One unusable crop must not fail the whole rebuild, and must
                # not be silently averaged into the baseline either.
                skipped.append(str(item.sample_id))
                logger.warning(
                    "Color baseline skipped crop %s: %s", item.sample_id, exc
                )
                continue
            hsv_rows.append(sampled.hsv)
            lab_rows.append(sampled.lab)
            coverages.append(sampled.coverage)
            dominant_fractions.append(sampled.dominant_fraction)
        if not hsv_rows:
            raise ColorBaselineError(
                f"所有 {len(evidence)} 個裁切都沒有足夠的顏色證據，無法建立基準。"
            )
        hsv_values = np.concatenate(hsv_rows, axis=0)
        lab_values = np.concatenate(lab_rows, axis=0)
        return {
            # The number of crops the statistics were actually built from, not
            # the number offered. Reporting the offered count would overstate
            # the evidence behind a baseline that had dropped crops.
            "count": len(hsv_rows),
            "skipped_crops": skipped,
            "hsv_mean": _hsv_trimmed_mean(hsv_values).tolist(),
            "hsv_min": np.percentile(hsv_values, 1, axis=0).tolist(),
            "hsv_max": np.percentile(hsv_values, 99, axis=0).tolist(),
            "lab_mean": _trimmed_mean(lab_values).tolist(),
            "lab_min": np.percentile(lab_values, 1, axis=0).tolist(),
            "lab_max": np.percentile(lab_values, 99, axis=0).tolist(),
            "hsv_p10": np.percentile(hsv_values, 10, axis=0).tolist(),
            "hsv_p90": np.percentile(hsv_values, 90, axis=0).tolist(),
            "lab_p10": np.percentile(lab_values, 10, axis=0).tolist(),
            "lab_p90": np.percentile(lab_values, 90, axis=0).tolist(),
            "coverage_mean": float(np.mean(coverages)),
            # How much of each crop's chromatic content belonged to its
            # dominant hue, averaged. Below 1.0 means detection boxes were
            # catching more than the one wire they name.
            "dominant_fraction_mean": float(np.mean(dominant_fractions)),
        }


class ColorBaselineCandidateStore:
    """Persist and verify deterministic immutable baseline candidates."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def commit(
        self,
        *,
        product: str,
        area: str,
        model_type: str,
        build: ColorBaselineBuild,
    ) -> ColorBaselineCandidate:
        normalized = {
            "schema_version": COLOR_BASELINE_SCHEMA_VERSION,
            "product": _required_segment(product, "product"),
            "area": _required_segment(area, "area"),
            "model_type": _required_segment(model_type, "model_type").lower(),
            "status": build.status,
            "algorithm": ALGORITHM_VERSION,
            "evidence_sha256": build.evidence_sha256,
            "color_model_sha256": _payload_sha256(build.model_payload),
            "report_sha256": _payload_sha256(build.report_payload),
        }
        candidate_id = hashlib.sha256(_canonical_json(normalized)).hexdigest()[:24]
        destination = self.root / candidate_id
        if destination.is_dir():
            return self.load(destination / "manifest.json")
        staging = self.root / f".{candidate_id}.{uuid4().hex}.tmp"
        self.root.mkdir(parents=True, exist_ok=True)
        staging.mkdir(parents=False, exist_ok=False)
        try:
            color_model_path = staging / "color_stats.json"
            report_path = staging / "report.json"
            color_model_path.write_bytes(_canonical_json(build.model_payload))
            report_path.write_bytes(_canonical_json(build.report_payload))
            manifest = {
                **normalized,
                "candidate_id": candidate_id,
                "display_version": f"color-base-{candidate_id[:8]}",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "colors": list(build.model_payload["summary"]),
                "color_model_path": "color_stats.json",
                "report_path": "report.json",
            }
            (staging / "manifest.json").write_bytes(_canonical_json(manifest))
            try:
                os.replace(staging, destination)
            except OSError:
                if not destination.is_dir():
                    raise
            return self.load(destination / "manifest.json")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def list_candidates(
        self,
        *,
        product: str = "",
        area: str = "",
        model_type: str = "",
    ) -> tuple[ColorBaselineCandidate, ...]:
        if not self.root.is_dir():
            return ()
        candidates: list[ColorBaselineCandidate] = []
        for manifest in sorted(self.root.glob("*/manifest.json")):
            candidate = self.load(manifest)
            if product and candidate.product != product:
                continue
            if area and candidate.area != area:
                continue
            if model_type and candidate.model_type != model_type.lower():
                continue
            candidates.append(candidate)
        return tuple(
            sorted(
                candidates,
                key=lambda item: (item.created_at, item.display_version),
                reverse=True,
            )
        )

    def load(self, manifest_path: str | Path) -> ColorBaselineCandidate:
        manifest = Path(manifest_path).expanduser().resolve()
        _require_within(manifest, self.root)
        if manifest.is_symlink() or not manifest.is_file():
            raise ColorBaselineError("顏色基準候選 manifest 不存在。")
        payload = _read_json(manifest)
        if payload.get("schema_version") != COLOR_BASELINE_SCHEMA_VERSION:
            raise ColorBaselineError("不支援的顏色基準候選格式。")
        candidate_id = str(payload.get("candidate_id") or "")
        if not candidate_id or manifest.parent.name != candidate_id:
            raise ColorBaselineError("顏色基準候選目錄與 ID 不一致。")
        identity = {
            key: payload.get(key)
            for key in (
                "schema_version",
                "product",
                "area",
                "model_type",
                "status",
                "algorithm",
                "evidence_sha256",
                "color_model_sha256",
                "report_sha256",
            )
        }
        expected_id = hashlib.sha256(_canonical_json(identity)).hexdigest()[:24]
        if expected_id != candidate_id:
            raise ColorBaselineError("顏色基準候選 ID 驗證失敗。")
        model_path = (manifest.parent / str(payload["color_model_path"])).resolve()
        report_path = (manifest.parent / str(payload["report_path"])).resolve()
        _require_within(model_path, manifest.parent)
        _require_within(report_path, manifest.parent)
        if model_path.is_symlink() or report_path.is_symlink():
            raise ColorBaselineError("顏色基準候選不可使用符號連結。")
        if sha256_file(model_path) != str(payload["color_model_sha256"]):
            raise ColorBaselineError("顏色基準候選模型完整性驗證失敗。")
        if sha256_file(report_path) != str(payload["report_sha256"]):
            raise ColorBaselineError("顏色基準候選報告完整性驗證失敗。")
        return ColorBaselineCandidate(
            candidate_id=candidate_id,
            display_version=str(payload["display_version"]),
            product=str(payload["product"]),
            area=str(payload["area"]),
            model_type=str(payload["model_type"]),
            status=str(payload["status"]),
            algorithm=str(payload["algorithm"]),
            created_at=str(payload["created_at"]),
            color_model_path=model_path,
            report_path=report_path,
            manifest_path=manifest,
            color_model_sha256=str(payload["color_model_sha256"]),
            report_sha256=str(payload["report_sha256"]),
            evidence_sha256=str(payload["evidence_sha256"]),
            colors=tuple(str(value) for value in payload.get("colors") or ()),
        )


def collect_confirmed_ok_evidence(
    *,
    repository: Any,
    inference_service: Any,
    records: Sequence[Any],
    inference_type: str,
    expected_colors: Sequence[str] = DEFAULT_COLORS,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    roi_policy: ColorRoiPolicy | None = None,
) -> tuple[ColorCropEvidence, ...]:
    """Compatibility wrapper for acceptance-only baseline evidence."""
    selected_records = tuple(
        record
        for record in records
        if str(record.review_status).casefold() == "confirmed" and str(record.expected_verdict).upper() == "OK"
    )
    samples = tuple(
        ColorBaselineImageSample(
            sample_id=str(record.sample_id),
            image_path=Path(repository.image_file(record)).resolve(),
            image_sha256=str(record.image_sha256),
            product=str(getattr(record, "product", "")),
            area=str(getattr(record, "area", "")),
            source_kind="acceptance",
            source_manifest=str(getattr(repository, "manifest_path", "")),
        )
        for record in selected_records
    )
    return collect_color_baseline_evidence(
        inference_service=inference_service,
        samples=samples,
        inference_type=inference_type,
        expected_colors=expected_colors,
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        roi_policy=roi_policy,
    )


def collect_color_baseline_evidence(
    *,
    inference_service: Any,
    samples: Sequence[ColorBaselineImageSample],
    inference_type: str,
    expected_colors: Sequence[str] = DEFAULT_COLORS,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    roi_policy: ColorRoiPolicy | None = None,
) -> tuple[ColorCropEvidence, ...]:
    """Rerun the selected detector and crop known colors from verified OK images."""
    selected = tuple(samples)
    color_lookup = {str(color).casefold(): str(color) for color in expected_colors}
    evidence: list[ColorCropEvidence] = []
    for index, record in enumerate(selected, start=1):
        _raise_if_cancelled(cancel_callback)
        image_path = record.image_path
        _frame, result = inference_service.detect_raw(
            record,
            image_path,
            inference_type=inference_type,
            cancel_cb=cancel_callback,
        )
        if result.error:
            raise ColorBaselineError(f"{record.sample_id} 推論失敗：{result.error}")
        crop_source = result.processed_image
        if (
            crop_source is None
            or not isinstance(crop_source, np.ndarray)
            or crop_source.size == 0
        ):
            raise ColorBaselineError(
                f"{record.sample_id} 推論結果缺少 processed_image；"
                "為避免座標系錯誤，已拒絕建立顏色基準。"
            )
        for item in result.items:
            canonical = color_lookup.get(str(item.label).casefold())
            if canonical is None:
                continue
            crop = extract_bbox_roi(
                crop_source,
                item.bbox_xyxy,
                min_size=8,
                policy=roi_policy,
            )
            if crop is None:
                continue
            evidence.append(
                ColorCropEvidence(
                    sample_id=str(record.sample_id),
                    color=canonical,
                    image_bgr=crop.copy(),
                    source_sha256=str(record.image_sha256),
                )
            )
        if progress_callback is not None:
            progress_callback(index, len(selected), str(record.sample_id))
    if not selected:
        raise ColorBaselineError("沒有人工確認為 OK 的驗收照片。")
    if not evidence:
        raise ColorBaselineError("選取模型未產生可用的五色元件裁切。")
    return tuple(evidence)


def _build_outlier_filter_report(
    grouped: Mapping[str, Sequence[ColorCropEvidence]],
    *,
    z_score_threshold: float,
    maximum_auto_exclusion_fraction: float,
    minimum_sample_count: int,
    sample_size: int,
    cancel_callback: Callable[[], bool] | None,
) -> ColorBaselineOutlierFilterReport:
    total_sample_ids = {
        item.sample_id
        for color_evidence in grouped.values()
        for item in color_evidence
    }
    raw_findings: list[
        tuple[str, int, str, tuple[str, ...], tuple[tuple[str, float], ...]]
    ] = []
    locally_eligible_ids: set[str] = set()
    all_candidate_ids: set[str] = set()

    for color_key in sorted(grouped):
        _raise_if_cancelled(cancel_callback)
        color_evidence = tuple(grouped[color_key])
        color = color_evidence[0].color if color_evidence else color_key.title()
        features_by_sample = _lab_features_by_sample(
            color_evidence,
            sample_size=sample_size,
        )
        sample_count = len(features_by_sample)
        if sample_count < minimum_sample_count:
            raw_findings.append(
                (color, sample_count, "INSUFFICIENT_SAMPLE_COUNT", (), ())
            )
            continue

        sample_ids = tuple(sorted(features_by_sample))
        features = np.vstack([features_by_sample[sample_id] for sample_id in sample_ids])
        center = np.median(features, axis=0)
        median_absolute_deviation = np.median(
            np.abs(features - center),
            axis=0,
        )
        robust_scale = np.maximum(
            median_absolute_deviation * 1.4826,
            np.full(3, 3.0, dtype=np.float32),
        )
        distances = np.linalg.norm((features - center) / robust_scale, axis=1)
        scored_candidates = tuple(
            sorted(
                (
                    (sample_id, float(distance))
                    for sample_id, distance in zip(
                        sample_ids,
                        distances,
                        strict=True,
                    )
                    if distance > z_score_threshold
                ),
                key=lambda item: (-item[1], item[0]),
            )
        )
        candidate_ids = tuple(sorted(sample_id for sample_id, _ in scored_candidates))
        all_candidate_ids.update(candidate_ids)
        maximum_local_exclusions = int(
            np.floor(sample_count * maximum_auto_exclusion_fraction)
        )
        if not candidate_ids:
            status = "NO_OUTLIERS"
        elif len(candidate_ids) <= maximum_local_exclusions:
            status = "ELIGIBLE_FOR_AUTO_EXCLUSION"
            locally_eligible_ids.update(candidate_ids)
        else:
            status = "SYSTEMATIC_SHIFT_NOT_FILTERED"
        raw_findings.append(
            (color, sample_count, status, candidate_ids, scored_candidates)
        )

    maximum_global_exclusions = int(
        np.floor(len(total_sample_ids) * maximum_auto_exclusion_fraction)
    )
    global_limit_exceeded = (
        bool(locally_eligible_ids)
        and len(locally_eligible_ids) > maximum_global_exclusions
    )
    excluded_ids = (
        ()
        if global_limit_exceeded
        else tuple(sorted(locally_eligible_ids))
    )
    excluded_id_set = set(excluded_ids)
    findings: list[ColorBaselineOutlierColorFinding] = []
    for color, sample_count, status, candidate_ids, scores in raw_findings:
        if status == "ELIGIBLE_FOR_AUTO_EXCLUSION":
            if global_limit_exceeded:
                final_status = "GLOBAL_LIMIT_NOT_FILTERED"
                color_excluded_ids: tuple[str, ...] = ()
            else:
                final_status = "AUTO_EXCLUDED"
                color_excluded_ids = tuple(
                    sample_id
                    for sample_id in candidate_ids
                    if sample_id in excluded_id_set
                )
        else:
            final_status = status
            color_excluded_ids = ()
        findings.append(
            ColorBaselineOutlierColorFinding(
                color=color,
                sample_count=sample_count,
                status=final_status,
                candidate_sample_ids=candidate_ids,
                excluded_sample_ids=color_excluded_ids,
                scores=scores,
            )
        )

    if excluded_ids:
        report_status = "AUTO_EXCLUDED"
    elif all_candidate_ids:
        report_status = "SYSTEMATIC_SHIFT_NOT_FILTERED"
    else:
        report_status = "NO_OUTLIERS"
    return ColorBaselineOutlierFilterReport(
        status=report_status,
        total_sample_count=len(total_sample_ids),
        z_score_threshold=z_score_threshold,
        maximum_auto_exclusion_fraction=maximum_auto_exclusion_fraction,
        candidate_sample_ids=tuple(sorted(all_candidate_ids)),
        excluded_sample_ids=excluded_ids,
        findings=tuple(findings),
    )


def _lab_features_by_sample(
    evidence: Sequence[ColorCropEvidence],
    *,
    sample_size: int,
) -> dict[str, np.ndarray]:
    features: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in evidence:
        try:
            sampled = _sample_color_pixels(
                item.image_bgr,
                item.color,
                sample_size=sample_size,
            )
        except _InsufficientColorPixels:
            continue
        features[item.sample_id].append(_trimmed_mean(sampled.lab))
    # ``sample_features`` can now be empty: every crop for a sample may have
    # been skipped for carrying no color evidence. Stacking an empty list
    # raises, and a sample with no features has no outlier score to offer.
    return {
        sample_id: np.mean(np.vstack(sample_features), axis=0)
        for sample_id, sample_features in features.items()
        if sample_features
    }


@dataclass(frozen=True)
class _SampledPixels:
    """The pixels one crop contributes to a color's baseline.

    ``dominant_fraction`` is how much of the crop's chromatic content the
    dominant-hue window kept. A clean crop of one wire keeps essentially all of
    it; a box that caught a neighbour keeps noticeably less, which is the
    per-crop measure of exactly the pollution that used to enter the baseline
    unnoticed.
    """

    hsv: np.ndarray
    lab: np.ndarray
    coverage: float
    dominant_fraction: float


class _InsufficientColorPixels(Exception):
    """One crop carries too little color evidence to contribute to a baseline.

    Internal to this module: callers drop the crop and record that they did,
    rather than letting one unusable crop fail a whole rebuild.
    """

    def __init__(self, color: str, selected: int, coverage: float) -> None:
        super().__init__(
            f"{color} 裁切僅有 {selected} 個有效像素（覆蓋率 {coverage:.3f}），"
            "不足以作為顏色基準。"
        )
        self.color = color
        self.selected = selected
        self.coverage = coverage


def _sample_color_pixels(
    image_bgr: np.ndarray,
    color: str,
    *,
    sample_size: int,
) -> _SampledPixels:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
        raise ColorBaselineError("顏色裁切必須是非空 BGR 影像。")
    height, width = image_bgr.shape[:2]
    margin_y = int(height * 0.15)
    margin_x = int(width * 0.15)
    center = image_bgr[
        margin_y : height - margin_y,
        margin_x : width - margin_x,
    ]
    if center.size == 0:
        center = image_bgr
    resized = cv2.resize(
        center,
        (sample_size, sample_size),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB).astype(np.float32)
    normalized = color.casefold()
    if normalized == "black":
        # Black is defined by the absence of chroma, so there is no dominant
        # hue to find and nothing a hue window could usefully exclude.
        mask = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] < 110)
        dominant_fraction = 1.0
    else:
        chromatic = hsv[:, :, 1] >= 20
        mask = _dominant_hue_mask(hsv[:, :, 0], chromatic)
        chromatic_count = int(np.count_nonzero(chromatic))
        dominant_fraction = (
            float(np.count_nonzero(mask)) / chromatic_count
            if chromatic_count
            else 0.0
        )
    selected = int(np.count_nonzero(mask))
    coverage = float(selected) / max(mask.size, 1)
    if selected < _minimum_sampled_pixels(sample_size):
        # Too little of this crop carries the kind of pixel the color is made
        # of. The previous behavior was to discard the mask and sample *every*
        # pixel instead, so a crop that is mostly unsaturated background
        # produced a "color baseline" built out of that background -- the
        # recorded coverage said 0.00 while the statistics came from thousands
        # of background pixels. A crop with no color evidence has to be
        # dropped, not padded.
        raise _InsufficientColorPixels(color, selected, coverage)
    return _SampledPixels(
        hsv=hsv[mask].reshape(-1, 3),
        lab=lab[mask].reshape(-1, 3),
        coverage=coverage,
        dominant_fraction=dominant_fraction,
    )


def _dominant_hue_mask(
    hue: np.ndarray,
    chromatic: np.ndarray,
    *,
    window: float = DOMINANT_HUE_WINDOW,
    bins: int = DOMINANT_HUE_BINS,
) -> np.ndarray:
    """Narrow a chromatic mask to the pixels around the crop's dominant hue.

    A crop is evidence for *one* color, but a detection box drawn around one
    wire routinely catches part of the next. Selecting every saturated pixel
    fed those neighbours into the baseline: recorded hue ranges then overlapped
    each other so badly that Red spanned 2..78 and Green 16..94 in the same
    file, and Red's mean landed on 20.7 -- amber.

    The dominant hue is located on a circular histogram, so a color sitting on
    the 0/179 seam is found as one cluster rather than split into two at
    opposite ends of the axis, and the window kept around it wraps for the same
    reason.

    Returns a mask over the same shape as ``chromatic``; never widens it.
    """
    if not np.any(chromatic):
        return chromatic
    values = np.asarray(hue, dtype=np.float64)
    selected = values[chromatic]
    edges = np.linspace(0.0, 180.0, bins + 1)
    counts, _ = np.histogram(np.mod(selected, 180.0), bins=edges)
    # Fold the histogram circularly before picking the peak, so a cluster
    # straddling the seam is not beaten by a smaller contiguous one.
    folded = counts + np.roll(counts, 1) + np.roll(counts, -1)
    peak = int(np.argmax(folded))
    center = float((edges[peak] + edges[peak + 1]) / 2.0)
    distance = np.abs(np.mod(values - center + 90.0, 180.0) - 90.0)
    return chromatic & (distance <= window)


def _minimum_sampled_pixels(sample_size: int) -> int:
    """Fewest masked pixels a crop must contribute to a baseline.

    This is an absolute pixel floor, not a fraction of the crop: the historical
    check compared the count against ``sample_size`` while the mask holds
    ``sample_size ** 2`` pixels, so the effective floor has always been
    ``1 / sample_size`` of the crop -- about 1.6% at the default 64. The value
    is preserved rather than "corrected" to a fraction, because raising it
    would reject crops the line currently accepts, which is a tuning decision
    rather than a bug fix. Naming it makes the choice visible.
    """
    return max(1, sample_size)


def _trimmed_mean(values: np.ndarray) -> np.ndarray:
    lower = np.percentile(values, 5, axis=0)
    upper = np.percentile(values, 95, axis=0)
    clipped = np.clip(values, lower, upper)
    return np.mean(clipped, axis=0)


def _circular_trimmed_hue_mean(hue: np.ndarray) -> float:
    """Trimmed mean of hue samples on OpenCV's 0..179 circle.

    Samples are re-expressed as signed offsets from the circular center, so the
    5/95 trim runs on a continuous axis instead of across the 0/179 seam.
    """
    values = np.asarray(hue, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    center = circular_hue_mean(values)
    offsets = np.mod(values - center + 90.0, 180.0) - 90.0
    lower = float(np.percentile(offsets, 5))
    upper = float(np.percentile(offsets, 95))
    trimmed = np.clip(offsets, lower, upper)
    return float(np.mod(center + float(np.mean(trimmed)), 180.0))


def _hsv_trimmed_mean(values: np.ndarray) -> np.ndarray:
    """Trimmed HSV mean whose hue channel is averaged on its circle.

    Hue is periodic, so averaging it linearly places the mean of red samples at
    3 and 178 near 90 -- which is green. Inference compares this value with
    circular distance, so producing it linearly here made the two ends of the
    contract disagree and cost genuine red parts their score. Saturation and
    value are ordinary linear channels and keep the plain trimmed mean.
    """
    result = np.asarray(_trimmed_mean(values), dtype=np.float64)
    result[0] = _circular_trimmed_hue_mean(np.asarray(values)[:, 0])
    return result


def _correct_predictions(
    checker: StatsColorChecker,
    evidence: Sequence[ColorCropEvidence],
    expected_color: str,
) -> int:
    expected = expected_color.casefold()
    correct = 0
    for item in evidence:
        result = checker.check(item.image_bgr)
        if bool(result.is_ok) and result.best_color.casefold() == expected:
            correct += 1
    return correct


def _center_drift(
    previous: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> tuple[float | None, float | None]:
    try:
        previous_hsv = np.asarray(previous["hsv_mean"], dtype=np.float32)
        candidate_hsv = np.asarray(candidate["hsv_mean"], dtype=np.float32)
        hue_drift = _circular_hue_distance(float(previous_hsv[0]), float(candidate_hsv[0]))
    except (KeyError, TypeError, ValueError, IndexError):
        hue_drift = None
    try:
        previous_lab = np.asarray(previous["lab_mean"], dtype=np.float32)
        candidate_lab = np.asarray(candidate["lab_mean"], dtype=np.float32)
        lab_drift = float(np.linalg.norm(previous_lab - candidate_lab))
    except (KeyError, TypeError, ValueError):
        lab_drift = None
    return hue_drift, lab_drift


def _checker_from_payload(payload: Mapping[str, Any]) -> StatsColorChecker:
    from tempfile import TemporaryDirectory

    temporary = TemporaryDirectory(prefix="color-baseline-check-")
    try:
        path = Path(temporary.name) / "color_stats.json"
        path.write_bytes(_canonical_json(dict(payload)))
        return StatsColorChecker.from_json(path)
    finally:
        temporary.cleanup()


def _validate_and_group_evidence(
    evidence: Sequence[ColorCropEvidence],
    expected_colors: Sequence[str],
) -> dict[str, tuple[ColorCropEvidence, ...]]:
    allowed = {color.casefold(): color for color in expected_colors}
    grouped: dict[str, list[ColorCropEvidence]] = defaultdict(list)
    for item in evidence:
        normalized = str(item.color).casefold()
        if normalized not in allowed:
            continue
        if not str(item.sample_id).strip():
            raise ColorBaselineError("顏色裁切缺少 sample_id。")
        _sample_color_pixels(item.image_bgr, item.color, sample_size=16)
        grouped[normalized].append(item)
    return {color.casefold(): tuple(grouped[color.casefold()]) for color in expected_colors}


def _resolve_expected_colors(
    expected_colors: Sequence[str],
    summary: Mapping[str, Any],
) -> tuple[str, ...]:
    lookup = {str(color).casefold(): str(color) for color in summary}
    resolved: list[str] = []
    for requested in expected_colors:
        existing = lookup.get(str(requested).casefold())
        if existing is None:
            raise ColorBaselineError(f"舊基準缺少必要色別：{requested}")
        resolved.append(existing)
    if len({color.casefold() for color in resolved}) != len(resolved):
        raise ColorBaselineError("必要色別不可重複。")
    return tuple(resolved)


def _read_stats_payload(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ColorBaselineError(f"舊顏色基準不存在：{path}")
    payload = _read_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict) or not summary:
        raise ColorBaselineError("舊顏色基準缺少 summary。")
    return payload


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ColorBaselineError(f"JSON 無法讀取：{path}") from exc
    if not isinstance(payload, dict):
        raise ColorBaselineError(f"JSON 根節點必須是物件：{path}")
    return payload


def _evidence_digest(
    evidence: Sequence[ColorCropEvidence],
    metadata: Mapping[str, Any] | None = None,
) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        evidence,
        key=lambda value: (
            value.color.casefold(),
            value.sample_id,
            value.source_sha256,
            hashlib.sha256(np.ascontiguousarray(value.image_bgr).tobytes()).hexdigest(),
        ),
    ):
        digest.update(item.color.casefold().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.source_sha256.encode("ascii", errors="ignore"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(np.ascontiguousarray(item.image_bgr).tobytes()).digest())
    if metadata:
        digest.update(b"\0lineage\0")
        digest.update(_canonical_json(metadata))
    return digest.hexdigest()


def _normalized_evidence_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if metadata is None:
        return {}
    try:
        encoded = json.dumps(
            dict(metadata),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ColorBaselineError("Color baseline evidence metadata is not JSON-safe.") from exc
    if not isinstance(normalized, dict):
        raise ColorBaselineError("Color baseline evidence metadata must be a mapping.")
    return normalized


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(payload))).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _clamped_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ColorBaselineError("元件 bbox 必須包含四個座標。")
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _circular_hue_distance(left: float, right: float) -> float:
    difference = abs(left - right)
    return min(difference, 180.0 - difference)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _required_segment(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise ColorBaselineError(f"{label} 無效：{value!r}")
    return normalized


def _require_within(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ColorBaselineError("顏色基準候選路徑超出允許目錄。") from exc


def _raise_if_cancelled(
    cancel_callback: Callable[[], bool] | None,
) -> None:
    if cancel_callback is not None and cancel_callback():
        raise ColorBaselineCancelled("顏色基準重建已取消。")
