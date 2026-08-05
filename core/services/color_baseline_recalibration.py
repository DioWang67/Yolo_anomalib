"""Rebuild immutable Stats Color baselines from verified human-reviewed OK evidence.

The module intentionally separates numerical rebuilding from append-only
persistence.  It never changes a model config, an active color revision, or an
inspection release pointer.
"""

from __future__ import annotations

import hashlib
import json
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
from core.stats_color_checker import StatsColorChecker

COLOR_BASELINE_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "stats-robust-v2"
OUTLIER_FILTER_ALGORITHM = "per-color-sample-lab-mad-v1"
DEFAULT_COLORS = ("Black", "Green", "Orange", "Red", "Yellow")


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
            "note": self.note,
        }
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
        maximum_lab_drift: float = 35.0,
        maximum_accuracy_regression: float = 0.02,
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
        self.minimum_crops_per_color = minimum_crops_per_color
        self.minimum_holdout_crops = minimum_holdout_crops
        self.holdout_fraction = holdout_fraction
        self.maximum_hue_drift = maximum_hue_drift
        self.maximum_lab_drift = maximum_lab_drift
        self.maximum_accuracy_regression = maximum_accuracy_regression
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
                )
            )

        model_payload["recalibration"]["preserved_by_safety"] = sorted(
            rejected_proposals
        )
        status = "INCOMPLETE" if incomplete else "READY"
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
                "maximum_lab_drift": self.maximum_lab_drift,
                "maximum_accuracy_regression": self.maximum_accuracy_regression,
            },
            "preserved_by_safety": sorted(rejected_proposals),
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
        for item in evidence:
            hsv, lab, coverage = _sample_color_pixels(
                item.image_bgr,
                item.color,
                sample_size=self.sample_size,
            )
            hsv_rows.append(hsv)
            lab_rows.append(lab)
            coverages.append(coverage)
        hsv_values = np.concatenate(hsv_rows, axis=0)
        lab_values = np.concatenate(lab_rows, axis=0)
        return {
            "count": len(evidence),
            "hsv_mean": _trimmed_mean(hsv_values).tolist(),
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
    )


def collect_color_baseline_evidence(
    *,
    inference_service: Any,
    samples: Sequence[ColorBaselineImageSample],
    inference_type: str,
    expected_colors: Sequence[str] = DEFAULT_COLORS,
    progress_callback: Callable[[int, int, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
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
        height, width = crop_source.shape[:2]
        for item in result.items:
            canonical = color_lookup.get(str(item.label).casefold())
            if canonical is None:
                continue
            x1, y1, x2, y2 = _clamped_bbox(
                item.bbox_xyxy,
                width=width,
                height=height,
            )
            crop = crop_source[y1:y2, x1:x2]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
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
        _hsv, lab, _coverage = _sample_color_pixels(
            item.image_bgr,
            item.color,
            sample_size=sample_size,
        )
        features[item.sample_id].append(_trimmed_mean(lab))
    return {
        sample_id: np.mean(np.vstack(sample_features), axis=0)
        for sample_id, sample_features in features.items()
    }


def _sample_color_pixels(
    image_bgr: np.ndarray,
    color: str,
    *,
    sample_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
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
        mask = (hsv[:, :, 1] < 80) & (hsv[:, :, 2] < 110)
    else:
        mask = hsv[:, :, 1] >= 20
    coverage = float(np.count_nonzero(mask)) / max(mask.size, 1)
    if np.count_nonzero(mask) < sample_size:
        mask = np.ones(mask.shape, dtype=bool)
    return (
        hsv[mask].reshape(-1, 3),
        lab[mask].reshape(-1, 3),
        coverage,
    )


def _trimmed_mean(values: np.ndarray) -> np.ndarray:
    lower = np.percentile(values, 5, axis=0)
    upper = np.percentile(values, 95, axis=0)
    clipped = np.clip(values, lower, upper)
    return np.mean(clipped, axis=0)


def _correct_predictions(
    checker: StatsColorChecker,
    evidence: Sequence[ColorCropEvidence],
    expected_color: str,
) -> int:
    return sum(checker.check(item.image_bgr).best_color.casefold() == expected_color.casefold() for item in evidence)


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
