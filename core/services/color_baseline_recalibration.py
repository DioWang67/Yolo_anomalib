"""Rebuild immutable Stats Color baselines from confirmed acceptance evidence.

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
DEFAULT_COLORS = ("Black", "Green", "Orange", "Red", "Yellow")


class ColorBaselineError(ValueError):
    """Raised when baseline evidence or an immutable candidate is invalid."""


class ColorBaselineCancelled(RuntimeError):
    """Raised when an operator cancels baseline collection or rebuilding."""


@dataclass(frozen=True)
class ColorCropEvidence:
    """One component crop tied to a confirmed acceptance sample."""

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

    @property
    def previous_accuracy(self) -> float | None:
        return _ratio(self.previous_holdout_correct, self.holdout_crops)

    @property
    def candidate_accuracy(self) -> float | None:
        return _ratio(self.candidate_holdout_correct, self.holdout_crops)

    def to_dict(self) -> dict[str, Any]:
        return {
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


@dataclass(frozen=True)
class ColorBaselineBuild:
    """Pure numerical output ready for immutable persistence."""

    status: str
    model_payload: dict[str, Any]
    report_payload: dict[str, Any]
    evidence_sha256: str
    color_reports: tuple[ColorBaselineColorReport, ...]


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
        self.minimum_crops_per_color = minimum_crops_per_color
        self.minimum_holdout_crops = minimum_holdout_crops
        self.holdout_fraction = holdout_fraction
        self.maximum_hue_drift = maximum_hue_drift
        self.maximum_lab_drift = maximum_lab_drift
        self.maximum_accuracy_regression = maximum_accuracy_regression
        self.sample_size = sample_size

    def build(
        self,
        *,
        base_model_path: str | Path,
        evidence: Sequence[ColorCropEvidence],
        expected_colors: Sequence[str] = DEFAULT_COLORS,
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
        evidence_sha256 = _evidence_digest(evidence)
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
        }

        candidate_checker = _checker_from_payload(model_payload)
        previous_checker = StatsColorChecker.from_json(base_path)
        color_reports: list[ColorBaselineColorReport] = []
        unsafe = False
        incomplete = False
        for color in canonical_colors:
            _raise_if_cancelled(cancel_callback)
            training, holdout = split_by_color[color]
            state, note = preliminary_states[color]
            previous_correct = _correct_predictions(previous_checker, holdout, color)
            candidate_correct = _correct_predictions(candidate_checker, holdout, color)
            hue_drift, lab_drift = _center_drift(
                base_summary[color],
                candidate_summary[color],
            )
            if state == "PRESERVED_INSUFFICIENT":
                incomplete = True
            else:
                previous_accuracy = _ratio(previous_correct, len(holdout))
                candidate_accuracy = _ratio(candidate_correct, len(holdout))
                drift_reasons: list[str] = []
                if hue_drift is not None and hue_drift > self.maximum_hue_drift:
                    drift_reasons.append(f"Hue 中心位移 {hue_drift:.1f} 超過 {self.maximum_hue_drift:.1f}")
                if lab_drift is not None and lab_drift > self.maximum_lab_drift:
                    drift_reasons.append(f"Lab 中心位移 {lab_drift:.1f} 超過 {self.maximum_lab_drift:.1f}")
                if (
                    previous_accuracy is not None
                    and candidate_accuracy is not None
                    and candidate_accuracy + self.maximum_accuracy_regression < previous_accuracy
                ):
                    drift_reasons.append("holdout 辨色率低於舊基準，超過允許退步幅度")
                if drift_reasons:
                    state = "REVIEW_REQUIRED"
                    note = "；".join(drift_reasons) + "。"
                    unsafe = True
            color_reports.append(
                ColorBaselineColorReport(
                    color=color,
                    state=state,
                    total_crops=len(training) + len(holdout),
                    training_crops=len(training),
                    holdout_crops=len(holdout),
                    previous_holdout_correct=previous_correct,
                    candidate_holdout_correct=candidate_correct,
                    hue_drift=hue_drift,
                    lab_drift=lab_drift,
                    note=note,
                )
            )

        status = "REVIEW_REQUIRED" if unsafe else ("INCOMPLETE" if incomplete else "READY")
        report_payload = {
            "schema_version": COLOR_BASELINE_SCHEMA_VERSION,
            "algorithm": ALGORITHM_VERSION,
            "status": status,
            "evidence_sha256": evidence_sha256,
            "base_model_path": str(base_path),
            "base_model_sha256": sha256_file(base_path),
            "color_reports": [item.to_dict() for item in color_reports],
            "limitations": [
                "資料只取人工確認為 OK 的元件裁切；不代表已有真實顏色缺陷 NG。",
                "候選仍須用既有驗收矩陣測試，通過後才能建立與啟用發布組合。",
            ],
        }
        return ColorBaselineBuild(
            status=status,
            model_payload=model_payload,
            report_payload=report_payload,
            evidence_sha256=evidence_sha256,
            color_reports=tuple(color_reports),
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
    """Rerun a selected detector and crop known colors from confirmed OK images."""
    selected = tuple(
        record
        for record in records
        if str(record.review_status).casefold() == "confirmed" and str(record.expected_verdict).upper() == "OK"
    )
    color_lookup = {str(color).casefold(): str(color) for color in expected_colors}
    evidence: list[ColorCropEvidence] = []
    for index, record in enumerate(selected, start=1):
        _raise_if_cancelled(cancel_callback)
        image_path = repository.image_file(record)
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


def _evidence_digest(evidence: Sequence[ColorCropEvidence]) -> str:
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
    return digest.hexdigest()


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
