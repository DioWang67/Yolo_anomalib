from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from core.services.color_baseline_evidence import ColorBaselineEvidenceProvider
from core.services.color_baseline_recalibration import (
    ColorBaselineError,
    collect_color_baseline_evidence,
)
from core.types import DetectionItem, DetectionResult


@dataclass(frozen=True)
class _AcceptanceRecord:
    sample_id: str
    image_sha256: str
    product: str = "Cable1"
    area: str = "A"
    review_status: str = "confirmed"
    expected_verdict: str = "OK"


class _AcceptanceRepository:
    def __init__(
        self,
        root: Path,
        records: tuple[_AcceptanceRecord, ...],
        paths: dict[str, Path],
    ) -> None:
        self.manifest_path = root / "ground_truth.csv"
        self._records = records
        self._paths = paths

    def records(self):
        return self._records

    def image_file(self, record):
        return self._paths[record.sample_id]


class _InferenceService:
    def detect_raw(self, _record, _path, *, inference_type, cancel_cb):
        assert inference_type == "yolo"
        processed = np.zeros((60, 80, 3), dtype=np.uint8)
        processed[10:50, 20:60, 2] = 220
        return processed.copy(), DetectionResult(
            status="PASS",
            items=[DetectionItem("Red", 0.95, (20, 10, 60, 50))],
            processed_image=processed,
        )


def _write_image(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_feedback(path: Path, rows: list[dict[str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "sample_id",
        "image_sha256",
        "product",
        "area",
        "model_type",
        "checker_type",
        "actual_is_ok",
        "output_image",
        "product_verdict",
        "detection_verdict",
        "color_verdict",
        "review_label",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _feedback_row(
    *,
    sample_id: str,
    digest: str,
    image_path: Path,
    actual_is_ok: bool,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "image_sha256": digest,
        "product": "Cable1",
        "area": "A",
        "model_type": "yolo",
        "checker_type": "stats",
        "actual_is_ok": "1" if actual_is_ok else "0",
        "output_image": str(image_path.resolve()),
        "product_verdict": "ok" if actual_is_ok else "ng",
        "detection_verdict": "correct",
        "color_verdict": "actually_ok" if actual_is_ok else "actually_ng",
        "review_label": "confirmed_ok" if actual_is_ok else "confirmed_ng",
    }


def test_provider_merges_feedback_deduplicates_and_excludes_truth_conflicts(
    tmp_path: Path,
) -> None:
    acceptance_image = tmp_path / "acceptance" / "images" / "acc.png"
    feedback_image = tmp_path / "training" / "color_review" / "images" / "fb.png"
    conflict_image = tmp_path / "training" / "color_review" / "images" / "ng.png"
    acceptance_sha = _write_image(acceptance_image, b"acceptance")
    feedback_sha = _write_image(feedback_image, b"feedback")
    conflict_sha = _write_image(conflict_image, b"conflict")
    records = (
        _AcceptanceRecord("ACC-1", acceptance_sha),
        _AcceptanceRecord("ACC-CONFLICT", conflict_sha),
    )
    repository = _AcceptanceRepository(
        tmp_path / "acceptance",
        records,
        {"ACC-1": acceptance_image, "ACC-CONFLICT": conflict_image},
    )
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [
            _feedback_row(
                sample_id="duplicate",
                digest=acceptance_sha,
                image_path=acceptance_image,
                actual_is_ok=True,
            ),
            _feedback_row(
                sample_id="new-ok",
                digest=feedback_sha,
                image_path=feedback_image,
                actual_is_ok=True,
            ),
            _feedback_row(
                sample_id="confirmed-ng",
                digest=conflict_sha,
                image_path=conflict_image,
                actual_is_ok=False,
            ),
        ],
    )

    snapshot = ColorBaselineEvidenceProvider(
        product="Cable1",
        area="A",
        model_type="yolo",
    ).collect(
        acceptance_repository=repository,
        feedback_manifest=feedback,
    )

    assert snapshot.selected_count == 2
    assert snapshot.selected_acceptance_count == 1
    assert snapshot.selected_feedback_count == 1
    assert snapshot.duplicate_count == 1
    assert snapshot.confirmed_ng_count == 1
    assert snapshot.conflict_count == 1
    assert {sample.source_kind for sample in snapshot.samples} == {
        "acceptance",
        "color_review",
    }
    lineage = snapshot.to_report_dict()
    assert lineage["schema_version"] == 2
    assert lineage["counts"]["selected_total"] == 2
    assert len(lineage["samples"]) == 2
    assert lineage["excluded_samples"][0]["reason_code"] == "TRUTH_CONFLICT"


def test_provider_rejects_unbounded_feedback_path_without_losing_valid_acceptance(
    tmp_path: Path,
) -> None:
    acceptance_image = tmp_path / "acceptance" / "images" / "acc.png"
    outside_image = tmp_path / "outside.png"
    acceptance_sha = _write_image(acceptance_image, b"acceptance")
    outside_sha = _write_image(outside_image, b"outside")
    repository = _AcceptanceRepository(
        tmp_path / "acceptance",
        (_AcceptanceRecord("ACC-1", acceptance_sha),),
        {"ACC-1": acceptance_image},
    )
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [
            _feedback_row(
                sample_id="outside",
                digest=outside_sha,
                image_path=outside_image,
                actual_is_ok=True,
            )
        ],
    )

    snapshot = ColorBaselineEvidenceProvider(
        product="Cable1",
        area="A",
        model_type="yolo",
    ).collect(
        acceptance_repository=repository,
        feedback_manifest=feedback,
    )

    assert snapshot.selected_count == 1
    assert snapshot.selected_feedback_count == 0
    assert snapshot.invalid_count == 1
    assert len(snapshot.excluded_samples) == 1
    exclusion = snapshot.excluded_samples[0]
    assert exclusion.sample_id == "outside"
    assert exclusion.reason_code == "IMAGE_NOT_FOUND"
    assert snapshot.to_report_dict()["excluded_samples"][0]["reason"]


def test_provider_excludes_mixed_product_ng_color_false_reject(
    tmp_path: Path,
) -> None:
    acceptance_image = tmp_path / "acceptance" / "images" / "acc.png"
    feedback_image = tmp_path / "training" / "color_review" / "images" / "false-reject.png"
    acceptance_digest = _write_image(acceptance_image, b"acceptance")
    digest = _write_image(feedback_image, b"false-reject")
    row = _feedback_row(
        sample_id="false-reject",
        digest=digest,
        image_path=feedback_image,
        actual_is_ok=True,
    )
    row.update(
        product_verdict="ng",
        detection_verdict="correct",
        color_verdict="actually_ok",
        review_label="color_false_reject",
    )
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [row],
    )

    snapshot = ColorBaselineEvidenceProvider(
        product="Cable1",
        area="A",
        model_type="yolo",
    ).collect(
        acceptance_repository=_AcceptanceRepository(
            tmp_path / "acceptance",
            (_AcceptanceRecord("ACC-1", acceptance_digest),),
            {"ACC-1": acceptance_image},
        ),
        feedback_manifest=feedback,
    )

    assert snapshot.selected_acceptance_count == 1
    assert snapshot.selected_feedback_count == 0
    assert snapshot.invalid_count == 1
    assert len(snapshot.excluded_samples) == 1
    exclusion = snapshot.excluded_samples[0]
    assert exclusion.sample_id == "false-reject"
    assert exclusion.reason_code == "MIXED_PRODUCT_NG_COLOR_OK"
    assert "整張照片的全部元件" in exclusion.reason


def test_provider_skips_invalid_row_with_unresolvable_output_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    acceptance_image = tmp_path / "acceptance" / "images" / "acc.png"
    acceptance_digest = _write_image(acceptance_image, b"acceptance")
    row = _feedback_row(
        sample_id="invalid-path",
        digest=hashlib.sha256(b"invalid-path").hexdigest(),
        image_path=tmp_path / "unused.png",
        actual_is_ok=True,
    )
    row.update(
        output_image="C:\\" + ("x" * 40_000),
        product_verdict="ng",
        detection_verdict="correct",
        color_verdict="actually_ok",
        review_label="color_false_reject",
    )
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [row],
    )
    original_resolve = Path.resolve

    def _raise_for_untrusted_path(path, *args, **kwargs):
        if len(str(path)) > 1_024:
            raise OSError("path too long")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _raise_for_untrusted_path)

    snapshot = ColorBaselineEvidenceProvider(
        product="Cable1",
        area="A",
        model_type="yolo",
    ).collect(
        acceptance_repository=_AcceptanceRepository(
            tmp_path / "acceptance",
            (_AcceptanceRecord("ACC-1", acceptance_digest),),
            {"ACC-1": acceptance_image},
        ),
        feedback_manifest=feedback,
    )

    assert snapshot.selected_acceptance_count == 1
    assert snapshot.selected_feedback_count == 0
    assert snapshot.invalid_count == 1
    assert snapshot.excluded_samples[0].sample_id == "invalid-path"
    assert snapshot.excluded_samples[0].reason_code == "MIXED_PRODUCT_NG_COLOR_OK"


def test_provider_fails_closed_for_unresolvable_feedback_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_resolve = Path.resolve

    def _raise_for_untrusted_path(path, *args, **kwargs):
        if len(str(path)) > 1_024:
            raise OSError("path too long")
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", _raise_for_untrusted_path)

    with pytest.raises(ColorBaselineError, match="manifest path is invalid"):
        ColorBaselineEvidenceProvider(
            product="Cable1",
            area="A",
            model_type="yolo",
        ).collect(
            acceptance_repository=_AcceptanceRepository(
                tmp_path / "acceptance",
                (),
                {},
            ),
            feedback_manifest="C:\\" + ("x" * 40_000),
        )


def test_provider_supports_relocated_feedback_images(tmp_path: Path) -> None:
    image_path = tmp_path / "training" / "color_review" / "images" / "case-1.png"
    digest = _write_image(image_path, b"relocated")
    repository = _AcceptanceRepository(tmp_path / "acceptance", (), {})
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [
            _feedback_row(
                sample_id="case-1",
                digest=digest,
                image_path=tmp_path / "old-workspace" / "case-1.png",
                actual_is_ok=True,
            )
        ],
    )

    snapshot = ColorBaselineEvidenceProvider(
        product="Cable1",
        area="A",
        model_type="yolo",
    ).collect(
        acceptance_repository=repository,
        feedback_manifest=feedback,
    )

    assert snapshot.selected_feedback_count == 1
    assert snapshot.samples[0].image_path == image_path.resolve()


def test_merged_snapshot_flows_into_component_crop_collection(tmp_path: Path) -> None:
    image_path = tmp_path / "training" / "color_review" / "images" / "case-1.png"
    digest = _write_image(image_path, b"feedback")
    repository = _AcceptanceRepository(tmp_path / "acceptance", (), {})
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [
            _feedback_row(
                sample_id="case-1",
                digest=digest,
                image_path=image_path,
                actual_is_ok=True,
            )
        ],
    )
    snapshot = ColorBaselineEvidenceProvider(
        product="Cable1",
        area="A",
        model_type="yolo",
    ).collect(
        acceptance_repository=repository,
        feedback_manifest=feedback,
    )

    evidence = collect_color_baseline_evidence(
        inference_service=_InferenceService(),
        samples=snapshot.samples,
        inference_type="yolo",
    )

    assert len(evidence) == 1
    assert evidence[0].sample_id == "color-review-case-1"
    assert evidence[0].color == "Red"
    assert evidence[0].image_bgr.shape == (40, 40, 3)


def test_provider_fails_when_only_ng_or_invalid_images_exist(tmp_path: Path) -> None:
    ng_image = tmp_path / "training" / "color_review" / "images" / "ng.png"
    digest = _write_image(ng_image, b"ng")
    repository = _AcceptanceRepository(tmp_path / "acceptance", (), {})
    feedback = _write_feedback(
        tmp_path / "training" / "color_review" / "feedback.csv",
        [
            _feedback_row(
                sample_id="ng",
                digest=digest,
                image_path=ng_image,
                actual_is_ok=False,
            )
        ],
    )

    with pytest.raises(ColorBaselineError, match="No verified OK images"):
        ColorBaselineEvidenceProvider(
            product="Cable1",
            area="A",
            model_type="yolo",
        ).collect(
            acceptance_repository=repository,
            feedback_manifest=feedback,
        )


def test_provider_rejects_unsafe_scope_segments() -> None:
    with pytest.raises(ColorBaselineError, match="product is invalid"):
        ColorBaselineEvidenceProvider(
            product="../Cable1",
            area="A",
            model_type="yolo",
        )
