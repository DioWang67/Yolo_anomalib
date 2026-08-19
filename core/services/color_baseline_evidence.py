"""Build a validated, deduplicated image snapshot for color baseline rebuilding."""

from __future__ import annotations

import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from core.services.color_baseline_recalibration import (
    ColorBaselineError,
    ColorBaselineImageSample,
)

EVIDENCE_LINEAGE_SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AcceptanceEvidenceRepository(Protocol):
    """Read-only acceptance contract required by the evidence provider."""

    manifest_path: Path

    def records(self) -> Sequence[Any]: ...

    def image_file(self, record: Any) -> Path: ...


@dataclass(frozen=True)
class ColorBaselineEvidenceExclusion:
    """One rejected evidence claim with enough context for operator review."""

    sample_id: str
    source_kind: str
    source_manifest: str
    image_path: str
    image_sha256: str
    reason_code: str
    reason: str

    def to_report_dict(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "source_kind": self.source_kind,
            "source_manifest": self.source_manifest,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ColorBaselineEvidenceSnapshot:
    """Immutable image selection plus the audit facts that produced it."""

    samples: tuple[ColorBaselineImageSample, ...]
    selected_acceptance_count: int
    selected_feedback_count: int
    confirmed_ng_count: int
    duplicate_count: int
    conflict_count: int
    invalid_count: int
    acceptance_manifest: str
    feedback_manifest: str
    excluded_samples: tuple[ColorBaselineEvidenceExclusion, ...] = ()

    @property
    def selected_count(self) -> int:
        return len(self.samples)

    def to_report_dict(self) -> dict[str, Any]:
        """Return stable, JSON-safe lineage for the immutable candidate report."""
        return {
            "schema_version": EVIDENCE_LINEAGE_SCHEMA_VERSION,
            "counts": {
                "selected_total": self.selected_count,
                "selected_acceptance_ok": self.selected_acceptance_count,
                "selected_color_review_ok": self.selected_feedback_count,
                "color_review_or_acceptance_ng": self.confirmed_ng_count,
                "deduplicated": self.duplicate_count,
                "truth_conflicts": self.conflict_count,
                "invalid_or_missing": self.invalid_count,
            },
            "manifests": {
                "acceptance": self.acceptance_manifest,
                "color_review": self.feedback_manifest,
            },
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "image_sha256": sample.image_sha256,
                    "source_kind": sample.source_kind,
                    "source_manifest": sample.source_manifest,
                }
                for sample in self.samples
            ],
            "excluded_samples": [
                exclusion.to_report_dict()
                for exclusion in self.excluded_samples
            ],
        }


class ColorBaselineEvidenceProvider:
    """Merge acceptance and color-review truth without mutating either source."""

    def __init__(
        self,
        *,
        product: str,
        area: str,
        model_type: str,
        checker_type: str = "stats",
    ) -> None:
        self.product = _required_text(product, "product")
        self.area = _required_text(area, "area")
        self.model_type = _required_text(model_type, "model_type").casefold()
        self.checker_type = _required_text(checker_type, "checker_type").casefold()

    def collect(
        self,
        *,
        acceptance_repository: AcceptanceEvidenceRepository,
        feedback_manifest: str | Path | None,
    ) -> ColorBaselineEvidenceSnapshot:
        """Create one deterministic snapshot and exclude every truth conflict."""
        acceptance_records = tuple(acceptance_repository.records())
        acceptance_manifest = str(
            Path(getattr(acceptance_repository, "manifest_path", "")).resolve()
        )
        try:
            feedback_path = (
                Path(feedback_manifest).expanduser().resolve()
                if feedback_manifest is not None
                else None
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ColorBaselineError(
                f"Color review manifest path is invalid: {feedback_manifest!r}."
            ) from exc

        invalid_count = 0
        duplicate_count = 0
        exclusions: list[ColorBaselineEvidenceExclusion] = []
        claims_by_hash: dict[str, set[str]] = {}
        invalid_truth_hashes: set[str] = set()
        acceptance_ok_by_hash: dict[str, Any] = {}
        negative_hashes: set[str] = set()

        for record in acceptance_records:
            if not self._matches_target(record):
                continue
            if str(getattr(record, "review_status", "")).casefold() != "confirmed":
                continue
            verdict = str(getattr(record, "expected_verdict", "")).upper()
            if verdict not in {"OK", "NG"}:
                continue
            digest = _normalized_sha256(getattr(record, "image_sha256", ""))
            if digest is None:
                invalid_count += 1
                exclusions.append(
                    _acceptance_exclusion(
                        record,
                        acceptance_manifest=acceptance_manifest,
                        reason_code="INVALID_IMAGE_SHA256",
                        reason="驗收資料的影像 SHA-256 缺失或格式錯誤。",
                    )
                )
                continue
            claims_by_hash.setdefault(digest, set()).add(verdict)
            if verdict == "NG":
                negative_hashes.add(digest)
                continue
            if digest in acceptance_ok_by_hash:
                duplicate_count += 1
                continue
            acceptance_ok_by_hash[digest] = record

        feedback_by_hash: dict[str, list[dict[str, str]]] = {}
        if feedback_path is not None and _is_regular_file(feedback_path):
            feedback_rows = _read_feedback_rows(feedback_path)
            for row in feedback_rows:
                if not self._matches_feedback_scope(row):
                    continue
                digest = _normalized_sha256(row.get("image_sha256"))
                truth = str(row.get("actual_is_ok") or "").strip()
                if digest is None:
                    invalid_count += 1
                    exclusions.append(
                        _feedback_exclusion(
                            row,
                            feedback_path=feedback_path,
                            reason_code="INVALID_IMAGE_SHA256",
                            reason="顏色覆核資料的影像 SHA-256 缺失或格式錯誤。",
                        )
                    )
                    continue
                if truth not in {"0", "1"}:
                    invalid_count += 1
                    exclusions.append(
                        _feedback_exclusion(
                            row,
                            feedback_path=feedback_path,
                            reason_code="INVALID_HUMAN_VERDICT",
                            reason="顏色覆核資料缺少有效的人工 OK／NG 真值。",
                            image_sha256=digest,
                        )
                    )
                    continue
                positive_validation_issue = (
                    _positive_feedback_validation_issue(row)
                    if truth == "1"
                    else None
                )
                if positive_validation_issue is not None:
                    invalid_count += 1
                    invalid_truth_hashes.add(digest)
                    sample_id = str(row.get("sample_id") or "").strip()
                    resolved_feedback_image = _feedback_image_path(
                        feedback_path,
                        (row,),
                        (sample_id,) if sample_id else (),
                    )
                    exclusions.append(
                        _feedback_exclusion(
                            row,
                            feedback_path=feedback_path,
                            reason_code=positive_validation_issue[0],
                            reason=positive_validation_issue[1],
                            image_path=(
                                str(resolved_feedback_image)
                                if resolved_feedback_image is not None
                                else ""
                            ),
                            image_sha256=digest,
                        )
                    )
                    continue
                verdict = "OK" if truth == "1" else "NG"
                claims_by_hash.setdefault(digest, set()).add(verdict)
                if verdict == "NG":
                    negative_hashes.add(digest)
                feedback_by_hash.setdefault(digest, []).append(row)

        conflicted_hashes = {
            digest for digest, claims in claims_by_hash.items() if len(claims) > 1
        }
        for digest in sorted(conflicted_hashes):
            sample_ids = {
                str(getattr(acceptance_ok_by_hash.get(digest), "sample_id", "")).strip()
            }
            sample_ids.update(
                str(row.get("sample_id") or "").strip()
                for row in feedback_by_hash.get(digest, ())
            )
            sample_ids.discard("")
            exclusions.append(
                ColorBaselineEvidenceExclusion(
                    sample_id="、".join(sorted(sample_ids)) or digest[:12],
                    source_kind="merged",
                    source_manifest=f"{acceptance_manifest} | {feedback_path or ''}",
                    image_path="",
                    image_sha256=digest,
                    reason_code="TRUTH_CONFLICT",
                    reason="同一張影像同時存在人工 OK 與 NG 真值。",
                )
            )
        blocked_hashes = conflicted_hashes | invalid_truth_hashes
        selected_by_hash: dict[str, ColorBaselineImageSample] = {}

        for digest, record in sorted(
            acceptance_ok_by_hash.items(),
            key=lambda item: str(getattr(item[1], "sample_id", "")),
        ):
            if digest in blocked_hashes:
                continue
            try:
                acceptance_image = Path(
                    acceptance_repository.image_file(record)
                ).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                invalid_count += 1
                exclusions.append(
                    _acceptance_exclusion(
                        record,
                        acceptance_manifest=acceptance_manifest,
                        reason_code="INVALID_IMAGE_PATH",
                        reason=f"驗收影像路徑無法安全解析：{exc}",
                        image_sha256=digest,
                    )
                )
                continue
            validation_issue = _image_validation_issue(
                acceptance_image,
                digest,
            )
            if validation_issue is not None:
                invalid_count += 1
                exclusions.append(
                    _acceptance_exclusion(
                        record,
                        acceptance_manifest=acceptance_manifest,
                        reason_code=validation_issue[0],
                        reason=validation_issue[1],
                        image_path=str(acceptance_image),
                        image_sha256=digest,
                    )
                )
                continue
            selected_by_hash[digest] = ColorBaselineImageSample(
                sample_id=str(getattr(record, "sample_id", "")),
                image_path=acceptance_image,
                image_sha256=digest,
                product=self.product,
                area=self.area,
                source_kind="acceptance",
                source_manifest=acceptance_manifest,
            )

        selected_feedback_count = 0
        for digest, scoped_rows in sorted(feedback_by_hash.items()):
            if digest in blocked_hashes or digest in negative_hashes:
                continue
            sample_ids = sorted(
                {
                    str(row.get("sample_id") or "").strip()
                    for row in scoped_rows
                    if str(row.get("sample_id") or "").strip()
                }
            )
            if not sample_ids:
                invalid_count += 1
                exclusions.append(
                    _feedback_exclusion(
                        scoped_rows[0],
                        feedback_path=feedback_path,
                        reason_code="MISSING_SAMPLE_ID",
                        reason="顏色覆核資料缺少 sample ID。",
                        image_sha256=digest,
                    )
                )
                continue
            duplicate_count += max(0, len(sample_ids) - 1)
            if digest in selected_by_hash:
                duplicate_count += 1
                continue
            feedback_image = _feedback_image_path(
                feedback_path,
                scoped_rows,
                sample_ids,
            )
            if feedback_image is None:
                invalid_count += 1
                exclusions.append(
                    _feedback_exclusion(
                        scoped_rows[0],
                        feedback_path=feedback_path,
                        reason_code="IMAGE_NOT_FOUND",
                        reason="找不到唯一且位於顏色覆核資料夾內的影像檔。",
                        image_sha256=digest,
                    )
                )
                continue
            validation_issue = _image_validation_issue(feedback_image, digest)
            if validation_issue is not None:
                invalid_count += 1
                exclusions.append(
                    _feedback_exclusion(
                        scoped_rows[0],
                        feedback_path=feedback_path,
                        reason_code=validation_issue[0],
                        reason=validation_issue[1],
                        image_path=str(feedback_image),
                        image_sha256=digest,
                    )
                )
                continue
            selected_by_hash[digest] = ColorBaselineImageSample(
                sample_id=f"color-review-{sample_ids[0]}",
                image_path=feedback_image,
                image_sha256=digest,
                product=self.product,
                area=self.area,
                source_kind="color_review",
                source_manifest=str(feedback_path),
            )
            selected_feedback_count += 1

        samples = tuple(
            sorted(
                selected_by_hash.values(),
                key=lambda sample: (sample.source_kind, sample.sample_id),
            )
        )
        if not samples:
            raise ColorBaselineError(
                "No verified OK images are available from acceptance or color review."
            )
        selected_acceptance_count = sum(
            sample.source_kind == "acceptance" for sample in samples
        )
        return ColorBaselineEvidenceSnapshot(
            samples=samples,
            selected_acceptance_count=selected_acceptance_count,
            selected_feedback_count=selected_feedback_count,
            confirmed_ng_count=len(negative_hashes),
            duplicate_count=duplicate_count,
            conflict_count=len(conflicted_hashes),
            invalid_count=invalid_count,
            acceptance_manifest=acceptance_manifest,
            feedback_manifest=str(feedback_path) if feedback_path is not None else "",
            excluded_samples=tuple(
                sorted(
                    exclusions,
                    key=lambda item: (
                        item.reason_code,
                        item.source_kind,
                        item.sample_id,
                    ),
                )
            ),
        )

    def _matches_target(self, record: Any) -> bool:
        return (
            str(getattr(record, "product", "")).casefold()
            == self.product.casefold()
            and str(getattr(record, "area", "")).casefold() == self.area.casefold()
        )

    def _matches_feedback_scope(self, row: Mapping[str, str]) -> bool:
        return (
            str(row.get("product") or "").casefold() == self.product.casefold()
            and str(row.get("area") or "").casefold() == self.area.casefold()
            and str(row.get("model_type") or "").casefold() == self.model_type
            and str(row.get("checker_type") or "").casefold() == self.checker_type
        )


def _read_feedback_rows(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            required = {
                "sample_id",
                "image_sha256",
                "product",
                "area",
                "model_type",
                "checker_type",
                "actual_is_ok",
                "output_image",
            }
            missing = sorted(required - fields)
            if missing:
                raise ColorBaselineError(
                    "Color-review feedback is missing required columns: "
                    + ", ".join(missing)
                )
            return tuple(dict(row) for row in reader)
    except (OSError, csv.Error) as exc:
        raise ColorBaselineError(f"Unable to read color-review feedback: {path}") from exc


def _positive_feedback_validation_issue(
    row: Mapping[str, str],
) -> tuple[str, str] | None:
    product_verdict = str(row.get("product_verdict") or "").strip().casefold()
    detection_verdict = str(row.get("detection_verdict") or "").strip().casefold()
    if detection_verdict not in {"", "correct"}:
        return (
            "INCONSISTENT_POSITIVE_REVIEW",
            "人工標為 OK，但偵測覆核不是正確，不能當成正常顏色基準。",
        )
    if product_verdict in {"", "ok"}:
        return None
    review_label = str(row.get("review_label") or "").strip().casefold()
    color_verdict = str(row.get("color_verdict") or "").strip().casefold()
    is_mixed_product_failure = (
        product_verdict == "ng"
        and review_label == "color_false_reject"
        and color_verdict in {"actually_ok", "ok"}
    )
    if is_mixed_product_failure:
        return (
            "MIXED_PRODUCT_NG_COLOR_OK",
            "該顏色項目人工確認為 OK，但整體產品真值為 NG；完整基準重建會擷取整張照片的全部元件，為避免異常元件污染基準而排除。",
        )
    return (
        "INCONSISTENT_POSITIVE_REVIEW",
        "人工標為 OK，但整體產品或覆核欄位無法證明這是有效正常樣本。",
    )


def _acceptance_exclusion(
    record: Any,
    *,
    acceptance_manifest: str,
    reason_code: str,
    reason: str,
    image_path: str = "",
    image_sha256: str = "",
) -> ColorBaselineEvidenceExclusion:
    declared_path = str(getattr(record, "image_path", "") or "").strip()
    return ColorBaselineEvidenceExclusion(
        sample_id=str(getattr(record, "sample_id", "") or "").strip(),
        source_kind="acceptance",
        source_manifest=acceptance_manifest,
        image_path=image_path or declared_path,
        image_sha256=image_sha256
        or str(getattr(record, "image_sha256", "") or "").strip(),
        reason_code=reason_code,
        reason=reason,
    )


def _feedback_exclusion(
    row: Mapping[str, str],
    *,
    feedback_path: Path | None,
    reason_code: str,
    reason: str,
    image_path: str = "",
    image_sha256: str = "",
) -> ColorBaselineEvidenceExclusion:
    return ColorBaselineEvidenceExclusion(
        sample_id=str(row.get("sample_id") or "").strip(),
        source_kind="color_review",
        source_manifest=str(feedback_path) if feedback_path is not None else "",
        image_path=image_path or str(row.get("output_image") or "").strip(),
        image_sha256=image_sha256
        or str(row.get("image_sha256") or "").strip(),
        reason_code=reason_code,
        reason=reason,
    )


def _feedback_image_path(
    manifest_path: Path | None,
    rows: Sequence[Mapping[str, str]],
    sample_ids: Sequence[str],
) -> Path | None:
    if manifest_path is None:
        return None
    feedback_root = _safe_resolve_path(manifest_path.parent)
    if feedback_root is None:
        return None

    declared: set[Path] = set()
    for row in rows:
        raw_path = str(row.get("output_image") or "").strip()
        if not raw_path:
            continue
        try:
            unresolved_candidate = Path(raw_path).expanduser()
            if unresolved_candidate.is_symlink():
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        candidate = _safe_resolve_path(unresolved_candidate)
        if candidate is not None and _is_within(candidate, feedback_root):
            declared.add(candidate)

    existing = sorted(
        (path for path in declared if _is_regular_file(path)),
        key=str,
    )
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        return None

    images_root = feedback_root / "images"
    try:
        if images_root.is_symlink() or not images_root.is_dir():
            return None
        image_entries = tuple(images_root.iterdir())
    except (OSError, RuntimeError, ValueError):
        return None

    relocated: list[Path] = []
    for path in image_entries:
        try:
            if path.is_symlink() or path.stem not in sample_ids:
                continue
        except (OSError, RuntimeError, ValueError):
            continue
        resolved = _safe_resolve_path(path)
        if (
            resolved is not None
            and _is_within(resolved, feedback_root)
            and _is_regular_file(resolved)
        ):
            relocated.append(resolved)
    relocated.sort(key=str)
    return relocated[0] if len(relocated) == 1 else None


def _image_validation_issue(
    path: Path,
    expected_sha256: str,
) -> tuple[str, str] | None:
    try:
        if path.is_symlink():
            return "UNSAFE_SYMBOLIC_LINK", "影像檔是符號連結，基於安全政策不予使用。"
        if not path.is_file():
            return "IMAGE_NOT_FOUND", "影像檔不存在。"
    except (OSError, RuntimeError, ValueError):
        return "INVALID_IMAGE_PATH", "影像檔路徑無法安全解析。"
    actual_sha256 = _sha256_file(path)
    if not actual_sha256:
        return "IMAGE_READ_FAILED", "影像檔無法讀取。"
    if actual_sha256 != expected_sha256:
        return "IMAGE_SHA256_MISMATCH", "影像內容與 manifest 的 SHA-256 不一致。"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (OSError, RuntimeError, ValueError):
        return ""
    return digest.hexdigest()


def _normalized_sha256(value: object) -> str | None:
    normalized = str(value or "").strip().casefold()
    return normalized if _SHA256_PATTERN.fullmatch(normalized) else None


def _required_text(value: str, field: str) -> str:
    normalized = str(value).strip()
    if (
        not normalized
        or normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
    ):
        raise ColorBaselineError(f"Color baseline {field} is invalid: {value!r}.")
    return normalized


def _safe_resolve_path(value: str | Path) -> Path | None:
    try:
        return Path(value).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _is_regular_file(path: Path) -> bool:
    try:
        return not path.is_symlink() and path.is_file()
    except (OSError, RuntimeError, ValueError):
        return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True
