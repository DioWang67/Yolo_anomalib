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

EVIDENCE_LINEAGE_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AcceptanceEvidenceRepository(Protocol):
    """Read-only acceptance contract required by the evidence provider."""

    manifest_path: Path

    def records(self) -> Sequence[Any]: ...

    def image_file(self, record: Any) -> Path: ...


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
        feedback_path = (
            Path(feedback_manifest).expanduser().resolve()
            if feedback_manifest is not None
            else None
        )

        invalid_count = 0
        duplicate_count = 0
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
        if feedback_path is not None and feedback_path.is_file():
            feedback_rows = _read_feedback_rows(feedback_path)
            for row in feedback_rows:
                if not self._matches_feedback_scope(row):
                    continue
                digest = _normalized_sha256(row.get("image_sha256"))
                truth = str(row.get("actual_is_ok") or "").strip()
                if digest is None or truth not in {"0", "1"}:
                    invalid_count += 1
                    continue
                if truth == "1" and not _positive_feedback_is_consistent(row):
                    invalid_count += 1
                    invalid_truth_hashes.add(digest)
                    continue
                verdict = "OK" if truth == "1" else "NG"
                claims_by_hash.setdefault(digest, set()).add(verdict)
                if verdict == "NG":
                    negative_hashes.add(digest)
                feedback_by_hash.setdefault(digest, []).append(row)

        conflicted_hashes = {
            digest for digest, claims in claims_by_hash.items() if len(claims) > 1
        }
        blocked_hashes = conflicted_hashes | invalid_truth_hashes
        selected_by_hash: dict[str, ColorBaselineImageSample] = {}

        for digest, record in sorted(
            acceptance_ok_by_hash.items(),
            key=lambda item: str(getattr(item[1], "sample_id", "")),
        ):
            if digest in blocked_hashes:
                continue
            acceptance_image = Path(
                acceptance_repository.image_file(record)
            ).resolve()
            if not _verified_image(acceptance_image, digest):
                invalid_count += 1
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
            if feedback_image is None or not _verified_image(feedback_image, digest):
                invalid_count += 1
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


def _positive_feedback_is_consistent(row: Mapping[str, str]) -> bool:
    product_verdict = str(row.get("product_verdict") or "").strip().casefold()
    detection_verdict = str(row.get("detection_verdict") or "").strip().casefold()
    return (
        product_verdict in {"", "ok"}
        and detection_verdict in {"", "correct"}
    )


def _feedback_image_path(
    manifest_path: Path | None,
    rows: Sequence[Mapping[str, str]],
    sample_ids: Sequence[str],
) -> Path | None:
    if manifest_path is None:
        return None
    feedback_root = manifest_path.parent.resolve()
    declared: set[Path] = set()
    for row in rows:
        raw_path = str(row.get("output_image") or "").strip()
        if not raw_path:
            continue
        candidate = Path(raw_path).expanduser().resolve()
        if _is_within(candidate, feedback_root):
            declared.add(candidate)
    existing = sorted(
        (path for path in declared if path.is_file() and not path.is_symlink()),
        key=str,
    )
    if len(existing) == 1:
        return existing[0]
    if len(existing) > 1:
        return None

    images_root = feedback_root / "images"
    if not images_root.is_dir():
        return None
    relocated = sorted(
        (
            path.resolve()
            for path in images_root.iterdir()
            if path.is_file()
            and not path.is_symlink()
            and path.stem in sample_ids
        ),
        key=str,
    )
    return relocated[0] if len(relocated) == 1 else None


def _verified_image(path: Path, expected_sha256: str) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    return _sha256_file(path) == expected_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
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


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
