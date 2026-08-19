"""Independent model-acceptance data and inference services.

The acceptance application reuses :class:`core.detection_system.DetectionSystem`
for production-equivalent verdicts while keeping its evidence outside the
production result store and all training datasets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import yaml

from core.services.acceptance_artifacts import (
    AcceptanceArtifactBundle,
    AcceptanceArtifactError,
    resolve_configured_model_weight,
)
from core.station_data import load_station_data_paths
from core.types import DetectionResult
from tools.cross_process_lock import (
    CrossProcessLockTimeoutError,
    cross_process_file_lock,
)

if TYPE_CHECKING:  # pragma: no cover
    from core.detection_system import DetectionSystem

ACCEPTANCE_REASON_CODES = (
    "MISSING",
    "WRONG_COMPONENT",
    "UNEXPECTED_COMPONENT",
    "POSITION_SHIFT",
    "COLOR_MISMATCH",
    "COUNT_MISMATCH",
    "SEQUENCE_MISMATCH",
    "OTHER",
)
SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
MANIFEST_FIELDS = (
    "sample_id",
    "image_path",
    "image_sha256",
    "product",
    "area",
    "expected_verdict",
    "expected_reasons",
    "defect_class",
    "source_batch",
    "captured_at",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "notes",
    "machine_status",
    "machine_reasons",
    "model_version",
    "model_sha256",
    "runtime_config_sha256",
    "color_model_sha256",
    "acceptance_run_id",
    "artifact_bundle_sha256",
    "color_revision_contract_sha256",
    "color_revision_overrides_json",
    "include_active_color_revisions",
    "inference_at",
    "latency_ms",
    "error",
    "color_check_status",
    "color_details_json",
)


class AcceptanceDataError(ValueError):
    """Raised when acceptance evidence is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class AcceptanceRecord:
    """One immutable view of an acceptance sample and its evidence."""

    sample_id: str
    image_path: str
    image_sha256: str
    product: str
    area: str
    expected_verdict: str = ""
    expected_reasons: str = ""
    defect_class: str = ""
    source_batch: str = ""
    captured_at: str = ""
    review_status: str = "pending"
    reviewed_by: str = ""
    reviewed_at: str = ""
    notes: str = ""
    machine_status: str = ""
    machine_reasons: str = ""
    model_version: str = ""
    model_sha256: str = ""
    runtime_config_sha256: str = ""
    color_model_sha256: str = ""
    acceptance_run_id: str = ""
    artifact_bundle_sha256: str = ""
    color_revision_contract_sha256: str = ""
    color_revision_overrides_json: str = ""
    include_active_color_revisions: str = ""
    inference_at: str = ""
    latency_ms: str = ""
    error: str = ""
    color_check_status: str = ""
    color_details_json: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> AcceptanceRecord:
        values = {field: str(row.get(field) or "").strip() for field in MANIFEST_FIELDS}
        values["review_status"] = values["review_status"] or "pending"
        return cls(**values)

    def with_changes(self, **changes: str) -> AcceptanceRecord:
        return replace(self, **changes)


def verified_acceptance_image_path(
    dataset_root: str | Path,
    record: AcceptanceRecord,
    *,
    verify_checksum: bool = True,
) -> Path:
    """Resolve one evidence image without following symbolic-link components."""

    root = Path(dataset_root).expanduser().resolve()
    relative = Path(record.image_path)
    if relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise AcceptanceDataError(
            f"Acceptance image escapes its evidence directory: {record.image_path}"
        )
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise AcceptanceDataError(
                f"Acceptance image is missing or unsafe: {record.sample_id}"
            )
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise AcceptanceDataError(
            f"Acceptance image escapes its evidence directory: {record.image_path}"
        )
    if not resolved.is_file():
        raise AcceptanceDataError(
            f"Acceptance image is missing or unsafe: {record.sample_id}"
        )
    if verify_checksum and _sha256_file(resolved) != record.image_sha256.lower():
        raise AcceptanceDataError(
            f"Acceptance image checksum mismatch: {record.sample_id}"
        )
    return resolved


@dataclass(frozen=True)
class AcceptanceMetrics:
    """Confusion counts and yield indicators from confirmed samples."""

    confirmed: int
    pending: int
    inferred: int
    tp: int
    fp: int
    fn: int
    tn: int
    errors: int
    #: Confirmed samples whose human verdict is neither OK nor NG. They are
    #: corrupt evidence, so they are excluded from ``confirmed`` and from every
    #: rate rather than being counted as a decided sample.
    malformed: int = 0

    @property
    def true_yield(self) -> float | None:
        return _rate(self.tn + self.fp, self.confirmed)

    @property
    def machine_yield(self) -> float | None:
        return _rate(self.tn + self.fn, self.tp + self.fp + self.fn + self.tn)

    @property
    def escape_rate(self) -> float | None:
        return _rate(self.fn, self.tp + self.fn)

    @property
    def overkill_rate(self) -> float | None:
        return _rate(self.fp, self.fp + self.tn)


@dataclass(frozen=True)
class ModelIdentity:
    version: str
    sha256: str
    runtime_config_sha256: str = ""
    color_model_sha256: str = ""


@dataclass(frozen=True)
class AcceptanceInferenceOutcome:
    """Serializable inference evidence plus a display-only annotated frame."""

    sample_id: str
    machine_status: str
    machine_reasons: tuple[str, ...]
    model_version: str
    model_sha256: str
    inference_at: str
    latency_ms: float
    error: str
    annotated_frame: np.ndarray | None = None
    runtime_config_sha256: str = ""
    color_model_sha256: str = ""
    color_check_status: str = ""
    color_details_json: str = "[]"


@dataclass(frozen=True)
class AcceptanceSnapshot:
    """Immutable manifest snapshot used as a comparison baseline."""

    snapshot_id: str
    root: Path
    manifest_path: Path
    summary_path: Path


@dataclass(frozen=True)
class AcceptanceComparison:
    """Machine-decision changes against one immutable snapshot."""

    snapshot_id: str
    baseline_version: str
    current_version: str
    common_samples: int
    improved: int
    regressed: int
    unchanged_correct: int
    unchanged_incorrect: int
    baseline_false_positives: int
    current_false_positives: int
    baseline_false_negatives: int
    current_false_negatives: int
    changed_sample_ids: tuple[str, ...]


class AcceptanceRepository:
    """Thread-safe, atomically persisted acceptance manifest repository."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest_path = self.root / "ground_truth.csv"
        self.images_dir = self.root / "images"
        self._lock = threading.RLock()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        with self._exclusive_mutation():
            if not self.manifest_path.exists():
                self._write_records(())

    def records(self) -> tuple[AcceptanceRecord, ...]:
        with self._lock:
            return tuple(self._read_records())

    def manifest_sha256(self) -> str:
        with self._lock:
            return _sha256_file(self.manifest_path)

    def image_file(self, record: AcceptanceRecord) -> Path:
        return verified_acceptance_image_path(
            self.root,
            record,
            verify_checksum=False,
        )

    def verified_image_file(self, record: AcceptanceRecord) -> Path:
        return verified_acceptance_image_path(self.root, record)

    def import_images(
        self,
        paths: Iterable[str | Path],
        *,
        product: str,
        area: str,
        source_batch: str = "",
    ) -> tuple[AcceptanceRecord, ...]:
        normalized_product = _required_segment(product, "product")
        normalized_area = _required_segment(area, "area")
        with self._exclusive_mutation():
            records = self._read_records()
            by_hash = {record.image_sha256: record for record in records}
            imported: list[AcceptanceRecord] = []
            changed = False
            for raw_path in paths:
                source = Path(raw_path).expanduser().resolve()
                if not source.is_file():
                    continue
                suffix = source.suffix.lower()
                if suffix not in SUPPORTED_IMAGE_SUFFIXES:
                    continue
                digest = _sha256_file(source)
                existing = by_hash.get(digest)
                if existing is not None:
                    imported.append(existing)
                    continue
                sample_id = _unique_sample_id(digest, records)
                relative_path = Path("images") / f"{sample_id}{suffix}"
                destination = self.root / relative_path
                shutil.copy2(source, destination)
                record = AcceptanceRecord(
                    sample_id=sample_id,
                    image_path=relative_path.as_posix(),
                    image_sha256=digest,
                    product=normalized_product,
                    area=normalized_area,
                    source_batch=source_batch.strip(),
                    captured_at=datetime.fromtimestamp(source.stat().st_mtime)
                    .astimezone()
                    .isoformat(timespec="seconds"),
                )
                records.append(record)
                by_hash[digest] = record
                imported.append(record)
                changed = True
            if changed:
                self._write_records(records)
            return tuple(imported)

    def confirm(
        self,
        sample_id: str,
        *,
        verdict: str,
        reasons: Sequence[str] = (),
        reviewed_by: str,
        defect_class: str = "",
        notes: str = "",
    ) -> AcceptanceRecord:
        normalized_verdict = verdict.strip().upper()
        if normalized_verdict not in {"OK", "NG"}:
            raise AcceptanceDataError("Expected verdict must be OK or NG.")
        reviewer = reviewed_by.strip()
        if not reviewer:
            raise AcceptanceDataError("Reviewer is required before confirmation.")
        normalized_reasons = _normalize_reasons(reasons)
        if normalized_verdict == "OK":
            normalized_reasons = ()
            defect_class = ""
        elif not normalized_reasons:
            raise AcceptanceDataError("At least one reason is required for an NG sample.")
        return self._update(
            sample_id,
            expected_verdict=normalized_verdict,
            expected_reasons="|".join(normalized_reasons),
            defect_class=defect_class.strip(),
            review_status="confirmed",
            reviewed_by=reviewer,
            reviewed_at=_now_iso(),
            notes=notes.strip(),
        )

    def save_inference(self, outcome: AcceptanceInferenceOutcome) -> AcceptanceRecord:
        return self._update(
            outcome.sample_id,
            **_inference_changes(outcome),
        )

    def save_inference_batch(
        self,
        outcomes: Sequence[AcceptanceInferenceOutcome],
        *,
        run_id: str,
        artifact_bundle: AcceptanceArtifactBundle,
        expected_manifest_sha256: str,
    ) -> tuple[tuple[AcceptanceRecord, ...], str]:
        """Atomically commit one completed run using checksum compare-and-swap."""

        normalized_run_id = _required_segment(run_id, "acceptance run ID")
        by_id = {outcome.sample_id: outcome for outcome in outcomes}
        if not by_id or len(by_id) != len(outcomes):
            raise AcceptanceDataError(
                "Inference batch must contain unique, non-empty outcomes."
            )
        expected_identity = (
            artifact_bundle.version,
            artifact_bundle.model_weight.sha256.lower(),
            artifact_bundle.model_config.sha256.lower(),
            (
                artifact_bundle.color_model.sha256.lower()
                if artifact_bundle.color_model is not None
                else ""
            ),
        )
        if any(
            (
                outcome.model_version,
                outcome.model_sha256.lower(),
                outcome.runtime_config_sha256.lower(),
                outcome.color_model_sha256.lower(),
            )
            != expected_identity
            for outcome in outcomes
        ):
            raise AcceptanceDataError(
                "Inference batch identity does not match its artifact bundle."
            )
        try:
            with self._exclusive_mutation():
                current_sha256 = _sha256_file(self.manifest_path)
                if current_sha256 != expected_manifest_sha256.lower():
                    raise AcceptanceDataError(
                        "Acceptance manifest changed during inference; batch commit was rejected."
                    )
                records = self._read_records()
                record_ids = {record.sample_id for record in records}
                missing = sorted(set(by_id) - record_ids)
                if missing:
                    raise AcceptanceDataError(
                        f"Inference batch contains {len(missing)} unknown samples."
                    )
                metadata = {
                    "acceptance_run_id": normalized_run_id,
                    "artifact_bundle_sha256": artifact_bundle.bundle_sha256,
                    "color_revision_contract_sha256": (
                        artifact_bundle.color_revision_contract_sha256
                    ),
                    "color_revision_overrides_json": json.dumps(
                        dict(artifact_bundle.color_revision_overrides),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "include_active_color_revisions": (
                        "true"
                        if artifact_bundle.include_active_color_revisions
                        else "false"
                    ),
                }
                updated_records = [
                    _committed_record(
                        record,
                        by_id,
                        metadata,
                        artifact_bundle.bundle_sha256,
                    )
                    for record in records
                ]
                self._write_records(updated_records)
                committed_sha256 = _sha256_file(self.manifest_path)
                committed = tuple(
                    record
                    for record in updated_records
                    if record.sample_id in by_id
                )
                return committed, committed_sha256
        except CrossProcessLockTimeoutError as exc:
            raise AcceptanceDataError(
                "Another process is updating the acceptance manifest."
            ) from exc

    def create_snapshot(
        self,
        *,
        label: str = "",
        require_completed_run: bool = False,
    ) -> AcceptanceSnapshot:
        """Create an immutable manifest and image-hash inventory."""
        with self._exclusive_mutation():
            records = self._read_records()
            if require_completed_run:
                # The property being protected is that every result came from
                # one identical artifact combination, which is exactly what an
                # equal ``artifact_bundle_sha256`` states. Requiring a single
                # run ID instead would additionally forbid re-running the few
                # samples that errored transiently, which loses no evidence.
                bundle_ids = {record.artifact_bundle_sha256 for record in records}
                run_ids = {record.acceptance_run_id for record in records}
                if (
                    not records
                    or len(bundle_ids) != 1
                    or "" in bundle_ids
                    or "" in run_ids
                    or any(
                        record.machine_status not in {"OK", "NG"}
                        for record in records
                    )
                    or any(
                        record.review_status == "confirmed"
                        and record.expected_verdict not in {"OK", "NG"}
                        for record in records
                    )
                    or any(
                        not _is_completed_acceptance_run(self.root, run_id)
                        for run_id in run_ids
                    )
                ):
                    raise AcceptanceDataError(
                        "A formal snapshot requires every sample to carry a result "
                        "from one identical artifact bundle, produced by completed "
                        "runs, with no pending, error, or malformed results."
                    )
            timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
            safe_label = _safe_label(label)
            snapshot_id = timestamp + (f"-{safe_label}" if safe_label else "")
            snapshots_root = self.root / "snapshots"
            snapshots_root.mkdir(parents=True, exist_ok=True)
            snapshot_root = snapshots_root / snapshot_id
            if snapshot_root.exists():
                raise AcceptanceDataError(f"Acceptance snapshot already exists: {snapshot_id}")
            temporary_root = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=snapshots_root))
            try:
                manifest_path = temporary_root / "ground_truth.csv"
                _write_manifest_file(manifest_path, records)
                inventory = [
                    {
                        "sample_id": record.sample_id,
                        "image_path": record.image_path,
                        "image_sha256": record.image_sha256,
                    }
                    for record in records
                ]
                (temporary_root / "image_inventory.json").write_text(
                    json.dumps(inventory, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                metrics = calculate_acceptance_metrics(records)
                summary = {
                    "schema_version": 1,
                    "snapshot_id": snapshot_id,
                    "created_at": _now_iso(),
                    "record_count": len(records),
                    "confirmed_count": metrics.confirmed,
                    "model_versions": sorted({record.model_version for record in records if record.model_version}),
                    "runtime_config_sha256": sorted(
                        {record.runtime_config_sha256 for record in records if record.runtime_config_sha256}
                    ),
                    "color_model_sha256": sorted(
                        {record.color_model_sha256 for record in records if record.color_model_sha256}
                    ),
                    "acceptance_run_ids": sorted(
                        {record.acceptance_run_id for record in records if record.acceptance_run_id}
                    ),
                    "artifact_bundle_sha256": sorted(
                        {
                            record.artifact_bundle_sha256
                            for record in records
                            if record.artifact_bundle_sha256
                        }
                    ),
                    "manifest_sha256": _sha256_file(manifest_path),
                    "metrics": _metrics_mapping(metrics),
                }
                summary_path = temporary_root / "snapshot.json"
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                os.replace(temporary_root, snapshot_root)
            finally:
                if temporary_root.exists():
                    shutil.rmtree(temporary_root)
            return AcceptanceSnapshot(
                snapshot_id=snapshot_id,
                root=snapshot_root,
                manifest_path=snapshot_root / "ground_truth.csv",
                summary_path=snapshot_root / "snapshot.json",
            )

    def snapshots(self) -> tuple[AcceptanceSnapshot, ...]:
        snapshots_root = self.root / "snapshots"
        if not snapshots_root.is_dir():
            return ()
        snapshots: list[AcceptanceSnapshot] = []
        for root in sorted(snapshots_root.iterdir()):
            manifest_path = root / "ground_truth.csv"
            summary_path = root / "snapshot.json"
            if root.is_dir() and manifest_path.is_file() and summary_path.is_file():
                snapshots.append(
                    AcceptanceSnapshot(
                        snapshot_id=root.name,
                        root=root,
                        manifest_path=manifest_path,
                        summary_path=summary_path,
                    )
                )
        return tuple(snapshots)

    def compare_with_snapshot(self, snapshot: AcceptanceSnapshot) -> AcceptanceComparison:
        with self._lock:
            current_records = self._read_records()
            baseline_records = _read_manifest_file(snapshot.manifest_path)
        current = {record.sample_id: record for record in current_records}
        baseline = {record.sample_id: record for record in baseline_records}
        common_ids = sorted(set(current) & set(baseline))
        improved = regressed = unchanged_correct = unchanged_incorrect = 0
        changed: list[str] = []
        for sample_id in common_ids:
            old = baseline[sample_id]
            new = current[sample_id]
            old_correct = _machine_decision_correct(old)
            new_correct = _machine_decision_correct(new)
            if old.machine_status != new.machine_status or old.machine_reasons != new.machine_reasons:
                changed.append(sample_id)
            if not old_correct and new_correct:
                improved += 1
            elif old_correct and not new_correct:
                regressed += 1
            elif old_correct:
                unchanged_correct += 1
            else:
                unchanged_incorrect += 1
        baseline_metrics = calculate_acceptance_metrics(baseline_records)
        current_metrics = calculate_acceptance_metrics(current_records)
        return AcceptanceComparison(
            snapshot_id=snapshot.snapshot_id,
            baseline_version=_single_version(baseline_records),
            current_version=_single_version(current_records),
            common_samples=len(common_ids),
            improved=improved,
            regressed=regressed,
            unchanged_correct=unchanged_correct,
            unchanged_incorrect=unchanged_incorrect,
            baseline_false_positives=baseline_metrics.fp,
            current_false_positives=current_metrics.fp,
            baseline_false_negatives=baseline_metrics.fn,
            current_false_negatives=current_metrics.fn,
            changed_sample_ids=tuple(changed),
        )

    def export_backup_zip(self, destination: str | Path) -> Path:
        """Export truth, images, labels, and snapshots as a portable ZIP."""
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.suffix.lower() != ".zip":
            destination_path = destination_path.with_suffix(".zip")
        temporary_path = destination_path.with_name(f".{destination_path.name}.{os.getpid()}.tmp")
        try:
            with zipfile.ZipFile(
                temporary_path,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as archive:
                for path in sorted(self.root.rglob("*")):
                    if not path.is_file():
                        continue
                    relative = path.relative_to(self.root)
                    # ``locks`` holds one zero-information byte per lock file, and
                    # a lock held by another process makes it unreadable on
                    # Windows, which would abort an otherwise valid backup.
                    if relative.parts and relative.parts[0] in {"backups", "locks"}:
                        continue
                    if path.resolve() in {
                        destination_path,
                        temporary_path.resolve(),
                    }:
                        continue
                    archive.write(path, relative.as_posix())
            os.replace(temporary_path, destination_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return destination_path

    def _update(self, sample_id: str, **changes: str) -> AcceptanceRecord:
        try:
            with self._exclusive_mutation():
                records = self._read_records()
                for index, record in enumerate(records):
                    if record.sample_id != sample_id:
                        continue
                    updated = record.with_changes(**changes)
                    records[index] = updated
                    self._write_records(records)
                    return updated
        except CrossProcessLockTimeoutError as exc:
            raise AcceptanceDataError(
                "Another process is updating the acceptance manifest."
            ) from exc
        raise AcceptanceDataError(f"Acceptance sample not found: {sample_id}")

    @contextmanager
    def _exclusive_mutation(self) -> Iterator[None]:
        with self._lock:
            with cross_process_file_lock(
                self.root / "locks" / "acceptance-manifest.lock"
            ):
                yield

    def _read_records(self) -> list[AcceptanceRecord]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return []
            records = [AcceptanceRecord.from_mapping(row) for row in reader]
        seen_ids: set[str] = set()
        for record in records:
            if not record.sample_id:
                raise AcceptanceDataError("Acceptance manifest contains an empty sample_id.")
            if record.sample_id in seen_ids:
                raise AcceptanceDataError(f"Acceptance manifest contains duplicate sample_id: {record.sample_id}")
            seen_ids.add(record.sample_id)
            self.image_file(record)
        return records

    def _write_records(self, records: Sequence[AcceptanceRecord]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".ground_truth.", suffix=".csv.tmp", dir=self.root)
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            _write_manifest_file(temporary_path, records)
            os.replace(temporary_path, self.manifest_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


class AcceptanceInferenceService:
    """Read-only adapter over the production detection orchestrator."""

    def __init__(
        self,
        *,
        project_root: str | Path,
        system_factory: Callable[..., DetectionSystem] | None = None,
        models_root: str | Path | None = None,
        global_config_path: str | Path | None = None,
        model_identity: ModelIdentity | None = None,
        color_revisions_root: str | Path | None = None,
        color_revision_overrides: Mapping[str, str] | None = None,
        include_active_color_revisions: bool = True,
        model_config_overrides: Mapping[tuple[str, str, str], str | Path] | None = None,
        model_weight_path_override: str | Path | None = None,
        color_model_path_override: str | Path | None = None,
    ):
        if system_factory is None:
            from core.detection_system import DetectionSystem

            system_factory = DetectionSystem
        self.project_root = Path(project_root).expanduser().resolve()
        self.data_paths = load_station_data_paths(self.project_root)
        self.models_root = (
            Path(models_root).expanduser().resolve()
            if models_root is not None
            else self.data_paths.models
        )
        resolved_color_revisions_root = (
            Path(color_revisions_root).expanduser().resolve()
            if color_revisions_root is not None
            else self.data_paths.color_revisions
        )
        self._temporary_config_root: tempfile.TemporaryDirectory[str] | None = None
        config_overrides = dict(model_config_overrides or {})
        model_weight_sha256 = ""
        color_model_sha256 = ""
        model_weight: Path | None = None
        if model_weight_path_override is not None:
            model_weight = Path(model_weight_path_override).expanduser().resolve()
            if model_weight.is_symlink() or not model_weight.is_file():
                raise AcceptanceDataError("Selected model weight is missing or unsafe.")
            model_weight_sha256 = _sha256_file(model_weight)
            if (
                model_identity is not None
                and model_identity.sha256
                and model_identity.sha256.lower() != model_weight_sha256
            ):
                raise AcceptanceDataError(
                    "Selected model weight checksum does not match its identity."
                )
            if not config_overrides:
                raise AcceptanceDataError(
                    "A version-matched model config is required for a model weight."
                )
            try:
                for config_path in config_overrides.values():
                    configured_weight = resolve_configured_model_weight(
                        config_path,
                        models_root=self.models_root,
                    )
                    if configured_weight.path != model_weight:
                        raise AcceptanceDataError(
                            "Selected model config does not reference the selected weight."
                        )
            except AcceptanceArtifactError as exc:
                raise AcceptanceDataError(str(exc)) from exc
        color_model: Path | None = None
        if color_model_path_override is not None:
            color_model = Path(color_model_path_override).expanduser().resolve()
            if color_model.is_symlink() or not color_model.is_file():
                raise AcceptanceDataError("Selected color baseline is missing or unsafe.")
            color_model_sha256 = _sha256_file(color_model)
        if color_model is not None:
            if not config_overrides:
                raise AcceptanceDataError(
                    "A version-matched model config is required for artifact overrides."
                )
            self._temporary_config_root = tempfile.TemporaryDirectory(prefix="acceptance-color-config-")
            try:
                temporary_root = Path(self._temporary_config_root.name)
                staged_overrides: dict[tuple[str, str, str], Path] = {}
                for index, (scope, raw_config_path) in enumerate(config_overrides.items()):
                    source_config = Path(raw_config_path).expanduser().resolve()
                    payload = yaml.safe_load(source_config.read_text(encoding="utf-8")) or {}
                    if not isinstance(payload, dict):
                        raise AcceptanceDataError("Selected model config must be a YAML mapping.")
                    if model_weight is not None:
                        # The staged file lives in a temporary directory, so a
                        # formerly config-relative path must be made absolute.
                        payload["weights"] = str(model_weight)
                    payload["enable_color_check"] = True
                    payload["color_checker_type"] = "stats"
                    payload["color_model_path"] = str(color_model)
                    staged_path = temporary_root / f"config-{index}.yaml"
                    staged_path.write_text(
                        yaml.safe_dump(
                            payload,
                            allow_unicode=True,
                            sort_keys=False,
                        ),
                        encoding="utf-8",
                    )
                    staged_overrides[scope] = staged_path
            except (OSError, TypeError, ValueError, yaml.YAMLError):
                self._temporary_config_root.cleanup()
                self._temporary_config_root = None
                raise
            config_overrides = staged_overrides
        identity_changes: dict[str, str] = {}
        if model_weight_sha256:
            identity_changes["sha256"] = model_weight_sha256
        if color_model_sha256:
            identity_changes["color_model_sha256"] = color_model_sha256
        self._model_identity = (
            replace(model_identity, **identity_changes)
            if model_identity is not None and identity_changes
            else model_identity
        )
        try:
            self._system = system_factory(
                config_path=str(
                    Path(global_config_path).expanduser().resolve()
                    if global_config_path is not None
                    else self.project_root / "config.yaml"
                ),
                initialize_camera=False,
                models_root=str(self.models_root),
                color_revisions_root=str(resolved_color_revisions_root),
                color_revision_overrides=dict(color_revision_overrides or {}),
                include_active_color_revisions=(include_active_color_revisions),
                model_config_overrides=config_overrides,
                include_active_inspection_release=False,
            )
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            if self._temporary_config_root is not None:
                self._temporary_config_root.cleanup()
                self._temporary_config_root = None
            raise

    def close(self) -> None:
        try:
            self._system.shutdown()
        finally:
            if self._temporary_config_root is not None:
                self._temporary_config_root.cleanup()
                self._temporary_config_root = None

    def infer(
        self,
        record: AcceptanceRecord,
        image_path: str | Path,
        *,
        inference_type: str,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> AcceptanceInferenceOutcome:
        identity = self._model_identity or load_model_identity(
            self.models_root,
            record.product,
            record.area,
            inference_type,
        )
        try:
            frame, result = self.detect_raw(
                record,
                image_path,
                inference_type=inference_type,
                cancel_cb=cancel_cb,
            )
            machine_status = _machine_status(result)
            reasons = machine_reason_codes(result)
            color_status, color_details = color_evidence(result)
            error = result.error or ""
            annotated = render_detection_frame(frame, result)
            latency_ms = result.latency * 1000.0
        except (OSError, RuntimeError, ValueError) as exc:
            machine_status = "ERROR"
            reasons = ()
            color_status = "ERROR"
            color_details = ()
            error = str(exc)
            annotated = None
            latency_ms = 0.0
        return AcceptanceInferenceOutcome(
            sample_id=record.sample_id,
            machine_status=machine_status,
            machine_reasons=reasons,
            model_version=identity.version,
            model_sha256=identity.sha256,
            runtime_config_sha256=identity.runtime_config_sha256,
            color_model_sha256=identity.color_model_sha256,
            inference_at=_now_iso(),
            latency_ms=latency_ms,
            error=error,
            color_check_status=color_status,
            color_details_json=json.dumps(
                color_details,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            annotated_frame=annotated,
        )

    def detect_raw(
        self,
        record: AcceptanceRecord,
        image_path: str | Path,
        *,
        inference_type: str,
        cancel_cb: Callable[[], bool] | None = None,
    ) -> tuple[np.ndarray, DetectionResult]:
        """Return the source frame and production-equivalent raw detections.

        The method is read-only and deliberately does not persist inference
        results.  Baseline rebuilding uses the bounding boxes to obtain
        component crops without modifying human acceptance labels.
        """
        frame = read_image(Path(image_path))
        result = self._system.detect(
            record.product,
            record.area,
            inference_type,
            frame=frame,
            cancel_cb=cancel_cb,
            persist=False,
        )
        return frame, result


def calculate_acceptance_metrics(
    records: Sequence[AcceptanceRecord],
) -> AcceptanceMetrics:
    tp = fp = fn = tn = errors = malformed = confirmed = 0
    inferred = sum(bool(record.machine_status) for record in records)
    for record in records:
        if record.review_status != "confirmed":
            continue
        if record.expected_verdict not in {"OK", "NG"}:
            # Counted, never dropped and never raised from here. This helper is
            # on the display and reporting path, so a corrupt row must not stop
            # a manifest from being read; treating it as absent would instead
            # report a clean sheet for a manifest that is not clean. Rejection
            # belongs to the two decision points: the acceptance gate and a
            # formal snapshot.
            malformed += 1
            continue
        confirmed += 1
        actual_ng = record.expected_verdict == "NG"
        if record.machine_status == "ERROR":
            errors += 1
            continue
        if record.machine_status not in {"OK", "NG"}:
            continue
        predicted_ng = record.machine_status == "NG"
        if actual_ng and predicted_ng:
            tp += 1
        elif not actual_ng and predicted_ng:
            fp += 1
        elif actual_ng and not predicted_ng:
            fn += 1
        else:
            tn += 1
    return AcceptanceMetrics(
        confirmed=confirmed,
        pending=len(records) - confirmed - malformed,
        inferred=inferred,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        errors=errors,
        malformed=malformed,
    )


def _inference_changes(outcome: AcceptanceInferenceOutcome) -> dict[str, str]:
    return {
        "machine_status": outcome.machine_status,
        "machine_reasons": "|".join(outcome.machine_reasons),
        "model_version": outcome.model_version,
        "model_sha256": outcome.model_sha256,
        "runtime_config_sha256": outcome.runtime_config_sha256,
        "color_model_sha256": outcome.color_model_sha256,
        "inference_at": outcome.inference_at,
        "latency_ms": f"{outcome.latency_ms:.3f}",
        "error": outcome.error,
        "color_check_status": outcome.color_check_status,
        "color_details_json": outcome.color_details_json,
    }


def _committed_record(
    record: AcceptanceRecord,
    by_id: Mapping[str, AcceptanceInferenceOutcome],
    metadata: Mapping[str, str],
    bundle_sha256: str,
) -> AcceptanceRecord:
    """Update this run's samples, keep same-bundle evidence, clear the rest.

    A formal snapshot needs every result to come from one identical artifact
    combination -- not from one invocation.  Clearing by bundle rather than by
    run is what lets an operator re-run the three samples that failed on a
    transient I/O error without discarding the other 197, which were produced
    by the very same pinned bundle and are therefore still comparable.

    A record whose bundle is empty carries no artifact evidence at all, so it
    can never match and is always cleared.
    """

    if record.sample_id in by_id:
        return record.with_changes(
            **_inference_changes(by_id[record.sample_id]),
            **metadata,
        )
    if record.artifact_bundle_sha256 == bundle_sha256:
        return record
    return record.with_changes(**_cleared_inference_changes())


def _cleared_inference_changes() -> dict[str, str]:
    """Remove stale machine evidence while preserving human ground truth."""

    return dict.fromkeys(
        (
            "machine_status",
            "machine_reasons",
            "model_version",
            "model_sha256",
            "runtime_config_sha256",
            "color_model_sha256",
            "acceptance_run_id",
            "artifact_bundle_sha256",
            "color_revision_contract_sha256",
            "color_revision_overrides_json",
            "include_active_color_revisions",
            "inference_at",
            "latency_ms",
            "error",
            "color_check_status",
            "color_details_json",
        ),
        "",
    )


def load_model_identity(
    models_root: str | Path,
    product: str,
    area: str,
    inference_type: str,
) -> ModelIdentity:
    root = Path(models_root).expanduser().resolve()
    selected_type = "yolo" if inference_type.lower() == "fusion" else inference_type
    station_root = root / product / area / selected_type
    manifest = station_root / "deployment_manifest.yaml"
    config_path = station_root / "config.yaml"
    config_sha256 = _sha256_file(config_path) if config_path.is_file() else ""
    if not manifest.is_file():
        return ModelIdentity(
            version="unknown",
            sha256="",
            runtime_config_sha256=config_sha256,
        )
    try:
        payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AcceptanceDataError(f"Unable to read deployment manifest: {manifest}") from exc
    if not isinstance(payload, Mapping):
        raise AcceptanceDataError(f"Deployment manifest must be a mapping: {manifest}")
    return ModelIdentity(
        version=str(payload.get("deployed_version") or "unknown").strip(),
        sha256=str(payload.get("weight_sha256") or "").strip().lower(),
        runtime_config_sha256=config_sha256,
        color_model_sha256=str(payload.get("color_model_sha256") or "").strip().lower(),
    )


def machine_reason_codes(result: DetectionResult) -> tuple[str, ...]:
    reasons: set[str] = set()
    if result.missing_items:
        reasons.add("MISSING")
    if result.over_items:
        reasons.add("COUNT_MISMATCH")
    if result.unexpected_items:
        reasons.add("UNEXPECTED_COMPONENT")
    if result.color_check and not bool(result.color_check.get("is_ok", True)):
        reasons.add("COLOR_MISMATCH")
    if result.sequence_check and not bool(result.sequence_check.get("is_ok", True)):
        reasons.add("SEQUENCE_MISMATCH")
    if any(str(item.metadata.get("position_status") or "").upper() == "WRONG" for item in result.items):
        reasons.add("POSITION_SHIFT")
    decision = result.metadata.get("decision")
    if isinstance(decision, Mapping):
        raw_reasons = decision.get("reasons")
        if isinstance(raw_reasons, Sequence) and not isinstance(raw_reasons, str):
            for value in raw_reasons:
                normalized = str(value).strip().upper()
                if normalized in ACCEPTANCE_REASON_CODES:
                    reasons.add(normalized)
    if _machine_status(result) == "NG" and not reasons:
        reasons.add("OTHER")
    return tuple(sorted(reasons))


def color_evidence(
    result: DetectionResult,
) -> tuple[str, tuple[dict[str, Any], ...]]:
    """Normalize per-item color evidence without trusting backend-specific extras."""
    color_check = result.color_check
    if not isinstance(color_check, Mapping):
        return "NOT_RUN", ()
    raw_items = color_check.get("items")
    if not isinstance(raw_items, Sequence) or isinstance(raw_items, str):
        raw_items = ()
    items: list[dict[str, Any]] = []
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            continue
        try:
            index = int(raw_item.get("index", position))
            diff = float(raw_item.get("diff"))
            threshold = float(raw_item.get("threshold"))
        except (TypeError, ValueError):
            continue
        items.append(
            {
                "index": index,
                "detector_class": str(raw_item.get("class_name") or raw_item.get("class") or "").strip(),
                "predicted_color": str(raw_item.get("best_color") or "").strip(),
                "diff": diff,
                "threshold": threshold,
                "is_ok": bool(raw_item.get("is_ok", False)),
            }
        )
    error = str(color_check.get("error") or "").strip()
    if error:
        return "ERROR", tuple(items)
    return (
        "PASS" if bool(color_check.get("is_ok", False)) else "FAIL",
        tuple(items),
    )


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise AcceptanceDataError(f"Acceptance image not found: {path}")
    encoded = np.fromfile(str(path), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise AcceptanceDataError(f"Unable to decode acceptance image: {path}")
    return frame


def render_detection_frame(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    if result.result_frame is not None and result.result_frame.size:
        return result.result_frame.copy()
    annotated = frame.copy()
    for item in result.items:
        x1, y1, x2, y2 = (int(value) for value in item.bbox_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (36, 196, 68), 2)
        caption = f"{item.label} {item.confidence:.2f}"
        cv2.putText(
            annotated,
            caption,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (36, 196, 68),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _machine_status(result: DetectionResult) -> str:
    if result.status == "PASS":
        return "OK"
    if result.status in {"FAIL", "DETECTION_FAIL"}:
        return "NG"
    return "ERROR"


def _normalize_reasons(reasons: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(reason).strip().upper() for reason in reasons if str(reason).strip()}))
    unsupported = set(normalized) - set(ACCEPTANCE_REASON_CODES)
    if unsupported:
        raise AcceptanceDataError("Unsupported acceptance reason(s): " + ", ".join(sorted(unsupported)))
    return normalized


def _required_segment(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
        raise AcceptanceDataError(f"Invalid {label}: {value!r}")
    return normalized


def _is_completed_acceptance_run(root: Path, run_id: str) -> bool:
    try:
        normalized_run_id = _required_segment(run_id, "acceptance run ID")
        state_path = root / "runs" / normalized_run_id / "state.json"
        if state_path.is_symlink() or not state_path.is_file():
            return False
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (AcceptanceDataError, OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and payload.get("state") == "COMPLETED"


def _unique_sample_id(digest: str, records: Sequence[AcceptanceRecord]) -> str:
    used = {record.sample_id for record in records}
    for length in range(12, len(digest) + 1, 2):
        candidate = f"ACC-{digest[:length].upper()}"
        if candidate not in used:
            return candidate
    raise AcceptanceDataError("Unable to allocate a unique sample ID.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest_file(path: Path, records: Sequence[AcceptanceRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _read_manifest_file(path: Path) -> tuple[AcceptanceRecord, ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return tuple(AcceptanceRecord.from_mapping(row) for row in reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AcceptanceDataError(f"Unable to read acceptance snapshot manifest: {path}") from exc


def load_acceptance_manifest(path: str | Path) -> tuple[AcceptanceRecord, ...]:
    """Load a manifest without creating or mutating an acceptance repository."""
    return _read_manifest_file(Path(path).expanduser().resolve())


def _safe_label(value: str) -> str:
    normalized = "".join(
        character for character in value.strip() if character.isalnum() or character in {"-", "_", "."}
    )
    return normalized[:80]


def _machine_decision_correct(record: AcceptanceRecord) -> bool:
    return (
        record.review_status == "confirmed"
        and record.expected_verdict in {"OK", "NG"}
        and record.expected_verdict == record.machine_status
    )


def _single_version(records: Sequence[AcceptanceRecord]) -> str:
    versions = sorted({record.model_version for record in records if record.model_version})
    return versions[0] if len(versions) == 1 else ", ".join(versions) or "unknown"


def _metrics_mapping(metrics: AcceptanceMetrics) -> dict[str, Any]:
    return {
        "confirmed": metrics.confirmed,
        "pending": metrics.pending,
        "inferred": metrics.inferred,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "tn": metrics.tn,
        "errors": metrics.errors,
        "malformed": metrics.malformed,
        "true_yield": metrics.true_yield,
        "machine_yield": metrics.machine_yield,
        "escape_rate": metrics.escape_rate,
        "overkill_rate": metrics.overkill_rate,
    }


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator
