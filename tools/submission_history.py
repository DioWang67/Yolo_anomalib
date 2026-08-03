"""Append-only operator submission history and legacy job reconstruction."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.retraining_options import RetrainingOptions
from core.training_batch_version import (
    format_training_batch_version,
    parse_training_batch_semver,
    validate_training_batch_version,
)
from tools.record_visibility import load_hidden_record_ids

SUBMISSION_ACTIONS = frozenset({"direct", "annotation", "color", "portable"})


class DuplicateTrainingBatchVersionError(ValueError):
    """A product/area already owns the requested human-readable version."""


@dataclass(frozen=True)
class SubmissionHistoryRecord:
    """One operator submission that can be opened without re-submitting it."""

    submission_id: str
    submitted_at: datetime | None
    action: str
    product: str
    area: str
    case_count: int
    ready_count: int
    pending_count: int
    color_feedback_count: int
    job_id: str
    source_type: str
    manifest_path: Path | None
    handoff_path: Path | None
    batch_version: str = ""
    training_options: tuple[tuple[str, int | str], ...] = ()


def record_submission_history(
    data_root: str | Path,
    source_manifest: str | Path,
    *,
    action: str,
    product: str,
    area: str,
    case_count: int,
    ready_count: int = 0,
    pending_count: int = 0,
    color_feedback_count: int = 0,
    job_id: str = "",
    handoff_path: str | Path | None = None,
    batch_version: str = "",
    training_options: Mapping[str, Any] | None = None,
) -> SubmissionHistoryRecord:
    """Persist an idempotent immutable copy of one submitted review manifest."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in SUBMISSION_ACTIONS:
        raise ValueError(f"Unsupported submission action: {action}")
    manifest = Path(source_manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Submission manifest not found: {manifest}")
    manifest_bytes = manifest.read_bytes()
    normalized_batch_version = (
        validate_training_batch_version(
            batch_version,
            product=product,
            area=area,
        )
        if str(batch_version or "").strip()
        else ""
    )
    normalized_training_options = (
        RetrainingOptions.from_mapping(training_options).to_dict()
        if training_options is not None
        else {}
    )
    digest = hashlib.sha256()
    digest.update(normalized_action.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(job_id or "").encode("utf-8"))
    digest.update(b"\0")
    if normalized_batch_version:
        digest.update(normalized_batch_version.encode("utf-8"))
        digest.update(b"\0")
    digest.update(manifest_bytes)
    submission_hash = digest.hexdigest()
    submission_id = submission_hash[:24]
    history_dir = (
        Path(data_root).expanduser().resolve()
        / ".operator_handoff"
        / "submissions"
        / submission_id
    )
    record_path = history_dir / "submission.json"
    existing = _read_json(record_path)
    if existing.get("submission_hash") == submission_hash:
        return _record_from_payload(existing, record_path)

    if normalized_batch_version:
        claim_training_batch_version(
            Path(data_root).expanduser().resolve(),
            batch_version=normalized_batch_version,
            product=product,
            area=area,
            submission_hash=submission_hash,
            owner_job_id=str(job_id or ""),
            record_path=record_path,
        )

    history_dir.mkdir(parents=True, exist_ok=True)
    immutable_manifest = history_dir / "manifest.csv"
    _write_bytes_atomic(immutable_manifest, manifest_bytes)
    submitted_at = datetime.now(timezone.utc)
    payload = {
        "schema_version": 1,
        "submission_id": submission_id,
        "submission_hash": submission_hash,
        "submitted_at": submitted_at.isoformat(),
        "action": normalized_action,
        "product": str(product or ""),
        "area": str(area or ""),
        "case_count": max(0, int(case_count)),
        "ready_count": max(0, int(ready_count)),
        "pending_count": max(0, int(pending_count)),
        "color_feedback_count": max(0, int(color_feedback_count)),
        "job_id": str(job_id or ""),
        "batch_version": normalized_batch_version,
        "training_options": normalized_training_options,
        "manifest_path": str(immutable_manifest.resolve()),
        "handoff_path": (
            str(Path(handoff_path).expanduser().resolve()) if handoff_path else ""
        ),
    }
    _write_json_atomic(record_path, payload)
    return _record_from_payload(payload, record_path)


def load_submission_history(data_root: str | Path) -> list[SubmissionHistoryRecord]:
    """Load explicit submissions plus reconstructable legacy jobs and color data."""
    root = Path(data_root).expanduser().resolve()
    records: list[SubmissionHistoryRecord] = []
    hidden_ids = load_hidden_record_ids(root, "submission_history")
    represented_jobs: set[str] = set()
    submissions_root = root / ".operator_handoff" / "submissions"
    if submissions_root.is_dir():
        for record_path in submissions_root.glob("*/submission.json"):
            payload = _read_json(record_path)
            if payload.get("schema_version") != 1:
                continue
            record = _record_from_payload(payload, record_path)
            if record.submission_id in hidden_ids:
                continue
            records.append(record)
            if record.job_id:
                represented_jobs.add(record.job_id)

    jobs_root = root / ".operator_handoff" / "jobs"
    if jobs_root.is_dir():
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            handoff_path = job_dir / "handoff.json"
            handoff = _read_json(handoff_path)
            job_id = str(handoff.get("job_id") or job_dir.name)
            if not handoff or job_id in represented_jobs:
                continue
            target = _single_target(handoff)
            sample_ids = _string_list(target.get("sample_ids"))
            pending_ids = _string_list(target.get("pending_sample_ids"))
            submission_id = f"legacy-job-{job_id}"
            if submission_id in hidden_ids:
                continue
            records.append(
                SubmissionHistoryRecord(
                    submission_id=submission_id,
                    submitted_at=_parse_datetime(handoff.get("created_at")),
                    action="annotation" if pending_ids else "direct",
                    product=str(target.get("product") or ""),
                    area=str(target.get("area") or ""),
                    case_count=len(sample_ids),
                    ready_count=_safe_int(target.get("ready_count")),
                    pending_count=len(pending_ids),
                    color_feedback_count=0,
                    job_id=job_id,
                    source_type="legacy_job",
                    manifest_path=None,
                    handoff_path=handoff_path.resolve(),
                    batch_version=str(handoff.get("batch_version") or ""),
                    training_options=_training_options_tuple(
                        handoff.get("training_options")
                    ),
                )
            )

    represented_color_targets = {
        (record.product, record.area)
        for record in records
        if record.action == "color"
    }
    for feedback_path in root.glob("*/*/color_review/feedback.csv"):
        rows = _read_csv(feedback_path)
        if not rows:
            continue
        product = str(rows[0].get("product") or "")
        area = str(rows[0].get("area") or "")
        if (product, area) in represented_color_targets:
            continue
        color_sample_ids = {
            str(row.get("sample_id") or "") for row in rows if row.get("sample_id")
        }
        submission_id = f"legacy-color-{product}-{area}"
        if submission_id in hidden_ids:
            continue
        records.append(
            SubmissionHistoryRecord(
                submission_id=submission_id,
                submitted_at=datetime.fromtimestamp(
                    feedback_path.stat().st_mtime, tz=timezone.utc
                ),
                action="color",
                product=product,
                area=area,
                case_count=len(color_sample_ids),
                ready_count=0,
                pending_count=0,
                color_feedback_count=len(rows),
                job_id="",
                source_type="color_feedback",
                manifest_path=feedback_path.resolve(),
                handoff_path=None,
                batch_version="",
                training_options=(),
            )
        )

    return sorted(
        records,
        key=lambda item: (
            item.submitted_at.timestamp() if item.submitted_at else 0.0,
            item.submission_id,
        ),
        reverse=True,
    )


def load_submission_entries(
    record: SubmissionHistoryRecord,
) -> list[tuple[int, dict[str, str]]]:
    """Load read-only thumbnail rows for one submission history record."""
    if record.source_type == "submission" and record.manifest_path:
        rows = _read_csv(record.manifest_path)
    elif record.source_type == "legacy_job" and record.handoff_path:
        rows = _load_legacy_job_rows(record)
    elif record.source_type == "color_feedback" and record.manifest_path:
        rows = _load_color_feedback_rows(record.manifest_path)
    else:
        rows = []
    return list(enumerate(rows))


def _load_legacy_job_rows(
    record: SubmissionHistoryRecord,
) -> list[dict[str, str]]:
    handoff_path = record.handoff_path
    if handoff_path is None:
        return []
    handoff = _read_json(handoff_path)
    target = _single_target(handoff)
    sample_ids = set(_string_list(target.get("sample_ids")))
    if not sample_ids:
        return []
    job_dir = handoff_path.parent
    rows: list[dict[str, str]] = []
    for manifest_path in job_dir.glob("dataset/*/*/metadata/review_dataset_manifest.csv"):
        target_root = manifest_path.parents[1]
        for row in _read_csv(manifest_path):
            if str(row.get("sample_id") or "") not in sample_ids:
                continue
            output_image = Path(str(row.get("output_image") or ""))
            immutable_image = target_root / "raw" / "images" / output_image.name
            preview = immutable_image if immutable_image.is_file() else output_image
            normalized = dict(row)
            normalized["annotated_path"] = str(preview)
            normalized["preprocessed_path"] = str(preview)
            normalized["original_path"] = str(row.get("source_image") or preview)
            normalized["training_selected"] = "0"
            rows.append(normalized)
    return sorted(rows, key=lambda row: str(row.get("timestamp") or ""))


def _load_color_feedback_rows(path: Path) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for row in _read_csv(path):
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in unique:
            continue
        normalized = dict(row)
        preview = str(row.get("output_image") or row.get("source_image") or "")
        normalized["annotated_path"] = preview
        normalized["preprocessed_path"] = preview
        normalized["original_path"] = str(row.get("source_image") or preview)
        normalized["training_selected"] = "0"
        unique[sample_id] = normalized
    return sorted(unique.values(), key=lambda row: str(row.get("timestamp") or ""))


def _record_from_payload(
    payload: dict[str, Any], record_path: Path
) -> SubmissionHistoryRecord:
    manifest_text = str(payload.get("manifest_path") or "")
    handoff_text = str(payload.get("handoff_path") or "")
    return SubmissionHistoryRecord(
        submission_id=str(payload.get("submission_id") or record_path.parent.name),
        submitted_at=_parse_datetime(payload.get("submitted_at")),
        action=str(payload.get("action") or "unknown"),
        product=str(payload.get("product") or ""),
        area=str(payload.get("area") or ""),
        case_count=_safe_int(payload.get("case_count")),
        ready_count=_safe_int(payload.get("ready_count")),
        pending_count=_safe_int(payload.get("pending_count")),
        color_feedback_count=_safe_int(payload.get("color_feedback_count")),
        job_id=str(payload.get("job_id") or ""),
        source_type="submission",
        manifest_path=Path(manifest_text).resolve() if manifest_text else None,
        handoff_path=Path(handoff_text).resolve() if handoff_text else None,
        batch_version=str(payload.get("batch_version") or ""),
        training_options=_training_options_tuple(payload.get("training_options")),
    )


def suggest_next_training_batch_version(
    data_root: str | Path,
    *,
    product: str,
    area: str,
) -> str:
    """Suggest the next patch version for one product/area submission stream."""
    versions = [
        parsed
        for record in _load_explicit_submission_records(data_root)
        if record.product == product and record.area == area
        if (parsed := parse_training_batch_semver(record.batch_version)) is not None
    ]
    versions.extend(
        parsed
        for claim in _load_training_batch_claims(data_root)
        if str(claim.get("product") or "") == product
        and str(claim.get("area") or "") == area
        if (
            parsed := parse_training_batch_semver(
                str(claim.get("batch_version") or "")
            )
        )
        is not None
    )
    latest = max(versions) if versions else None
    next_version = (
        (0, 0, 1)
        if latest is None
        else (latest[0], latest[1], latest[2] + 1)
    )
    return format_training_batch_version(product, area, next_version)


def ensure_training_batch_version_available(
    data_root: str | Path,
    *,
    product: str,
    area: str,
    batch_version: str,
) -> str:
    """Validate a version and reject reuse before expensive export work starts."""
    normalized = validate_training_batch_version(
        batch_version,
        product=product,
        area=area,
    )
    claim_path = _training_batch_claim_path(
        Path(data_root).expanduser().resolve(),
        product=product,
        area=area,
        batch_version=normalized,
    )
    if claim_path.is_file() or any(
        record.product == product
        and record.area == area
        and record.batch_version.casefold() == normalized.casefold()
        for record in _load_explicit_submission_records(data_root)
    ):
        raise DuplicateTrainingBatchVersionError(
            f"補訓批次版本已存在：{normalized}。請使用下一個版本。"
        )
    return normalized


def _load_explicit_submission_records(
    data_root: str | Path,
) -> list[SubmissionHistoryRecord]:
    """Load immutable submissions, including entries hidden from the UI."""
    submissions_root = (
        Path(data_root).expanduser().resolve()
        / ".operator_handoff"
        / "submissions"
    )
    if not submissions_root.is_dir():
        return []
    records: list[SubmissionHistoryRecord] = []
    for record_path in submissions_root.glob("*/submission.json"):
        payload = _read_json(record_path)
        if payload.get("schema_version") == 1:
            records.append(_record_from_payload(payload, record_path))
    return records


def claim_training_batch_version(
    data_root: str | Path,
    *,
    batch_version: str,
    product: str,
    area: str,
    submission_hash: str,
    owner_job_id: str = "",
    record_path: str | Path | None = None,
) -> None:
    """Atomically reserve one target-scoped version across GUI processes."""
    normalized = validate_training_batch_version(
        batch_version,
        product=product,
        area=area,
    )
    normalized_owner_job_id = str(owner_job_id or "").strip()
    normalized_submission_hash = str(submission_hash or "").strip()
    if not normalized_owner_job_id and not normalized_submission_hash:
        raise ValueError("A batch-version claim requires an owner identity.")
    claim_path = _training_batch_claim_path(
        Path(data_root).expanduser().resolve(),
        product=product,
        area=area,
        batch_version=normalized,
    )
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "batch_version": normalized,
        "product": str(product or ""),
        "area": str(area or ""),
        "submission_hash": normalized_submission_hash,
        "owner_job_id": normalized_owner_job_id,
        "record_path": (
            str(Path(record_path).expanduser().resolve()) if record_path else ""
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | getattr(os, "O_BINARY", 0),
        )
    except FileExistsError:
        existing = _read_json(claim_path)
        same_job = normalized_owner_job_id and (
            str(existing.get("owner_job_id") or "") == normalized_owner_job_id
        )
        same_submission = not normalized_owner_job_id and (
            str(existing.get("submission_hash") or "")
            == normalized_submission_hash
        )
        if same_job or same_submission:
            return
        raise DuplicateTrainingBatchVersionError(
            f"補訓批次版本已存在：{normalized}。請使用下一個版本。"
        ) from None
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        claim_path.unlink(missing_ok=True)
        raise


def _training_batch_claim_path(
    data_root: Path,
    *,
    product: str,
    area: str,
    batch_version: str,
) -> Path:
    scope_digest = hashlib.sha256(
        f"{product}\0{area}".casefold().encode("utf-8")
    ).hexdigest()[:24]
    version_digest = hashlib.sha256(
        batch_version.casefold().encode("utf-8")
    ).hexdigest()[:24]
    return (
        data_root
        / ".operator_handoff"
        / "submission_versions"
        / scope_digest
        / f"{version_digest}.json"
    )


def _load_training_batch_claims(data_root: str | Path) -> list[dict[str, Any]]:
    claims_root = (
        Path(data_root).expanduser().resolve()
        / ".operator_handoff"
        / "submission_versions"
    )
    if not claims_root.is_dir():
        return []
    return [
        payload
        for claim_path in claims_root.glob("*/*.json")
        if (payload := _read_json(claim_path)).get("schema_version") == 1
    ]


def _training_options_tuple(
    value: Any,
) -> tuple[tuple[str, int | str], ...]:
    if not isinstance(value, Mapping) or not value:
        return ()
    try:
        normalized = RetrainingOptions.from_mapping(value).to_dict()
    except ValueError:
        return ()
    return tuple(sorted(normalized.items()))


def _single_target(payload: dict[str, Any]) -> dict[str, Any]:
    targets = payload.get("targets")
    if not isinstance(targets, list) or len(targets) != 1:
        return {}
    return dict(targets[0]) if isinstance(targets[0], dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _safe_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone() if parsed.tzinfo else parsed.astimezone()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    _write_bytes_atomic(path, data)
