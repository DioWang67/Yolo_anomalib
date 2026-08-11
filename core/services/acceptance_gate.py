"""Headless, fail-closed acceptance gate for candidate model bundles."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from core.services.model_acceptance import (
    AcceptanceDataError,
    AcceptanceInferenceOutcome,
    AcceptanceInferenceService,
    AcceptanceRecord,
    ModelIdentity,
    calculate_acceptance_metrics,
    load_acceptance_manifest,
)


@dataclass(frozen=True)
class AcceptanceGatePolicy:
    """Non-regression limits applied to one frozen acceptance snapshot."""

    min_confirmed: int
    max_false_positives: int
    max_false_negatives: int
    max_regressions: int = 0
    require_all_confirmed: bool = True
    require_no_errors: bool = True
    require_baseline_predictions: bool = True


@dataclass(frozen=True)
class AcceptanceGateResult:
    """Published gate decision and its immutable report location."""

    passed: bool
    failures: tuple[str, ...]
    report_path: Path


InferenceServiceFactory = Callable[..., AcceptanceInferenceService]
ProgressCallback = Callable[[int, int, str], None]


def run_candidate_acceptance(
    *,
    project_root: str | Path,
    models_root: str | Path,
    global_config_path: str | Path,
    color_revisions_root: str | Path | None,
    dataset_root: str | Path,
    snapshot_manifest_path: str | Path,
    report_path: str | Path,
    product: str,
    area: str,
    inference_type: str,
    model_identity: ModelIdentity,
    policy: AcceptanceGatePolicy,
    color_revision_overrides: Mapping[str, str] | None = None,
    include_active_color_revisions: bool = True,
    color_revision_contract: Mapping[str, Any] | None = None,
    color_revision_contract_validator: Callable[[], Sequence[str]] | None = None,
    service_factory: InferenceServiceFactory = AcceptanceInferenceService,
    progress_callback: ProgressCallback | None = None,
) -> AcceptanceGateResult:
    """Run a candidate without changing deployed files or human truth."""
    resolved_dataset_root = Path(dataset_root).expanduser().resolve()
    resolved_snapshot = Path(snapshot_manifest_path).expanduser().resolve()
    resolved_report = Path(report_path).expanduser().resolve()
    snapshot_sha256 = _sha256_file(resolved_snapshot)
    records = load_acceptance_manifest(resolved_snapshot)
    failures = _validate_snapshot(
        records,
        dataset_root=resolved_dataset_root,
        product=product,
        area=area,
        policy=policy,
    )

    candidate_records: list[AcceptanceRecord] = []
    sample_results: list[dict[str, Any]] = []
    if not failures:
        service = service_factory(
            project_root=project_root,
            models_root=models_root,
            global_config_path=global_config_path,
            model_identity=model_identity,
            color_revisions_root=color_revisions_root,
            color_revision_overrides=dict(color_revision_overrides or {}),
            include_active_color_revisions=include_active_color_revisions,
        )
        try:
            total = len(records)
            for index, record in enumerate(records, start=1):
                try:
                    image_path = _verified_image_path(
                        resolved_dataset_root,
                        record,
                    )
                except AcceptanceDataError as exc:
                    failures.append(str(exc))
                    break
                outcome = service.infer(
                    record,
                    image_path,
                    inference_type=inference_type,
                )
                candidate_records.append(_record_with_outcome(record, outcome))
                sample_results.append(_sample_result(record, outcome))
                if progress_callback is not None:
                    progress_callback(index, total, record.sample_id)
        finally:
            service.close()
    if _sha256_file(resolved_snapshot) != snapshot_sha256:
        failures.append("acceptance snapshot changed while inference was running")
    if color_revision_contract_validator is not None:
        try:
            failures.extend(
                str(failure)
                for failure in color_revision_contract_validator()
                if str(failure)
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(f"active color revision contract changed: {exc}")

    baseline_metrics = calculate_acceptance_metrics(records)
    candidate_metrics = calculate_acceptance_metrics(candidate_records)
    comparison = _compare(records, candidate_records)
    failures.extend(
        _evaluate_policy(
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            policy=policy,
        )
    )
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "passed": not failures,
        "failures": failures,
        "target": {
            "product": product,
            "area": area,
            "inference_type": inference_type,
        },
        "candidate": asdict(model_identity),
        "dataset": {
            "root": str(resolved_dataset_root),
            "snapshot_manifest": str(resolved_snapshot),
            "snapshot_manifest_sha256": snapshot_sha256,
            "image_inventory_sha256": _image_inventory_sha256(records),
            "record_count": len(records),
        },
        "policy": asdict(policy),
        "color_revisions": dict(color_revision_contract or {}),
        "baseline_metrics": _metrics_payload(baseline_metrics),
        "metrics": _metrics_payload(candidate_metrics),
        "comparison": comparison,
        "sample_results": sample_results,
    }
    _write_json_atomic(resolved_report, payload)
    return AcceptanceGateResult(
        passed=not failures,
        failures=tuple(failures),
        report_path=resolved_report,
    )


def _validate_snapshot(
    records: Sequence[AcceptanceRecord],
    *,
    dataset_root: Path,
    product: str,
    area: str,
    policy: AcceptanceGatePolicy,
) -> list[str]:
    failures: list[str] = []
    confirmed = sum(record.review_status == "confirmed" for record in records)
    if confirmed < policy.min_confirmed:
        failures.append(
            f"confirmed samples {confirmed} below required {policy.min_confirmed}"
        )
    if policy.require_all_confirmed and confirmed != len(records):
        failures.append(
            f"snapshot contains {len(records) - confirmed} unconfirmed samples"
        )
    mismatched_targets = [
        record.sample_id
        for record in records
        if record.product != product or record.area != area
    ]
    if mismatched_targets:
        failures.append(
            f"snapshot target mismatch for {len(mismatched_targets)} samples"
        )
    if policy.require_baseline_predictions:
        missing_predictions = [
            record.sample_id
            for record in records
            if record.machine_status not in {"OK", "NG"}
        ]
        if missing_predictions:
            failures.append(
                f"baseline prediction missing for {len(missing_predictions)} samples"
            )
    for record in records:
        try:
            _verified_image_path(
                dataset_root,
                record,
                verify_checksum=False,
            )
        except AcceptanceDataError as exc:
            failures.append(str(exc))
            if len(failures) >= 20:
                failures.append("additional snapshot validation failures omitted")
                break
    return failures


def _verified_image_path(
    dataset_root: Path,
    record: AcceptanceRecord,
    *,
    verify_checksum: bool = True,
) -> Path:
    image_path = (dataset_root / record.image_path).resolve()
    if not image_path.is_relative_to(dataset_root):
        raise AcceptanceDataError(
            f"acceptance image escapes dataset root: {record.sample_id}"
        )
    if not image_path.is_file():
        raise AcceptanceDataError(
            f"acceptance image missing: {record.sample_id}"
        )
    if verify_checksum and _sha256_file(image_path) != record.image_sha256.lower():
        raise AcceptanceDataError(
            f"acceptance image checksum mismatch: {record.sample_id}"
        )
    return image_path


def _record_with_outcome(
    record: AcceptanceRecord,
    outcome: AcceptanceInferenceOutcome,
) -> AcceptanceRecord:
    return replace(
        record,
        machine_status=outcome.machine_status,
        machine_reasons="|".join(outcome.machine_reasons),
        model_version=outcome.model_version,
        model_sha256=outcome.model_sha256,
        runtime_config_sha256=outcome.runtime_config_sha256,
        color_model_sha256=outcome.color_model_sha256,
        inference_at=outcome.inference_at,
        latency_ms=f"{outcome.latency_ms:.3f}",
        error=outcome.error,
        color_check_status=outcome.color_check_status,
        color_details_json=outcome.color_details_json,
    )


def _sample_result(
    record: AcceptanceRecord,
    outcome: AcceptanceInferenceOutcome,
) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "expected_verdict": record.expected_verdict,
        "baseline_status": record.machine_status,
        "candidate_status": outcome.machine_status,
        "candidate_reasons": list(outcome.machine_reasons),
        "latency_ms": outcome.latency_ms,
        "error": outcome.error,
        "color_check_status": outcome.color_check_status,
        "color_details": json.loads(outcome.color_details_json or "[]"),
    }


def _compare(
    baseline: Sequence[AcceptanceRecord],
    candidate: Sequence[AcceptanceRecord],
) -> dict[str, Any]:
    baseline_by_id = {record.sample_id: record for record in baseline}
    candidate_by_id = {record.sample_id: record for record in candidate}
    common_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    improved_ids: list[str] = []
    regressed_ids: list[str] = []
    changed_ids: list[str] = []
    for sample_id in common_ids:
        old = baseline_by_id[sample_id]
        new = candidate_by_id[sample_id]
        if (
            old.machine_status != new.machine_status
            or old.machine_reasons != new.machine_reasons
        ):
            changed_ids.append(sample_id)
        old_correct = _is_correct(old)
        new_correct = _is_correct(new)
        if not old_correct and new_correct:
            improved_ids.append(sample_id)
        elif old_correct and not new_correct:
            regressed_ids.append(sample_id)
    return {
        "common_samples": len(common_ids),
        "improved": len(improved_ids),
        "regressed": len(regressed_ids),
        "changed": len(changed_ids),
        "improved_sample_ids": improved_ids,
        "regressed_sample_ids": regressed_ids,
        "changed_sample_ids": changed_ids,
    }


def _evaluate_policy(
    *,
    candidate_metrics: Any,
    comparison: dict[str, Any],
    policy: AcceptanceGatePolicy,
) -> list[str]:
    failures: list[str] = []
    if candidate_metrics.fp > policy.max_false_positives:
        failures.append(
            f"false positives {candidate_metrics.fp} exceed "
            f"{policy.max_false_positives}"
        )
    if candidate_metrics.fn > policy.max_false_negatives:
        failures.append(
            f"false negatives {candidate_metrics.fn} exceed "
            f"{policy.max_false_negatives}"
        )
    if comparison["regressed"] > policy.max_regressions:
        failures.append(
            f"regressed samples {comparison['regressed']} exceed "
            f"{policy.max_regressions}"
        )
    if policy.require_no_errors and candidate_metrics.errors:
        failures.append(f"inference errors: {candidate_metrics.errors}")
    return failures


def _is_correct(record: AcceptanceRecord) -> bool:
    if record.review_status != "confirmed":
        return False
    if record.expected_verdict not in {"OK", "NG"}:
        return False
    return record.machine_status == record.expected_verdict


def _metrics_payload(metrics: Any) -> dict[str, Any]:
    decided = metrics.tp + metrics.fp + metrics.fn + metrics.tn
    accuracy = (
        (metrics.tp + metrics.tn) / decided
        if decided
        else None
    )
    return {
        "confirmed": metrics.confirmed,
        "pending": metrics.pending,
        "inferred": metrics.inferred,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "tn": metrics.tn,
        "errors": metrics.errors,
        "accuracy": accuracy,
        "escape_rate": metrics.escape_rate,
        "overkill_rate": metrics.overkill_rate,
    }


def _image_inventory_sha256(records: Sequence[AcceptanceRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.sample_id):
        digest.update(record.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(record.image_sha256.lower().encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
