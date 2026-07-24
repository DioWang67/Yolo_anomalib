#!/usr/bin/env python3
"""Read-only audit for legacy review semantics and derived workflow states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.review_workflow import (
        blocking_violations,
        derive_review_semantics,
        derive_workflow_state,
        record_identity,
        validate_record_consistency,
    )
except ModuleNotFoundError as exc:
    if exc.name != "tools":
        raise
    from review_workflow import (  # type: ignore[no-redef]
        blocking_violations,
        derive_review_semantics,
        derive_workflow_state,
        record_identity,
        validate_record_consistency,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDITED_FIELDS = (
    "review_selected",
    "review_outcome",
    "review_label",
    "failure_category",
    "skip_reason",
    "product_verdict",
    "detection_verdict",
    "color_verdict",
    "action_route",
    "training_selected",
    "submission_status",
    "job_status",
)


def audit_review_workflow(
    manifest_paths: list[str | Path],
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit manifests and optional job/submission artifacts without mutation."""
    scopes: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    state_counts: Counter[str] = Counter()
    violation_counts: Counter[str] = Counter()
    inconsistent_records: list[dict[str, Any]] = []
    seen_manifests: set[Path] = set()

    discovered = [Path(path).expanduser().resolve() for path in manifest_paths]
    root = Path(data_root).expanduser().resolve() if data_root is not None else None
    if root is not None:
        discovered.extend(root.glob("*/*/metadata/review_dataset_manifest.csv"))
        discovered.extend(
            root.glob(".operator_handoff/submissions/*/manifest.csv")
        )
        discovered.extend(
            root.glob(
                ".operator_handoff/jobs/*/dataset/*/*/metadata/"
                "review_dataset_manifest.csv"
            )
        )

    for manifest in sorted({path.resolve() for path in discovered}):
        if manifest in seen_manifests:
            continue
        seen_manifests.add(manifest)
        context = _manifest_context(manifest, root, errors)
        try:
            rows = _read_csv_strict(manifest)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            errors.append({"scope": str(manifest), "error": str(exc)})
            continue
        scope_state_counts: Counter[str] = Counter()
        scope_inconsistent = 0
        for row_index, row in enumerate(rows, start=2):
            record = {**row, **context}
            state = derive_workflow_state(record)
            semantics = derive_review_semantics(record)
            violations = validate_record_consistency(record)
            blocking = blocking_violations(violations)
            state_counts[state.value] += 1
            scope_state_counts[state.value] += 1
            for violation in violations:
                violation_counts[violation.code] += 1
            if violations:
                scope_inconsistent += 1
                inconsistent_records.append(
                    {
                        "scope": str(manifest),
                        "row_number": row_index,
                        "sample_id": record_identity(record),
                        "workflow_state": state.value,
                        "blocking": bool(blocking),
                        "violations": [
                            {
                                "code": violation.code,
                                "message": violation.message,
                                "suggestion": violation.suggestion,
                                "blocking": violation.blocking,
                            }
                            for violation in violations
                        ],
                        "original_fields": {
                            field: str(record.get(field) or "")
                            for field in AUDITED_FIELDS
                        },
                        "derived_semantics": semantics.to_dict(),
                    }
                )
        scopes.append(
            {
                "type": "manifest",
                "path": str(manifest),
                "record_count": len(rows),
                "workflow_state_counts": dict(sorted(scope_state_counts.items())),
                "inconsistent_record_count": scope_inconsistent,
            }
        )

    if root is not None:
        _audit_job_statuses(
            root,
            scopes=scopes,
            errors=errors,
            state_counts=state_counts,
            violation_counts=violation_counts,
            inconsistent_records=inconsistent_records,
        )

    blocking_count = sum(
        1 for record in inconsistent_records if record["blocking"]
    )
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dry_run_only",
        "mutation_performed": False,
        "summary": {
            "scopes_checked": len(scopes),
            "records_checked": sum(
                int(scope.get("record_count", 0)) for scope in scopes
            ),
            "blocking_inconsistency_count": blocking_count,
            "diagnostic_record_count": len(inconsistent_records),
            "error_count": len(errors),
        },
        "workflow_state_counts": dict(sorted(state_counts.items())),
        "violation_type_counts": dict(sorted(violation_counts.items())),
        "scopes": scopes,
        "inconsistent_records": inconsistent_records,
        "errors": errors,
    }


def _manifest_context(
    manifest: Path,
    data_root: Path | None,
    errors: list[dict[str, str]],
) -> dict[str, str]:
    parts = {part.lower() for part in manifest.parts}
    context: dict[str, str] = {}
    if "submissions" in parts:
        context["submission_status"] = "submitted"
        context["submission_id"] = manifest.parent.name
        submission = _read_json(manifest.parent / "submission.json", errors)
        job_id = str(submission.get("job_id") or "")
        if job_id and data_root is not None:
            status = _read_json(
                data_root / ".operator_handoff" / "jobs" / job_id / "status.json",
                errors,
            )
            context["job_id"] = job_id
            context["job_status"] = str(status.get("state") or "unknown")
    elif "jobs" in parts:
        try:
            jobs_index = [part.lower() for part in manifest.parts].index("jobs")
            job_id = manifest.parts[jobs_index + 1]
        except (ValueError, IndexError):
            job_id = ""
        if job_id and data_root is not None:
            status = _read_json(
                data_root / ".operator_handoff" / "jobs" / job_id / "status.json",
                errors,
            )
            context.update(
                {
                    "submission_status": "submitted",
                    "job_id": job_id,
                    "job_status": str(status.get("state") or "unknown"),
                }
            )
    return context


def _audit_job_statuses(
    data_root: Path,
    *,
    scopes: list[dict[str, Any]],
    errors: list[dict[str, str]],
    state_counts: Counter[str],
    violation_counts: Counter[str],
    inconsistent_records: list[dict[str, Any]],
) -> None:
    jobs_root = data_root / ".operator_handoff" / "jobs"
    if not jobs_root.is_dir():
        return
    for status_path in sorted(jobs_root.glob("*/status.json")):
        error_count_before = len(errors)
        payload = _read_json(status_path, errors)
        if len(errors) != error_count_before:
            continue
        record = {
            **payload,
            "job_id": str(payload.get("job_id") or status_path.parent.name),
            "job_status": str(payload.get("state") or "unknown"),
        }
        state = derive_workflow_state(record)
        violations = validate_record_consistency(record)
        blocking = blocking_violations(violations)
        state_counts[state.value] += 1
        for violation in violations:
            violation_counts[violation.code] += 1
        if violations:
            inconsistent_records.append(
                {
                    "scope": str(status_path),
                    "row_number": 0,
                    "sample_id": record_identity(record),
                    "workflow_state": state.value,
                    "blocking": bool(blocking),
                    "violations": [
                        {
                            "code": violation.code,
                            "message": violation.message,
                            "suggestion": violation.suggestion,
                            "blocking": violation.blocking,
                        }
                        for violation in violations
                    ],
                    "original_fields": {
                        field: str(record.get(field) or "")
                        for field in AUDITED_FIELDS
                    },
                    "derived_semantics": derive_review_semantics(record).to_dict(),
                }
            )
        scopes.append(
            {
                "type": "job_status",
                "path": str(status_path),
                "record_count": 1,
                "workflow_state_counts": {state.value: 1},
                "inconsistent_record_count": 1 if violations else 0,
            }
        )


def _read_csv_strict(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Review manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        if not reader.fieldnames:
            raise ValueError(f"Review manifest header is missing: {path}")
        return [dict(row) for row in reader]


def _read_json(path: Path, errors: list[dict[str, str]]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append({"scope": str(path), "error": str(exc)})
        return {}
    if not isinstance(payload, dict):
        errors.append({"scope": str(path), "error": "JSON root is not an object"})
        return {}
    return dict(payload)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="Review manifest to audit; may be repeated",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Optional training data root containing submissions and jobs",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifests = list(args.manifest)
    if not manifests:
        manifests = sorted(PROJECT_ROOT.glob("review_manifest*.csv"))
    report = audit_review_workflow(manifests, data_root=args.data_root)
    if args.output is not None:
        _write_json_atomic(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    summary = report["summary"]
    if summary["error_count"]:
        return 1
    return 2 if summary["blocking_inconsistency_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
