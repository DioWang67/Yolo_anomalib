"""Report the report_only versus suppress verdict difference for a pinned evidence window.

This produces the A/B report that section 10 of
``docs/pilot/CROSS_CLASS_DUPLICATE_DETECTION_PROPOSAL.md`` lists as an unchecked
blocking gate, and the pilot-progress count that section 15 gates on. Both are
read-only: no result file, config or database is modified.

Two ledgers, never merged
-------------------------
*Gate progress* counts inspections the line actually ran in ``report_only``.
*A/B evidence* compares verdicts under both modes for every comparable snapshot.
A snapshot recorded under ``suppress`` is legitimate A/B material but cannot
count toward a report-only gate, so mixing the two would inflate the pilot count
with runs that were never report-only.

What this report cannot prove
-----------------------------
Snapshots record the config hash and model version but not the code revision
that produced them, so "this evidence belongs to the post-fix code" is an
operator declaration recorded here, never a fact derived from the data. The
2026-08-05 regression went unnoticed for fourteen days precisely because config
looked correct while code behaviour had changed, so ``--code-revision`` is
required for any written report and appears in it as a declaration.

Full-shift coverage is likewise not computed. Shift boundaries are not in the
snapshots, so the report states the observed span and distinct dates and leaves
the shift judgement to the named approver rather than inventing a definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The audit tool's selection, policy-loading and safe-output helpers are shared
# deliberately rather than reimplemented: one implementation of "which snapshots
# are in scope" is what keeps the two reports from producing numbers nobody can
# reconcile.
from tools.audit_cross_class_duplicates import (  # noqa: E402
    AuditFilters,
    _load_filters,
    _snapshot_filter_reason,
    _update_inventory_digest,
    _validate_output_destination,
    _verify_policy_source,
    _write_json_atomic,
)
from tools.audit_cross_class_duplicates import _load_policy as _load_audit_policy  # noqa: E402
from tools.duplicate_filter_ab import (  # noqa: E402
    AB_POLICY_VERSION,
    MODE_REPORT_ONLY,
    TRANSITION_FAIL_TO_PASS,
    TRANSITION_PASS_TO_FAIL,
    TRANSITION_REASONS_ONLY,
    TRANSITION_UNCHANGED,
    ReplayError,
    compare_snapshot,
)

AB_REPORT_SCHEMA_VERSION = 1

#: Section 15 pilot threshold: at least one full shift or 500 supervised
#: inspections, whichever completes later.
DEFAULT_REQUIRED_INSPECTIONS = 500


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare report_only and suppress verdicts across saved snapshots "
            "and report pilot gate progress. Nothing is modified."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="One snapshot JSON file or a directory containing result snapshots.",
    )
    parser.add_argument("--product", help="Exact snapshot product filter.")
    parser.add_argument("--area", help="Exact snapshot area filter.")
    parser.add_argument(
        "--start-time",
        help="Inclusive local/ISO-8601 lower bound of the evidence window.",
    )
    parser.add_argument(
        "--end-time",
        help="Inclusive local/ISO-8601 upper bound of the evidence window.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Load the duplicate-filter policy and runtime configuration from a "
            "model config instead of each snapshot's embedded config."
        ),
    )
    parser.add_argument(
        "--code-revision",
        help=(
            "The code revision this evidence window belongs to. Required to "
            "write a report: snapshots cannot prove it, so it is recorded as an "
            "operator declaration."
        ),
    )
    parser.add_argument(
        "--required-inspections",
        type=int,
        default=DEFAULT_REQUIRED_INSPECTIONS,
        help=(
            "Report-only inspections the section 15 gate requires "
            f"(default {DEFAULT_REQUIRED_INSPECTIONS})."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print a machine-readable JSON report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Atomically write a schema-versioned JSON report.",
    )
    # Thresholds exist only so _load_policy can build a policy without a config.
    parser.add_argument("--iou", type=float, default=0.90)
    parser.add_argument("--center-ratio", type=float, default=0.10)
    parser.add_argument("--area-similarity", type=float, default=0.80)
    return parser


def compare_path(
    path: Path,
    *,
    filter_options: dict[str, Any],
    filters: AuditFilters | None = None,
    replay_config: dict[str, Any] | None = None,
    replay_config_source: str = "snapshot.config",
    required_inspections: int = DEFAULT_REQUIRED_INSPECTIONS,
) -> dict[str, Any]:
    """Replay every in-scope snapshot under both modes and assemble the report."""
    active_filters = filters or AuditFilters()
    files = [path] if path.is_file() else sorted(path.rglob("*_config_snapshot.json"))
    source_root = path.parent if path.is_file() else path

    comparisons: list[dict[str, Any]] = []
    fidelity_mismatches: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    filter_reason_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    recorded_mode_counts: Counter[str] = Counter()
    reason_drift_count = 0
    selected_snapshot_count = 0
    observed_timestamps: list[str] = []
    inventory_digest = hashlib.sha256()

    for snapshot_path in files:
        relative_path = snapshot_path.relative_to(source_root).as_posix()
        snapshot_sha256 = ""
        inventory_recorded = False
        try:
            if snapshot_path.is_symlink():
                raise ValueError("snapshot cannot be a symbolic link")
            raw_snapshot = snapshot_path.read_bytes()
            snapshot_sha256 = hashlib.sha256(raw_snapshot).hexdigest()
            payload = json.loads(raw_snapshot.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("snapshot root must be a JSON object")

            filter_reason = _snapshot_filter_reason(payload, active_filters)
            if filter_reason is not None:
                filter_reason_counts[filter_reason] += 1
                _update_inventory_digest(
                    inventory_digest,
                    relative_path=relative_path,
                    snapshot_sha256=snapshot_sha256,
                    filter_outcome="filtered:" + filter_reason,
                )
                inventory_recorded = True
                continue
            selected_snapshot_count += 1
            _update_inventory_digest(
                inventory_digest,
                relative_path=relative_path,
                snapshot_sha256=snapshot_sha256,
                filter_outcome="selected",
            )
            inventory_recorded = True

            config_payload = (
                replay_config if replay_config is not None else payload.get("config")
            )
            if not isinstance(config_payload, dict):
                raise ReplayError(
                    replay_config_source + " is missing or not a JSON mapping",
                    code="replay_config_unavailable",
                )
            comparison = compare_snapshot(
                payload,
                filter_options=filter_options,
                config_payload=config_payload,
            )
        except ReplayError as exc:
            if not inventory_recorded:
                _update_inventory_digest(
                    inventory_digest,
                    relative_path=relative_path,
                    snapshot_sha256=snapshot_sha256,
                    filter_outcome="error:" + exc.code,
                )
            # A snapshot that cannot be replayed is excluded and named, never
            # dropped: an unexplained gap between scanned and compared counts
            # reads as coverage that was never achieved.
            excluded_reason_counts[exc.code] += 1
            excluded.append(
                {
                    "path": str(snapshot_path),
                    "sha256": snapshot_sha256,
                    "reason": exc.code,
                    "detail": str(exc),
                }
            )
            continue
        except (OSError, TypeError, ValueError) as exc:
            if not inventory_recorded:
                _update_inventory_digest(
                    inventory_digest,
                    relative_path=relative_path,
                    snapshot_sha256=snapshot_sha256,
                    filter_outcome="error:snapshot_error",
                )
            errors.append({"path": str(snapshot_path), "error": str(exc)})
            continue

        recorded_mode_counts[comparison.fidelity.recorded_mode] += 1
        timestamp = str(payload.get("timestamp") or "")
        if timestamp:
            observed_timestamps.append(timestamp)

        record = {
            "path": str(snapshot_path),
            "sha256": snapshot_sha256,
            "inspection_id": str(payload.get("inspection_id") or ""),
            "timestamp": timestamp,
            "product": str(payload.get("product") or ""),
            "area": str(payload.get("area") or ""),
            "model_version": str(
                (payload.get("model_info") or {}).get("model_version") or ""
            ),
            "config_hash": str(payload.get("config_hash") or ""),
            **comparison.to_dict(),
        }

        if not comparison.is_usable:
            fidelity_mismatches.append(record)
            continue
        if not comparison.fidelity.reasons_reproduced:
            reason_drift_count += 1
        transition_counts[comparison.transition] += 1
        comparisons.append(record)

    report_only_count = int(recorded_mode_counts.get(MODE_REPORT_ONLY, 0))
    return {
        "policy_version": AB_POLICY_VERSION,
        "source": {
            "path": str(path),
            "snapshot_inventory_sha256": inventory_digest.hexdigest(),
            "snapshot_inventory_scope": (
                "all_discovered_relative_path_raw_sha256_filter_outcome"
            ),
        },
        "filters": active_filters.to_dict(),
        "replay_config_source": replay_config_source,
        "filter_options": dict(filter_options),
        "scanned_count": len(files),
        "selected_snapshot_count": selected_snapshot_count,
        "filter_skipped_snapshot_count": sum(filter_reason_counts.values()),
        "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
        "recorded_mode_counts": dict(sorted(recorded_mode_counts.items())),
        "gate": _build_gate(
            report_only_count=report_only_count,
            required_inspections=required_inspections,
            observed_timestamps=observed_timestamps,
        ),
        "ab": _build_ab_section(comparisons, transition_counts),
        "fidelity": {
            "status_reproduced_count": len(comparisons),
            "status_mismatch_count": len(fidelity_mismatches),
            "reason_drift_count": reason_drift_count,
            "mismatches": fidelity_mismatches,
        },
        "excluded_snapshot_count": len(excluded),
        "excluded_reason_counts": dict(sorted(excluded_reason_counts.items())),
        "excluded_snapshots": excluded,
        "errors": errors,
    }


def _build_ab_section(
    comparisons: list[dict[str, Any]], transition_counts: Counter[str]
) -> dict[str, Any]:
    """Assemble the A/B ledger, stating what its zeroes do and do not prove.

    ``fail_to_pass=0`` alone is ambiguous: it can mean suppression introduced no
    escape, or that no board in the window exercised the decision at all. Both
    read identically to an approver, so a window containing no verdict-changing
    board says so explicitly rather than presenting an untested policy as a
    clean sheet.
    """
    counts = {
        key: int(transition_counts.get(key, 0))
        for key in (
            TRANSITION_UNCHANGED,
            TRANSITION_REASONS_ONLY,
            TRANSITION_FAIL_TO_PASS,
            TRANSITION_PASS_TO_FAIL,
        )
    }
    verdict_changing = counts[TRANSITION_FAIL_TO_PASS] + counts[TRANSITION_PASS_TO_FAIL]
    candidate_count = sum(
        1 for record in comparisons if record["suppress"]["suppressed_count"] > 0
    )
    notes: list[str] = []
    if candidate_count == 0:
        notes.append(
            "no snapshot in this window produced a suppression candidate, so the "
            "policy is unexercised here and the transition counts prove nothing "
            "about it"
        )
    elif verdict_changing == 0:
        notes.append(
            f"{candidate_count} snapshots produced suppression candidates but none "
            "changed a board verdict: every candidate board failed for an "
            "additional independent reason. This window shows no new escape and "
            "no new false reject, and equally cannot demonstrate the benefit"
        )
    return {
        "comparable_snapshot_count": len(comparisons),
        "suppression_candidate_snapshot_count": candidate_count,
        "verdict_changing_snapshot_count": verdict_changing,
        "transition_counts": counts,
        "evidence_notes": notes,
        "comparisons": comparisons,
    }


def _build_gate(
    *,
    report_only_count: int,
    required_inspections: int,
    observed_timestamps: list[str],
) -> dict[str, Any]:
    """Assemble section 15 gate progress.

    ``full_shift_confirmed`` is deliberately null: shift boundaries are not
    recorded in snapshots, and a computed guess would be indistinguishable in the
    report from an attested fact. Because the gate takes whichever of the two
    conditions completes later, an unconfirmed shift keeps the gate unsatisfied.
    """
    ordered = sorted(observed_timestamps)
    distinct_dates = sorted({value[:10] for value in ordered if len(value) >= 10})
    count_satisfied = report_only_count >= required_inspections
    return {
        "required_inspections": required_inspections,
        "report_only_inspection_count": report_only_count,
        "remaining_inspections": max(0, required_inspections - report_only_count),
        "inspection_count_satisfied": count_satisfied,
        "report_only_evidence_available": report_only_count > 0,
        "full_shift_confirmed": None,
        "full_shift_evidence": (
            "not computable from snapshots; requires a named shift attestation"
        ),
        "observed_first_timestamp": ordered[0] if ordered else "",
        "observed_last_timestamp": ordered[-1] if ordered else "",
        "observed_distinct_dates": distinct_dates,
        "gate_satisfied": False,
        "gate_blockers": _gate_blockers(
            count_satisfied=count_satisfied,
            report_only_count=report_only_count,
            required_inspections=required_inspections,
        ),
    }


def _gate_blockers(
    *, count_satisfied: bool, report_only_count: int, required_inspections: int
) -> list[str]:
    blockers: list[str] = []
    if report_only_count == 0:
        blockers.append(
            "no report_only inspections in scope: the pilot has not produced any "
            "report-only evidence yet"
        )
    elif not count_satisfied:
        blockers.append(
            f"report_only inspections {report_only_count} below required "
            f"{required_inspections}"
        )
    blockers.append("full shift coverage requires a named attestation")
    blockers.append(
        "section 15 named approvals (process, AI, software, wire order) are not "
        "recorded in this report"
    )
    return blockers


def _versioned_report(report: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": AB_REPORT_SCHEMA_VERSION, **report}


def _print_summary(report: dict[str, Any]) -> None:
    gate = report["gate"]
    ab = report["ab"]
    fidelity = report["fidelity"]
    transitions = ab["transition_counts"]
    print(
        "Scanned={scanned_count}, selected={selected_snapshot_count}, "
        "comparable={comparable}".format(
            comparable=ab["comparable_snapshot_count"], **report
        )
    )
    print(
        "Recorded modes: "
        + (
            ", ".join(
                f"{mode}={count}"
                for mode, count in report["recorded_mode_counts"].items()
            )
            or "none"
        )
    )
    print(
        "Gate: report_only={count}/{required} (remaining={remaining}), "
        "full_shift=UNCONFIRMED, satisfied={satisfied}".format(
            count=gate["report_only_inspection_count"],
            required=gate["required_inspections"],
            remaining=gate["remaining_inspections"],
            satisfied=gate["gate_satisfied"],
        )
    )
    for blocker in gate["gate_blockers"]:
        print("  BLOCKER | " + blocker)
    print(
        f"A/B transitions: unchanged={transitions[TRANSITION_UNCHANGED]}, reasons_only={transitions[TRANSITION_REASONS_ONLY]}, "
        f"fail_to_pass={transitions[TRANSITION_FAIL_TO_PASS]}, pass_to_fail={transitions[TRANSITION_PASS_TO_FAIL]}"
    )
    print(
        "A/B scope: candidates={candidates}, verdict_changing={changing}".format(
            candidates=ab["suppression_candidate_snapshot_count"],
            changing=ab["verdict_changing_snapshot_count"],
        )
    )
    for note in ab["evidence_notes"]:
        print("  NOTE | " + note)
    print(
        "Fidelity: reproduced={ok}, status_mismatch={bad}, "
        "reason_drift={drift}".format(
            ok=fidelity["status_reproduced_count"],
            bad=fidelity["status_mismatch_count"],
            drift=fidelity["reason_drift_count"],
        )
    )
    # Every verdict-changing board is listed in full. A truncated list here
    # would read as a clean sheet for the boards it left out.
    for record in ab["comparisons"]:
        if record["transition"] in {TRANSITION_UNCHANGED}:
            continue
        print(
            "{transition} | {timestamp} | model={model} | raw={raw} -> eff={eff} | "
            "{before} -> {after}".format(
                transition=record["transition"].upper(),
                timestamp=record["timestamp"],
                model=record["model_version"],
                raw=record["report_only"]["raw_count"],
                eff=record["suppress"]["effective_count"],
                before=",".join(record["report_only"]["fail_reasons"]) or "PASS",
                after=",".join(record["suppress"]["fail_reasons"]) or "PASS",
            )
        )
        print("  " + record["path"])
    for record in fidelity["mismatches"]:
        print(
            "FIDELITY | {timestamp} | recorded={recorded} ({mode}) != "
            "replayed={replayed} | {path}".format(
                timestamp=record["timestamp"],
                recorded=record["fidelity"]["recorded_status"],
                mode=record["fidelity"]["recorded_mode"],
                replayed=record["fidelity"]["replayed_status"],
                path=record["path"],
            )
        )
    for reason, count in report["excluded_reason_counts"].items():
        print(f"EXCLUDED | {reason} | {count}")
    for error in report["errors"]:
        print("ERROR | {path} | {error}".format(**error))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path.exists():
        print(f"ERROR | Path does not exist: {args.path}", file=sys.stderr)
        return 1
    source_is_directory = args.path.is_dir()
    if not args.path.is_file() and not source_is_directory:
        print(f"ERROR | Path is not a file or directory: {args.path}", file=sys.stderr)
        return 1
    if args.output_json is not None and not str(args.code_revision or "").strip():
        print(
            "ERROR | --code-revision is required to write a report: snapshots "
            "cannot prove which code produced them",
            file=sys.stderr,
        )
        return 1
    if args.required_inspections < 1:
        print("ERROR | --required-inspections must be at least 1", file=sys.stderr)
        return 1

    try:
        source_path = args.path.expanduser().resolve(strict=True)
        output_path = _validate_output_destination(
            output_path=args.output_json,
            source_path=source_path,
            source_is_directory=source_is_directory,
            config_path=args.config,
        )
        filters = _load_filters(args)
        policy, policy_source, replay_config = _load_audit_policy(args)
        replay_config_source = (
            "model_config[{path};sha256={sha}]".format(
                path=policy_source["path"], sha=policy_source["sha256"]
            )
            if policy_source.get("kind") == "model_config"
            else "snapshot.config"
        )
        report = compare_path(
            source_path,
            filter_options=policy.to_dict(),
            filters=filters,
            replay_config=replay_config,
            replay_config_source=replay_config_source,
            required_inspections=args.required_inspections,
        )
        report["policy_source"] = policy_source
        report["evidence_window"] = {
            "declared_code_revision": str(args.code_revision or "").strip(),
            "code_revision_is_operator_declared": True,
            "code_revision_evidence": (
                "snapshots record config_hash and model_version but not the code "
                "revision; this value is declared, not derived"
            ),
            "start_time": filters.to_dict()["start_time"],
            "end_time": filters.to_dict()["end_time"],
        }
        if report["scanned_count"] == 0:
            report["errors"].append(
                {"path": str(source_path), "error": "no snapshot files found"}
            )
        elif report["selected_snapshot_count"] == 0:
            report["errors"].append(
                {
                    "path": str(source_path),
                    "code": "no_snapshots_in_selected_scope",
                    "error": "no snapshots matched the selected evidence window",
                }
            )
        policy_error = _verify_policy_source(policy_source)
        if policy_error:
            report["errors"].append(
                {"path": str(policy_source.get("path") or ""), "error": policy_error}
            )
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR | A/B report could not be completed: {exc}", file=sys.stderr)
        return 1

    if output_path is not None:
        try:
            _write_json_atomic(output_path, _versioned_report(report))
        except (OSError, TypeError, ValueError) as exc:
            print(f"ERROR | A/B report could not be written: {exc}", file=sys.stderr)
            return 1

    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_summary(report)

    # A fidelity mismatch means the report describes code that is not the code
    # in the tree, so it exits non-zero even though every snapshot was read.
    if report["errors"] or report["fidelity"]["status_mismatch_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
