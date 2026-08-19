"""Read-only replay audit for cross-class duplicate detections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import resolve_position_check_enabled  # noqa: E402
from core.pipeline.registry import validate_duplicate_filter_order  # noqa: E402
from core.services.cross_class_duplicate_filter import (  # noqa: E402
    DuplicateFilterPolicy,
    analyze_cross_class_duplicates,
    duplicate_filter_color_block_status,
    duplicate_filter_position_block_status,
)
from tools.collect_review_cases import (  # noqa: E402
    normalize_time_bound,
)

AUDIT_REPORT_SCHEMA_VERSION = 1


class SnapshotPositionGuardError(ValueError):
    """Raised when a snapshot cannot pass the runtime position-safety guard."""

    def __init__(self, *, status: str, state: str, evidence: str) -> None:
        self.status = status
        self.state = state
        self.evidence = evidence
        super().__init__(f"{status}: position check state={state}; evidence={evidence}")


class SnapshotFilterError(ValueError):
    """Raised when active filters cannot safely classify a snapshot."""

    def __init__(self, message: str, *, code: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AuditFilters:
    product: str = ""
    area: str = ""
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "product": self.product or None,
            "area": self.area or None,
            "start_time": (self.start_time.isoformat() if self.start_time is not None else None),
            "end_time": (self.end_time.isoformat() if self.end_time is not None else None),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan result snapshots and report boxes that satisfy the conservative "
            "cross-class duplicate policy. No result files are modified."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="One snapshot JSON file or a directory containing result snapshots.",
    )
    parser.add_argument("--iou", type=float, default=0.90)
    parser.add_argument("--center-ratio", type=float, default=0.10)
    parser.add_argument("--area-similarity", type=float, default=0.80)
    parser.add_argument("--product", help="Exact snapshot product filter.")
    parser.add_argument("--area", help="Exact snapshot area filter.")
    parser.add_argument(
        "--start-time",
        help="Inclusive local/ISO-8601 snapshot timestamp lower bound.",
    )
    parser.add_argument(
        "--end-time",
        help="Inclusive local/ISO-8601 snapshot timestamp upper bound.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "Load the exact cross_class_duplicate_filter policy from a model "
            "config. When set, policy threshold flags are ignored."
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
    return parser


def audit_path(
    path: Path,
    policy: DuplicateFilterPolicy,
    *,
    filters: AuditFilters | None = None,
    replay_config: dict[str, Any] | None = None,
    replay_config_source: str = "snapshot.config",
) -> dict[str, Any]:
    active_filters = filters or AuditFilters()
    files = [path] if path.is_file() else sorted(path.rglob("*_config_snapshot.json"))
    records: list[dict[str, Any]] = []
    blocked_records: list[dict[str, Any]] = []
    runtime_skipped_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    filter_reason_counts: Counter[str] = Counter()
    blocked_reason_counts: Counter[str] = Counter()
    runtime_skip_reason_counts: Counter[str] = Counter()
    selected_snapshot_count = 0
    no_detection_snapshot_count = 0
    analyzed_snapshot_count = 0
    source_root = path.parent if path.is_file() else path
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
                    filter_outcome=f"filtered:{filter_reason}",
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
            snapshot_status = str(payload.get("status") or "").upper()
            if snapshot_status == "INFERENCE_ERROR":
                skip_reason = "inference_error_status"
                runtime_skip_reason_counts[skip_reason] += 1
                runtime_skipped_records.append(
                    {
                        "path": str(snapshot_path),
                        "sha256": snapshot_sha256,
                        "inspection_id": str(payload.get("inspection_id") or ""),
                        "timestamp": str(payload.get("timestamp") or ""),
                        "product": str(payload.get("product") or ""),
                        "area": str(payload.get("area") or ""),
                        "status": snapshot_status,
                        "reason": skip_reason,
                    }
                )
                continue
            raw_detections = payload.get("raw_detections")
            if raw_detections is not None and not isinstance(raw_detections, list):
                raise ValueError("snapshot raw_detections must be a list")
            detections = raw_detections if raw_detections is not None else payload.get("detections")
            if detections is None:
                detections = []
            if not isinstance(detections, list) or not all(isinstance(item, dict) for item in detections):
                raise ValueError("snapshot detections must be a list of objects")
            if not detections:
                no_detection_snapshot_count += 1
                continue
            position_state, position_evidence = _snapshot_position_check_state(
                payload,
                replay_config=replay_config,
                replay_config_source=replay_config_source,
            )
            position_block_status = duplicate_filter_position_block_status(
                require_position_disabled=policy.require_position_disabled,
                position_state=position_state,
            )
            if position_block_status is not None:
                if position_state == "unknown":
                    raise SnapshotPositionGuardError(
                        status=position_block_status,
                        state=position_state,
                        evidence=position_evidence,
                    )
                blocked_reason_counts[position_block_status] += 1
                blocked_records.append(
                    {
                        "path": str(snapshot_path),
                        "sha256": snapshot_sha256,
                        "inspection_id": str(payload.get("inspection_id") or ""),
                        "timestamp": str(payload.get("timestamp") or ""),
                        "product": str(payload.get("product") or ""),
                        "area": str(payload.get("area") or ""),
                        "snapshot_duplicate_filter_status": (_snapshot_duplicate_filter_status(payload)),
                        "reason": position_block_status,
                        "position_check_state": position_state,
                        "position_check_evidence": position_evidence,
                    }
                )
                continue
            color_result = payload.get("color_result")
            if color_result is None:
                color_result = {}
            if not isinstance(color_result, dict):
                raise ValueError("snapshot color_result must be a JSON object")
            color_items = color_result.get("items")
            if color_items is None:
                color_items = []
            if not isinstance(color_items, list) or not all(isinstance(item, dict) for item in color_items):
                raise ValueError("snapshot color_result.items must be a list of objects")
            indexed_color_items = {int(item.get("index", position)): item for position, item in enumerate(color_items)}
            if len(indexed_color_items) != len(color_items):
                raise ValueError("snapshot color_result.items contains duplicate indices")
            color_block_status = duplicate_filter_color_block_status(
                has_color_items=bool(indexed_color_items)
            )
            if color_block_status is not None:
                blocked_reason_counts[color_block_status] += 1
                blocked_records.append(
                    {
                        "path": str(snapshot_path),
                        "sha256": snapshot_sha256,
                        "inspection_id": str(payload.get("inspection_id") or ""),
                        "timestamp": str(payload.get("timestamp") or ""),
                        "product": str(payload.get("product") or ""),
                        "area": str(payload.get("area") or ""),
                        "snapshot_duplicate_filter_status": (
                            _snapshot_duplicate_filter_status(payload)
                        ),
                        "reason": color_block_status,
                        "position_check_state": position_state,
                        "position_check_evidence": position_evidence,
                    }
                )
                continue
            model_info = payload.get("model_info")
            if model_info is None:
                model_info = {}
            if not isinstance(model_info, dict):
                raise ValueError("snapshot model_info must be a JSON mapping")
            analysis = analyze_cross_class_duplicates(
                detections=detections,
                color_items_by_index=indexed_color_items,
                policy=policy,
            )
            analyzed_snapshot_count += 1
            if analysis["invalid_indices"]:
                raise ValueError(f"snapshot contains invalid detections at indices {analysis['invalid_indices']}")
        except (OSError, TypeError, ValueError) as exc:
            if not inventory_recorded:
                error_code = exc.code if isinstance(exc, SnapshotFilterError) else "snapshot_error"
                _update_inventory_digest(
                    inventory_digest,
                    relative_path=relative_path,
                    snapshot_sha256=snapshot_sha256,
                    filter_outcome=f"error:{error_code}",
                )
            error = {"path": str(snapshot_path), "error": str(exc)}
            if isinstance(exc, SnapshotPositionGuardError):
                error.update(
                    {
                        "code": exc.status,
                        "position_check_state": exc.state,
                        "position_check_evidence": exc.evidence,
                    }
                )
            elif isinstance(exc, SnapshotFilterError):
                error["code"] = exc.code
            errors.append(error)
            continue
        if analysis["candidate_count"] <= 0:
            continue
        records.append(
            {
                "path": str(snapshot_path),
                "sha256": snapshot_sha256,
                "inspection_id": str(payload.get("inspection_id") or ""),
                "timestamp": str(payload.get("timestamp") or ""),
                "status": str(payload.get("status") or ""),
                "model_version": str(model_info.get("model_version") or ""),
                "snapshot_duplicate_filter_status": (_snapshot_duplicate_filter_status(payload)),
                "position_guard": {
                    "required": policy.require_position_disabled,
                    "state": position_state,
                    "evidence": position_evidence,
                },
                "raw_count": analysis["raw_count"],
                "candidate_count": analysis["candidate_count"],
                "proposed_suppressions": analysis["proposed_suppressions"],
            }
        )
    return {
        "source": {
            "path": str(path),
            "snapshot_inventory_sha256": inventory_digest.hexdigest(),
            "snapshot_inventory_scope": ("all_discovered_relative_path_raw_sha256_filter_outcome"),
        },
        "filters": active_filters.to_dict(),
        "policy": policy.to_dict(),
        "scanned_count": len(files),
        "selected_snapshot_count": selected_snapshot_count,
        "filter_skipped_snapshot_count": sum(filter_reason_counts.values()),
        "filter_reason_counts": dict(sorted(filter_reason_counts.items())),
        "no_detection_snapshot_count": no_detection_snapshot_count,
        "runtime_skipped_snapshot_count": len(runtime_skipped_records),
        "runtime_skip_reason_counts": dict(sorted(runtime_skip_reason_counts.items())),
        "analyzed_snapshot_count": analyzed_snapshot_count,
        "blocked_snapshot_count": len(blocked_records),
        "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
        "matched_snapshot_count": len(records),
        "proposed_suppression_count": sum(len(record["proposed_suppressions"]) for record in records),
        "records": records,
        "blocked_records": blocked_records,
        "runtime_skipped_records": runtime_skipped_records,
        "errors": errors,
    }


def _update_inventory_digest(
    inventory_digest: Any,
    *,
    relative_path: str,
    snapshot_sha256: str,
    filter_outcome: str,
) -> None:
    inventory_digest.update(relative_path.encode("utf-8"))
    inventory_digest.update(b"\0")
    inventory_digest.update((snapshot_sha256 or "RAW_SHA256_UNAVAILABLE").encode("ascii"))
    inventory_digest.update(b"\0")
    inventory_digest.update(filter_outcome.encode("utf-8"))
    inventory_digest.update(b"\n")


def _snapshot_filter_reason(
    payload: dict[str, Any],
    filters: AuditFilters,
) -> str | None:
    if filters.product:
        product = payload.get("product")
        if not isinstance(product, str) or not product.strip():
            raise SnapshotFilterError(
                "snapshot product must be a non-empty string when a product "
                "filter is active",
                code="invalid_snapshot_product",
            )
        if product != filters.product:
            return "product"
    if filters.area:
        area = payload.get("area")
        if not isinstance(area, str) or not area.strip():
            raise SnapshotFilterError(
                "snapshot area must be a non-empty string when an area filter "
                "is active",
                code="invalid_snapshot_area",
            )
        if area != filters.area:
            return "area"
    if filters.start_time is not None or filters.end_time is not None:
        timestamp = payload.get("timestamp")
        if timestamp is None or not str(timestamp).strip():
            raise SnapshotFilterError(
                "snapshot timestamp is required when a time filter is active",
                code="invalid_snapshot_timestamp",
            )
        try:
            observed = normalize_time_bound(
                str(timestamp),
                field_name="snapshot timestamp",
            )
        except ValueError as exc:
            raise SnapshotFilterError(
                str(exc),
                code="invalid_snapshot_timestamp",
            ) from exc
        if observed is None:
            raise SnapshotFilterError(
                "snapshot timestamp is required when a time filter is active",
                code="invalid_snapshot_timestamp",
            )
        if filters.start_time is not None and observed < filters.start_time:
            return "timestamp"
        if filters.end_time is not None and observed > filters.end_time:
            return "timestamp"
    return None


def _snapshot_duplicate_filter_status(payload: dict[str, Any]) -> str:
    duplicate_filter = payload.get("duplicate_filter")
    if not isinstance(duplicate_filter, dict):
        return ""
    return str(duplicate_filter.get("status") or "").strip()


def _snapshot_position_check_state(
    payload: dict[str, Any],
    *,
    replay_config: dict[str, Any] | None,
    replay_config_source: str,
) -> tuple[str, str]:
    product = payload.get("product")
    area = payload.get("area")
    if not isinstance(product, str) or not product.strip():
        return "unknown", "snapshot product is missing or invalid"
    if not isinstance(area, str) or not area.strip():
        return "unknown", "snapshot area is missing or invalid"
    config = replay_config if replay_config is not None else payload.get("config")
    if not isinstance(config, dict):
        return (
            "unknown",
            f"{replay_config_source} is missing or not a JSON mapping",
        )
    position_config = config.get("position_config", {})
    if not isinstance(position_config, dict):
        return "unknown", f"{replay_config_source}.position_config is not a mapping"
    try:
        enabled = resolve_position_check_enabled(position_config, product, area)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return (
            "unknown",
            f"{replay_config_source}.position_config could not resolve product={product}, area={area}: {exc}",
        )
    return (
        "enabled" if enabled else "disabled",
        f"{replay_config_source}.position_config resolved product={product}, area={area}, enabled={enabled}",
    )


def _versioned_report(report: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": AUDIT_REPORT_SCHEMA_VERSION, **report}


def _load_filters(args: argparse.Namespace) -> AuditFilters:
    start_time = normalize_time_bound(args.start_time, field_name="start_time")
    end_time = normalize_time_bound(args.end_time, field_name="end_time")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("start_time must not be later than end_time")
    return AuditFilters(
        product=str(args.product or "").strip(),
        area=str(args.area or "").strip(),
        start_time=start_time,
        end_time=end_time,
    )


def _validate_output_destination(
    *,
    output_path: Path | None,
    source_path: Path,
    source_is_directory: bool,
    config_path: Path | None,
) -> Path | None:
    if output_path is None:
        return None
    if output_path.is_symlink():
        raise ValueError("audit report destination cannot be a symbolic link")

    destination = output_path.expanduser().resolve(strict=False)
    if source_is_directory:
        if destination == source_path or source_path in destination.parents:
            raise ValueError("audit report destination must be outside the snapshot scan directory")
    elif destination == source_path:
        raise ValueError("audit report destination must not overwrite the snapshot input")

    if config_path is not None:
        canonical_config = config_path.expanduser().resolve(strict=False)
        if destination == canonical_config:
            raise ValueError("audit report destination must not overwrite the model config")
    if destination.suffix.lower() != ".json":
        raise ValueError("audit report destination must use a .json suffix")
    return destination


def _load_policy(
    args: argparse.Namespace,
) -> tuple[DuplicateFilterPolicy, dict[str, Any], dict[str, Any] | None]:
    if args.config is None:
        policy = DuplicateFilterPolicy.from_options(
            {
                "mode": "report_only",
                "iou_threshold": args.iou,
                "center_distance_ratio_max": args.center_ratio,
                "area_similarity_min": args.area_similarity,
                "require_same_verified_class": True,
                "require_color_check_pass": True,
                "require_different_raw_class": True,
                "require_position_disabled": True,
            }
        )
        return policy, {"kind": "command_line_defaults"}, None

    candidate = args.config
    if candidate.is_symlink():
        raise ValueError("model config cannot be a symbolic link")
    config_path = candidate.resolve()
    raw_config = config_path.read_bytes()
    payload = yaml.safe_load(raw_config.decode("utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("model config must contain a YAML mapping")
    pipeline = payload.get("pipeline")
    if not isinstance(pipeline, list):
        raise ValueError("model config pipeline must contain a list")
    validate_duplicate_filter_order(pipeline)
    normalized_pipeline = [str(raw_name).strip().lower() for raw_name in pipeline]
    if "cross_class_duplicate_filter" not in normalized_pipeline:
        raise ValueError("model config pipeline does not enable cross_class_duplicate_filter")
    steps = payload.get("steps", {})
    if not isinstance(steps, dict):
        raise ValueError("model config steps must contain a YAML mapping")
    options = steps.get("cross_class_duplicate_filter", {})
    if not isinstance(options, dict) or not options.get("enabled", True):
        raise ValueError("model config cross_class_duplicate_filter must be enabled")
    policy = DuplicateFilterPolicy.from_options(options)
    policy_source = {
        "kind": "model_config",
        "path": str(config_path),
        "sha256": hashlib.sha256(raw_config).hexdigest(),
    }
    return policy, policy_source, payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError("audit report destination cannot be a symbolic link")
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except (OSError, TypeError, ValueError):
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def _verify_policy_source(policy_source: dict[str, Any]) -> str:
    if policy_source.get("kind") != "model_config":
        return ""
    path = Path(str(policy_source.get("path") or ""))
    try:
        if path.is_symlink() or not path.is_file():
            return "model config became missing or unsafe during audit"
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"model config could not be re-verified after audit: {exc}"
    if actual_sha256 != policy_source.get("sha256"):
        return "model config changed while duplicate audit was running"
    return ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.path.exists():
        print(f"ERROR | Path does not exist: {args.path}", file=sys.stderr)
        return 1
    source_is_file = args.path.is_file()
    source_is_directory = args.path.is_dir()
    if not source_is_file and not source_is_directory:
        print(f"ERROR | Path is not a file or directory: {args.path}", file=sys.stderr)
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
        policy, policy_source, replay_config = _load_policy(args)
        replay_config_source = (
            f"model_config[{policy_source['path']};sha256={policy_source['sha256']}]"
            if policy_source.get("kind") == "model_config"
            else "snapshot.config"
        )
        report = audit_path(
            source_path,
            policy,
            filters=filters,
            replay_config=replay_config,
            replay_config_source=replay_config_source,
        )
        report["policy_source"] = policy_source
        report["position_config_source"] = (
            policy_source if policy_source.get("kind") == "model_config" else {"kind": "snapshot_config"}
        )
        if report["scanned_count"] == 0:
            report["errors"].append({"path": str(source_path), "error": "no snapshot files found"})
        elif report["selected_snapshot_count"] == 0:
            report["errors"].append(
                {
                    "path": str(source_path),
                    "code": "no_snapshots_in_selected_scope",
                    "error": "no snapshots matched the selected audit scope",
                }
            )
        policy_error = _verify_policy_source(policy_source)
        if policy_error:
            report["errors"].append({"path": str(policy_source.get("path") or ""), "error": policy_error})
    except (OSError, TypeError, ValueError) as exc:
        print(f"ERROR | Audit could not be completed: {exc}", file=sys.stderr)
        return 1
    if output_path is not None:
        try:
            _write_json_atomic(output_path, _versioned_report(report))
        except (OSError, TypeError, ValueError) as exc:
            print(f"ERROR | Audit report could not be written: {exc}", file=sys.stderr)
            return 1
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            "Scanned={scanned_count}, matched={matched_snapshot_count}, "
            "proposed_suppressions={proposed_suppression_count}".format(**report)
        )
        for record in report["records"]:
            print(
                f"{record['timestamp']} | model={record['model_version']} | "
                f"raw={record['raw_count']} | {record['path']}"
            )
            for suppression in record["proposed_suppressions"]:
                print(
                    "  "
                    f"#{suppression['suppressed_index']} "
                    f"{suppression['suppressed_raw_class']} -> "
                    f"#{suppression['kept_index']} {suppression['kept_raw_class']} | "
                    f"verified={suppression['verified_class']} | "
                    f"IoU={suppression['iou']:.3f}"
                )
        for error in report["errors"]:
            print(f"ERROR | {error['path']} | {error['error']}")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
