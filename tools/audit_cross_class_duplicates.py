"""Read-only replay audit for cross-class duplicate detections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.services.cross_class_duplicate_filter import (  # noqa: E402
    DuplicateFilterPolicy,
    analyze_cross_class_duplicates,
)


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
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print a machine-readable JSON report.",
    )
    return parser


def audit_path(path: Path, policy: DuplicateFilterPolicy) -> dict[str, Any]:
    files = (
        [path]
        if path.is_file()
        else sorted(path.rglob("*_config_snapshot.json"))
    )
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for snapshot_path in files:
        try:
            payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            detections = payload.get("raw_detections") or payload.get("detections") or []
            color_items = (payload.get("color_result") or {}).get("items") or []
            indexed_color_items = {
                int(item.get("index", position)): item
                for position, item in enumerate(color_items)
                if isinstance(item, dict)
            }
            analysis = analyze_cross_class_duplicates(
                detections=detections,
                color_items_by_index=indexed_color_items,
                policy=policy,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            errors.append({"path": str(snapshot_path), "error": str(exc)})
            continue
        if analysis["candidate_count"] <= 0:
            continue
        records.append(
            {
                "path": str(snapshot_path),
                "inspection_id": str(payload.get("inspection_id") or ""),
                "timestamp": str(payload.get("timestamp") or ""),
                "status": str(payload.get("status") or ""),
                "model_version": str(
                    (payload.get("model_info") or {}).get("model_version") or ""
                ),
                "raw_count": analysis["raw_count"],
                "candidate_count": analysis["candidate_count"],
                "proposed_suppressions": analysis["proposed_suppressions"],
            }
        )
    return {
        "policy": policy.to_dict(),
        "scanned_count": len(files),
        "matched_snapshot_count": len(records),
        "proposed_suppression_count": sum(
            len(record["proposed_suppressions"]) for record in records
        ),
        "records": records,
        "errors": errors,
    }


def main() -> int:
    args = build_parser().parse_args()
    if not args.path.exists():
        raise SystemExit(f"Path does not exist: {args.path}")
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
    report = audit_path(args.path.resolve(), policy)
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
