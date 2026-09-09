"""Reconstruct training provenance for weights deployed before it was recorded.

Models deployed before provenance stamping carry only ``dataset_hash`` -- an
MD5 over relative paths, sizes and mtimes. It cannot be reversed to a file
list, so the images behind those weights are genuinely unrecoverable from the
artifact alone.

What *can* be done is a guess: a run's ``args.yaml`` names the dataset it
trained from, and submission history holds an immutable per-batch image
manifest for each operator job. Matching the two by product/area and time
usually identifies the batch. Usually is not always -- two jobs for the same
station on the same day are indistinguishable this way, and a from-scratch
dataset never passed through submission history at all.

So every record this writes is marked ``provenance_confidence: inferred``,
and the GUI renders that differently from a stamped ``recorded``. Treat the
result as a lead, not as evidence: it is not admissible for the acceptance
and release gates, which require identity the runtime itself produced.

Usage::

    python tools/backfill_training_provenance.py --models models --dry-run
    python tools/backfill_training_provenance.py --models models --apply
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_paths(models_root: Path) -> list[Path]:
    return sorted(models_root.glob("*/*/*/weights/*.manifest.yaml"))


def _parse_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _candidate_submissions(
    history: list[Any], product: str, area: str, trained_at: datetime | None
) -> list[Any]:
    """Submissions for one station that could have produced a weight.

    A batch submitted after training finished cannot be its source, so the
    list is cut at ``trained_at``. Without this the newest batch wins every
    time and every historical weight gets attributed to the same recent job
    -- a confident-looking answer that is simply wrong.
    """
    matches = []
    for record in history:
        if record.product != product or record.area != area or not record.job_id:
            continue
        submitted = _parse_timestamp(record.submitted_at)
        if trained_at is not None and submitted is not None and submitted > trained_at:
            continue
        matches.append((submitted, record))
    matches.sort(
        key=lambda pair: pair[0] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return [record for _submitted, record in matches]


def backfill(
    models_root: Path, training_data_root: Path, *, apply: bool
) -> tuple[int, int]:
    """Return (examined, updated). Writes only when ``apply`` is true."""
    from tools.submission_history import load_submission_history

    try:
        history = load_submission_history(training_data_root)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not read submission history: {exc}", file=sys.stderr)
        return (0, 0)

    examined = 0
    updated = 0
    for manifest_path in _manifest_paths(models_root):
        manifest = _load_yaml(manifest_path)
        if not manifest:
            continue
        examined += 1
        if manifest.get("dataset_id"):
            continue  # Already stamped at deploy time; never overwrite.
        product = str(manifest.get("product") or "")
        area = str(manifest.get("area") or "")
        if not product or not area:
            print(f"SKIP  {manifest_path.name}: manifest names no product/area")
            continue
        trained_at = _parse_timestamp(manifest.get("trained_at"))
        if trained_at is None:
            print(f"SKIP  {manifest_path.name}: no trained_at to order batches by")
            continue
        candidates = _candidate_submissions(history, product, area, trained_at)
        if not candidates:
            print(
                f"SKIP  {manifest_path.name}: no {product}/{area} batch submitted "
                f"before {trained_at.date()}"
            )
            continue
        chosen = candidates[0]
        submitted = _parse_timestamp(chosen.submitted_at)
        gap = (
            f", {(trained_at - submitted).days}d before training"
            if submitted is not None
            else ""
        )
        print(
            f"MATCH {manifest_path.name} -> job {chosen.job_id} "
            f"({chosen.batch_version or 'unversioned'}, "
            f"{chosen.case_count} cases{gap})"
        )
        updated += 1
        if not apply:
            continue
        manifest["dataset_id"] = f"inferred:{chosen.submission_id}"
        manifest["training_job_id"] = chosen.job_id
        manifest["provenance_confidence"] = "inferred"
        manifest["dataset_image_count"] = int(chosen.case_count or 0)
        manifest_path.write_text(
            yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return (examined, updated)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument(
        "--training-data",
        type=Path,
        help="Training project data root (defaults to the workspace layout)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Report only")
    group.add_argument("--apply", action="store_true", help="Write the manifests")
    args = parser.parse_args(argv)

    training_data = args.training_data
    if training_data is None:
        from core.path_utils import project_root
        from core.workspace import load_workspace_paths

        training_data = load_workspace_paths(project_root()).training_data

    examined, updated = backfill(
        args.models.resolve(), Path(training_data), apply=bool(args.apply)
    )
    verb = "updated" if args.apply else "would update"
    print(f"\n{examined} manifest(s) examined; {verb} {updated}.")
    if updated and not args.apply:
        print("Re-run with --apply to write them.")
    if updated and args.apply:
        print(
            "All written records are marked inferred. They are a lead for "
            "humans, not evidence for the acceptance or release gates."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
