"""Migrate exported review samples from letterboxed images to camera originals."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tools.export_review_dataset import (
    MISSING_LABEL_BASELINE,
    _read_image_size,
    _reverse_letterbox_bbox,
    _sha256_file,
    _write_text_atomic,
)


@dataclass(frozen=True)
class MigrationReport:
    ready_count: int
    pending_count: int
    skipped_count: int
    backup_dir: Path | None


@dataclass(frozen=True)
class _PlannedSample:
    row: dict[str, str]
    source_original: Path
    output_image: Path
    output_label: Path
    converted_label: str | None
    pending: bool


def migrate_review_dataset_originals(
    output_root: str | Path, *, apply: bool = False
) -> MigrationReport:
    """Plan or apply an atomic, backup-backed original-image migration."""
    root = Path(output_root).resolve()
    ready_manifests = sorted(root.glob("*/*/metadata/review_dataset_manifest.csv"))
    pending_manifests = sorted(root.glob("*/*/review_pending/manifest.csv"))
    manifest_rows: dict[Path, list[dict[str, str]]] = {}
    plans: list[_PlannedSample] = []
    skipped_count = 0

    for manifest_path, pending in [
        *((path, False) for path in ready_manifests),
        *((path, True) for path in pending_manifests),
    ]:
        rows = _read_csv(manifest_path)
        manifest_rows[manifest_path] = rows
        for row in rows:
            plan = _plan_sample(row, pending=pending)
            if plan is None:
                skipped_count += 1
                continue
            plans.append(plan)

    ready_count = sum(not plan.pending for plan in plans)
    pending_count = sum(plan.pending for plan in plans)
    if not apply or not plans:
        return MigrationReport(ready_count, pending_count, skipped_count, None)

    backup_dir = root.parent / "data_migration_backups" / datetime.now().strftime(
        "original-images-%Y%m%d-%H%M%S"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    backed_up: set[Path] = set()
    try:
        for manifest_path in manifest_rows:
            _backup_path(manifest_path, root, backup_dir, backed_up)
        _backup_path(
            root / "metadata" / "review_dataset_manifest.csv",
            root,
            backup_dir,
            backed_up,
        )
        _backup_path(
            root / ".operator_handoff" / "pending.csv",
            root,
            backup_dir,
            backed_up,
        )
        for plan in plans:
            _backup_path(plan.output_image, root, backup_dir, backed_up)
            if plan.output_label.is_file():
                _backup_path(plan.output_label, root, backup_dir, backed_up)

        for plan in plans:
            _copy_file_atomic(plan.source_original, plan.output_image)
            if plan.converted_label is not None:
                _write_text_atomic(plan.output_label, plan.converted_label)
            plan.row["source_image"] = str(plan.source_original)
            plan.row["image_sha256"] = _sha256_file(plan.source_original)
            if plan.pending:
                plan.row["detection_source_image"] = str(
                    _preprocessed_peer(plan.source_original)
                )
                plan.row["label_baseline_sha256"] = (
                    _sha256_file(plan.output_label)
                    if plan.output_label.is_file()
                    else MISSING_LABEL_BASELINE
                )

        for manifest_path, rows in manifest_rows.items():
            _write_csv_atomic(manifest_path, rows)
        _rebuild_master_manifests(root, manifest_rows)
    except (OSError, ValueError, csv.Error):
        _restore_backup(root, backup_dir)
        raise

    return MigrationReport(ready_count, pending_count, skipped_count, backup_dir)


def _plan_sample(row: dict[str, str], *, pending: bool) -> _PlannedSample | None:
    current_source = Path(str(row.get("source_image") or ""))
    source_original = _original_peer(current_source)
    if source_original == current_source or not source_original.is_file():
        return None
    output_image = Path(str(row.get("output_image") or ""))
    output_label = Path(str(row.get("output_label") or ""))
    if not output_image.is_file():
        raise ValueError(f"Missing exported image: {output_image}")

    processed_size = _read_image_size(current_source)
    original_size = _read_image_size(source_original)
    if processed_size is None or original_size is None:
        raise ValueError(f"Unreadable source pair: {current_source}")
    converted_label = None
    if output_label.is_file():
        converted_label = _convert_yolo_label(
            output_label.read_text(encoding="utf-8"),
            original_size=original_size,
            letterbox_size=processed_size,
        )
    return _PlannedSample(
        row=row,
        source_original=source_original,
        output_image=output_image,
        output_label=output_label,
        converted_label=converted_label,
        pending=pending,
    )


def _convert_yolo_label(
    content: str,
    *,
    original_size: tuple[int, int],
    letterbox_size: tuple[int, int],
) -> str:
    converted: list[str] = []
    canvas_width, canvas_height = letterbox_size
    original_width, original_height = original_size
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label at line {line_number}: {raw_line}")
        class_id = int(parts[0])
        center_x, center_y, box_width, box_height = map(float, parts[1:])
        x1 = (center_x - box_width / 2.0) * canvas_width
        y1 = (center_y - box_height / 2.0) * canvas_height
        x2 = (center_x + box_width / 2.0) * canvas_width
        y2 = (center_y + box_height / 2.0) * canvas_height
        projected = _reverse_letterbox_bbox(
            (x1, y1, x2, y2),
            original_size=(float(original_width), float(original_height)),
            letterbox_size=(float(canvas_width), float(canvas_height)),
        )
        if projected is None:
            raise ValueError(f"Label lies outside the camera image at line {line_number}")
        px1, py1, px2, py2 = projected
        converted.append(
            f"{class_id} {((px1 + px2) / 2.0) / original_width:.8f} "
            f"{((py1 + py2) / 2.0) / original_height:.8f} "
            f"{(px2 - px1) / original_width:.8f} "
            f"{(py2 - py1) / original_height:.8f}"
        )
    return "\n".join(converted) + ("\n" if converted else "")


def _original_peer(path: Path) -> Path:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "preprocessed":
            parts[index] = "original"
            return Path(*parts)
    return path


def _preprocessed_peer(path: Path) -> Path:
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part.lower() == "original":
            parts[index] = "preprocessed"
            return Path(*parts)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    if (
        any("detection_source_image" in row for row in rows)
        and "detection_source_image" not in fields
    ):
        insert_at = (
            fields.index("source_image") + 1
            if "source_image" in fields
            else len(fields)
        )
        fields.insert(insert_at, "detection_source_image")
    temporary = path.with_name(f".{path.name}.original-migration.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _rebuild_master_manifests(
    root: Path, manifest_rows: dict[Path, list[dict[str, str]]]
) -> None:
    ready_rows: list[dict[str, str]] = []
    pending_rows: list[dict[str, str]] = []
    for path, rows in manifest_rows.items():
        if path.parent.name == "metadata":
            ready_rows.extend(rows)
        else:
            pending_rows.extend(rows)
    if ready_rows:
        _write_csv_atomic(root / "metadata" / "review_dataset_manifest.csv", ready_rows)
    if pending_rows:
        _write_csv_atomic(root / ".operator_handoff" / "pending.csv", pending_rows)


def _backup_path(path: Path, root: Path, backup: Path, seen: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in seen or not path.is_file():
        return
    relative = resolved.relative_to(root)
    destination = backup / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved, destination)
    seen.add(resolved)


def _restore_backup(root: Path, backup: Path) -> None:
    for source in sorted(backup.rglob("*")):
        if not source.is_file():
            continue
        destination = root / source.relative_to(backup)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_file_atomic(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.original-migration.tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = migrate_review_dataset_originals(args.output_root, apply=args.apply)
    print(
        f"ready={report.ready_count} pending={report.pending_count} "
        f"skipped={report.skipped_count} backup={report.backup_dir or '-'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
