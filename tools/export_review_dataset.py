from __future__ import annotations

"""Export reviewed production cases into a dataset curation folder.

This tool does not invent YOLO labels. It copies reviewed evidence images into a
raw/images folder and optionally creates empty label placeholders so operators
can annotate them before running Yolo11_auto_train.
"""

import argparse
import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_LABELS = {"confirmed_ng", "false_positive", "false_negative", "uncertain"}


@dataclass(frozen=True)
class ExportedReviewItem:
    """One copied review image prepared for annotation or retraining."""

    source_manifest: str
    review_label: str
    review_note: str
    source_image: str
    output_image: str
    output_label: str
    product: str
    area: str
    status: str
    decision_reasons: str
    model_version: str


def export_review_dataset(
    manifest_csv: str | Path,
    output_dir: str | Path,
    *,
    include_labels: set[str] | None = None,
    source_kind: str = "failure_crops",
    create_label_placeholders: bool = True,
) -> list[ExportedReviewItem]:
    """Export reviewed cases into ``raw/images`` and ``raw/labels``.

    Args:
        manifest_csv: Review manifest with filled ``review_label`` values.
        output_dir: Destination dataset curation directory.
        include_labels: Review labels to export.
        source_kind: ``failure_crops``, ``annotated``, or ``both``.
        create_label_placeholders: Create empty YOLO label files for annotation.

    Returns:
        Exported item records.
    """
    include = include_labels or set(DEFAULT_LABELS)
    manifest_path = Path(manifest_csv)
    output_root = Path(output_dir)
    images_dir = output_root / "raw" / "images"
    labels_dir = output_root / "raw" / "labels"
    metadata_dir = output_root / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    exported: list[ExportedReviewItem] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle), start=1):
            review_label = str(row.get("review_label") or "").strip()
            if review_label not in include:
                continue
            for source_image in _source_images(row, source_kind):
                source_path = Path(source_image)
                if not source_path.exists():
                    continue
                output_name = _build_output_name(row, row_index, review_label, source_path)
                output_image = images_dir / output_name
                shutil.copy2(source_path, output_image)
                output_label = labels_dir / f"{output_image.stem}.txt"
                if create_label_placeholders and not output_label.exists():
                    output_label.write_text("", encoding="utf-8")
                exported.append(
                    ExportedReviewItem(
                        source_manifest=str(manifest_path),
                        review_label=review_label,
                        review_note=str(row.get("review_note") or ""),
                        source_image=str(source_path),
                        output_image=str(output_image),
                        output_label=str(output_label) if create_label_placeholders else "",
                        product=str(row.get("product") or ""),
                        area=str(row.get("area") or ""),
                        status=str(row.get("status") or ""),
                        decision_reasons=str(row.get("decision_reasons") or ""),
                        model_version=str(row.get("model_version") or ""),
                    )
                )

    _write_export_manifest(exported, metadata_dir / "review_dataset_manifest.csv")
    return exported


def _source_images(row: dict[str, str], source_kind: str) -> list[str]:
    values: list[str] = []
    if source_kind in {"failure_crops", "both"}:
        crops = str(row.get("failure_crop_paths") or "")
        values.extend([item for item in crops.split("|") if item])
    if source_kind in {"annotated", "both"}:
        annotated = str(row.get("annotated_path") or "")
        if annotated:
            values.append(annotated)
    return values


def _build_output_name(
    row: dict[str, str],
    row_index: int,
    review_label: str,
    source_path: Path,
) -> str:
    product = _safe_name(row.get("product") or "unknown")
    area = _safe_name(row.get("area") or "unknown")
    reason = _safe_name(row.get("decision_reasons") or "review")
    return f"{row_index:06d}_{review_label}_{product}_{area}_{reason}_{source_path.name}"


def _safe_name(value: str) -> str:
    text = str(value or "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _write_export_manifest(items: list[ExportedReviewItem], path: Path) -> None:
    fieldnames = list(ExportedReviewItem.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([asdict(item) for item in items])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", required=True, help="Reviewed review_manifest.csv path")
    parser.add_argument("--output-dir", required=True, help="Destination dataset curation directory")
    parser.add_argument(
        "--source-kind",
        choices=["failure_crops", "annotated", "both"],
        default="failure_crops",
        help="Which image evidence to export",
    )
    parser.add_argument(
        "--include-label",
        action="append",
        default=None,
        help="Review label to include. Repeatable. Defaults to all non-empty review labels.",
    )
    parser.add_argument(
        "--no-label-placeholders",
        action="store_true",
        help="Do not create empty YOLO .txt label placeholders",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    include = set(args.include_label) if args.include_label else None
    items = export_review_dataset(
        args.manifest_csv,
        args.output_dir,
        include_labels=include,
        source_kind=args.source_kind,
        create_label_placeholders=not args.no_label_placeholders,
    )
    print(f"Exported {len(items)} review images to {args.output_dir}")
    print("Annotate generated raw/images and raw/labels before Yolo11_auto_train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
