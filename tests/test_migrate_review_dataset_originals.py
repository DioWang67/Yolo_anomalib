import csv
import shutil

from PIL import Image

from tools.migrate_review_dataset_originals import migrate_review_dataset_originals


def test_migration_replaces_letterbox_with_original_and_preserves_backup(tmp_path):
    result_root = tmp_path / "Result" / "20260716" / "Cable1" / "A" / "FAIL"
    processed = result_root / "preprocessed" / "yolo" / "case.jpg"
    original = result_root / "original" / "yolo" / "case.jpg"
    processed.parent.mkdir(parents=True)
    original.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), (128, 128, 128)).save(processed)
    Image.new("RGB", (300, 200), (20, 40, 60)).save(original)

    output_root = tmp_path / "training-data"
    target = output_root / "Cable1" / "A"
    output_image = target / "raw" / "images" / "review_legacy.jpg"
    output_label = target / "raw" / "labels" / "review_legacy.txt"
    output_image.parent.mkdir(parents=True)
    output_label.parent.mkdir(parents=True)
    shutil.copy2(processed, output_image)
    output_label.write_text(
        "0 0.35000000 0.40100000 0.30000000 0.19800000\n",
        encoding="utf-8",
    )
    manifest = target / "metadata" / "review_dataset_manifest.csv"
    manifest.parent.mkdir(parents=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_image",
                "output_image",
                "output_label",
                "sample_id",
                "image_sha256",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_image": str(processed),
                "output_image": str(output_image),
                "output_label": str(output_label),
                "sample_id": "legacy",
                "image_sha256": "old-hash",
            }
        )

    dry_run = migrate_review_dataset_originals(output_root)
    assert dry_run.ready_count == 1
    assert dry_run.backup_dir is None
    with Image.open(output_image) as image:
        assert image.size == (100, 100)

    report = migrate_review_dataset_originals(output_root, apply=True)

    assert report.ready_count == 1
    assert report.pending_count == 0
    assert report.backup_dir is not None
    with Image.open(output_image) as image:
        assert image.size == (300, 200)
    assert output_label.read_text(encoding="utf-8") == (
        "0 0.35000000 0.35000000 0.30000000 0.30000000\n"
    )
    migrated = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert migrated[0]["source_image"] == str(original)
    assert migrated[0]["sample_id"] == "legacy"
    assert (
        report.backup_dir / output_image.relative_to(output_root)
    ).is_file()
