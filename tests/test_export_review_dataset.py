import csv
from pathlib import Path

from tools.export_review_dataset import export_review_dataset


def test_export_review_dataset_copies_reviewed_failure_crops(tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"png")
    manifest = tmp_path / "review_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "product",
                "area",
                "status",
                "detector",
                "decision_reasons",
                "model_version",
                "weights",
                "inference_time",
                "config_snapshot_path",
                "annotated_path",
                "failure_crop_paths",
                "review_label",
                "review_note",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "product": "PCBA",
                "area": "TOP",
                "status": "FAIL",
                "decision_reasons": "MISSING",
                "model_version": "1.2.3",
                "failure_crop_paths": str(crop),
                "review_label": "false_positive",
                "review_note": "operator confirmed OK",
            }
        )

    exported = export_review_dataset(manifest, tmp_path / "dataset")

    assert len(exported) == 1
    output_image = Path(exported[0].output_image)
    output_label = Path(exported[0].output_label)
    assert output_image.exists()
    assert output_label.exists()
    assert output_label.read_text(encoding="utf-8") == ""
    assert "false_positive_PCBA_TOP_MISSING" in output_image.name


def test_export_review_dataset_skips_unreviewed_rows(tmp_path):
    crop = tmp_path / "crop.png"
    crop.write_bytes(b"png")
    manifest = tmp_path / "review_manifest.csv"
    manifest.write_text(
        "product,area,status,decision_reasons,failure_crop_paths,review_label,review_note\n"
        f"PCBA,TOP,FAIL,MISSING,{crop},,\n",
        encoding="utf-8",
    )

    exported = export_review_dataset(manifest, tmp_path / "dataset")

    assert exported == []
