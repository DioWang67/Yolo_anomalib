import hashlib
import json
import zipfile
from pathlib import Path

from tools.portable_training_package import export_portable_training_package


def _write_portable_fixture(tmp_path: Path) -> Path:
    data_root = tmp_path / "training" / "data"
    dataset = data_root / "Cable1" / "A"
    images = dataset / "raw" / "images"
    labels = dataset / "raw" / "labels"
    metadata = dataset / "metadata"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    metadata.mkdir(parents=True)
    image = images / "review_sample.jpg"
    label = labels / "review_sample.txt"
    image.write_bytes(b"image")
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    (metadata / "review_dataset_manifest.csv").write_text(
        "sample_id,output_image,output_label,annotation_status\n"
        f"sample,{image},{label},verified_annotation\n",
        encoding="utf-8",
    )
    models = tmp_path / "inference" / "models"
    station = models / "Cable1" / "A" / "yolo"
    weights = station / "weights"
    weights.mkdir(parents=True)
    runtime = weights / "best.pt"
    runtime.write_bytes(b"model")
    (station / "config.yaml").write_text(
        "weights: models/Cable1/A/yolo/weights/best.pt\n", encoding="utf-8"
    )
    job_id = "job-portable"
    job_dir = data_root / ".operator_handoff" / "jobs" / job_id
    job_dir.mkdir(parents=True)
    handoff = job_dir / "handoff.json"
    status = job_dir / "status.json"
    handoff.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "job_id": job_id,
                "submission_hash": hashlib.sha256(b"submission").hexdigest(),
                "source_manifest": "",
                "data_root": str(data_root),
                "status_path": str(status),
                "inference_models_dir": str(models),
                "training_options": {
                    "epochs": 20,
                    "augmentations_per_image": 20,
                    "batch": 8,
                    "imgsz": 640,
                },
                "targets": [
                    {
                        "product": "Cable1",
                        "area": "A",
                        "dataset_root": str(dataset),
                        "sample_ids": ["sample"],
                        "pending_sample_ids": [],
                        "class_names": ["Black"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status.write_text(json.dumps({"state": "queued"}), encoding="utf-8")
    return handoff


def test_portable_package_contains_dataset_model_contract_and_checksums(tmp_path):
    handoff = _write_portable_fixture(tmp_path)

    report = export_portable_training_package(handoff, tmp_path / "transfer.zip")

    assert report.package_path.is_file()
    assert report.product == "Cable1"
    assert report.area == "A"
    with zipfile.ZipFile(report.package_path) as archive:
        metadata = json.loads(archive.read("package.json"))
        names = set(archive.namelist())
        assert "payload/data/Cable1/A/raw/images/review_sample.jpg" in names
        assert "payload/data/Cable1/A/raw/labels/review_sample.txt" in names
        assert "payload/models/Cable1/A/yolo/config.yaml" in names
        assert "payload/models/Cable1/A/yolo/weights/best.pt" in names
        for relative, contract in metadata["files"].items():
            content = archive.read(relative)
            assert len(content) == contract["size"]
            assert hashlib.sha256(content).hexdigest() == contract["sha256"]
