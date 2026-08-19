"""Build a verified, self-contained operator retraining ZIP package."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PORTABLE_TRAINING_PACKAGE_SCHEMA = 1
PACKAGE_METADATA_NAME = "package.json"
SUPPORTED_OPERATOR_HANDOFF_SCHEMAS = frozenset({3, 4, 5, 6})


class PortableTrainingPackageError(ValueError):
    """Raised when a training package cannot be created safely."""


@dataclass(frozen=True)
class PortableTrainingPackageReport:
    """Summary of one completed portable package export."""

    package_path: Path
    package_id: str
    job_id: str
    product: str
    area: str
    file_count: int
    total_bytes: int


def export_portable_training_package(
    handoff_path: str | Path,
    destination: str | Path,
) -> PortableTrainingPackageReport:
    """Package dataset, annotation queue, configuration, and baseline artifacts."""
    handoff = Path(handoff_path).expanduser().resolve()
    payload = _read_json_mapping(handoff, "operator handoff")
    if int(payload.get("schema_version", 0)) not in SUPPORTED_OPERATOR_HANDOFF_SCHEMAS:
        raise PortableTrainingPackageError(
            "Portable export requires a schema-v3 through schema-v6 operator handoff."
        )
    targets = payload.get("targets")
    if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], dict):
        raise PortableTrainingPackageError(
            "Portable export requires exactly one product and area."
        )
    target = targets[0]
    product = _safe_segment(target.get("product"), "product")
    area = _safe_segment(target.get("area"), "area")
    job_id = _safe_segment(payload.get("job_id"), "job_id")
    data_root = Path(str(payload.get("data_root") or "")).expanduser().resolve()
    dataset_root = Path(str(target.get("dataset_root") or "")).expanduser().resolve()
    if not dataset_root.is_relative_to(data_root) or not dataset_root.is_dir():
        raise PortableTrainingPackageError("The operator dataset path is missing or unsafe.")
    models_root = Path(
        str(payload.get("inference_models_dir") or "")
    ).expanduser().resolve()
    model_station = (models_root / product / area / "yolo").resolve()
    if not model_station.is_relative_to(models_root) or not model_station.is_dir():
        raise PortableTrainingPackageError(
            f"The deployed model station is unavailable: {product}/{area}"
        )

    destination_path = Path(destination).expanduser().resolve()
    if destination_path.suffix.lower() != ".zip":
        destination_path = destination_path.with_suffix(".zip")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    submission_hash = str(payload.get("submission_hash") or "").strip().lower()
    if not submission_hash:
        submission_hash = _sha256_file(handoff)
    package_id = f"{job_id}-{submission_hash[:12]}"
    temporary_zip = destination_path.with_name(
        f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="portable-training-", dir=destination_path.parent
        ) as temporary_directory:
            staging = Path(temporary_directory)
            payload_root = staging / "payload"
            portable_dataset = payload_root / "data" / product / area
            for directory_name in ("raw", "metadata", "review_pending", "color_review"):
                source = dataset_root / directory_name
                if source.is_dir():
                    _copy_tree_safely(source, portable_dataset / directory_name)
            _copy_tree_safely(
                model_station,
                payload_root / "models" / product / area / "yolo",
            )
            job_destination = payload_root / "job"
            job_destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(handoff, job_destination / "handoff.json")
            status_path = handoff.parent / "status.json"
            if status_path.is_file():
                shutil.copy2(status_path, job_destination / "status.json")
            annotation_images = handoff.parent / "annotation_images"
            if annotation_images.is_dir():
                _copy_tree_safely(
                    annotation_images, job_destination / "annotation_images"
                )
            source_manifest = Path(str(payload.get("source_manifest") or ""))
            if source_manifest.is_file():
                shutil.copy2(
                    source_manifest,
                    payload_root / "source_review_manifest.csv",
                )

            readme = staging / "README.txt"
            readme.write_text(
                "Portable YOLO operator retraining package\n"
                "Copy this ZIP to a computer with Yolo11_auto_train, then run:\n"
                "  import_operator_training.bat <this-package.zip>\n"
                "The importer verifies every file before annotation or training starts.\n",
                encoding="utf-8",
            )
            files = _build_inventory(staging)
            package_metadata = {
                "schema_version": PORTABLE_TRAINING_PACKAGE_SCHEMA,
                "package_id": package_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_job_id": job_id,
                "submission_hash": submission_hash,
                "product": product,
                "area": area,
                "handoff_path": "payload/job/handoff.json",
                "dataset_path": f"payload/data/{product}/{area}",
                "models_path": "payload/models",
                "training_options": payload.get("training_options") or {},
                "class_names": target.get("class_names") or [],
                "sample_ids": target.get("sample_ids") or [],
                "pending_sample_ids": target.get("pending_sample_ids") or [],
                "files": files,
            }
            (staging / PACKAGE_METADATA_NAME).write_text(
                json.dumps(package_metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with zipfile.ZipFile(
                temporary_zip,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                allowZip64=True,
            ) as archive:
                for source in sorted(staging.rglob("*")):
                    if source.is_file():
                        archive.write(source, source.relative_to(staging).as_posix())
        temporary_zip.replace(destination_path)
    finally:
        temporary_zip.unlink(missing_ok=True)

    total_bytes = sum(int(item["size"]) for item in files.values())
    return PortableTrainingPackageReport(
        package_path=destination_path,
        package_id=package_id,
        job_id=job_id,
        product=product,
        area=area,
        file_count=len(files),
        total_bytes=total_bytes,
    )


def _copy_tree_safely(source_root: Path, destination_root: Path) -> None:
    resolved_root = source_root.resolve()
    for source in sorted(resolved_root.rglob("*")):
        if source.is_symlink():
            raise PortableTrainingPackageError(
                f"Symbolic links are not allowed in a portable package: {source}"
            )
        if not source.is_file():
            continue
        relative = source.relative_to(resolved_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _build_inventory(root: Path) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == PACKAGE_METADATA_NAME:
            continue
        relative = path.relative_to(root).as_posix()
        inventory[relative] = {
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
    return inventory


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise PortableTrainingPackageError(f"Missing {description}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PortableTrainingPackageError(
            f"Unable to read {description}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PortableTrainingPackageError(f"Invalid {description} JSON.")
    return payload


def _safe_segment(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or any(
        not (character.isalnum() or character in "._-") for character in text
    ):
        raise PortableTrainingPackageError(f"Invalid {field_name}: {text!r}")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
