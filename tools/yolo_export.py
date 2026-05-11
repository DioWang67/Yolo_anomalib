"""Export YOLO artifacts to deployment-friendly runtimes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_FORMATS = {"onnx", "openvino", "engine", "torchscript"}


@dataclass(frozen=True)
class ExportRequest:
    """Validated YOLO export request."""

    weights: Path
    export_format: str
    imgsz: int | tuple[int, int]
    half: bool = False
    int8: bool = False
    dynamic: bool = False
    device: str | None = None
    project: Path | None = None
    name: str | None = None
    dry_run: bool = False


def parse_imgsz(value: str) -> int | tuple[int, int]:
    """Parse an Ultralytics image size value.

    Args:
        value: Either ``640`` or ``640,480``.

    Returns:
        Integer square size or ``(height, width)`` tuple.

    Raises:
        ValueError: If the value is not a positive integer or pair.
    """
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        size = int(parts[0])
        if size <= 0:
            raise ValueError("imgsz must be positive")
        return size
    if len(parts) == 2:
        height, width = int(parts[0]), int(parts[1])
        if height <= 0 or width <= 0:
            raise ValueError("imgsz dimensions must be positive")
        return height, width
    raise ValueError("imgsz must be like 640 or 640,480")


def build_export_kwargs(request: ExportRequest) -> dict[str, Any]:
    """Build keyword arguments for ``ultralytics.YOLO.export``."""
    if request.export_format not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(f"Unsupported export format: {request.export_format}. Use one of: {supported}")
    if request.half and request.int8:
        raise ValueError("Use either half precision or int8 quantization, not both")

    kwargs: dict[str, Any] = {
        "format": request.export_format,
        "imgsz": request.imgsz,
    }
    if request.half:
        kwargs["half"] = True
    if request.int8:
        kwargs["int8"] = True
    if request.dynamic:
        kwargs["dynamic"] = True
    if request.device:
        kwargs["device"] = request.device
    if request.project:
        kwargs["project"] = str(request.project)
    if request.name:
        kwargs["name"] = request.name
    return kwargs


def export_yolo_model(request: ExportRequest) -> dict[str, Any]:
    """Export one YOLO model and return a manifest dictionary.

    Args:
        request: Validated export request.

    Returns:
        Manifest with input weights, export options and exported artifact path.

    Raises:
        FileNotFoundError: If the source weights path does not exist.
        RuntimeError: If Ultralytics is not installed.
    """
    if not request.weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {request.weights}")

    kwargs = build_export_kwargs(request)
    exported_path: str | None = None
    if not request.dry_run:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required for YOLO export") from exc
        model = YOLO(str(request.weights))
        exported = model.export(**kwargs)
        exported_path = str(exported)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_weights": str(request.weights),
        "dry_run": request.dry_run,
        "exported_path": exported_path,
        "options": kwargs,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write an export manifest as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--format", required=True, choices=sorted(SUPPORTED_FORMATS))
    parser.add_argument("--imgsz", default="640")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the YOLO export CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    request = ExportRequest(
        weights=args.weights,
        export_format=args.format,
        imgsz=parse_imgsz(args.imgsz),
        half=args.half,
        int8=args.int8,
        dynamic=args.dynamic,
        device=args.device,
        project=args.project,
        name=args.name,
        dry_run=args.dry_run,
    )
    manifest = export_yolo_model(request)
    if args.manifest:
        write_manifest(manifest, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
