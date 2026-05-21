from __future__ import annotations

"""Post-process the PyInstaller output folder."""

import shutil
import sys
from pathlib import Path


def _copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_models(source_root: Path, dist_root: Path) -> None:
    source_models = source_root / "models"
    destination_models = dist_root / "models"
    if not source_models.is_dir():
        raise FileNotFoundError(f"Required models directory not found: {source_models}")
    if destination_models.exists():
        shutil.rmtree(destination_models)
    shutil.copytree(source_models, destination_models)


def _rewrite_packaged_yolo_configs(dist_root: Path) -> int:
    """Switch packaged YOLO configs from ONNX to PyTorch weights.

    ONNX Runtime can fail to initialize inside the frozen executable on some
    Windows environments. The packaged model folders include matching .pt
    weights, so the release bundle uses those instead of .onnx files.
    """
    changed = 0
    for config_path in (dist_root / "models").rglob("config.yaml"):
        text = config_path.read_text(encoding="utf-8")
        updated = text.replace(".onnx", ".pt")
        if updated != text:
            config_path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def _copy_onnxruntime_dlls(dist_root: Path) -> None:
    internal = dist_root / "_internal"
    capi = internal / "onnxruntime" / "capi"
    if not capi.is_dir():
        return
    for dll_name in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
        source = capi / dll_name
        if source.exists():
            shutil.copy2(source, internal / dll_name)


def postprocess(source_root: Path, dist_root: Path) -> None:
    _copy_file(source_root / "config.yaml", dist_root / "config.yaml")
    _copy_file(source_root / "config.example.yaml", dist_root / "config.example.yaml")
    _copy_models(source_root, dist_root)
    changed = _rewrite_packaged_yolo_configs(dist_root)
    _copy_onnxruntime_dlls(dist_root)
    print(f"[OK] postprocess complete; rewritten_config_count={changed}")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: postprocess_build.py <source_root> <dist_root>", file=sys.stderr)
        return 2

    source_root = Path(sys.argv[1]).resolve()
    dist_root = Path(sys.argv[2]).resolve()
    try:
        postprocess(source_root, dist_root)
    except Exception as exc:
        print(f"[ERROR] build postprocess failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
