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


def _remove_conflicting_pyqt_runtime_dlls(dist_root: Path) -> list[str]:
    """Remove PyQt-bundled MSVC runtime DLLs that conflict with ONNX Runtime."""
    qt_bin = dist_root / "_internal" / "PyQt5" / "Qt5" / "bin"
    removed: list[str] = []
    for dll_name in (
        "MSVCP140.dll",
        "MSVCP140_1.dll",
        "VCRUNTIME140.dll",
        "VCRUNTIME140_1.dll",
    ):
        dll_path = qt_bin / dll_name
        if dll_path.exists():
            dll_path.unlink()
            removed.append(str(dll_path.relative_to(dist_root)))
    return removed


def postprocess(source_root: Path, dist_root: Path) -> None:
    _copy_file(source_root / "config.yaml", dist_root / "config.yaml")
    _copy_file(source_root / "config.example.yaml", dist_root / "config.example.yaml")
    _copy_models(source_root, dist_root)
    removed = _remove_conflicting_pyqt_runtime_dlls(dist_root)
    print(f"[OK] postprocess complete; removed_pyqt_runtime_dlls={len(removed)}")


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
