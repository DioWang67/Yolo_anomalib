"""Write a manifest for packaged Hikrobot runtime files.

The manifest is used as a stable baseline for comparing future PyInstaller
outputs without needing an older working build.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_runtime_manifest(bundle_root: Path) -> list[str]:
    """Return manifest lines for the packaged Hikrobot runtime.

    Args:
        bundle_root: PyInstaller onedir output root containing ``_internal``.

    Returns:
        Sorted manifest lines with runtime files and expected environment paths.

    Raises:
        FileNotFoundError: If the packaged runtime directory does not exist.
    """
    runtime_dir = bundle_root / "_internal" / "Runtime"
    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"Runtime directory not found: {runtime_dir}")

    lines = [
        "# Hikrobot runtime manifest",
        f"# Bundle root: {bundle_root}",
        "",
        "[Files]",
    ]
    runtime_files = sorted(
        path
        for path in runtime_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dll", ".cti"}
    )
    for path in runtime_files:
        relative_path = path.relative_to(bundle_root)
        lines.append(f"{relative_path} | {path.stat().st_size} bytes")

    lines.extend(
        [
            "",
            "[Environment]",
            r"GENICAM_GENTL64_PATH=_internal\Runtime",
            r"MVCAM_GENICAM_CLPROTOCOL=_internal\Runtime\CLProtocol",
            r"PATH runtime entry=_internal\Runtime",
        ]
    )
    return lines


def main() -> int:
    """Write the runtime manifest file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest output path. Defaults to runtime_manifest_20260528.txt in bundle root.",
    )
    args = parser.parse_args()

    output_path = args.output or args.bundle_root / "runtime_manifest_20260528.txt"
    lines = build_runtime_manifest(args.bundle_root.resolve())
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote runtime manifest: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
