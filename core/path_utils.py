"""Path utilities for robust project-root resolution and relative path handling."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    """Return the project root directory.

    Resolution order:
    1. ``YOLO11_ROOT`` environment variable (if set)
    2. ``sys.executable`` parent (if PyInstaller-frozen)
    3. Two parents up from this file (``core/path_utils.py`` → repo root)
    """
    env = os.getenv("YOLO11_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        return p

    if getattr(sys, "frozen", False):
        # PyInstaller packaged mode: use the executable's directory
        return Path(sys.executable).parent.resolve()

    # core/path_utils.py -> repo root
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str | None) -> Path | None:
    """Resolve a relative path against known base directories.

    Tries multiple base directories in priority order and returns the first
    existing match.  Falls back to the project root even when the path does
    not exist on disk.

    Args:
        p: Path string to resolve (may be ``None`` or empty).

    Returns:
        Resolved :class:`Path`, or ``None`` if *p* is falsy.
    """
    if not p:
        return None
    path = Path(p)
    if path.is_absolute():
        return path
    # Try multiple bases to be robust across environments
    bases: list[Path] = []
    env = os.getenv("YOLO11_ROOT")
    if env:
        bases.append(Path(env).expanduser())

    if getattr(sys, "frozen", False):
        # PyInstaller packaged mode
        bases.append(Path(sys.executable).parent.resolve())
        if hasattr(sys, "_MEIPASS"):
            bases.append(Path(sys._MEIPASS).resolve())
    else:
        # Repo root derived from this file
        bases.append(Path(__file__).resolve().parents[1])

    # Current working directory (IDE/run-time)
    try:
        bases.append(Path.cwd())
    except OSError:
        pass

    for base in bases:
        try:
            cand = (base / path).resolve()
            if cand.exists():
                return cand
        except (OSError, ValueError):
            continue

    # Fallback to repo-root join even if not exists
    if getattr(sys, "frozen", False):
        return (Path(sys.executable).parent.resolve() / path).resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()
