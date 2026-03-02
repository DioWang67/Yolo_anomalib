from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
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
    except Exception:
        pass

    for base in bases:
        try:
            cand = (base / path).resolve()
            if cand.exists():
                return cand
        except Exception:
            continue
            
    # Fallback to repo-root join even if not exists
    if getattr(sys, "frozen", False):
        return (Path(sys.executable).parent.resolve() / path).resolve()
    return (Path(__file__).resolve().parents[1] / path).resolve()


"""Path utilities for robust project-root resolution and relative path handling."""
