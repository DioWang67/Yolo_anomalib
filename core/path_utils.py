from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.getenv("YOLO11_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        return p
    # core/path_utils.py -> core -> repo root
    return Path(__file__).resolve().parents[2]


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
    # Repo root derived from this file
    bases.append(Path(__file__).resolve().parents[2])
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
    return (Path(__file__).resolve().parents[2] / path).resolve()
"""Path utilities for robust project-root resolution and relative path handling."""
