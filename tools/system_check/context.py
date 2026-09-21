"""Locate the installation under test and read its configuration.

The checker has to work in three layouts:

1. A source checkout, run as ``python -m tools.system_check``.
2. Next to a PyInstaller one-dir build, where data files live under
   ``_internal/`` and models sit beside the executable.
3. Anywhere at all, with ``--app-root`` naming the installation explicitly.

Everything here is read-only. Configuration is parsed with ``yaml.safe_load``
so a malformed or hostile config cannot execute code.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.system_check.spec import (
    DEFAULT_INFERENCE_TIMEOUT_S,
    ModelTarget,
)

_CONFIG_NAME = "config.yaml"
_LOCAL_CONFIG_NAME = "config.local.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Parse one YAML mapping, returning ``{}`` when unusable.

    Never raises: a station with a truncated config must still get a report
    that says so rather than a traceback.
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        loaded = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 - malformed YAML is a finding, not a crash
        return {}
    return loaded if isinstance(loaded, dict) else {}


@dataclass(frozen=True)
class AppContext:
    """Where the installation is and what it is configured to do.

    Args:
        app_root: Directory holding ``config.yaml``, ``models/`` and ``Runtime/``
            (or, for a frozen build, the directory holding the executable).
        data_root: Directory holding bundled data. Equals ``app_root`` for a
            source checkout and ``app_root/_internal`` for PyInstaller >= 6.
        config: Merged ``config.yaml`` + ``config.local.yaml`` mapping.
        config_path: The config file actually read, or ``None``.
        frozen: Whether the checker itself is running from a frozen build.
    """

    app_root: Path
    data_root: Path
    config: dict[str, Any]
    config_path: Path | None
    frozen: bool

    @property
    def runtime_dir(self) -> Path:
        """Hikrobot SDK directory for this layout."""
        candidate = self.data_root / "Runtime"
        if candidate.is_dir():
            return candidate
        return self.app_root / "Runtime"

    @property
    def models_dir(self) -> Path:
        """Model bundle root. Models are external data, never inside ``_internal``."""
        return self.app_root / "models"

    @property
    def result_dir(self) -> Path:
        """Directory inspection evidence is written to, per ``output_dir``."""
        configured = str(self.config.get("output_dir") or "./Result")
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = self.app_root / path
        return path

    def anomalib_enabled(self) -> bool:
        """Whether the anomalib backend is switched on for this station."""
        return bool(self.config.get("enable_anomalib", False))

    def sync_enabled(self) -> bool:
        """Whether outbound inspection sync is switched on for this station."""
        return bool(self.config.get("inspection_sync_enabled", False))


#: How far up from the executable to look for a neighbouring installation.
_NEARBY_SEARCH_DEPTH = 3


def _candidate_roots(explicit: str | Path | None) -> list[Path]:
    """Return app-root candidates in resolution order."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.environ.get("YOLO11_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    if getattr(sys, "frozen", False):
        here = Path(sys.executable).resolve().parent
        candidates.append(here)
        # The checker ships as its own folder, so the installation is almost
        # never the folder the exe sits in - it is a sibling or an ancestor.
        # Without this an operator who just double-clicks the exe gets "no
        # model bundle found" and has to learn --app-root, which is exactly
        # the command line they were trying to avoid.
        candidates.extend(_nearby_installations(here))
    else:
        # tools/system_check/context.py -> repo root
        candidates.append(Path(__file__).resolve().parents[2])

    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.extend(_nearby_installations(cwd))
    return candidates


def _nearby_installations(start: Path) -> list[Path]:
    """Find installations in ``start``'s ancestors and their children.

    Ordered nearest-first. Read-only and failure-tolerant: an unreadable or
    permission-denied directory is skipped rather than raised.
    """
    found: list[Path] = []
    seen: set[Path] = set()
    current = start
    for _ in range(_NEARBY_SEARCH_DEPTH):
        parent = current.parent
        if parent == current:
            break
        for candidate in (parent, *_children(parent)):
            resolved = candidate if candidate.is_absolute() else candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved != start and _looks_like_installation(resolved):
                found.append(resolved)
        current = parent
    return found


def _children(path: Path) -> list[Path]:
    """List immediate subdirectories, tolerating an unreadable directory."""
    try:
        return sorted(item for item in path.iterdir() if item.is_dir())
    except OSError:
        return []


def _looks_like_installation(path: Path) -> bool:
    """Return whether ``path`` holds an inference installation.

    Requires a marker that a *deployed* installation has. ``models`` alone is
    not enough on its own to distinguish a real installation from an arbitrary
    folder, but combined with the others it is what every layout shares.
    """
    if not path.is_dir():
        return False
    markers = (
        path / _CONFIG_NAME,
        path / "yolo11_inference.exe",
        path / "Runtime",
        path / "_internal" / "Runtime",
        path / "models",
    )
    return any(marker.exists() for marker in markers)


def resolve_context(app_root: str | Path | None = None) -> AppContext:
    """Find the installation and load its effective configuration.

    Args:
        app_root: Explicit installation directory, or ``None`` to auto-detect.

    Returns:
        A populated :class:`AppContext`. Falls back to the first candidate even
        when no marker file is present, so the checker can still report *why*
        the layout looks wrong.
    """
    candidates = _candidate_roots(app_root)
    chosen = next((path for path in candidates if _looks_like_installation(path)), candidates[0])
    chosen = chosen.resolve()

    internal = chosen / "_internal"
    data_root = internal if internal.is_dir() else chosen

    config: dict[str, Any] = {}
    config_path: Path | None = None
    base = chosen / _CONFIG_NAME
    if base.is_file():
        config = _load_yaml(base)
        config_path = base
    local = chosen / _LOCAL_CONFIG_NAME
    if local.is_file():
        # Matches the documented override semantics: top-level keys replace.
        config = {**config, **_load_yaml(local)}

    return AppContext(
        app_root=chosen,
        data_root=data_root,
        config=config,
        config_path=config_path,
        frozen=bool(getattr(sys, "frozen", False)),
    )


def _coerce_imgsz(value: Any) -> tuple[int, int]:
    """Normalize a config ``imgsz`` into ``(height, width)``."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return (640, 640)
    if isinstance(value, (int, float)) and value > 0:
        return int(value), int(value)
    return (640, 640)


def _expected_item_count(bundle: dict[str, Any], product: str, area: str) -> int | None:
    """Return how many items ``expected_items`` declares for this bundle.

    The benchmark uses this to say whether its frame exercised postprocessing
    at production scale. ``None`` when the bundle declares nothing, so the
    report makes no claim rather than assuming one item.
    """
    expected = bundle.get("expected_items")
    if not isinstance(expected, dict):
        return None
    by_area = expected.get(product)
    if not isinstance(by_area, dict):
        return None
    items = by_area.get(area)
    if not isinstance(items, list):
        return None
    return len(items) or None


def _resolve_weights(context: AppContext, bundle_dir: Path, raw: str) -> Path | None:
    """Resolve a config ``weights`` value against the roots the app searches."""
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    for base in (context.app_root, bundle_dir, bundle_dir.parent, Path.cwd()):
        resolved = base / candidate
        if resolved.exists():
            return resolved.resolve()
    return None


def discover_model_targets(context: AppContext) -> list[ModelTarget]:
    """Enumerate benchmarkable model bundles under ``models/``.

    A bundle qualifies when its ``config.yaml`` names weights that exist on
    disk. Bundles are returned in ``product/area`` order; the active product
    from the root config, when present, is moved to the front so the default
    benchmark exercises what this station actually runs.

    Args:
        context: Resolved installation.

    Returns:
        Model targets, possibly empty.
    """
    models_dir = context.models_dir
    if not models_dir.is_dir():
        return []

    targets: list[ModelTarget] = []
    try:
        bundle_configs = sorted(models_dir.glob("*/*/*/config.yaml"))
    except OSError:
        return []

    for config_path in bundle_configs:
        bundle_dir = config_path.parent
        # models/<product>/<area>/<backend>/config.yaml
        try:
            backend = bundle_dir.name
            area = bundle_dir.parent.name
            product = bundle_dir.parent.parent.name
        except IndexError:  # pragma: no cover - glob guarantees the depth
            continue
        if backend != "yolo":
            # Only the YOLO path is benchmarkable without torch/anomalib.
            continue

        bundle = _load_yaml(config_path)
        weights = _resolve_weights(context, bundle_dir, str(bundle.get("weights") or ""))
        if weights is None:
            continue

        targets.append(
            ModelTarget(
                product=product,
                area=area,
                weights=str(weights),
                imgsz=_coerce_imgsz(bundle.get("imgsz")),
                conf_thres=float(bundle.get("conf_thres", 0.25) or 0.25),
                iou_thres=float(bundle.get("iou_thres", 0.45) or 0.45),
                timeout_s=float(
                    bundle.get("timeout", context.config.get("timeout", DEFAULT_INFERENCE_TIMEOUT_S))
                    or DEFAULT_INFERENCE_TIMEOUT_S
                ),
                device=str(bundle.get("device", "cpu") or "cpu"),
                config_path=str(config_path),
                expected_item_count=_expected_item_count(bundle, product, area),
            )
        )

    active_product = str(context.config.get("current_product") or "")
    active_area = str(context.config.get("current_area") or "")
    if active_product:
        targets.sort(
            key=lambda target: (
                target.product != active_product,
                bool(active_area) and target.area != active_area,
                target.product,
                target.area,
            )
        )
    return targets
