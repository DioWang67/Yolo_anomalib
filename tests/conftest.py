# tests/conftest.py
import atexit
import hashlib
import os
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_WORKSPACE_ENV_VAR = "YOLO11_WORKSPACE_ROOT"
_WORKSPACE_MANIFEST_NAME = "workspace.yaml"


class _WorkspaceGuardError(RuntimeError):
    """Raised when live workspace integrity cannot be established."""


class _UnsafeLiveWorkspaceError(pytest.UsageError):
    """Raised when pytest configuration can route writes into live data."""


@dataclass(frozen=True)
class _PathState:
    """Immutable content and metadata identity for one filesystem entry."""

    kind: str
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int
    sha256: str | None


_TreeSnapshot = dict[str, _PathState]
_RootsSnapshot = dict[Path, _TreeSnapshot]
_LIVE_SNAPSHOT_KEY = pytest.StashKey[_RootsSnapshot]()


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either resolved path contains the other."""
    return first == second or first in second.parents or second in first.parents


def _find_live_workspace_manifest() -> Path | None:
    """Find the manifest belonging to this checkout, ignoring environment state."""
    for directory in (ROOT, *ROOT.parents):
        manifest_path = directory / _WORKSPACE_MANIFEST_NAME
        if manifest_path.is_file():
            return manifest_path.resolve()
    return None


_LIVE_WORKSPACE_MANIFEST = _find_live_workspace_manifest()
_LIVE_WORKSPACE_ROOT = _LIVE_WORKSPACE_MANIFEST.parent if _LIVE_WORKSPACE_MANIFEST is not None else ROOT.resolve()


def _load_workspace_manifest(manifest_path: Path):
    """Parse a manifest through production validation and normalize errors."""
    from core import workspace as workspace_module

    try:
        return workspace_module._load_manifest(manifest_path)  # noqa: SLF001
    except workspace_module.WorkspaceConfigurationError as exc:
        raise _WorkspaceGuardError(f"Invalid pytest workspace manifest {manifest_path}: {exc}") from exc


def _workspace_targets_from_manifest(manifest_path: Path) -> tuple[Path, ...]:
    """Resolve every configured path through the production workspace parser."""
    paths = _load_workspace_manifest(manifest_path)
    return (
        paths.root,
        paths.training_project,
        paths.inference_project,
        paths.training_data,
        paths.inference_models,
        paths.station_data,
        paths.inference_results,
        paths.inference_artifacts,
    )


def _assert_safe_workspace_environment(
    environ: Mapping[str, str],
    *,
    live_workspace_root: Path,
) -> Path | None:
    """Reject an explicit pytest workspace that can reach live station data."""
    configured_value = environ.get(_WORKSPACE_ENV_VAR, "").strip()
    if not configured_value:
        return None

    configured_path = Path(configured_value).expanduser().resolve()
    configured_root = (
        configured_path.parent if configured_path.name.casefold() == _WORKSPACE_MANIFEST_NAME else configured_path
    )
    live_root = live_workspace_root.resolve()
    if _paths_overlap(configured_root, live_root):
        raise _UnsafeLiveWorkspaceError(
            f"Refusing unsafe {_WORKSPACE_ENV_VAR}: configured pytest root "
            f"{configured_root} overlaps the live workspace {live_root}. "
            "Use an isolated temporary workspace."
        )

    manifest_path = (
        configured_path
        if configured_path.name.casefold() == _WORKSPACE_MANIFEST_NAME
        else configured_root / _WORKSPACE_MANIFEST_NAME
    )
    try:
        targets = _workspace_targets_from_manifest(manifest_path)
    except _WorkspaceGuardError as exc:
        raise pytest.UsageError(f"Unsafe pytest workspace configuration: {exc}") from exc
    overlapping = next(
        (target for target in targets if _paths_overlap(target, live_root)),
        None,
    )
    if overlapping is not None:
        raise _UnsafeLiveWorkspaceError(
            f"Refusing unsafe {_WORKSPACE_ENV_VAR}: configured pytest path "
            f"{overlapping} overlaps the live workspace {live_root}. "
            "Use an isolated temporary workspace."
        )
    return configured_root


_assert_safe_workspace_environment(
    os.environ,
    live_workspace_root=_LIVE_WORKSPACE_ROOT,
)


def _create_isolated_workspace() -> tuple[tempfile.TemporaryDirectory[str], Path]:
    """Create the minimal paired-workspace contract used by pytest."""
    temporary_directory = tempfile.TemporaryDirectory(prefix="yolo11-pytest-workspace-")
    workspace_root = Path(temporary_directory.name).resolve()
    try:
        _validated_external_basetemp(
            workspace_root,
            source="isolated pytest workspace",
        )
    except pytest.UsageError:
        temporary_directory.cleanup()
        raise
    for relative_directory in (
        "Yolo11_auto_train/data",
        "yolo11_inference/models",
        "station_data/yolo11_inference",
        "Result",
        "release_artifacts/yolo11_inference",
    ):
        (workspace_root / relative_directory).mkdir(parents=True, exist_ok=True)
    (workspace_root / _WORKSPACE_MANIFEST_NAME).write_text(
        """schema_version: 1
projects:
  training: Yolo11_auto_train
  inference: yolo11_inference
paths:
  training_data: Yolo11_auto_train/data
  inference_models: yolo11_inference/models
  station_data: station_data/yolo11_inference
  inference_results: Result
  inference_artifacts: release_artifacts/yolo11_inference
""",
        encoding="utf-8",
    )
    return temporary_directory, workspace_root


def _discovery_starts_in_checkout(start: str | Path | None) -> bool:
    """Identify discovery rooted anywhere inside the live paired workspace."""
    if start is None:
        return True
    resolved = Path(start).expanduser().resolve()
    anchor = resolved.parent if resolved.is_file() else resolved
    return anchor == _LIVE_WORKSPACE_ROOT or _LIVE_WORKSPACE_ROOT in anchor.parents


def _install_default_workspace_isolation(isolated_workspace_root: Path) -> None:
    """Route checkout-local implicit discovery to an isolated test workspace."""
    from core import workspace as workspace_module

    original_load_workspace_paths = workspace_module.load_workspace_paths
    isolated_manifest = isolated_workspace_root / _WORKSPACE_MANIFEST_NAME

    def load_without_environment(start: str | Path | None):
        """Apply production discovery order while ignoring pytest's default env."""
        anchor = workspace_module._directory_anchor(start)  # noqa: SLF001
        seen: set[Path] = set()
        for parent in workspace_module._ancestors(anchor):  # noqa: SLF001
            candidate = (parent / workspace_module.WORKSPACE_FILENAME).resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return workspace_module._load_manifest(candidate)  # noqa: SLF001
        return workspace_module._legacy_workspace_paths(anchor)  # noqa: SLF001

    @wraps(original_load_workspace_paths)
    def isolated_load_workspace_paths(start: str | Path | None = None):
        try:
            configured_root = _assert_safe_workspace_environment(
                os.environ,
                live_workspace_root=_LIVE_WORKSPACE_ROOT,
            )
        except _UnsafeLiveWorkspaceError:
            raise
        except pytest.UsageError:
            # Preserve the production WorkspaceConfigurationError contract for
            # malformed external roots; it also fails before returning paths.
            return original_load_workspace_paths(start)

        if configured_root is not None and configured_root != isolated_workspace_root:
            return original_load_workspace_paths(start)
        if _discovery_starts_in_checkout(start):
            return workspace_module._load_manifest(isolated_manifest)  # noqa: SLF001
        return load_without_environment(start)

    workspace_module.load_workspace_paths = isolated_load_workspace_paths
    station_data_module = sys.modules.get("core.station_data")
    if station_data_module is not None:
        station_data_module.load_workspace_paths = isolated_load_workspace_paths


def _validated_external_basetemp(path: str | Path, *, source: str) -> Path:
    """Resolve a pytest temp root and reject workspace descendants."""
    resolved = Path(path).expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "workspace.yaml").is_file():
            raise pytest.UsageError(
                f"{source}={resolved} lies inside the workspace at {parent}. "
                "Workspace discovery would resolve every test's tmp_path to the "
                "real station data and let tests publish into production "
                "evidence. Choose a basetemp outside the workspace, such as one "
                "under the system temporary directory."
            )
    return resolved


_ISOLATED_WORKSPACE, _ISOLATED_WORKSPACE_ROOT = _create_isolated_workspace()
atexit.register(_ISOLATED_WORKSPACE.cleanup)
os.environ[_WORKSPACE_ENV_VAR] = str(_ISOLATED_WORKSPACE_ROOT)
_install_default_workspace_isolation(_ISOLATED_WORKSPACE_ROOT)


def pytest_configure(config):
    """Refuse to run when ``--basetemp`` sits inside a workspace.

    Workspace discovery walks *upwards* for ``workspace.yaml``, so a basetemp
    under the workspace makes every test's ``tmp_path`` resolve to the real
    station data. This workspace keeps its scratch directories in ``.tmp/``,
    which is inside the workspace, so applying that convention to pytest is an
    easy and previously unguarded mistake: it published a stub color model into
    the live color-profile store, where acceptance offered it as a selectable
    variant until it failed on every image of a 250-sample matrix combination.

    Refused up front rather than detected afterwards, because by the time a
    write is visible the production evidence has already been modified.
    """
    basetemp = getattr(config.option, "basetemp", None)
    if basetemp is None:
        return
    _validated_external_basetemp(basetemp, source="--basetemp")


def _live_station_data_roots() -> tuple[Path, ...]:
    """Locate live data without allowing the environment to redirect the guard."""
    if _LIVE_WORKSPACE_MANIFEST is not None:
        paths = _load_workspace_manifest(_LIVE_WORKSPACE_MANIFEST)
        return (paths.station_data,)

    # A standalone legacy checkout stores mutable station state in the source
    # tree. Protect only its known mutable roots so normal pytest cache and
    # coverage files do not produce false positives. Runtime logs are
    # intentionally excluded: application initialization creates dated
    # diagnostic logs during normal tests, and those logs are not station
    # evidence or an input to release decisions.
    return tuple(
        ROOT / name
        for name in (
            ".color_baselines",
            ".color_profiles",
            ".color_revisions",
            ".inspection_releases",
            ".processing_runs",
            ".review_repairs",
            "acceptance",
            "acceptance_reports",
            "Result",
        )
    )


try:
    _LIVE_STATION_DATA_ROOTS = _live_station_data_roots()
except _WorkspaceGuardError as exc:
    raise pytest.UsageError(f"Unable to protect live station data: {exc}") from exc


def _entry_state(path: Path, *, kind: str, digest: str | None) -> _PathState:
    """Read portable metadata without following filesystem links."""
    try:
        details = path.lstat()
    except OSError as exc:
        raise _WorkspaceGuardError(f"Unable to stat guarded path {path}: {exc}") from exc
    return _PathState(
        kind=kind,
        mode=details.st_mode,
        size=details.st_size,
        mtime_ns=details.st_mtime_ns,
        ctime_ns=details.st_ctime_ns,
        device=details.st_dev,
        inode=details.st_ino,
        sha256=digest,
    )


def _file_state_with_sha256(path: Path) -> _PathState:
    """Hash one file and reject a concurrent mutation during the read."""
    before = _entry_state(path, kind="file", digest=None)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise _WorkspaceGuardError(f"Unable to hash guarded file {path}: {exc}") from exc
    after = _entry_state(path, kind="file", digest=None)
    if before != after:
        raise _WorkspaceGuardError(f"Guarded file changed while its snapshot was read: {path}")
    return replace(after, sha256=digest.hexdigest())


def _is_access_denied(exc: OSError) -> bool:
    """Recognize portable PermissionError and native Windows access denial."""
    return isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5


def _snapshot_tree(root: Path) -> _TreeSnapshot:
    """Recursively snapshot relative paths, contents, and filesystem metadata."""
    snapshot: _TreeSnapshot = {}
    try:
        root.lstat()
    except FileNotFoundError:
        snapshot["."] = _PathState("missing", 0, 0, 0, 0, 0, 0, None)
        return snapshot
    except OSError as exc:
        raise _WorkspaceGuardError(f"Unable to stat guarded root {root}: {exc}") from exc

    def visit(path: Path, relative_path: str) -> None:
        try:
            details = path.lstat()
        except OSError as exc:
            raise _WorkspaceGuardError(f"Unable to stat guarded path {path}: {exc}") from exc

        if stat.S_ISLNK(details.st_mode):
            before = _entry_state(path, kind="symlink", digest=None)
            try:
                target = os.readlink(path)
            except OSError as exc:
                raise _WorkspaceGuardError(f"Unable to read guarded symlink {path}: {exc}") from exc
            after = _entry_state(path, kind="symlink", digest=None)
            if before != after:
                raise _WorkspaceGuardError(f"Guarded symlink changed while its snapshot was read: {path}")
            digest = hashlib.sha256(os.fsencode(target)).hexdigest()
            snapshot[relative_path] = replace(after, sha256=digest)
            return
        if stat.S_ISREG(details.st_mode):
            snapshot[relative_path] = _file_state_with_sha256(path)
            return
        if not stat.S_ISDIR(details.st_mode):
            snapshot[relative_path] = _entry_state(
                path,
                kind="other",
                digest=None,
            )
            return

        initial_state = _entry_state(path, kind="directory", digest=None)
        snapshot[relative_path] = initial_state
        try:
            with os.scandir(path) as iterator:
                children = sorted(iterator, key=lambda entry: entry.name)
        except OSError as exc:
            if _is_access_denied(exc):
                snapshot[relative_path] = replace(
                    initial_state,
                    kind="opaque_directory",
                )
                return
            raise _WorkspaceGuardError(f"Unable to enumerate guarded directory {path}: {exc}") from exc
        initial_names = tuple(child.name for child in children)
        for child in children:
            child_relative = child.name if relative_path == "." else f"{relative_path}/{child.name}"
            visit(Path(child.path), child_relative)
        try:
            with os.scandir(path) as iterator:
                final_names = tuple(sorted(entry.name for entry in iterator))
        except OSError as exc:
            raise _WorkspaceGuardError(f"Unable to re-enumerate guarded directory {path}: {exc}") from exc
        final_state = _entry_state(path, kind="directory", digest=None)
        if initial_names != final_names or initial_state != final_state:
            raise _WorkspaceGuardError(f"Guarded directory changed while its snapshot was read: {path}")

    visit(root, ".")
    return snapshot


def _snapshot_roots(roots: tuple[Path, ...]) -> _RootsSnapshot:
    """Capture every configured live root; unreadable roots fail closed."""
    return {root: _snapshot_tree(root) for root in roots}


def _describe_snapshot_changes(
    before: _TreeSnapshot,
    after: _TreeSnapshot,
) -> tuple[str, ...]:
    """Return deterministic relative-path diagnostics for two snapshots."""
    before_paths = set(before)
    after_paths = set(after)
    changes = [f"added: {path}" for path in sorted(after_paths - before_paths)]
    changes.extend(f"removed: {path}" for path in sorted(before_paths - after_paths))
    changes.extend(f"modified: {path}" for path in sorted(before_paths & after_paths) if before[path] != after[path])
    return tuple(changes)


def _describe_root_changes(
    before: _RootsSnapshot,
    after: _RootsSnapshot,
) -> tuple[str, ...]:
    """Attach each protected root to its recursive change diagnostics."""
    changes: list[str] = []
    roots = sorted(set(before) | set(after), key=str)
    for root in roots:
        if root not in before:
            changes.append(f"added guarded root: {root}")
            continue
        if root not in after:
            changes.append(f"removed guarded root: {root}")
            continue
        changes.extend(f"{root}: {change}" for change in _describe_snapshot_changes(before[root], after[root]))
    return tuple(changes)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    """Snapshot live station data before collection imports test modules."""
    try:
        session.config.stash[_LIVE_SNAPSHOT_KEY] = _snapshot_roots(_LIVE_STATION_DATA_ROOTS)
    except _WorkspaceGuardError as exc:
        raise pytest.UsageError(f"Unable to snapshot live station data before tests: {exc}") from exc


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the session if a live station-data path or its content changed."""
    del exitstatus
    before = session.config.stash.get(_LIVE_SNAPSHOT_KEY, None)
    if before is None:
        session.exitstatus = int(pytest.ExitCode.TESTS_FAILED)
        return
    try:
        after = _snapshot_roots(_LIVE_STATION_DATA_ROOTS)
    except _WorkspaceGuardError as exc:
        changes = (f"snapshot failed: {exc}",)
    else:
        changes = _describe_root_changes(before, after)
    if not changes:
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    details = "\n".join(f"  - {change}" for change in changes[:50])
    if len(changes) > 50:
        details += f"\n  - ... {len(changes) - 50} additional changes"
    message = f"Live station data changed during pytest; refusing a successful result:\n{details}"
    if reporter is not None:
        reporter.write_sep("=", "LIVE STATION DATA INTEGRITY FAILURE")
        reporter.write_line(message)
    session.exitstatus = int(pytest.ExitCode.TESTS_FAILED)


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch, tmp_path_factory):
    """
    Automatically add the system temporary directory to the allowed roots
    of the global path_validator for all tests.
    This prevents 'SecurityError' when tests use 'tmp_path' fixture to create config files.
    """
    try:
        from core.security import path_validator

        # Get system temp dir (where pytest tmp_path lives)
        # On Windows: C:\Users\ADMIN~1\AppData\Local\Temp
        # On Linux: /tmp
        temp_dir = _validated_external_basetemp(
            tempfile.gettempdir(),
            source="system temporary directory",
        )

        # ``--basetemp`` may intentionally live on another disk when the system
        # temp volume cannot satisfy production's free-space reserve.  Permit
        # only this pytest session's isolated root; production roots are not
        # changed and workspace-local basetemp is rejected in pytest_configure.
        pytest_temp_dir = _validated_external_basetemp(
            tmp_path_factory.getbasetemp(),
            source="actual pytest basetemp",
        )
        current_roots = list(path_validator.allowed_roots)
        for allowed_temp_dir in (temp_dir, pytest_temp_dir):
            if allowed_temp_dir not in current_roots:
                current_roots.append(allowed_temp_dir)

        monkeypatch.setattr(path_validator, "allowed_roots", current_roots)

    except ImportError:
        pass
