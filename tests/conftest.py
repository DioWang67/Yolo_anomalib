# tests/conftest.py
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    resolved = Path(basetemp).expanduser().resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "workspace.yaml").is_file():
            raise pytest.UsageError(
                f"--basetemp={resolved} lies inside the workspace at {parent}. "
                "Workspace discovery would resolve every test's tmp_path to the "
                "real station data and let tests publish into production "
                "evidence. Choose a basetemp outside the workspace, such as one "
                "under the system temporary directory."
            )


def _real_station_data_dirs() -> tuple[Path, ...]:
    """Locate the live station-data directories a test must never write into.

    ``load_workspace_paths()`` finds a workspace by walking *upwards* for
    ``workspace.yaml``. A pytest ``--basetemp`` inside the workspace therefore
    resolves a test's own ``tmp_path`` to the real station data, and any code
    that rediscovers paths from a caller-supplied root -- as the release builder
    does from ``models_root.parent`` -- publishes test artifacts into production
    evidence. That happened: a stub color model reached the real profile store
    and was offered as a selectable acceptance variant until it failed on every
    image of a 250-sample matrix combination.
    """
    try:
        from core.station_data import load_station_data_paths

        paths = load_station_data_paths(ROOT)
    except Exception:  # noqa: BLE001 - a guard must never break collection
        return ()
    return tuple(
        directory
        for directory in (
            paths.color_baselines,
            paths.color_profiles,
            paths.color_revisions,
            paths.inspection_releases,
            paths.acceptance,
        )
        if directory.is_dir()
    )


_REAL_STATION_DATA_DIRS = _real_station_data_dirs()


def _station_data_entries() -> dict[Path, frozenset[str]]:
    return {
        directory: frozenset(entry.name for entry in directory.iterdir())
        for directory in _REAL_STATION_DATA_DIRS
        if directory.is_dir()
    }


@pytest.fixture(autouse=True)
def forbid_writes_to_real_station_data():
    """Fail the test that publishes into live station data, not a later audit.

    Checked per test rather than once per session so the failure names the
    culprit. Reading a handful of small directories twice costs far less than
    finding the leak weeks later from a manifest's recorded source path.
    """
    if not _REAL_STATION_DATA_DIRS:
        yield
        return
    before = _station_data_entries()
    yield
    for directory, names in _station_data_entries().items():
        created = sorted(names - before.get(directory, frozenset()))
        if created:
            raise AssertionError(
                "Test wrote into live station data "
                f"{directory}: {', '.join(created)}. Pass an explicit "
                "tmp_path-based root instead of letting workspace discovery "
                "walk up into the real workspace."
            )


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch):
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
        temp_dir = Path(tempfile.gettempdir()).resolve()

        # Add to allowed roots
        current_roots = list(path_validator.allowed_roots)
        current_roots.append(temp_dir)

        # Also add the specific pytest temp root if possible,
        # but system temp should cover it as long as validator establishes parenthood correctly

        monkeypatch.setattr(path_validator, "allowed_roots", current_roots)

    except ImportError:
        pass
