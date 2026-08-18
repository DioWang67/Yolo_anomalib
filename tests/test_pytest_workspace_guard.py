import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import conftest as pytest_config
import pytest
from conftest import (
    _ISOLATED_WORKSPACE_ROOT,
    _LIVE_WORKSPACE_ROOT,
    _WORKSPACE_ENV_VAR,
    ROOT,
    _assert_safe_workspace_environment,
    _describe_snapshot_changes,
    _install_default_workspace_isolation,
    _live_station_data_roots,
    _snapshot_tree,
    _validated_external_basetemp,
)

WORKSPACE_MANIFEST = """schema_version: 1
projects:
  training: Yolo11_auto_train
  inference: yolo11_inference
paths:
  training_data: Yolo11_auto_train/data
  inference_models: yolo11_inference/models
  station_data: station_data/yolo11_inference
  inference_results: Result
  inference_artifacts: release_artifacts/yolo11_inference
"""


def _write_workspace_manifest(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "workspace.yaml").write_text(WORKSPACE_MANIFEST, encoding="utf-8")


def test_actual_pytest_basetemp_guard_rejects_workspace_descendant(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")

    with pytest.raises(pytest.UsageError, match="lies inside the workspace"):
        _validated_external_basetemp(
            workspace / ".tmp" / "pytest",
            source="actual pytest basetemp",
        )


def test_actual_pytest_basetemp_guard_accepts_external_directory(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external" / "pytest"

    assert (
        _validated_external_basetemp(
            external,
            source="actual pytest basetemp",
        )
        == external.resolve()
    )


@pytest.mark.parametrize(
    "start",
    (ROOT, _LIVE_WORKSPACE_ROOT, _LIVE_WORKSPACE_ROOT / "station_data"),
)
def test_checkout_discovery_is_routed_to_isolated_workspace(start: Path) -> None:
    from core.workspace import load_workspace_paths

    paths = load_workspace_paths(start)

    assert Path(os.environ[_WORKSPACE_ENV_VAR]).resolve() == _ISOLATED_WORKSPACE_ROOT
    assert paths.root == _ISOLATED_WORKSPACE_ROOT
    assert not paths.station_data.is_relative_to(_LIVE_WORKSPACE_ROOT)


def test_preimported_station_data_uses_installed_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import station_data, workspace

    stale_station_alias = station_data.load_workspace_paths
    with monkeypatch.context() as context:
        context.setattr(
            workspace,
            "load_workspace_paths",
            workspace.load_workspace_paths,
        )
        context.setattr(
            station_data,
            "load_workspace_paths",
            station_data.load_workspace_paths,
        )
        _install_default_workspace_isolation(_ISOLATED_WORKSPACE_ROOT)

        assert station_data.load_workspace_paths is workspace.load_workspace_paths
        assert station_data.load_workspace_paths is not stale_station_alias
        assert station_data.load_station_data_paths(ROOT).root == (
            _ISOLATED_WORKSPACE_ROOT / "station_data" / "yolo11_inference"
        )


def test_dynamic_live_workspace_environment_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.workspace import load_workspace_paths

    monkeypatch.setenv(_WORKSPACE_ENV_VAR, str(_LIVE_WORKSPACE_ROOT))

    with pytest.raises(pytest.UsageError, match="Refusing unsafe"):
        load_workspace_paths(ROOT)


def test_standalone_guard_excludes_runtime_diagnostic_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pytest_config, "_LIVE_WORKSPACE_MANIFEST", None)

    guarded_roots = _live_station_data_roots()

    assert ROOT / "logs" not in guarded_roots
    assert ROOT / "Result" in guarded_roots


@pytest.mark.parametrize(
    "unsafe_root",
    (
        _LIVE_WORKSPACE_ROOT,
        _LIVE_WORKSPACE_ROOT / "station_data",
        _LIVE_WORKSPACE_ROOT / "workspace.yaml",
    ),
)
def test_workspace_environment_rejects_live_paths(unsafe_root: Path) -> None:
    with pytest.raises(pytest.UsageError, match="[Uu]nsafe"):
        _assert_safe_workspace_environment(
            {_WORKSPACE_ENV_VAR: str(unsafe_root)},
            live_workspace_root=_LIVE_WORKSPACE_ROOT,
        )


def test_workspace_environment_accepts_valid_isolated_root(tmp_path: Path) -> None:
    isolated_root = tmp_path / "isolated"
    _write_workspace_manifest(isolated_root)

    assert (
        _assert_safe_workspace_environment(
            {_WORKSPACE_ENV_VAR: str(isolated_root)},
            live_workspace_root=_LIVE_WORKSPACE_ROOT,
        )
        == isolated_root.resolve()
    )


def test_workspace_environment_rejects_manifest_symlink_to_live_workspace(
    tmp_path: Path,
) -> None:
    isolated_root = tmp_path / "isolated"
    _write_workspace_manifest(isolated_root)
    station_link = isolated_root / "station_data"
    try:
        station_link.symlink_to(
            _LIVE_WORKSPACE_ROOT / "station_data",
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    with pytest.raises(pytest.UsageError, match="[Uu]nsafe"):
        _assert_safe_workspace_environment(
            {_WORKSPACE_ENV_VAR: str(isolated_root)},
            live_workspace_root=_LIVE_WORKSPACE_ROOT,
        )


def test_recursive_snapshot_detects_same_size_existing_file_overwrite(
    tmp_path: Path,
) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_file = guarded_root / "acceptance" / "nested" / "result.json"
    guarded_file.parent.mkdir(parents=True)
    guarded_file.write_bytes(b"original")
    original_stat = guarded_file.stat()
    before = _snapshot_tree(guarded_root)

    guarded_file.write_bytes(b"tampered")
    os.utime(
        guarded_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
    )
    after = _snapshot_tree(guarded_root)

    assert before["acceptance/nested/result.json"].sha256 != after["acceptance/nested/result.json"].sha256
    assert "modified: acceptance/nested/result.json" in _describe_snapshot_changes(
        before,
        after,
    )


def test_recursive_snapshot_detects_nested_addition(tmp_path: Path) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_root.mkdir()
    before = _snapshot_tree(guarded_root)

    nested_file = guarded_root / "acceptance" / "new-run" / "report.json"
    nested_file.parent.mkdir(parents=True)
    nested_file.write_text("{}", encoding="utf-8")
    after = _snapshot_tree(guarded_root)

    assert "added: acceptance/new-run/report.json" in _describe_snapshot_changes(
        before,
        after,
    )


def test_recursive_snapshot_detects_metadata_only_change(tmp_path: Path) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_file = guarded_root / "result.json"
    guarded_root.mkdir()
    guarded_file.write_text("{}", encoding="utf-8")
    before = _snapshot_tree(guarded_root)

    original_stat = guarded_file.stat()
    os.utime(
        guarded_file,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )
    after = _snapshot_tree(guarded_root)

    assert before["result.json"].sha256 == after["result.json"].sha256
    assert "modified: result.json" in _describe_snapshot_changes(before, after)


def test_recursive_snapshot_records_permission_denied_directory_as_opaque(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_root = tmp_path / "station_data"
    opaque_directory = guarded_root / "acceptance" / "opaque-run"
    opaque_directory.mkdir(parents=True)
    (opaque_directory / "result.json").write_text("{}", encoding="utf-8")
    original_scandir = os.scandir

    def deny_opaque_directory(path: str | os.PathLike[str]):
        if Path(path).resolve() == opaque_directory.resolve():
            raise PermissionError(13, "simulated access denial", str(path))
        return original_scandir(path)

    monkeypatch.setattr(pytest_config.os, "scandir", deny_opaque_directory)
    before = _snapshot_tree(guarded_root)

    original_stat = opaque_directory.stat()
    os.utime(
        opaque_directory,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )
    after = _snapshot_tree(guarded_root)

    opaque_path = "acceptance/opaque-run"
    assert before[opaque_path].kind == "opaque_directory"
    assert after[opaque_path].kind == "opaque_directory"
    assert f"modified: {opaque_path}" in _describe_snapshot_changes(before, after)


def test_session_finish_fails_when_guarded_tree_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_root = tmp_path / "station_data"
    guarded_root.mkdir()
    before = _snapshot_tree(guarded_root)
    (guarded_root / "nested").mkdir()

    monkeypatch.setattr(
        pytest_config,
        "_LIVE_STATION_DATA_ROOTS",
        (guarded_root,),
    )
    session = SimpleNamespace(
        config=SimpleNamespace(
            stash={pytest_config._LIVE_SNAPSHOT_KEY: {guarded_root: before}},
            pluginmanager=SimpleNamespace(get_plugin=lambda name: None),
        ),
        exitstatus=int(pytest.ExitCode.OK),
    )

    pytest_config.pytest_sessionfinish(session, int(pytest.ExitCode.OK))

    assert session.exitstatus == int(pytest.ExitCode.TESTS_FAILED)


def test_pytest_startup_rejects_live_workspace_environment() -> None:
    environment = os.environ.copy()
    environment[_WORKSPACE_ENV_VAR] = str(_LIVE_WORKSPACE_ROOT)
    environment.pop("PYTEST_ADDOPTS", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            str(Path(__file__).resolve()),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == int(pytest.ExitCode.USAGE_ERROR), output
    assert f"Refusing unsafe {_WORKSPACE_ENV_VAR}" in output
