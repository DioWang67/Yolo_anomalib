"""Environment checks must degrade, never crash.

Every test here removes something a real station might be missing — psutil,
nvidia-smi, torch, a package, RAM, disk — and asserts the checker still returns
a result describing the gap.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

import pytest

from tools.system_check import context as context_module
from tools.system_check import sysinfo
from tools.system_check.checks import (
    CHECK_REGISTRY,
    dependency_check,
    disk_check,
    gpu_check,
    memory_check,
    runtime_check,
)
from tools.system_check.context import AppContext, discover_model_targets, resolve_context
from tools.system_check.results import CheckResult, Status, overall_label, overall_status, run_check
from tools.system_check.spec import VCRUNTIME_DLLS
from tools.system_check.sysinfo import BYTES_PER_GB, BYTES_PER_MB, DiskInfo, MemoryInfo


@pytest.fixture
def app_root(tmp_path: Path) -> Path:
    """A minimal installation layout the checker can resolve."""
    (tmp_path / "models" / "Widget" / "A" / "yolo" / "weights").mkdir(parents=True)
    (tmp_path / "Runtime").mkdir()
    (tmp_path / "config.yaml").write_text(
        "enable_anomalib: false\noutput_dir: ./Result\nwidth: 1280\nheight: 960\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def context(app_root: Path) -> AppContext:
    """A resolved context pointing at the temporary installation."""
    return resolve_context(app_root)


# --------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------


def test_run_check_isolates_exceptions() -> None:
    """A check that raises becomes UNKNOWN instead of ending the run."""

    def exploding() -> CheckResult:
        raise RuntimeError("driver query failed")

    results = run_check("boom", "Exploding check", exploding)

    assert len(results) == 1
    assert results[0].status is Status.UNKNOWN
    assert "RuntimeError" in results[0].detail
    assert "driver query failed" in results[0].detail


def test_run_check_does_not_swallow_keyboard_interrupt() -> None:
    """The operator must still be able to abort the tool."""

    def aborting() -> CheckResult:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_check("abort", "Aborting check", aborting)


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([Status.PASS, Status.SKIP], Status.PASS),
        ([Status.PASS, Status.WARNING], Status.WARNING),
        ([Status.PASS, Status.UNKNOWN], Status.WARNING),
        ([Status.WARNING, Status.FAIL], Status.FAIL),
    ],
)
def test_overall_status_rollup(statuses: list[Status], expected: Status) -> None:
    """An undetermined check degrades the verdict; it is not evidence of a pass."""
    results = [
        CheckResult(check_id=f"c{index}", title="t", status=status, detail="")
        for index, status in enumerate(statuses)
    ]
    assert overall_status(results) is expected


def test_overall_label_wording() -> None:
    assert overall_label(Status.PASS) == "PASS"
    assert overall_label(Status.WARNING) == "PASS WITH WARNINGS"
    assert overall_label(Status.FAIL) == "FAIL"


# --------------------------------------------------------------------------
# Every check survives a hostile environment
# --------------------------------------------------------------------------


def test_every_check_runs_without_psutil(monkeypatch: pytest.MonkeyPatch, context: AppContext) -> None:
    """With psutil, nvidia-smi and torch all absent, no check may raise."""
    monkeypatch.setattr(sysinfo, "_psutil", lambda: None)
    monkeypatch.setattr(sysinfo, "run_command", lambda *a, **k: (-1, "", "not found"))
    monkeypatch.setattr(gpu_check, "run_command", lambda *a, **k: (-1, "", "not found"))

    real_import = importlib.import_module

    def blocked(name: str, *args, **kwargs):
        if name in {"torch", "psutil", "cpuinfo"}:
            raise ImportError(f"{name} is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", blocked)
    monkeypatch.setattr(gpu_check.importlib, "import_module", blocked)

    collected: list[CheckResult] = []
    for check_id, title, func in CHECK_REGISTRY:

        def invoke(bound=func) -> list[CheckResult]:
            return list(bound(context))

        collected.extend(run_check(check_id, title, invoke))

    assert collected
    # An isolated crash would show up as an UNKNOWN naming the exception type.
    crashed = [item for item in collected if "Check raised" in item.detail]
    assert crashed == []


# --------------------------------------------------------------------------
# GPU / CUDA absence
# --------------------------------------------------------------------------


def test_missing_nvidia_smi_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """This project needs no GPU, so a missing driver tool must not fail it."""
    monkeypatch.setattr(
        gpu_check, "run_command", lambda *a, **k: (-1, "", "nvidia-smi not found on PATH")
    )
    results = gpu_check.check(context)
    inventory = next(item for item in results if item.check_id == "gpu.present")

    assert inventory.status is Status.PASS
    assert "does not need one" in inventory.detail


def test_cuda_unavailable_reported_from_torch_not_the_driver(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """A present driver with a CPU-only torch is the expected state, not a fault."""
    monkeypatch.setattr(
        gpu_check,
        "run_command",
        lambda *a, **k: (0, "NVIDIA RTX 3060, 560.00, 12288, 900\n", ""),
    )
    monkeypatch.setattr(
        gpu_check,
        "_torch_cuda",
        lambda: {
            "importable": True,
            "version": "2.4.1+cpu",
            "built_with_cuda": None,
            "is_available": False,
        },
    )

    results = gpu_check.check(context)
    cuda = next(item for item in results if item.check_id == "gpu.cuda")

    assert cuda.status is Status.PASS
    assert "CPU-only build" in cuda.detail
    assert "torch.cuda.is_available()=False" in (cuda.measured or "")


def test_torch_missing_skips_cuda_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """The standalone checker has no torch; that is a SKIP, not a FAIL."""
    monkeypatch.setattr(gpu_check, "run_command", lambda *a, **k: (-1, "", "absent"))
    monkeypatch.setattr(
        gpu_check, "_torch_cuda", lambda: {"importable": False, "error": "no module"}
    )
    results = gpu_check.check(context)
    cuda = next(item for item in results if item.check_id == "gpu.cuda")
    assert cuda.status is Status.SKIP


# --------------------------------------------------------------------------
# Low memory
# --------------------------------------------------------------------------


def test_low_available_memory_fails_against_the_configured_queue(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """Below the station's own image-queue ceiling is a measured failure."""
    monkeypatch.setattr(
        memory_check,
        "memory_info",
        lambda: MemoryInfo(total=4 * BYTES_PER_GB, available=100 * BYTES_PER_MB),
    )
    results = memory_check.check(context)
    available = next(item for item in results if item.check_id == "memory.available")

    assert available.status is Status.FAIL
    assert "256 MB" in (available.requirement or "")


def test_ample_memory_passes(monkeypatch: pytest.MonkeyPatch, context: AppContext) -> None:
    monkeypatch.setattr(
        memory_check,
        "memory_info",
        lambda: MemoryInfo(total=32 * BYTES_PER_GB, available=24 * BYTES_PER_GB),
    )
    results = memory_check.check(context)
    assert all(item.status is Status.PASS for item in results)


def test_unreadable_memory_is_unknown_not_zero(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    monkeypatch.setattr(memory_check, "memory_info", lambda: MemoryInfo(None, None))
    results = memory_check.check(context)
    assert results[0].status is Status.UNKNOWN


# --------------------------------------------------------------------------
# Low disk
# --------------------------------------------------------------------------


def test_low_disk_fails_at_the_stated_threshold(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """min_free_disk_mb is a real requirement, so below it is a FAIL."""
    monkeypatch.setattr(
        disk_check,
        "disk_info",
        lambda path: DiskInfo(str(path), "D:\\", 500 * BYTES_PER_GB, 200 * BYTES_PER_MB),
    )
    results = disk_check.check(context)

    assert results[0].status is Status.FAIL
    assert "1024 MB" in (results[0].requirement or "")


def test_thin_disk_margin_warns(monkeypatch: pytest.MonkeyPatch, context: AppContext) -> None:
    monkeypatch.setattr(
        disk_check,
        "disk_info",
        lambda path: DiskInfo(str(path), "D:\\", 500 * BYTES_PER_GB, 3 * BYTES_PER_GB),
    )
    assert disk_check.check(context)[0].status is Status.WARNING


def test_unreadable_disk_is_unknown(monkeypatch: pytest.MonkeyPatch, context: AppContext) -> None:
    monkeypatch.setattr(
        disk_check, "disk_info", lambda path: DiskInfo(str(path), "Z:\\", None, None)
    )
    assert disk_check.check(context)[0].status is Status.UNKNOWN


def test_station_override_of_min_free_disk_is_respected(
    monkeypatch: pytest.MonkeyPatch, app_root: Path
) -> None:
    """A station that raised min_free_disk_mb is judged against its own value."""
    (app_root / "config.yaml").write_text(
        "min_free_disk_mb: 20480\noutput_dir: ./Result\n", encoding="utf-8"
    )
    context = resolve_context(app_root)
    monkeypatch.setattr(
        disk_check,
        "disk_info",
        lambda path: DiskInfo(str(path), "D:\\", 500 * BYTES_PER_GB, 10 * BYTES_PER_GB),
    )
    result = disk_check.check(context)[0]
    assert result.status is Status.FAIL
    assert "20480 MB" in (result.requirement or "")


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------


def test_missing_blocking_package_fails(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """The five distributions start.bat gates on must FAIL when absent."""
    monkeypatch.setattr(dependency_check, "_interpreter_has_app", lambda: True)
    monkeypatch.setattr(dependency_check, "_bundle_versions", lambda ctx: ({}, {}))
    monkeypatch.setattr(dependency_check, "_import_results", list)
    monkeypatch.setattr(
        dependency_check,
        "_interpreter_versions",
        lambda: {"torch": "2.4.1", "torchvision": "0.19.1", "pyqt5": "5.15.11"},
    )

    results = dependency_check.check(context)
    by_id = {item.check_id: item for item in results}

    assert by_id["deps.version.onnxruntime"].status is Status.FAIL
    assert by_id["deps.version.jsonargparse"].status is Status.FAIL
    assert by_id["deps.version.torch"].status is Status.PASS


def test_advisory_drift_warns_but_does_not_fail(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """The launcher does not block on numpy drift, so neither does this tool."""
    monkeypatch.setattr(dependency_check, "_interpreter_has_app", lambda: True)
    monkeypatch.setattr(dependency_check, "_bundle_versions", lambda ctx: ({}, {}))
    monkeypatch.setattr(dependency_check, "_import_results", list)
    monkeypatch.setattr(
        dependency_check,
        "_interpreter_versions",
        lambda: {
            "torch": "2.4.1",
            "torchvision": "0.19.1",
            "onnxruntime": "1.23.2",
            "pyqt5": "5.15.11",
            "jsonargparse": "4.34.0",
            "numpy": "2.1.0",
        },
    )

    by_id = {item.check_id: item for item in dependency_check.check(context)}
    assert by_id["deps.version.numpy"].status is Status.WARNING
    assert "2.1.0" in (by_id["deps.version.numpy"].measured or "")


def test_blocking_version_mismatch_fails(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    monkeypatch.setattr(dependency_check, "_interpreter_has_app", lambda: True)
    monkeypatch.setattr(dependency_check, "_bundle_versions", lambda ctx: ({}, {}))
    monkeypatch.setattr(dependency_check, "_import_results", list)
    monkeypatch.setattr(
        dependency_check, "_interpreter_versions", lambda: {"torch": "2.6.0"}
    )
    by_id = {item.check_id: item for item in dependency_check.check(context)}
    assert by_id["deps.version.torch"].status is Status.FAIL
    assert "start.bat" in by_id["deps.version.torch"].detail


def test_bundle_mode_reports_unversioned_packages_as_unknown(app_root: Path) -> None:
    """A packaged build without metadata is "unknown version", not "missing"."""
    internal = app_root / "_internal"
    (internal / "torch-2.4.1.dist-info").mkdir(parents=True)
    (internal / "cv2").mkdir()

    by_id = {item.check_id: item for item in dependency_check.check(resolve_context(app_root))}

    assert by_id["deps.mode"].measured == "bundle"
    assert by_id["deps.version.torch"].status is Status.PASS
    assert by_id["deps.version.opencv-python"].status is Status.UNKNOWN
    assert by_id["python.version"].status is Status.SKIP


def test_bundle_mode_never_claims_a_package_is_missing(app_root: Path) -> None:
    """Pure-Python packages sit inside the PYZ, invisible on disk.

    openpyxl and pyserial are bundled exactly that way in the real build, so
    treating "not on disk" as "missing" would raise a false alarm on every
    packaged deployment.
    """
    (app_root / "_internal" / "onnxruntime-1.23.2.dist-info").mkdir(parents=True)

    results = dependency_check.check(resolve_context(app_root))
    absent = next(item for item in results if item.check_id == "deps.version.pyserial")

    assert absent.status is Status.UNKNOWN
    assert "not evidence of absence" in absent.detail
    assert not [item for item in results if item.status is Status.FAIL]


def test_packaged_target_is_read_from_the_bundle_even_from_a_dev_environment(
    app_root: Path,
) -> None:
    """The target decides the mode; a dev interpreter must not answer for it."""
    internal = app_root / "_internal"
    (internal / "torch-9.9.9.dist-info").mkdir(parents=True)

    by_id = {item.check_id: item for item in dependency_check.check(resolve_context(app_root))}

    assert by_id["deps.mode"].measured == "bundle"
    # The packaged build's version, not whatever this test environment has.
    assert by_id["deps.version.torch"].measured == "9.9.9"


def test_frozen_checker_against_a_source_checkout_claims_nothing(
    monkeypatch: pytest.MonkeyPatch, app_root: Path
) -> None:
    """The standalone exe must not report its own bundle as the application's.

    Its bundle carries onnxruntime, numpy and cv2 for the benchmark but no
    torch or PyQt5. Trusting it here would emit a FAIL for every pin the
    checker does not happen to carry.
    """
    monkeypatch.setattr(dependency_check.sys, "frozen", True, raising=False)
    results = dependency_check.check(resolve_context(app_root))
    by_id = {item.check_id: item for item in results}

    assert by_id["deps.mode"].measured == "unavailable"
    assert by_id["deps.mode"].status is Status.UNKNOWN
    assert by_id["python.version"].status is Status.UNKNOWN
    # No per-package verdicts at all, rather than invented failures.
    assert not [item for item in results if item.check_id.startswith("deps.version.")]
    assert not [item for item in results if item.check_id.startswith("deps.import.")]
    assert all(item.status is not Status.FAIL for item in results)


def test_anomalib_packages_skipped_only_when_absent(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """With the backend off and the packages gone, there is nothing to check."""
    monkeypatch.setattr(dependency_check, "_interpreter_has_app", lambda: True)
    monkeypatch.setattr(dependency_check, "_bundle_versions", lambda ctx: ({}, {}))
    monkeypatch.setattr(dependency_check, "_import_results", list)
    monkeypatch.setattr(dependency_check, "_interpreter_versions", lambda: {"torch": "2.4.1"})

    by_id = {item.check_id: item for item in dependency_check.check(context)}
    assert by_id["deps.anomalib"].status is Status.SKIP
    assert "deps.version.kornia" not in by_id


def test_anomalib_drift_is_reported_even_when_the_backend_is_disabled(
    monkeypatch: pytest.MonkeyPatch, context: AppContext
) -> None:
    """Skipping on the flag hid a real drift in a shipped artifact.

    yolo11_inference.spec collects anomalib into every build regardless of
    enable_anomalib, so the artifact carries whatever the build machine had.
    The production environment shipped kornia 0.6.9 against a pinned 0.8.1 and
    nothing caught it, because this station has the backend switched off.
    """
    assert context.anomalib_enabled() is False
    monkeypatch.setattr(dependency_check, "_interpreter_has_app", lambda: True)
    monkeypatch.setattr(dependency_check, "_bundle_versions", lambda ctx: ({}, {}))
    monkeypatch.setattr(dependency_check, "_import_results", list)
    monkeypatch.setattr(
        dependency_check,
        "_interpreter_versions",
        lambda: {"kornia": "0.6.9", "timm": "0.6.12", "anomalib": "1.2.0"},
    )

    by_id = {item.check_id: item for item in dependency_check.check(context)}

    assert by_id["deps.anomalib"].status is Status.PASS
    assert "collected into every build" in by_id["deps.anomalib"].detail
    assert by_id["deps.version.kornia"].status is Status.WARNING
    assert by_id["deps.version.kornia"].measured == "0.6.9"
    assert by_id["deps.version.timm"].status is Status.WARNING
    assert by_id["deps.version.anomalib"].status is Status.PASS


def test_corrupt_package_metadata_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken dist-info must not take down the whole dependency check."""

    def exploding(name: str) -> str:
        if name == "torch":
            raise ValueError("corrupt METADATA")
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", exploding)
    versions = dependency_check._interpreter_versions()
    assert versions["torch"] is None


# --------------------------------------------------------------------------
# Visual C++ runtimes
# --------------------------------------------------------------------------


def test_shipped_crt_in_runtime_is_not_reported_missing(
    monkeypatch: pytest.MonkeyPatch, app_root: Path
) -> None:
    """The VC120 CRT ships inside Runtime, so searching only the system lies.

    Windows resolves a dependent DLL from the loading module's own directory
    before the system path, which is why MvCameraControl.dll loads fine on a
    machine with no VC++ 2013 redistributable. Warning there is a false alarm.
    """
    for name in ("msvcr120.dll", "msvcp120.dll"):
        (app_root / "Runtime" / name).write_bytes(b"stub")

    def always_missing(name: str) -> object:
        raise OSError(f"{name} not found on the system path")

    monkeypatch.setattr(runtime_check, "_windll", always_missing)
    monkeypatch.setattr(runtime_check.os, "name", "nt")

    results = runtime_check._vcredist_results(resolve_context(app_root))
    by_id = {item.check_id: item for item in results}

    assert by_id["runtime.vc120"].status is Status.PASS
    assert "ship with the SDK" in by_id["runtime.vc120"].detail
    assert by_id["runtime.vc120"].data["from_bundle"] == ["msvcr120.dll", "msvcp120.dll"]
    # onnxruntime does not ship its CRT, so that one is still a real failure.
    assert by_id["runtime.vcredist"].status is Status.FAIL


def test_crt_missing_from_both_system_and_runtime_fails(
    monkeypatch: pytest.MonkeyPatch, app_root: Path
) -> None:
    def always_missing(name: str) -> object:
        raise OSError(f"{name} not found")

    monkeypatch.setattr(runtime_check, "_windll", always_missing)
    monkeypatch.setattr(runtime_check.os, "name", "nt")

    by_id = {
        item.check_id: item
        for item in runtime_check._vcredist_results(resolve_context(app_root))
    }
    assert by_id["runtime.vc120"].status is Status.FAIL
    assert "Restore the missing CRT files" in (by_id["runtime.vc120"].remedy or "")


def test_vcruntime_list_matches_onnxruntime_pe_imports() -> None:
    """Pinned to what onnxruntime actually imports, msvcp140_1 included."""
    assert set(VCRUNTIME_DLLS) == {
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
    }


# --------------------------------------------------------------------------
# Context resolution
# --------------------------------------------------------------------------


def test_frozen_checker_finds_a_sibling_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Double-clicking the exe must find the app without --app-root.

    The checker ships as its own folder, so the installation is a sibling, not
    the folder the exe sits in. Requiring --app-root here would defeat the
    point of an operator just double-clicking it.
    """
    deployment = tmp_path / "deployment"
    checker = deployment / "system_check"
    checker.mkdir(parents=True)
    (checker / "_internal").mkdir()
    app = deployment / "yolo11_inference"
    (app / "_internal" / "Runtime").mkdir(parents=True)
    (app / "models").mkdir()
    (app / "yolo11_inference.exe").write_bytes(b"stub")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(checker / "system_check.exe"), raising=False)
    monkeypatch.delenv("YOLO11_ROOT", raising=False)
    monkeypatch.chdir(checker)

    assert resolve_context().app_root == app.resolve()


def test_nearby_search_tolerates_an_unreadable_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permission error while scanning must not abort resolution."""
    monkeypatch.setattr(
        context_module.Path, "iterdir", lambda self: (_ for _ in ()).throw(PermissionError())
    )
    assert context_module._nearby_installations(tmp_path / "a" / "b") == []


def test_malformed_config_yaml_yields_empty_config(app_root: Path) -> None:
    """A truncated config is a finding, not a traceback."""
    (app_root / "config.yaml").write_text("weights: [unclosed\n", encoding="utf-8")
    context = resolve_context(app_root)
    assert context.config == {}
    assert context.result_dir == app_root / "Result"


def test_local_config_overrides_top_level_keys(app_root: Path) -> None:
    (app_root / "config.local.yaml").write_text("output_dir: D:/Other\n", encoding="utf-8")
    context = resolve_context(app_root)
    assert context.result_dir == Path("D:/Other")


def test_bundle_without_weights_is_not_offered(app_root: Path) -> None:
    """A config naming weights that do not exist yields no benchmark target."""
    bundle = app_root / "models" / "Widget" / "A" / "yolo"
    (bundle / "config.yaml").write_text("weights: weights/absent.onnx\n", encoding="utf-8")
    assert discover_model_targets(resolve_context(app_root)) == []


def test_model_target_reads_thresholds_from_the_bundle(app_root: Path) -> None:
    bundle = app_root / "models" / "Widget" / "A" / "yolo"
    (bundle / "weights" / "model.onnx").write_bytes(b"not-a-real-model")
    (bundle / "config.yaml").write_text(
        "weights: models/Widget/A/yolo/weights/model.onnx\n"
        "conf_thres: 0.4\niou_thres: 0.55\ntimeout: 3\nimgsz: [512, 512]\ndevice: cpu\n",
        encoding="utf-8",
    )
    target = discover_model_targets(resolve_context(app_root))[0]

    assert target.conf_thres == pytest.approx(0.4)
    assert target.iou_thres == pytest.approx(0.55)
    assert target.timeout_s == pytest.approx(3.0)
    assert target.imgsz == (512, 512)
