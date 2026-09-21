"""Python runtime and package checks.

Three inspection modes, because the honest answer differs by deployment:

``interpreter``
    The checker shares an interpreter with the application (source checkout,
    or the checker run inside the conda environment). Packages are probed by
    import and by ``importlib.metadata``, which is what the launcher's own
    gate — ``tools/check_runtime_environment.py`` — does.

``bundle``
    The checker is a standalone executable inspecting a PyInstaller build. The
    application's packages live inside *its* bundle and cannot be imported from
    this process, so versions are read from the ``.dist-info`` directories
    PyInstaller copied into ``_internal``. Only distributions the spec file
    calls ``copy_metadata`` on carry a version there; the rest can be confirmed
    present but not versioned, and are reported as such rather than as missing.

``unavailable``
    The standalone executable was pointed at a source checkout. Its own bundle
    carries onnxruntime, numpy and cv2 for the benchmark, so trusting it here
    would report the *checker's* packages as the application's and fail every
    pin the checker does not carry. Nothing is claimed instead.

Version drift on the five distributions ``check_runtime_environment.py`` gates
is reported as FAIL, matching what ``start.bat`` already refuses to launch on.
Drift elsewhere is a WARNING: the launcher does not block on it, but a numpy or
OpenCV difference can move a borderline colour verdict.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import re
import sys
from pathlib import Path

from tools.system_check.context import AppContext
from tools.system_check.results import CheckResult, Confidence, Status
from tools.system_check.spec import (
    PINNED_ADVISORY,
    PINNED_ANOMALIB,
    PINNED_BLOCKING,
    PYTHON_MAX_EXCLUSIVE,
    PYTHON_MIN,
    REQUIRED_IMPORTS,
    SUPPORTED_PYTHON_MINORS,
)

_DIST_INFO_RE = re.compile(r"^(?P<name>.+?)-(?P<version>\d[^-]*)\.dist-info$")


def check(context: AppContext) -> list[CheckResult]:
    """Report interpreter version and the state of every pinned package."""
    mode, versions, evidence = _collect_versions(context)
    results: list[CheckResult] = [_python_result(mode, context)]

    results.append(
        CheckResult(
            check_id="deps.mode",
            title="Package inspection mode",
            status=Status.UNKNOWN if mode == "unavailable" else Status.PASS,
            detail=_mode_detail(mode, evidence),
            measured=mode,
            requirement="Packages must be inspectable",
            confidence=(
                Confidence.UNKNOWN if mode == "unavailable" else Confidence.CONFIRMED
            ),
            remedy=(
                "Run `python -m tools.system_check` from the application's own "
                "environment, or point --app-root at a packaged build."
                if mode == "unavailable"
                else None
            ),
            data={"mode": mode, "evidence": evidence},
        )
    )

    if mode == "unavailable":
        # Reporting "torch is missing" here would be a lie: the checker cannot
        # see the target's packages at all, and the five blocking pins would
        # all come back FAIL against the checker's own bundle.
        return results

    if mode == "interpreter":
        results.extend(_import_results())

    results.extend(
        _version_results(
            PINNED_BLOCKING,
            versions,
            mode,
            failing=True,
            group="blocking",
            source="tools/check_runtime_environment.py:14-20",
        )
    )
    results.extend(
        _version_results(
            PINNED_ADVISORY,
            versions,
            mode,
            failing=False,
            group="advisory",
            source="requirements.txt",
        )
    )
    results.extend(_anomalib_results(context, versions, mode))
    return results


def _anomalib_results(
    context: AppContext, versions: dict[str, str | None], mode: str
) -> list[CheckResult]:
    """Check the anomalib stack, whose pins drift independently of the rest.

    Skipping this group whenever ``enable_anomalib`` is false hides real drift:
    the PyInstaller spec collects anomalib into every build regardless of the
    flag, so the versions in the artifact are whatever the build machine had.
    A station that later flips the flag inherits them. The group is therefore
    checked whenever the packages are actually present, and skipped only when
    they are both unused and absent.
    """
    enabled = context.anomalib_enabled()
    present = [name for name in PINNED_ANOMALIB if _canonical(name) in versions]

    if not enabled and not present:
        return [
            CheckResult(
                check_id="deps.anomalib",
                title="Anomalib backend packages",
                status=Status.SKIP,
                detail=(
                    "enable_anomalib is false and none of anomalib, lightning, "
                    "timm, kornia, einops, FrEIA or open-clip-torch is present."
                ),
                requirement="Only required when enable_anomalib is true",
                measured="backend disabled, packages absent",
                source="config.yaml enable_anomalib",
            )
        ]

    results = _version_results(
        PINNED_ANOMALIB,
        versions,
        mode,
        failing=False,
        group="anomalib",
        source="pyproject.toml:27-65",
    )
    if not enabled:
        results.insert(
            0,
            CheckResult(
                check_id="deps.anomalib",
                title="Anomalib backend packages",
                status=Status.PASS,
                detail=(
                    "enable_anomalib is false for this station, but "
                    f"{len(present)} of these packages are present and are "
                    "collected into every build by yolo11_inference.spec. Their "
                    "versions are checked anyway, because turning the flag on "
                    "would start using exactly these."
                ),
                requirement="Checked because present, not because required",
                measured=f"backend disabled, {len(present)} package(s) present",
                source="yolo11_inference.spec:41-42, config.yaml enable_anomalib",
                data={"present": present},
            ),
        )
    return results


# --------------------------------------------------------------------------
# Version collection
# --------------------------------------------------------------------------


def _collect_versions(context: AppContext) -> tuple[str, dict[str, str | None], dict[str, object]]:
    """Return ``(mode, {canonical_name: version_or_None}, evidence)``.

    The *target* decides the mode, not the checker. A packaged installation is
    always read from its own ``_internal`` metadata, even when the checker runs
    from an environment that happens to have the same packages installed.

    A frozen checker can never answer for a source checkout: its interpreter is
    its own bundle, which carries onnxruntime, numpy and cv2 for the benchmark.
    Trusting it there would report the *checker's* packages as the
    application's and fail every pin the checker does not carry. That case
    returns ``unavailable`` so the report says so instead of inventing
    failures.

    ``None`` as a version means "present but version unknown"; a name absent
    from the mapping means "not found".
    """
    bundle_versions, bundle_evidence = _bundle_versions(context)
    if bundle_versions:
        return "bundle", bundle_versions, bundle_evidence

    frozen_checker = bool(getattr(sys, "frozen", False))
    if not frozen_checker and _interpreter_has_app():
        return "interpreter", _interpreter_versions(), {"executable": sys.executable}

    return (
        "unavailable",
        {},
        {
            "frozen_checker": frozen_checker,
            "app_root": str(context.app_root),
            "data_root": str(context.data_root),
        },
    )


def _interpreter_has_app() -> bool:
    """Whether this interpreter can see the application's own dependencies."""
    try:
        importlib.metadata.version("onnxruntime")
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:  # noqa: BLE001
        return False
    return True


def _interpreter_versions() -> dict[str, str | None]:
    """Read installed distribution versions from this interpreter."""
    found: dict[str, str | None] = {}
    for name in (*PINNED_BLOCKING, *PINNED_ADVISORY, *PINNED_ANOMALIB):
        try:
            found[_canonical(name)] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:  # noqa: BLE001 - a corrupt dist-info is not fatal
            found[_canonical(name)] = None
    return found


#: Import names used to confirm a package is bundled when no metadata exists.
_BUNDLE_IMPORT_NAMES: dict[str, tuple[str, ...]] = {
    "opencv-python": ("cv2",),
    "pyyaml": ("yaml",),
    "pillow": ("PIL",),
    "pyserial": ("serial",),
    "scikit-learn": ("sklearn",),
    "open-clip-torch": ("open_clip",),
    "pyqt5": ("PyQt5",),
    "freia": ("FrEIA",),
}


def _bundle_versions(context: AppContext) -> tuple[dict[str, str | None], dict[str, object]]:
    """Read package evidence out of a PyInstaller ``_internal`` directory."""
    root = context.data_root
    if not root.is_dir():
        return {}, {}

    found: dict[str, str | None] = {}
    dist_infos: list[str] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return {}, {}

    for entry in entries:
        if not entry.is_dir():
            continue
        match = _DIST_INFO_RE.match(entry.name)
        if match:
            dist_infos.append(entry.name)
            found[_canonical(match.group("name"))] = match.group("version")

    for name in (*PINNED_BLOCKING, *PINNED_ADVISORY, *PINNED_ANOMALIB):
        canonical = _canonical(name)
        if canonical in found:
            continue
        for import_name in _BUNDLE_IMPORT_NAMES.get(canonical, (name,)):
            if _bundled_present(root, import_name):
                found[canonical] = None
                break

    evidence = {"data_root": str(root), "dist_info_count": len(dist_infos)}
    return found, evidence


def _bundled_present(root: Path, import_name: str) -> bool:
    """Whether a package directory or extension module sits in the bundle."""
    if (root / import_name).is_dir():
        return True
    try:
        return any(root.glob(f"{import_name}.*.pyd")) or (root / f"{import_name}.pyd").exists()
    except OSError:
        return False


def _canonical(name: str) -> str:
    """Normalize a distribution name for comparison (PEP 503)."""
    return re.sub(r"[-_.]+", "-", name).lower()


# --------------------------------------------------------------------------
# Result builders
# --------------------------------------------------------------------------


def _python_result(mode: str, context: AppContext) -> CheckResult:
    """Check the interpreter version against ``requires-python``."""
    actual = sys.version_info[:2]
    version_text = platform.python_version()
    requirement = (
        f">={PYTHON_MIN[0]}.{PYTHON_MIN[1]},"
        f"<{PYTHON_MAX_EXCLUSIVE[0]}.{PYTHON_MAX_EXCLUSIVE[1]}"
    )

    if mode == "bundle":
        return CheckResult(
            check_id="python.version",
            title="Python runtime",
            status=Status.SKIP,
            detail=(
                "The application under test is a packaged build with its own "
                "embedded interpreter, so the host's Python version does not "
                f"apply. (Checker itself is running {version_text}.)"
            ),
            requirement=requirement,
            measured=f"packaged build; checker on {version_text}",
            source="pyproject.toml:18",
        )

    if mode == "unavailable":
        return CheckResult(
            check_id="python.version",
            title="Python runtime",
            status=Status.UNKNOWN,
            detail=(
                "The target is a source checkout, and this checker cannot see "
                f"the interpreter that would run it. (Checker itself is "
                f"running {version_text}.)"
            ),
            requirement=requirement,
            measured="not determined",
            confidence=Confidence.UNKNOWN,
            source="pyproject.toml:18",
            remedy=(
                "Run `python -m tools.system_check` with the interpreter that "
                "launches the application."
            ),
        )

    if actual in SUPPORTED_PYTHON_MINORS:
        status, detail, remedy = (
            Status.PASS,
            f"Python {version_text}.",
            None,
        )
    elif PYTHON_MIN <= actual < PYTHON_MAX_EXCLUSIVE:
        status, detail, remedy = (
            Status.WARNING,
            (
                f"Python {version_text} satisfies requires-python but is not "
                "one of the two minors the launcher gate accepts."
            ),
            "Use Python 3.10 or 3.11 to match the tested runtime.",
        )
    else:
        status, detail, remedy = (
            Status.FAIL,
            (
                f"Python {version_text} is outside requires-python "
                f"({requirement}). The pinned torch and onnxruntime wheels have "
                "no build for it."
            ),
            "Install Python 3.10 or 3.11 (64-bit) and recreate the environment.",
        )

    return CheckResult(
        check_id="python.version",
        title="Python runtime",
        status=status,
        detail=detail,
        requirement=requirement,
        measured=version_text,
        source="pyproject.toml:18, tools/check_runtime_environment.py:13",
        remedy=remedy,
        data={"version": version_text, "executable": sys.executable},
    )


def _import_results() -> list[CheckResult]:
    """Probe each import the launcher gate requires."""
    results: list[CheckResult] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError, RuntimeError) as exc:
            results.append(
                CheckResult(
                    check_id=f"deps.import.{module_name}",
                    title=f"import {module_name}",
                    status=Status.FAIL,
                    detail=f"{type(exc).__name__}: {exc}",
                    requirement=f"{module_name} must import",
                    measured="import failed",
                    source="tools/check_runtime_environment.py:21-29",
                    remedy=_import_remedy(module_name),
                )
            )
            continue
        results.append(
            CheckResult(
                check_id=f"deps.import.{module_name}",
                title=f"import {module_name}",
                status=Status.PASS,
                detail=str(getattr(module, "__file__", "built-in")),
                requirement=f"{module_name} must import",
                measured="ok",
                source="tools/check_runtime_environment.py:21-29",
            )
        )
    return results


def _import_remedy(module_name: str) -> str:
    """Return the known remedy for a failed import."""
    if module_name.startswith("onnxruntime"):
        return (
            "Install onnxruntime==1.23.2 and the Microsoft Visual C++ "
            "Redistributable 2015-2022 x64."
        )
    if module_name.startswith("PyQt5"):
        return "Install PyQt5==5.15.11; the GUI entry point cannot start without it."
    return "Reinstall the environment from requirements.txt."


def _version_results(
    pins: dict[str, str],
    versions: dict[str, str | None],
    mode: str,
    *,
    failing: bool,
    group: str,
    source: str,
) -> list[CheckResult]:
    """Compare one group of pins against what was found."""
    results: list[CheckResult] = []
    for name, expected in pins.items():
        canonical = _canonical(name)
        check_id = f"deps.version.{canonical}"
        title = f"{name} {expected}"

        if canonical not in versions:
            if mode == "bundle":
                # Absence cannot be proven in a packaged build: PyInstaller
                # puts pure-Python packages inside the PYZ archive embedded in
                # the executable, where nothing on disk reveals them. openpyxl
                # and pyserial are bundled and invisible exactly this way, so
                # calling them missing would be a false alarm on every
                # packaged deployment.
                results.append(
                    CheckResult(
                        check_id=check_id,
                        title=title,
                        status=Status.UNKNOWN,
                        detail=(
                            f"{name} could not be located in the packaged "
                            "build. Pure-Python packages live inside the PYZ "
                            "archive embedded in the executable and leave no "
                            "trace on disk, so this is not evidence of "
                            "absence. verify_build.py checks build "
                            "completeness."
                        ),
                        requirement=f"{name}=={expected}",
                        measured="not visible on disk",
                        confidence=Confidence.UNKNOWN,
                        source=source,
                        data={"group": group, "expected": expected, "actual": None},
                    )
                )
                continue
            results.append(
                CheckResult(
                    check_id=check_id,
                    title=title,
                    status=Status.FAIL if failing else Status.WARNING,
                    detail=f"{name} was not found ({mode} inspection).",
                    requirement=f"{name}=={expected}",
                    measured="missing",
                    source=source,
                    remedy=f"pip install {name}=={expected}",
                    data={"group": group, "expected": expected, "actual": None},
                )
            )
            continue

        actual = versions[canonical]
        if actual is None:
            results.append(
                CheckResult(
                    check_id=check_id,
                    title=title,
                    status=Status.UNKNOWN,
                    detail=(
                        f"{name} is present in the bundle but carries no "
                        "metadata, so its version cannot be confirmed. Only "
                        "distributions the spec file copies metadata for are "
                        "versionable in a packaged build."
                    ),
                    requirement=f"{name}=={expected}",
                    measured="present, version unknown",
                    confidence=Confidence.UNKNOWN,
                    source=source,
                    data={"group": group, "expected": expected, "actual": None},
                )
            )
            continue

        if actual == expected:
            status, detail, remedy = Status.PASS, f"{name} {actual}.", None
        elif failing:
            status = Status.FAIL
            detail = (
                f"{name} is {actual}, not the pinned {expected}. "
                "start.bat refuses to launch on this mismatch."
            )
            remedy = f"pip install {name}=={expected}"
        else:
            status = Status.WARNING
            detail = (
                f"{name} is {actual}, not the pinned {expected}. The launcher "
                "does not block on this, but behaviour may differ from the "
                "tested runtime."
            )
            remedy = f"pip install {name}=={expected} to match the tested runtime."

        results.append(
            CheckResult(
                check_id=check_id,
                title=title,
                status=status,
                detail=detail,
                requirement=f"{name}=={expected}",
                measured=actual,
                source=source,
                remedy=remedy,
                data={"group": group, "expected": expected, "actual": actual},
            )
        )
    return results


def _mode_detail(mode: str, evidence: dict[str, object]) -> str:
    """Explain which inspection mode was used and why it matters."""
    if mode == "bundle":
        return (
            f"Reading packages from the packaged build at "
            f"{evidence.get('data_root')} "
            f"({evidence.get('dist_info_count')} distributions carry metadata). "
            "Versions without metadata are reported as UNKNOWN, not missing."
        )
    if mode == "unavailable":
        if evidence.get("frozen_checker"):
            return (
                f"{evidence.get('app_root')} is a source checkout, and this is "
                "the standalone checker, whose own bundle is not the "
                "application's environment. Package versions were not "
                "inspected, rather than reported from the wrong interpreter."
            )
        return (
            "Neither a packaged build nor an environment with the "
            "application's dependencies was found, so package versions were "
            "not inspected."
        )
    return f"Reading packages from the interpreter at {evidence.get('executable')}."
