"""Fail-fast validation for the supported source-checkout runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from dataclasses import asdict, dataclass

SUPPORTED_PYTHON = ((3, 10), (3, 11))
EXPECTED_DISTRIBUTIONS = {
    "torch": "2.4.1",
    "torchvision": "0.19.1",
    "onnxruntime": "1.23.2",
    "PyQt5": "5.15.11",
    "jsonargparse": "4.34.0",
}
REQUIRED_IMPORTS = (
    "torch",
    "torchvision",
    "onnxruntime",
    "cv2",
    "PyQt5.QtCore",
    "jsonargparse",
    "yaml",
)


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    passed: bool
    detail: str


def run_checks() -> tuple[RuntimeCheck, ...]:
    checks: list[RuntimeCheck] = []
    actual_python = sys.version_info[:2]
    checks.append(
        RuntimeCheck(
            name="python_version",
            passed=actual_python in SUPPORTED_PYTHON,
            detail=(
                "expected=3.10 or 3.11 "
                f"actual={platform.python_version()} executable={sys.executable}"
            ),
        )
    )
    for module_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
        except (ImportError, OSError, RuntimeError) as exc:
            checks.append(
                RuntimeCheck(
                    name=f"import:{module_name}",
                    passed=False,
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        checks.append(
            RuntimeCheck(
                name=f"import:{module_name}",
                passed=True,
                detail=str(getattr(module, "__file__", "built-in")),
            )
        )
    for distribution, expected_version in EXPECTED_DISTRIBUTIONS.items():
        try:
            actual_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            actual_version = ""
        checks.append(
            RuntimeCheck(
                name=f"version:{distribution}",
                passed=actual_version == expected_version,
                detail=f"expected={expected_version} actual={actual_version or 'missing'}",
            )
        )
    return tuple(checks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checks = run_checks()
    if args.as_json:
        print(
            json.dumps(
                [asdict(check) for check in checks],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for check in checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")
    return 0 if all(check.passed for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
