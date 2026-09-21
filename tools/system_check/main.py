"""Command-line entry point for the deployment preflight checker.

    python -m tools.system_check                      # check + benchmark
    python -m tools.system_check --save-baseline b.json
    python -m tools.system_check --baseline b.json    # compare with a baseline

Exit codes:

``0``
    PASS, or PASS WITH WARNINGS (use ``--strict`` to fail on warnings too).
``1``
    FAIL — at least one stated requirement is not met.
``2``
    The checker itself could not run.

The tool only reads. It never installs packages, edits the registry, touches
firewall rules, changes environment variables outside its own process, or
writes to production configuration or inspection evidence.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from tools.system_check.benchmark import cpu_benchmark as cpu_benchmark_module
from tools.system_check.benchmark.project_benchmark import (
    DEFAULT_TIMED_RUNS,
    DEFAULT_WARMUP_RUNS,
    BenchmarkError,
    BenchmarkResult,
    run_benchmark,
)
from tools.system_check.checks import CHECK_REGISTRY, CheckFunc
from tools.system_check.context import AppContext, discover_model_targets, resolve_context
from tools.system_check.report.comparison import BaselineError, build_comparison, load_baseline
from tools.system_check.report.reporter import build_report, render_text, write_json
from tools.system_check.results import CheckResult, Confidence, Status, run_check
from tools.system_check.spec import REQUIREMENTS, ModelTarget
from tools.system_check.verdicts import benchmark_checks

__version__ = "1.0.0"

_DEFAULT_TXT_NAME = "system_report.txt"
_DEFAULT_JSON_NAME = "system_report.json"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="system_check",
        description=(
            "Check whether this Windows machine can run yolo11_inference, and "
            "measure how fast it does."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"system_check {__version__}")
    parser.add_argument(
        "--app-root",
        type=Path,
        help="Installation directory to inspect. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Where to write system_report.txt and system_report.json.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline JSON to compare this machine against.",
    )
    parser.add_argument(
        "--save-baseline",
        type=Path,
        help="Also write this run's report to the given path, as a baseline.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Run environment checks only.",
    )
    parser.add_argument(
        "--skip-cpu-benchmark",
        action="store_true",
        help="Skip the synthetic CPU probe.",
    )
    parser.add_argument(
        "--benchmark-backend",
        choices=("auto", "onnx", "ultralytics"),
        default="auto",
        help=(
            "auto (default) uses onnxruntime for .onnx weights and ultralytics "
            "for .pt checkpoints."
        ),
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=DEFAULT_TIMED_RUNS,
        help=f"Timed inference runs (default {DEFAULT_TIMED_RUNS}; use 100+ for a meaningful P99).",
    )
    parser.add_argument(
        "--benchmark-warmup",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=f"Discarded warm-up runs (default {DEFAULT_WARMUP_RUNS}).",
    )
    parser.add_argument(
        "--benchmark-image",
        type=Path,
        help=(
            "Read this image instead of the deterministic synthetic frame. "
            "Read-only; two machines must use the same file to be comparable."
        ),
    )
    parser.add_argument("--product", help="Benchmark this product instead of the default.")
    parser.add_argument("--area", help="Benchmark this area instead of the default.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as failures.",
    )
    parser.add_argument(
        "--list-requirements",
        action="store_true",
        help="Print the requirements this tool derived from the repository, then exit.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show requirement provenance.")
    parser.add_argument("--quiet", action="store_true", help="Write report files, print nothing.")
    parser.add_argument("--debug", action="store_true", help="Print tracebacks on tool errors.")
    return parser


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_all_checks(context: AppContext) -> list[CheckResult]:
    """Run every registered check with exception isolation."""
    results: list[CheckResult] = []
    for check_id, title, func in CHECK_REGISTRY:
        # ``func`` is bound as a default so each closure keeps its own check
        # rather than the loop variable's final value.
        def invoke(bound: CheckFunc = func) -> list[CheckResult]:
            return list(bound(context))

        results.extend(run_check(check_id, title, invoke))
    return results


def select_target(
    context: AppContext, product: str | None, area: str | None
) -> ModelTarget | None:
    """Pick the model bundle to benchmark."""
    targets = discover_model_targets(context)
    if not targets:
        return None
    if product or area:
        for target in targets:
            if product and target.product.lower() != product.lower():
                continue
            if area and target.area.lower() != area.lower():
                continue
            return target
        return None
    return targets[0]


def _frame_size(context: AppContext) -> tuple[int, int]:
    """Return the camera resolution the synthetic benchmark frame should use."""

    def positive(key: str, fallback: int) -> int:
        try:
            value = int(context.config.get(key, fallback))
        except (TypeError, ValueError):
            return fallback
        return value if 16 <= value <= 20000 else fallback

    return positive("width", 3072), positive("height", 2048)


def run_project_benchmark(
    context: AppContext, args: argparse.Namespace, results: list[CheckResult]
) -> BenchmarkResult | None:
    """Run the production-path benchmark, recording any failure as a check.

    Returns ``None`` when the benchmark could not run; the reason is appended
    to ``results`` so it shows up in the report instead of vanishing.
    """
    target = select_target(context, args.product, args.area)
    if target is None:
        results.append(
            CheckResult(
                check_id="benchmark.model",
                title="Benchmark model",
                status=Status.UNKNOWN,
                detail=(
                    "No model bundle matched, so the performance benchmark did "
                    "not run. Environment checks above are unaffected."
                ),
                requirement="A model bundle to benchmark",
                measured="none selected",
                confidence=Confidence.UNKNOWN,
                remedy="Copy models/ next to the executable, or pass --product/--area.",
            )
        )
        return None

    width, height = _frame_size(context)
    gpu_present = any(
        result.check_id == "gpu.present" and result.data.get("available")
        for result in results
    )

    try:
        return run_benchmark(
            target,
            backend=args.benchmark_backend,
            warmup_runs=max(0, args.benchmark_warmup),
            timed_runs=max(1, args.benchmark_runs),
            frame_width=width,
            frame_height=height,
            image_path=str(args.benchmark_image) if args.benchmark_image else None,
            sample_gpu=gpu_present,
            progress=None if args.quiet else _progress,
        )
    except BenchmarkError as exc:
        results.append(
            CheckResult(
                check_id="benchmark.model",
                title="Benchmark model",
                status=Status.FAIL,
                detail=f"The performance benchmark could not run: {exc}",
                requirement="The configured model must load and infer",
                measured="benchmark failed",
                remedy=(
                    "Resolve the runtime failures listed above; the application "
                    "will hit the same error when it loads this model."
                ),
                data={"weights": target.weights},
            )
        )
        return None
    except Exception as exc:  # noqa: BLE001 - one broken model must not end the run
        results.append(
            CheckResult(
                check_id="benchmark.model",
                title="Benchmark model",
                status=Status.UNKNOWN,
                detail=(
                    f"The performance benchmark raised {type(exc).__name__}: {exc}. "
                    "Environment checks above are unaffected."
                ),
                requirement="The configured model must load and infer",
                measured="benchmark error",
                confidence=Confidence.UNKNOWN,
                data={"traceback": traceback.format_exc(limit=8)},
            )
        )
        return None
    finally:
        if not args.quiet:
            sys.stdout.write("\r" + " " * 40 + "\r")
            sys.stdout.flush()


def _progress(done: int, total: int) -> None:
    """Write a single-line progress indicator to stdout."""
    sys.stdout.write(f"\r  benchmarking {done}/{total} runs...")
    sys.stdout.flush()


def _guard_output_dir(output_dir: Path, context: AppContext) -> Path:
    """Refuse to write reports into the inspection evidence tree.

    Raises:
        ValueError: If ``output_dir`` is inside the configured result root.
    """
    resolved = output_dir.expanduser().resolve()
    try:
        result_root = context.result_dir.resolve()
    except OSError:
        return resolved
    if resolved == result_root or result_root in resolved.parents:
        raise ValueError(
            f"Refusing to write reports into the inspection result tree "
            f"({result_root}). Choose a different --output-dir."
        )
    return resolved


def _print_requirements() -> None:
    """Print the derived requirements table."""
    width = max(len(item.category) for item in REQUIREMENTS)
    for item in REQUIREMENTS:
        print(f"[{item.confidence.value:<9}] {item.category:<{width}}  {item.statement}")
        if item.source:
            print(f"{'':<12} {'':<{width}}  source: {item.source}")
        if item.notes:
            print(f"{'':<12} {'':<{width}}  note:   {item.notes}")


def _launched_by_double_click() -> bool:
    """Whether this process owns its console alone, i.e. Explorer started it.

    When a user double-clicks the executable, Windows creates a console just
    for it and destroys it the moment the process exits — the report would
    flash past unread. Run from an existing cmd or PowerShell window, the
    console is shared with the shell and must not be held open.

    ``GetConsoleProcessList`` reports how many processes are attached: one
    means us alone. Any failure answers "no", so an odd environment loses the
    convenience rather than hanging a script.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        buffer = (ctypes.c_uint * 8)()
        count = int(ctypes.windll.kernel32.GetConsoleProcessList(buffer, 8))
    except Exception:  # noqa: BLE001 - a missing console is not an error here
        return False
    return count == 1


def _hold_console(report_dir: Path, status_line: str) -> None:
    """Keep a double-clicked window open until the operator dismisses it."""
    print()
    print("=" * 78)
    print(f"  {status_line}")
    print(f"  Reports saved to: {report_dir}")
    print(f"    - {_DEFAULT_TXT_NAME}   (this report, as a text file)")
    print(f"    - {_DEFAULT_JSON_NAME}  (for tooling and --baseline comparison)")
    print("=" * 78)
    try:
        input("\nPress Enter to close this window...")
    except (EOFError, KeyboardInterrupt):
        return


def _force_utf8_stdout() -> None:
    """Make stdout UTF-8 so the report renders on a cp950/cp437 console.

    Best effort: a redirected or wrapped stream may not support it, and the
    report is written to file regardless.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            continue


def main(argv: list[str] | None = None) -> int:
    """Run the checker.

    Running it with no arguments at all is the intended everyday use: it finds
    the installation, runs every check, benchmarks the real model and writes
    both reports. Double-clicking the executable does exactly that and then
    holds the window open.
    """
    args = build_parser().parse_args(argv)
    _force_utf8_stdout()
    interactive = _launched_by_double_click() and not args.quiet

    if args.list_requirements:
        _print_requirements()
        return 0

    try:
        context = resolve_context(args.app_root)
        output_dir = _guard_output_dir(args.output_dir, context)
    except Exception as exc:  # noqa: BLE001
        print(f"system_check could not start: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        if interactive:
            _hold_console(Path.cwd(), "The checker could not start.")
        return 2

    if interactive:
        print(f"Checking installation: {context.app_root}")
        print("This takes about a minute. Please wait...\n")

    results = run_all_checks(context)

    benchmark: BenchmarkResult | None = None
    cpu_result: cpu_benchmark_module.CpuBenchmark | None = None
    if not args.skip_benchmark:
        benchmark = run_project_benchmark(context, args, results)
        if benchmark is not None:
            results.extend(benchmark_checks(benchmark))
        if not args.skip_cpu_benchmark:
            cpu_result = _safe_cpu_benchmark(results)

    # The comparison needs a report to compare, and it appends a check of its
    # own describing whether the baseline loaded. So: build a provisional
    # report to diff against, then build the final one once the check list is
    # complete, or the baseline result would be missing from its own report.
    comparison = None
    if args.baseline:
        provisional = build_report(
            context, results, benchmark, cpu_result, None, tool_version=__version__
        )
        comparison = _build_comparison(args.baseline, provisional, results)

    report = build_report(
        context, results, benchmark, cpu_result, comparison, tool_version=__version__
    )
    text = render_text(report, results, benchmark, cpu_result, comparison, verbose=args.verbose)

    exit_code = _write_outputs(report, text, output_dir, args, results)
    if not args.quiet:
        print(text)
    if interactive:
        _hold_console(output_dir, f"OVERALL COMPATIBILITY: {report['overall_label']}")
    return exit_code


def _safe_cpu_benchmark(results: list[CheckResult]) -> cpu_benchmark_module.CpuBenchmark | None:
    """Run the synthetic CPU probe, recording a failure rather than raising."""
    try:
        return cpu_benchmark_module.run()
    except Exception as exc:  # noqa: BLE001
        results.append(
            CheckResult(
                check_id="benchmark.cpu",
                title="CPU probe",
                status=Status.UNKNOWN,
                detail=f"The synthetic CPU probe failed: {type(exc).__name__}: {exc}",
                requirement="Informational only",
                measured="not measured",
                confidence=Confidence.UNKNOWN,
            )
        )
        return None


def _build_comparison(baseline_path: Path, report: dict[str, Any], results: list[CheckResult]):
    """Load and apply a baseline, recording any problem as a check result."""
    try:
        baseline = load_baseline(baseline_path)
    except BaselineError as exc:
        results.append(
            CheckResult(
                check_id="baseline.load",
                title="Baseline comparison",
                status=Status.UNKNOWN,
                detail=(
                    f"{exc} The environment checks and benchmark above are "
                    "unaffected; only the comparison is missing."
                ),
                requirement="A readable baseline JSON",
                measured="unusable",
                confidence=Confidence.UNKNOWN,
                remedy="Regenerate it with --save-baseline on the reference machine.",
            )
        )
        return None

    comparison = build_comparison(baseline, report, str(baseline_path))
    results.append(
        CheckResult(
            check_id="baseline.load",
            title="Baseline comparison",
            status=Status.PASS,
            detail=(
                f"Compared against {baseline_path.name} recorded "
                f"{comparison.baseline_recorded_at or 'at an unknown time'}. "
                "Differences are informational: acceptance is decided by the "
                "inference-timeout check, not by being slower than the baseline."
            ),
            requirement="A readable baseline JSON",
            measured="loaded",
            data={"warnings": list(comparison.warnings)},
        )
    )
    return comparison


def _write_outputs(
    report: dict[str, Any],
    text: str,
    output_dir: Path,
    args: argparse.Namespace,
    results: list[CheckResult],
) -> int:
    """Write reports and return the process exit code."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(report, output_dir / _DEFAULT_JSON_NAME)
        (output_dir / _DEFAULT_TXT_NAME).write_text(text, encoding="utf-8")
        if args.save_baseline:
            baseline_path = Path(args.save_baseline).expanduser()
            write_json(report, baseline_path)
            if not args.quiet:
                print(f"Baseline written to {baseline_path}")
    except OSError as exc:
        print(f"Report files could not be written: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        return 2

    if any(result.status is Status.FAIL for result in results):
        return 1
    if args.strict and any(
        result.status in (Status.WARNING, Status.UNKNOWN) for result in results
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
