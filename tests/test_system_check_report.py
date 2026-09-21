"""Baseline comparison and the CLI end to end.

The comparison tests exist mostly to pin one thing down: direction. A latency
that went up is a regression, a throughput that went up is an improvement, and
a report that gets that backwards is worse than no report at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.system_check import main as main_module
from tools.system_check.context import resolve_context
from tools.system_check.report.comparison import (
    BENCHMARK_METRICS,
    BaselineError,
    MetricSpec,
    build_comparison,
    compare_metric,
    dig,
    load_baseline,
)
from tools.system_check.report.reporter import build_report, render_text
from tools.system_check.results import Status

LATENCY = MetricSpec("total_mean_ms", "Total (mean)", "benchmark.total.mean_ms", "ms", False, 2)
THROUGHPUT = MetricSpec("fps", "Throughput", "benchmark.throughput_fps", "FPS", True, 2)


def _report(mean_ms: float, fps: float) -> dict:
    """A minimal report payload carrying just the compared fields."""
    return {
        "recorded_at": "2026-09-01T00:00:00+00:00",
        "system": {
            "os": {"edition": "Windows 11 Pro"},
            "cpu": {"model": "Test CPU", "logical_cores": 8},
            "memory": {"total_gb": 16.0},
            "disk": {"free_gb": 100.0},
            "gpu": {"name": None, "vram_total_mb": None},
        },
        "benchmark": {
            "backend": "onnx",
            "weights_sha_prefix": "abc123abc123",
            "source_frame": "synthetic:3072x2048",
            "imgsz": [640, 640],
            "timed_runs": 50,
            "total": {"mean_ms": mean_ms},
            "throughput_fps": fps,
        },
    }


# --------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------


def test_slower_target_is_worse_for_latency() -> None:
    """Latency up means the target regressed."""
    comparison = compare_metric(LATENCY, _report(28.0, 35.7), _report(41.0, 24.4))

    assert comparison.difference == pytest.approx(13.0)
    assert comparison.percent == pytest.approx(46.4, abs=0.2)
    assert comparison.verdict == "worse"


def test_faster_target_is_better_for_latency() -> None:
    comparison = compare_metric(LATENCY, _report(41.0, 24.4), _report(28.0, 35.7))

    assert comparison.difference == pytest.approx(-13.0)
    assert comparison.verdict == "better"


def test_higher_throughput_is_better() -> None:
    """The same arithmetic sign must mean the opposite for throughput."""
    comparison = compare_metric(THROUGHPUT, _report(41.0, 24.4), _report(28.0, 35.7))

    assert comparison.difference is not None and comparison.difference > 0
    assert comparison.verdict == "better"


def test_lower_throughput_is_worse() -> None:
    comparison = compare_metric(THROUGHPUT, _report(28.0, 35.7), _report(41.0, 24.4))

    assert comparison.difference is not None and comparison.difference < 0
    assert comparison.verdict == "worse"


def test_negligible_change_reads_as_same() -> None:
    assert compare_metric(LATENCY, _report(40.0, 25.0), _report(40.2, 25.0)).verdict == "same"


def test_every_benchmark_metric_declares_a_direction() -> None:
    """No metric may silently inherit a default direction."""
    for spec in BENCHMARK_METRICS:
        assert isinstance(spec.higher_is_better, bool)
    latency_like = [spec for spec in BENCHMARK_METRICS if spec.unit == "ms"]
    assert latency_like
    assert all(not spec.higher_is_better for spec in latency_like)
    assert next(spec for spec in BENCHMARK_METRICS if spec.key == "throughput_fps").higher_is_better


# --------------------------------------------------------------------------
# Damaged baselines
# --------------------------------------------------------------------------


def test_missing_baseline_file_raises_baseline_error(tmp_path: Path) -> None:
    with pytest.raises(BaselineError, match="not found"):
        load_baseline(tmp_path / "absent.json")


def test_malformed_baseline_raises_baseline_error(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("{ this is not json", encoding="utf-8")

    with pytest.raises(BaselineError, match="not valid JSON"):
        load_baseline(path)


def test_non_object_baseline_raises_baseline_error(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(BaselineError, match="JSON object"):
        load_baseline(path)


def test_baseline_missing_fields_degrades_to_unknown() -> None:
    """A baseline from an older tool must not crash the comparison."""
    comparison = build_comparison({"recorded_at": "2026-01-01"}, _report(40.0, 25.0), "b.json")

    latency = next(item for item in comparison.benchmark if item.key == "total_mean_ms")
    assert latency.verdict == "unknown"
    assert latency.baseline is None
    assert latency.target == pytest.approx(40.0)
    assert any("no benchmark section" in warning for warning in comparison.warnings)


def test_baseline_with_null_metric_does_not_divide_by_zero() -> None:
    baseline = _report(40.0, 25.0)
    baseline["benchmark"]["total"]["mean_ms"] = None

    comparison = compare_metric(LATENCY, baseline, _report(40.0, 25.0))
    assert comparison.verdict == "unknown"


def test_zero_baseline_yields_no_percentage() -> None:
    baseline = _report(0.0, 25.0)
    comparison = compare_metric(LATENCY, baseline, _report(40.0, 25.0))

    assert comparison.percent is None
    assert comparison.verdict == "worse"


def test_dig_tolerates_wrong_shapes() -> None:
    assert dig({"a": 1}, "a.b.c") is None
    assert dig(None, "a") is None
    assert dig({"a": {"b": 2}}, "a.b") == 2


# --------------------------------------------------------------------------
# Comparability warnings
# --------------------------------------------------------------------------


def test_different_backends_are_flagged_as_not_comparable() -> None:
    baseline = _report(40.0, 25.0)
    baseline["benchmark"]["backend"] = "ultralytics"

    comparison = build_comparison(baseline, _report(40.0, 25.0), "b.json")
    assert any("benchmark backend differs" in warning for warning in comparison.warnings)


def test_different_model_weights_are_flagged() -> None:
    baseline = _report(40.0, 25.0)
    baseline["benchmark"]["weights_sha_prefix"] = "deadbeefdead"

    comparison = build_comparison(baseline, _report(40.0, 25.0), "b.json")
    assert any("model weights differs" in warning for warning in comparison.warnings)


def test_matching_runs_produce_no_comparability_warning() -> None:
    comparison = build_comparison(_report(40.0, 25.0), _report(41.0, 24.0), "b.json")
    assert comparison.warnings == ()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


@pytest.fixture
def installation(tmp_path: Path) -> Path:
    """An installation with no model bundle, so the benchmark cannot run."""
    root = tmp_path / "app"
    (root / "models").mkdir(parents=True)
    (root / "Runtime").mkdir()
    (root / "config.yaml").write_text("enable_anomalib: false\n", encoding="utf-8")
    return root


def test_cli_writes_both_reports(installation: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    code = main_module.main(
        ["--app-root", str(installation), "--output-dir", str(out), "--skip-benchmark", "--quiet"]
    )

    assert code in (0, 1)
    payload = json.loads((out / "system_report.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["overall_status"] in {"PASS", "WARNING", "FAIL"}
    assert payload["checks"]
    assert payload["requirements"]
    assert (out / "system_report.txt").read_text(encoding="utf-8").startswith("=")


def test_cli_refuses_to_write_into_the_result_tree(installation: Path) -> None:
    """Reports must never land inside inspection evidence."""
    (installation / "config.yaml").write_text(
        f"output_dir: {(installation / 'Result').as_posix()}\n", encoding="utf-8"
    )
    code = main_module.main(
        [
            "--app-root",
            str(installation),
            "--output-dir",
            str(installation / "Result" / "reports"),
            "--skip-benchmark",
            "--quiet",
        ]
    )
    assert code == 2


def test_cli_records_a_benchmark_failure_as_a_check(
    monkeypatch: pytest.MonkeyPatch, installation: Path, tmp_path: Path
) -> None:
    """A benchmark that explodes must not take the environment report with it."""
    bundle = installation / "models" / "P" / "A" / "yolo"
    bundle.mkdir(parents=True)
    (bundle / "model.onnx").write_bytes(b"x")
    (bundle / "config.yaml").write_text("weights: model.onnx\n", encoding="utf-8")

    def exploding(*args, **kwargs):
        raise MemoryError("out of memory during warmup")

    monkeypatch.setattr(main_module, "run_benchmark", exploding)

    out = tmp_path / "out"
    main_module.main(
        ["--app-root", str(installation), "--output-dir", str(out), "--quiet"]
    )
    payload = json.loads((out / "system_report.json").read_text(encoding="utf-8"))
    entry = next(item for item in payload["checks"] if item["check_id"] == "benchmark.model")

    assert entry["status"] == "UNKNOWN"
    assert "MemoryError" in entry["detail"]
    # The environment checks still ran.
    assert any(item["check_id"] == "os.platform" for item in payload["checks"])


def test_cli_records_an_unusable_baseline_without_failing(
    installation: Path, tmp_path: Path
) -> None:
    baseline = tmp_path / "bad.json"
    baseline.write_text("not json at all", encoding="utf-8")
    out = tmp_path / "out"

    main_module.main(
        [
            "--app-root",
            str(installation),
            "--output-dir",
            str(out),
            "--baseline",
            str(baseline),
            "--skip-benchmark",
            "--quiet",
        ]
    )
    payload = json.loads((out / "system_report.json").read_text(encoding="utf-8"))
    entry = next(item for item in payload["checks"] if item["check_id"] == "baseline.load")

    assert entry["status"] == "UNKNOWN"
    assert payload["comparison"] is None


def test_cli_save_baseline_writes_the_same_schema(installation: Path, tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    out = tmp_path / "out"
    main_module.main(
        [
            "--app-root",
            str(installation),
            "--output-dir",
            str(out),
            "--save-baseline",
            str(baseline),
            "--skip-benchmark",
            "--quiet",
        ]
    )
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    produced = json.loads((out / "system_report.json").read_text(encoding="utf-8"))

    assert saved.keys() == produced.keys()
    assert saved["schema_version"] == produced["schema_version"]


def test_strict_mode_fails_on_warnings(
    monkeypatch: pytest.MonkeyPatch, installation: Path, tmp_path: Path
) -> None:
    from tools.system_check.results import CheckResult

    monkeypatch.setattr(
        main_module,
        "run_all_checks",
        lambda context: [
            CheckResult(check_id="x", title="t", status=Status.WARNING, detail="d")
        ],
    )
    args = ["--app-root", str(installation), "--output-dir", str(tmp_path / "o"), "--skip-benchmark", "--quiet"]

    assert main_module.main(args) == 0
    assert main_module.main([*args, "--strict"]) == 1


def test_double_clicked_run_holds_the_window_open(
    monkeypatch: pytest.MonkeyPatch, installation: Path, tmp_path: Path
) -> None:
    """Explorer destroys the console on exit, so the report must be held."""
    prompts: list[str] = []

    def record(prompt: str = "") -> str:
        prompts.append(prompt)
        return ""

    monkeypatch.setattr(main_module, "_launched_by_double_click", lambda: True)
    monkeypatch.setattr("builtins.input", record)

    out = tmp_path / "out"
    main_module.main(
        ["--app-root", str(installation), "--output-dir", str(out), "--skip-benchmark"]
    )

    assert prompts, "a double-clicked run must wait before closing"


def test_console_is_not_held_when_run_from_a_shell(
    monkeypatch: pytest.MonkeyPatch, installation: Path, tmp_path: Path
) -> None:
    """Holding a shared console would hang a script that invoked the checker."""

    def must_not_be_called(prompt: str = "") -> str:
        raise AssertionError("must not wait for input when run from a shell")

    monkeypatch.setattr(main_module, "_launched_by_double_click", lambda: False)
    monkeypatch.setattr("builtins.input", must_not_be_called)

    main_module.main(
        [
            "--app-root",
            str(installation),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-benchmark",
        ]
    )


def test_quiet_never_waits_even_if_double_clicked(
    monkeypatch: pytest.MonkeyPatch, installation: Path, tmp_path: Path
) -> None:
    """--quiet is what a scheduled task uses; it must never block."""

    def must_not_be_called(prompt: str = "") -> str:
        raise AssertionError("--quiet must not wait for input")

    monkeypatch.setattr(main_module, "_launched_by_double_click", lambda: True)
    monkeypatch.setattr("builtins.input", must_not_be_called)

    main_module.main(
        [
            "--app-root",
            str(installation),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-benchmark",
            "--quiet",
        ]
    )


def test_list_requirements_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    assert main_module.main(["--list-requirements"]) == 0
    printed = capsys.readouterr().out
    assert "CONFIRMED" in printed
    assert "UNKNOWN" in printed


def test_rendered_text_names_undetermined_requirements(installation: Path) -> None:
    """The report must say out loud what it did not check."""
    context = resolve_context(installation)
    results = main_module.run_all_checks(context)
    report = build_report(context, results, None, None, None, tool_version="test")
    text = render_text(report, results, None, None, None)

    assert "REQUIREMENTS THIS REPORT DOES NOT ASSERT" in text
    assert "OVERALL COMPATIBILITY" in text
    assert text.isascii()
