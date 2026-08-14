from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import app.acceptance.headless as headless
import app.cli as operator_cli


def _headless_args(tmp_path: Path, *, color: bool = True) -> list[str]:
    weight = tmp_path / "candidate.pt"
    config = tmp_path / "config.yaml"
    color_model = tmp_path / "color.json"
    weight.write_bytes(b"weight")
    config.write_text(
        "weights: candidate.pt\n"
        "enable_color_check: true\n"
        "color_checker_type: stats\n"
        "color_model_path: color.json\n",
        encoding="utf-8",
    )
    color_model.write_bytes(b"color")
    args = [
        "--project-root",
        str(tmp_path),
        "--models-root",
        str(tmp_path / "models"),
        "--global-config",
        str(config),
        "--color-revisions-root",
        str(tmp_path / ".color_revisions"),
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--snapshot-manifest",
        str(tmp_path / "manifest.json"),
        "--report",
        str(tmp_path / "report.json"),
        "--product",
        "Cable1",
        "--area",
        "A",
        "--candidate-version",
        "v9",
        "--candidate-weight",
        str(weight),
        "--candidate-config",
        str(config),
        "--min-confirmed",
        "12",
        "--max-false-positives",
        "1",
        "--max-false-negatives",
        "0",
        "--allow-pending",
        "--allow-errors",
    ]
    if color:
        args.extend(("--color-model", str(color_model)))
    return args


@pytest.mark.parametrize(
    ("passed", "expected_exit"),
    ((True, 0), (False, 2)),
)
def test_headless_main_builds_hashed_contract_and_reports_result(
    monkeypatch,
    tmp_path: Path,
    capsys,
    passed: bool,
    expected_exit: int,
) -> None:
    captured: dict[str, object] = {}

    def run_candidate_acceptance(**kwargs):
        captured.update(kwargs)
        callback = kwargs["progress_callback"]
        callback(1, 50, "first")
        callback(2, 50, "silent")
        callback(25, 50, "quarter")
        callback(50, 50, "last")
        return SimpleNamespace(
            passed=passed,
            report_path=tmp_path / "report.json",
            failures=("false negatives exceeded",),
        )

    monkeypatch.setattr(headless, "run_candidate_acceptance", run_candidate_acceptance)

    assert headless.main(_headless_args(tmp_path)) == expected_exit

    bundle = captured["artifact_bundle"]
    policy = captured["policy"]
    assert bundle.version == "v9"
    assert bundle.model_weight.sha256 == hashlib.sha256(b"weight").hexdigest()
    assert bundle.model_config.sha256 == hashlib.sha256(
        (tmp_path / "config.yaml").read_bytes()
    ).hexdigest()
    assert bundle.color_model.sha256 == hashlib.sha256(b"color").hexdigest()
    assert bundle.color_model_mode == "override"
    assert policy.min_confirmed == 12
    assert not policy.require_all_confirmed
    assert not policy.require_no_errors
    assert bundle.color_revision_overrides == ()
    assert bundle.include_active_color_revisions is False
    assert captured["color_revision_contract"]["entries"] == []
    output = capsys.readouterr()
    assert "1/50" in output.out
    assert "25/50" in output.out
    assert "50/50" in output.out
    if passed:
        assert "PASSED" in output.out
    else:
        assert "BLOCKED" in output.err


def test_headless_optional_color_and_file_guards(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        headless,
        "run_candidate_acceptance",
        lambda **kwargs: captured.update(kwargs)
        or SimpleNamespace(passed=True, report_path="report.json", failures=()),
    )

    assert headless.main(_headless_args(tmp_path, color=False)) == 0
    bundle = captured["artifact_bundle"]
    assert bundle.color_model.sha256 == hashlib.sha256(b"color").hexdigest()
    assert bundle.color_model_mode == "embedded"
    assert headless._required_file(str(tmp_path / "candidate.pt"), "weight").is_file()
    with pytest.raises(FileNotFoundError, match="candidate weight not found"):
        headless._required_file(str(tmp_path / "missing.pt"), "candidate weight")


def _system() -> SimpleNamespace:
    return SimpleNamespace(
        logger=SimpleNamespace(logger=SimpleNamespace(info=Mock(), error=Mock())),
        detect=Mock(return_value=SimpleNamespace(status="PASS")),
        shutdown=Mock(),
    )


def test_operator_cli_rejects_missing_or_empty_model_roots(
    monkeypatch,
    tmp_path: Path,
) -> None:
    system = _system()
    missing = tmp_path / "missing"
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=missing),
    )
    operator_cli.run_cli(system)
    assert "找不到 models" in system.logger.logger.error.call_args.args[0]

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=models),
    )
    operator_cli.run_cli(system)
    assert "未找到任何機種" in system.logger.logger.error.call_args.args[0]


def test_operator_cli_validates_commands_runs_both_backends_and_quits(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    models = tmp_path / "models"
    for inference_type in ("yolo", "anomalib"):
        config = models / "Cable1" / "A" / inference_type / "config.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text("model: test", encoding="utf-8")
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=models),
    )
    answers = iter(
        (
            "Unknown",
            "Cable1",
            "bad-format",
            "B,yolo",
            "A,unsupported",
            "A,yolo",
            "A,anomalib",
            "quit",
        )
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    system = _system()
    system.detect.side_effect = (
        SimpleNamespace(status="ERROR"),
        SimpleNamespace(status="PASS"),
    )

    operator_cli.run_cli(system)

    assert system.detect.call_count == 2
    assert system.detect.call_args_list[0].args == ("Cable1", "A", "yolo")
    assert system.detect.call_args_list[1].args == ("Cable1", "A", "anomalib")
    system.shutdown.assert_called_once()
    output = capsys.readouterr().out
    assert "無此機種" in output
    assert "指令格式錯誤" in output
    assert "無此區域" in output
    assert "推理類型只能是" in output


def test_operator_cli_missing_config_keyboard_interrupt_and_runtime_error(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    models = tmp_path / "models"
    (models / "Cable1" / "A").mkdir(parents=True)
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=models),
    )

    answers = iter(("Cable1", "A,yolo", "quit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    system = _system()
    operator_cli.run_cli(system)
    assert "模型設定不存在" in capsys.readouterr().out

    config = models / "Cable1" / "A" / "yolo" / "config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("model: test", encoding="utf-8")
    answers = iter(("Cable1", "A,yolo", "quit"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    system = _system()
    system.detect.side_effect = RuntimeError("camera offline")
    operator_cli.run_cli(system)
    assert "camera offline" in system.logger.logger.error.call_args.args[0]

    answers = iter(("Cable1", "A,yolo"))
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))
    system = _system()
    system.detect.side_effect = KeyboardInterrupt
    operator_cli.run_cli(system)
    system.shutdown.assert_called_once()
    assert "中止" in capsys.readouterr().out


def test_operator_cli_ctrl_c_at_command_prompt_shuts_down_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    (models / "Cable1" / "A").mkdir(parents=True)
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=models),
    )
    calls = 0

    def interrupt_second_prompt(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return "Cable1"
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt_second_prompt)
    system = _system()
    operator_cli.run_cli(system)
    system.shutdown.assert_called_once()


def test_operator_cli_can_quit_before_selecting_product(
    monkeypatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    (models / "Cable1").mkdir(parents=True)
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=models),
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "quit")
    system = _system()
    operator_cli.run_cli(system)
    system.shutdown.assert_called_once()


def test_operator_cli_ctrl_c_at_product_prompt_shuts_down_cleanly(
    monkeypatch,
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    (models / "Cable1").mkdir(parents=True)
    monkeypatch.setattr(
        operator_cli,
        "load_station_data_paths",
        lambda _root: SimpleNamespace(models=models),
    )
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    system = _system()
    operator_cli.run_cli(system)
    system.shutdown.assert_called_once()
