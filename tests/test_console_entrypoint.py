from __future__ import annotations

from unittest.mock import Mock, patch

import main


def test_console_entrypoint_runs_interactive_cli_and_always_shuts_down() -> None:
    system = Mock()
    with (
        patch("main._create_detection_system", return_value=system),
        patch("main.run_cli") as run_cli,
    ):
        main.main([])

    run_cli.assert_called_once_with(system)
    system.shutdown.assert_called_once_with()


def test_console_entrypoint_passes_detection_arguments() -> None:
    result = Mock(status="PASS")
    system = Mock()
    system.detect.return_value = result
    with (
        patch("main._create_detection_system", return_value=system),
        patch("core.format_result.format_detection_result", return_value="PASS"),
        patch("builtins.print"),
    ):
        main.main(["--product", "Cable1", "--area", "A", "--type", "yolo"])

    system.detect.assert_called_once_with("Cable1", "A", "yolo", frame=None)
    system.shutdown.assert_called_once_with()
