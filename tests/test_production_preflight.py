from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from core.services.inspection_repository import InspectionRepository
from tools.production_preflight import (
    ProductionPreflightSettings,
    load_settings,
    main,
    run_preflight,
)

_NOW = datetime(2026, 7, 29, tzinfo=timezone.utc)


def _database(result_root: Path) -> Path:
    database_path = result_root / "inspection_records.sqlite3"
    InspectionRepository(database_path).upsert_snapshot(
        {
            "timestamp": _NOW.isoformat(),
            "status": "PASS",
            "detector": "yolo",
            "product": "Cable1",
            "equipment": {"station": "A"},
            "artifacts": {},
        },
        snapshot_path=result_root / "result.json",
    )
    return database_path


def test_preflight_runs_verified_backup_restore_drill(
    tmp_path: Path,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)

    checks = run_preflight(
        ProductionPreflightSettings(
            result_root=result_root,
            min_free_disk_mb=0,
        ),
        backup_restore_drill=True,
        now=_NOW,
    )

    by_name = {check.name: check for check in checks}
    assert by_name["result_folder_writable"].status == "PASS"
    assert by_name["free_disk"].status == "PASS"
    assert by_name["database_integrity"].status == "PASS"
    assert by_name["backup_restore_drill"].status == "PASS"
    assert Path(
        by_name["backup_restore_drill"].detail.removeprefix(
            "verified_backup="
        )
    ).is_file()
    assert list(result_root.glob(".preflight-*.tmp")) == []


def test_preflight_requires_sync_token_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("COMPANY_TOKEN", raising=False)
    checks = run_preflight(
        ProductionPreflightSettings(
            result_root=tmp_path,
            min_free_disk_mb=0,
            sync_enabled=True,
            sync_endpoint="https://company.example/inspections",
            sync_token_env="COMPANY_TOKEN",
        ),
        now=_NOW,
    )

    sync = next(
        check
        for check in checks
        if check.name == "company_sync_configuration"
    )
    assert sync.status == "FAIL"
    assert "COMPANY_TOKEN" in sync.detail


def test_preflight_can_reach_all_pass_release_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    result_root = tmp_path / "Result"
    _database(result_root)
    monkeypatch.setenv("COMPANY_TOKEN", "secret")

    checks = run_preflight(
        ProductionPreflightSettings(
            result_root=result_root,
            min_free_disk_mb=0,
            sync_enabled=True,
            sync_endpoint="https://company.example/inspections",
            sync_token_env="COMPANY_TOKEN",
        ),
        backup_restore_drill=True,
        now=datetime.now(timezone.utc),
    )

    assert {check.status for check in checks} == {"PASS"}


def test_load_settings_reads_global_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
global:
  min_free_disk_mb: 2048
  inspection_backup_interval_hours: 12
  inspection_sync_enabled: true
  inspection_sync_endpoint: https://company.example/inspections
  inspection_sync_api_token_env: COMPANY_TOKEN
""",
        encoding="utf-8",
    )

    settings = load_settings(tmp_path / "Result", config_path=config_path)

    assert settings.min_free_disk_mb == 2048
    assert settings.backup_interval_hours == 12
    assert settings.sync_enabled is True
    assert settings.sync_token_env == "COMPANY_TOKEN"


def test_strict_cli_blocks_unresolved_warnings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("COMPANY_TOKEN", "secret")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "min_free_disk_mb: 0\n"
        "inspection_sync_enabled: true\n"
        "inspection_sync_endpoint: https://company.example/inspections\n"
        "inspection_sync_api_token_env: COMPANY_TOKEN\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--result-root",
            os.fspath(tmp_path / "Result"),
            "--config",
            os.fspath(config_path),
            "--strict",
        ]
    )

    assert exit_code == 1
